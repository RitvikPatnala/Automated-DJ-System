import librosa
import beat_track
import structure
import numpy as np
import cue_point_selection

def load_audio(path, sr=44100):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

def extract_beats_custom(y, sr):
    # 1) Base beat times/BPM from librosa (or your custom method)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Early out if too few beats
    if len(beat_times) < 8:
        return float(tempo), beat_times, np.asarray([])

    # 2) Low-frequency energy per beat (proxy for kick strength)
    #    Use STFT mag and sum bins below ~150 Hz
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    lf_mask = freqs <= 150.0
    lf_env = S[lf_mask].sum(axis=0)        # frame-wise LF energy
    lf_times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=512)

    # Map each beat to LF energy by nearest STFT frame
    beat_idx_in_stft = np.searchsorted(lf_times, beat_times, side="left").clip(0, len(lf_times)-1)
    beat_lf = lf_env[beat_idx_in_stft]

    # 3) Pick 1-of-4 offset with maximum average LF energy at downbeats
    best_offset = 0
    best_score = -np.inf
    for off in range(4):
        idx = np.arange(off, len(beat_times), 4)
        if len(idx) == 0: 
            continue
        score = float(np.mean(beat_lf[idx]))
        if score > best_score:
            best_score = score
            best_offset = off

    downbeat_times = beat_times[best_offset::4]

    # 4) Half/double-time normalization of downbeats (bars ≈ 4 beats)
    spb = np.median(np.diff(beat_times))
    s_per_bar = 4 * spb
    db_diff = np.median(np.diff(downbeat_times)) if len(downbeat_times) > 1 else s_per_bar
    if 0.45 * s_per_bar <= db_diff <= 0.6 * s_per_bar:
        # double-time -> keep every other downbeat
        downbeat_times = downbeat_times[::2]
    elif 1.8 * s_per_bar <= db_diff <= 2.4 * s_per_bar:
        # half-time -> interpolate midpoints
        if len(downbeat_times) > 1:
            mids = (downbeat_times[:-1] + downbeat_times[1:]) / 2.0
            downbeat_times = np.sort(np.concatenate([downbeat_times, mids]))

    # 5) Return BPM measured from beats (or tempo)
    bpm = 60.0 / spb if spb > 0 else float(tempo)
    return float(bpm), beat_times, downbeat_times

def analyze_track(path):
    y, sr = load_audio(path, sr=None)
    print(len(y)/sr, "seconds")
    
    
    
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    print("librosa's bpm:", tempo)

    bpm, beat_times, downbeat_times = extract_beats_custom(y, sr)

    section_times = structure.make_sections_from_foote(y, sr, beat_times, downbeat_times)
    energy_sections = cue_point_selection.compute_section_energy_labels_rms(y, sr, section_times)

    return {
        "path": path,
        "y": y,
        "sr": sr,
        "bpm": float(bpm),
        "beat_times": beat_times,
        "downbeat_times": downbeat_times,
        "section_times": section_times,
        "energy_sections": energy_sections
    }