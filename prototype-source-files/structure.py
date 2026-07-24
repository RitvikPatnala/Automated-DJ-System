import numpy as np
import librosa
from scipy.signal import find_peaks

def extract_rms_mfcc(y, sr, hop_length=512, n_mfcc=20):
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]        
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)            

    def smooth(x, win=4):
        if x.ndim == 1:
            pad = np.pad(x, (win, win), mode='edge')
            x_s = np.convolve(pad, np.ones(win*2+1)/(win*2+1), mode='valid')
            return x_s[:x.shape[0]]
        else:
            out = np.empty_like(x)
            for i in range(x.shape[0]):
                out[i] = smooth(x[i], win=win)
            return out

    rms_s = smooth(rms, win=4)           
    mfcc_s = smooth(mfcc, win=4)        

    mfcc_s = librosa.util.normalize(mfcc_s, axis=0)

    return rms_s, mfcc_s, hop_length

def ssm_rms(rms_vec):
    T = rms_vec.shape[0]
    diff = np.abs(rms_vec[:, None] - rms_vec[None, :])  
    mx = diff.max() + 1e-6
    sim = 1.0 - (diff / mx)
    return sim.astype(np.float32)

def ssm_mfcc(mfcc_mat):
    # mfcc_mat: (F, T)
    X = mfcc_mat.T                                     
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)
    sim = np.dot(X, X.T)                        
    return sim.astype(np.float32)

def checkerboard_kernel(half_size):
    k = half_size
    K = np.zeros((2*k, 2*k), dtype=np.float32)
    K[:k, :k] = 1.0
    K[k:, k:] = 1.0
    K[:k, k:] = -1.0
    K[k:, :k] = -1.0
    return K

def novelty_from_ssm(ssm, half_size=16):
    T = ssm.shape[0]
    K = checkerboard_kernel(half_size)
    k = half_size
    novelty = np.zeros(T, dtype=np.float32)

    for t in range(k, T - k):
        patch = ssm[t-k:t+k, t-k:t+k]
        novelty[t] = np.sum(patch * K)

    mn, mx = novelty.min(), novelty.max()
    if mx > mn:
        novelty = (novelty - mn) / (mx - mn)
    else:
        novelty[:] = 0.0
    return novelty

def merge_novelty(nov_rms, nov_mfcc):
    return np.sqrt(nov_rms * nov_mfcc)

def pick_novelty_peaks(novelty, distance=10, top_k=12, height=0.2):
    peaks, _ = find_peaks(novelty, distance=distance, height=height)
    if peaks.size == 0:
        return peaks
    if peaks.size > top_k:
        # keep the biggest ones
        idx = np.argsort(novelty[peaks])[::-1][:top_k]
        peaks = np.sort(peaks[idx])
    return peaks
def ensure_start_boundary(snapped, downbeat_times=None, beat_times=None):
    if downbeat_times is not None and len(downbeat_times) > 0:
        start = float(downbeat_times[0])
    elif beat_times is not None and len(beat_times) > 0:
        start = float(beat_times[0])
    else:
        start = 0.0

    snapped.append(start)
    return snapped

def normalize_downbeats(bpm, beat_times, downbeat_times):
    if (len(beat_times) < 2) or (len(downbeat_times) < 2):
        return downbeat_times

    spb = np.median(np.diff(beat_times))                
    s_per_bar = 4 * spb                            
    db_diff = np.median(np.diff(downbeat_times))        

    if (0.45 * s_per_bar) <= db_diff <= 0.6 * s_per_bar:
        return np.asarray(downbeat_times)[::2]

    if (1.8 * s_per_bar) <= db_diff <= 2.4 * s_per_bar:
        db = np.asarray(downbeat_times)
        # insert midpoints between downbeats to halve spacing
        mids = (db[:-1] + db[1:]) / 2.0
        return np.sort(np.concatenate([db, mids]))

    return downbeat_times

def snap_peaks_to_downbeat_phrases(boundary_times, beat_times, downbeat_times, phrase_bars=8, tol_beats=0.5):

    if len(boundary_times) == 0 or len(beat_times) < 2 or len(downbeat_times) == 0:
        return np.asarray([], dtype=float)

    beat_times = np.asarray(beat_times, float)
    downbeat_times = np.asarray(downbeat_times, float)
    spb = np.median(np.diff(beat_times))  
    tol_sec = tol_beats * spb

    phrase_db = downbeat_times[::phrase_bars] if len(downbeat_times) >= phrase_bars else downbeat_times

    snapped = []
    for t in np.asarray(boundary_times, float):
        i = int(np.argmin(np.abs(phrase_db - t)))
        if abs(phrase_db[i] - t) <= tol_sec:
            snapped.append(float(phrase_db[i]))

    snapped.append(float(phrase_db[0]))
    snapped = np.unique(np.round(snapped, 6))
    snapped.sort()
    return snapped

def make_sections_from_foote(y, sr, beat_times, downbeat_times, phrase_bars=8,hop_length=512, n_mfcc=20, half_kernel=16, top_k=12):

        # 1) frame features
    rms_s, mfcc_s, hop_length = extract_rms_mfcc(
        y, sr,
        hop_length=hop_length,
        n_mfcc=n_mfcc
    )

    # 2) SSMs
    ssm_r = ssm_rms(rms_s)
    ssm_m = ssm_mfcc(mfcc_s)

    # 3) novelty per SSM
    nov_r = novelty_from_ssm(ssm_r, half_size=half_kernel)
    nov_m = novelty_from_ssm(ssm_m, half_size=half_kernel)

    # 4) merge with geometric mean
    nov = merge_novelty(nov_r, nov_m)

    # 5) pick boundary frames
    boundary_frames = pick_novelty_peaks(nov, distance=10, top_k=top_k, height=0.2)

    # 6) convert to seconds
    boundary_times = librosa.frames_to_time(boundary_frames, sr=sr, hop_length=hop_length)

    bpm_est = 60.0 / np.median(np.diff(beat_times)) if len(beat_times) > 1 else 120.0
    downbeat_times = normalize_downbeats(bpm_est, beat_times, downbeat_times)

    section_times = snap_peaks_to_downbeat_phrases( boundary_times, beat_times, downbeat_times, phrase_bars=phrase_bars, tol_beats=0.5)

    if section_times.size < 2 and len(downbeat_times) > 0 and len(beat_times) > 1:
        spb = np.median(np.diff(beat_times))
        phrase_len_sec = phrase_bars * 4 * spb
        t0 = float(downbeat_times[0])
        t_end = len(y) / sr
        section_times = np.arange(t0, t_end, phrase_len_sec)
        section_times = np.unique(np.round(section_times, 6))

    return section_times