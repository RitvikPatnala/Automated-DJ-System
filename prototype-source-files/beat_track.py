import librosa
import numpy as np

def compute_odf_mel(y, sr, n_mels=40, hop_length=512, n_fft=2048):
    # mel spectrogram: shape (n_mels, T)
    S = librosa.feature.melspectrogram(y=y,sr=sr,n_fft=n_fft,hop_length=hop_length,n_mels=n_mels,power=2.0)

    dS = np.diff(S, axis=1)
    # half-wave rectify: max(x, 0)
    dS = np.maximum(dS, 0.0)
    # sum over mel bins -> onset detection function
    odf = dS.sum(axis=0)  # shape (T-1,)
    return odf, hop_length

#autocorrelation + tempo detection curve
def autocorr_odf(odf, max_lag):
    M = len(odf)
    A = np.zeros(max_lag)
    for lag in range(1, max_lag):
        # up to M - lag terms
        product = odf[:M - lag] * odf[lag:]
        A[lag] = product.mean()
    return A

##tempo detection curve
def estimate_tempo_from_acf(A, sr, hop_length, bpm_min=60, bpm_max=200):
    min_lag = int(np.floor((60.0 / bpm_max) * sr / hop_length))
    max_lag = int(np.ceil((60.0 / bpm_min) * sr / hop_length))

    max_lag = min(max_lag, len(A) - 1)
    min_lag = max(min_lag, 1)

    best_lag = min_lag
    best_score = -1.0

    for L in range(min_lag, max_lag + 1):
        # build B(L)
        multiples = A[L::L]
        if len(multiples) == 0:
            continue
        B_L = multiples.mean()
        if B_L > best_score:
            best_score = B_L
            best_lag = L

    # convert lag to bpm
    beat_period_sec = best_lag * hop_length / float(sr)
    bpm = 60.0 / beat_period_sec
    return bpm, best_lag

def estimate_phase(odf, beat_lag):
    T = len(odf)
    best_phi = 0
    best_sum = -1.0
    for phi in range(beat_lag):
        vals = odf[phi:T:beat_lag]
        s = vals.sum()
        if s > best_sum:
            best_sum = s
            best_phi = phi
    return best_phi

def odf_beattrack(y, sr, n_mels=40, hop_length=512, bpm_min=60, bpm_max=200):
    # 1) ODF
    odf, hop_length = compute_odf_mel(
        y, sr,
        n_mels=n_mels,
        hop_length=hop_length
    )
    # 2) autocorrelation
    max_lag = int((60.0 / bpm_min) * sr / hop_length) + 1
    A = autocorr_odf(odf, max_lag)

    # 3) tempo detection curve
    bpm, beat_lag = estimate_tempo_from_acf(
        A, sr, hop_length,
        bpm_min=bpm_min,
        bpm_max=bpm_max
    )

    # 4) phase
    phi = estimate_phase(odf, beat_lag)

    # 5) construct beat frame indices
    odf_len = len(odf)
    beat_frames = np.arange(phi, odf_len, beat_lag)

    # 6) convert to seconds
    beat_times = librosa.frames_to_time(beat_frames,
                                        sr=sr,
                                        hop_length=hop_length)

    return bpm, beat_times

def per_beat_energy(y, sr, beat_times, frame_length=2048, hop_length=512):
    energies = []
    for t in beat_times:
        center = int(t * sr)
        start = max(0, center - frame_length // 2)
        end = min(len(y), center + frame_length // 2)
        frame = y[start:end]
        if len(frame) == 0:
            energies.append(0.0)
        else:
            energies.append(np.sqrt(np.mean(frame**2)))
    return np.array(energies, dtype=float)

def estimate_downbeats(y, sr, beat_times):
    if len(beat_times) == 0:
        return np.array([])

    energies = per_beat_energy(y, sr, beat_times)

    best_offset = 0
    best_score = -1.0
    for offset in range(4):
        idx = np.arange(offset, len(beat_times), 4)
        score = energies[idx].sum()
        if score > best_score:
            best_score = score
            best_offset = offset

    downbeat_times = beat_times[best_offset::4]
    return downbeat_times
