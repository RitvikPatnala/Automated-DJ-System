import numpy as np
import librosa

# ---------- small helpers ----------

def spb(bpm: float) -> float:
    return 60.0 / float(bpm)

def ensure_sr(y, sr_from, sr_to):
    return librosa.resample(y, orig_sr=sr_from, target_sr=sr_to) if sr_from != sr_to else y

def ramp_time_stretch(y, sr, rate_start, rate_end, ramp_seconds, chunk_out_s=0.5, xfade_s=0.02):
    if ramp_seconds <= 0 or len(y) == 0:
        return np.zeros(0, dtype=y.dtype), 0

    n_chunks = max(1, int(round(ramp_seconds / chunk_out_s)))
    rates = np.linspace(rate_start, rate_end, n_chunks)
    chunkN = int(round(chunk_out_s * sr))
    xfadeN = int(round(xfade_s * sr))

    out, used, prev = [], 0, None
    for r in rates:
        need_in = int(round(chunkN * r))
        if used + need_in > len(y):
            need_in = max(0, len(y) - used)
        if need_in <= 0:
            break

        seg_in  = y[used:used+need_in]
        seg_out = librosa.effects.time_stretch(y=seg_in, rate=r)

        # length-regularize to exactly chunkN
        if len(seg_out) > chunkN: seg_out = seg_out[:chunkN]
        elif len(seg_out) < chunkN: seg_out = np.pad(seg_out, (0, chunkN - len(seg_out)))

        if prev is None:
            out.append(seg_out)
        else:
            n = min(xfadeN, len(prev), len(seg_out))
            if n > 0:
                fo = np.linspace(1.0, 0.0, n)
                fi = np.linspace(0.0, 1.0, n)
                merged = np.concatenate([
                    prev[:-n],
                    prev[-n:] * fo + seg_out[:n] * fi,
                    seg_out[n:]
                ])
                out[-1] = merged
            else:
                out.append(seg_out)

        prev = out[-1]
        used += need_in

    return (np.concatenate(out) if out else np.zeros(0, dtype=y.dtype)), used


# ---------- main mixer ----------

def render_mix(trackA: dict, trackB: dict, A_mix_time: float, B_start_time: float, xfade_beats: int = 16, ramp_beats: int = 32):
    # 0) unify sample rates (use A's SR as master)
    sr = int(trackA["sr"])
    yA = np.asarray(trackA["y"], dtype=np.float32)
    yB = ensure_sr(np.asarray(trackB["y"], dtype=np.float32), trackB["sr"], sr)

    # 1) derive lengths
    spbA = spb(trackA["bpm"])
    L = int(round(xfade_beats * spbA * sr))  

    A_start = int(round(A_mix_time * sr))
    A_end   = A_start + L
    if A_start < 0: A_start = 0
    if A_end > len(yA):
        L = max(1, len(yA) - A_start)
        A_end = len(yA)

    # 2) envelopes
    fade_out = np.linspace(1.0, 0.0, L, dtype=np.float32)
    fade_in  = np.linspace(0.0, 1.0, L, dtype=np.float32)

    # 3) A segments
    A_pre  = yA[:A_start]
    A_xseg = yA[A_start:A_end] * fade_out

    # 4) BPM Matching
    rate_match = float(trackA["bpm"]) / float(trackB["bpm"])

    B_start_idx = int(round(B_start_time * sr))
    B_src_tail  = yB[B_start_idx:]  # B from its cue onward

    need_in = int(round(L * rate_match))
    if len(B_src_tail) < need_in:
        B_src_tail = np.pad(B_src_tail, (0, int(need_in - len(B_src_tail))))

    B_x_in  = B_src_tail[:need_in]
    B_x_out = librosa.effects.time_stretch(y=B_x_in, rate=rate_match)
    if len(B_x_out) > L: B_x_out = B_x_out[:L]
    elif len(B_x_out) < L: B_x_out = np.pad(B_x_out, (0, L - len(B_x_out)))
    B_xseg = B_x_out * fade_in

    # 5) After crossfade
    ramp_seconds = ramp_beats * spbA
    B_after_src = B_src_tail[need_in:]
    B_ramp_out, used = ramp_time_stretch(B_after_src, sr, rate_start=rate_match, rate_end=1.0, ramp_seconds=ramp_seconds, chunk_out_s=0.5, xfade_s=0.02)
    B_post = np.concatenate([B_ramp_out, B_after_src[used:]])

    # 6) Final Mix
    mix = np.concatenate([A_pre, yA[A_start:A_end], B_post]).astype(np.float32)

    return mix, {"A_pre": A_pre, "A_xseg": A_xseg, "B_xseg": B_xseg, "B_post": B_post, "sr": sr}
