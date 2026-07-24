import librosa
import numpy as np

def compute_section_energy_labels_rms(y, sr, section_times, hop_length=512):

    # Compute RMS 
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # Section Boundaries
    bounds = np.unique(np.clip(np.concatenate([section_times, [len(y)/sr]]), 0, len(y)/sr))

    # Compute median RMS per section
    scores = []
    sections = []
    for i in range(len(bounds) - 1):
        t0, t1 = float(bounds[i]), float(bounds[i+1])
        m = (times >= t0) & (times < t1)
        s = float(np.median(rms[m])) if np.any(m) else 0.0
        scores.append(s)
        sections.append({"start": t0, "end": t1, "label": None, "score": s})

    # Normalize scores
    scores = np.array(scores)
    mu, sd = np.mean(scores), np.std(scores) + 1e-9
    z = (scores - mu) / sd

    # Threshold
    for i, zi in enumerate(z):
        sections[i]["label"] = "high" if zi > 0 else "low"

    return sections

import numpy as np
import librosa

# ----------------- helpers -----------------

def snap_to_downbeat(downbeat_times, t):
    if downbeat_times is None:
        return float(t)
    db = np.asarray(downbeat_times)
    if db.size == 0:
        return float(t)
    return float(db[np.argmin(np.abs(db - t))])


def first_with_label(sections, label):
    return next((s for s in sections if s["label"] == label), None)

def last_with_label(sections, label):
    for s in reversed(sections):
        if s["label"] == label:
            return s
    return None

def largest_with_label(sections, label):
    cands = [s for s in sections if s["label"] == label]
    if not cands:
        return None
    return max(cands, key=lambda s: (s["end"] - s["start"]))

def peak_downbeat_in_span(track, t0, t1, hop_length=512):
    y, sr = track["y"], track["sr"]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(len(onset)), sr=sr, hop_length=hop_length)
    m = (times >= t0) & (times < t1)
    if np.any(m):
        peak_t = float(times[m][np.argmax(onset[m])])
    else:
        peak_t = float(t0)
    return snap_to_downbeat(track["downbeat_times"], peak_t)

def time_to_beat_index(t, beat_times):
    i = int(np.searchsorted(beat_times, t, side="left"))
    if i == len(beat_times):
        i -= 1
    elif i > 0 and (t - beat_times[i-1]) <= (beat_times[i] - t):
        i -= 1
    return i

def beat_to_bar_index(beat_idx, beats_per_bar=4):
    return beat_idx // beats_per_bar

def snap_time_to_previous_phrase_start(t, beat_times, phrase_bars=8, beats_per_bar=4):
    """Return the time of the **previous** phrase start at/ before t."""
    if len(beat_times) == 0:
        return float(t)
    i = time_to_beat_index(t, beat_times)
    bar_idx = beat_to_bar_index(i, beats_per_bar=beats_per_bar)
    # move back to the nearest phrase boundary ≤ current bar
    phrase_bar = bar_idx - (bar_idx % phrase_bars)
    phrase_beat_idx = phrase_bar * beats_per_bar
    phrase_beat_idx = max(0, min(phrase_beat_idx, len(beat_times)-1))
    return float(beat_times[phrase_beat_idx])

def choose_cue_points_from_energy(trackA, trackB, transition_type="relaxed",
                                  phrase_bars=8, beats_per_bar=4, hop_length=512):

    def phrase_start_downbeats_in_span(track, t0, t1):
        bt = np.asarray(track["beat_times"], float)
        db = np.asarray(track["downbeat_times"], float)
        if db.size == 0 or bt.size == 0:
            return np.asarray([], float)

        # map each downbeat to nearest beat index -> bar index -> keep phrase starts
        db_idx = [time_to_beat_index(t, bt) for t in db]
        bar_idx = np.asarray([beat_to_bar_index(i, beats_per_bar) for i in db_idx])
        mask_phrase = (bar_idx % phrase_bars) == 0
        cand = db[mask_phrase]
        return cand[(cand >= t0) & (cand < t1)]

    def onset_at_times(y, sr, times, hop_length=512):
        """Sample onset strength at arbitrary 'times' (nearest frame)."""
        if len(times) == 0:
            return np.asarray([], float)
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        onset_t = librosa.frames_to_time(np.arange(len(onset)), sr=sr, hop_length=hop_length)
        idx = np.searchsorted(onset_t, times, side="left").clip(0, len(onset_t)-1)
        return onset[idx]

    def pick_phrase_drop(track, span_start, span_end):
        y, sr = track["y"], track["sr"]
        bt = np.asarray(track["beat_times"], float)

        cands = phrase_start_downbeats_in_span(track, span_start, span_end)

        if cands.size > 0:
            scores = onset_at_times(y, sr, cands, hop_length=hop_length)
            return float(cands[int(np.argmax(scores))])

        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        onset_t = librosa.frames_to_time(np.arange(len(onset)), sr=sr, hop_length=hop_length)
        m = (onset_t >= span_start) & (onset_t < span_end)
        if np.any(m):
            peak_t = float(onset_t[m][np.argmax(onset[m])])
        else:
            peak_t = float(span_start)

        t_prev_phrase = snap_time_to_previous_phrase_start(peak_t, bt, phrase_bars, beats_per_bar)

        # ensure it's inside the span
        spb = np.median(np.diff(bt)) if len(bt) > 1 else (60.0 / float(track.get("bpm", 120.0)))
        phrase_len_sec = phrase_bars * beats_per_bar * spb
        in_span = (t_prev_phrase >= span_start) and (t_prev_phrase < span_end)
        if in_span:
            return float(t_prev_phrase)

        t_next_phrase = t_prev_phrase + phrase_len_sec
        if (t_next_phrase >= span_start) and (t_next_phrase < span_end):
            return float(t_next_phrase)

        first_phrase_after = snap_time_to_previous_phrase_start(span_start + 1e-6, bt, phrase_bars, beats_per_bar)
        if first_phrase_after < span_start:
            first_phrase_after += phrase_len_sec
        return float(first_phrase_after)

    # ---------- choose sections by energy ----------
    secA = trackA["energy_sections"]
    secB = trackB["energy_sections"]
    bpmA = float(trackA["bpm"]); spbA = 60.0 / bpmA

    tt = transition_type.lower()
    assert tt in {"relaxed", "rolling", "double_drop"}

    if tt == "relaxed":
        a_low = last_with_label(secA, "low") or secA[-1]
        A_mix_time = max(0.0, a_low["start"] - (phrase_bars * beats_per_bar) * spbA)
        A_mix_time = snap_to_downbeat(trackA["downbeat_times"], A_mix_time)

        b_low = first_with_label(secB, "low") or secB[0]
        B_start_time = snap_to_downbeat(trackB["downbeat_times"], b_low["start"])

        xfade_beats = max(32, phrase_bars * beats_per_bar)
        meta = {"A_section":"low(outro-ish)", "B_section":"low(intro-ish)"}

    elif tt == "rolling":
        a_high = largest_with_label(secA, "high") or secA[0]
        b_high = largest_with_label(secB, "high") or secB[0]

        A_mix_time = snap_to_downbeat(trackA["downbeat_times"], a_high["start"])
        B_start_time = snap_to_downbeat(trackB["downbeat_times"], b_high["start"])

        xfade_beats = max(16, phrase_bars * beats_per_bar // 2)
        meta = {"A_section":"high(main)", "B_section":"high(drop/main)"}

    else:
        a_high = largest_with_label(secA, "high") or secA[0]
        b_high = largest_with_label(secB, "high") or secB[0]

        A_drop = pick_phrase_drop(trackA, a_high["start"], a_high["start"]+3)
        B_drop = pick_phrase_drop(trackB, b_high["start"]+3, b_high["end"])

        A_mix_time  = float(A_drop)
        B_start_time = float(B_drop)

        xfade_beats = 4  # tight slam
        meta = {"align":"drop==drop (phrase-start)", "A_span":[a_high["start"], a_high["end"]],
                "B_span":[b_high["start"], b_high["end"]]}

    btA = np.asarray(trackA["beat_times"], float)
    btB = np.asarray(trackB["beat_times"], float)

    A_mix_time  = snap_time_to_previous_phrase_start(A_mix_time,  btA, phrase_bars, beats_per_bar)
    B_start_time = snap_time_to_previous_phrase_start(B_start_time, btB, phrase_bars, beats_per_bar)
    # B_start_time = _align_B_to_A_phrase(A_mix_time, trackA, trackB, phrase_bars, beats_per_bar)


    return float(A_mix_time), float(B_start_time), int(xfade_beats), meta
