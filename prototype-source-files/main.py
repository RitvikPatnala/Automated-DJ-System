import librosa
import numpy as np
import soundfile as sf
import mixing
import analysis_funcs
import track_select
import cue_point_selection
import matplotlib.pyplot as plt


trackA = analysis_funcs.analyze_track("/Users/ritvik/Documents/UIUC_Fall_2025/ECE 420/Automated DJ System/data/Martin Garrix - Animals (Official Video).mp3")
trackB = analysis_funcs.analyze_track("/Users/ritvik/Documents/UIUC_Fall_2025/ECE 420/Automated DJ System/data/Zedd - Clarity ft. Foxes.mp3")

A_mix_time, B_start_time, xfade_beats_1, meta_1 = cue_point_selection.choose_cue_points_from_energy(
    trackA, trackB, transition_type="relaxed"
)

final_mix_1, segs = mixing.render_mix(trackA, trackB,
                 A_mix_time=A_mix_time,
                 B_start_time=B_start_time,
                 xfade_beats=xfade_beats_1,
                 ramp_beats=32)

sf.write("auto_mix_1.wav", final_mix_1, trackA["sr"], subtype="PCM_16")

sr      = segs["sr"]
A_pre   = segs["A_pre"]
A_xseg  = segs["A_xseg"]
B_xseg  = segs["B_xseg"]
B_post  = segs["B_post"]

mix_len = len(final_mix_1)
a_contrib = np.zeros(mix_len, dtype=np.float32)
b_contrib = np.zeros(mix_len, dtype=np.float32)

# Segment boundaries in the final mix:
i = 0
# A_pre occupies [0 : len(A_pre)]
a_contrib[i:i+len(A_pre)] = A_pre
i += len(A_pre)

# Crossfade region occupies [i : i+len(A_xseg)]
a_contrib[i:i+len(A_xseg)] += A_xseg
b_contrib[i:i+len(B_xseg)] += B_xseg
i += len(A_xseg)

# Post region occupies [i : end]
b_contrib[i:i+len(B_post)] += B_post

# --- Time axis in seconds ---
t = np.arange(mix_len) / sr

# --- Plot (2 colors; shaded overlap) ---
plt.figure(figsize=(14, 4))
plt.plot(t, a_contrib, label="Track A contribution", linewidth=0.8, color="#1f77b4")
plt.plot(t, b_contrib, label="Track B contribution", linewidth=0.8, color="#ff7f0e")

# Shade the crossfade/overlap window
xfade_start_s = len(A_pre) / sr
xfade_end_s   = (len(A_pre) + len(A_xseg)) / sr
plt.axvspan(xfade_start_s, xfade_end_s, alpha=0.15, color="gray", label="Crossfade")

plt.title("Automix Waveform: Track Contributions and Overlap")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()


# trackA = analysis_funcs.analyze_track("/Users/ritvik/Documents/UIUC_Fall_2025/ECE 420/Automated DJ System/data/Martin Garrix - Animals (Official Video).mp3")
# trackB = analysis_funcs.analyze_track("/Users/ritvik/Documents/UIUC_Fall_2025/ECE 420/Automated DJ System/data/Zedd - Clarity ft. Foxes.mp3")


# A_mix_time, B_start_time, xfade_beats_1, meta_1 = cue_point_selection.choose_cue_points_from_energy(
#     trackA, trackB, transition_type="rolling"
# )

# final_mix_2 = mixing.render_mix(trackA, trackB,
#                  A_mix_time=A_mix_time,
#                  B_start_time=B_start_time,
#                  xfade_beats=xfade_beats_1,
#                  ramp_beats=32)

# sf.write("auto_mix_2.wav", final_mix_2, trackA["sr"], subtype="PCM_16")

# trackA = analysis_funcs.analyze_track("/Users/ritvik/Documents/UIUC_Fall_2025/ECE 420/Automated DJ System/data/Martin Garrix - Animals (Official Video).mp3")
# trackB = analysis_funcs.analyze_track("/Users/ritvik/Documents/UIUC_Fall_2025/ECE 420/Automated DJ System/data/Don't You Worry Child (Radio Edit).mp3")

# A_mix_time, B_start_time, xfade_beats_1, meta_1 = cue_point_selection.choose_cue_points_from_energy(
#     trackA, trackB, transition_type="double_drop"
# )

# final_mix_3 = mixing.render_mix(trackA, trackB,
#                  A_mix_time=A_mix_time,
#                  B_start_time=B_start_time,
#                  xfade_beats=xfade_beats_1,
#                  ramp_beats=32)

# sf.write("auto_mix_3.wav", final_mix_3, trackA["sr"], subtype="PCM_16")

