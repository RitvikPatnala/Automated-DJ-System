# Automated-DJ-System

An Android app that automatically mixes two songs like a DJ would — detecting beats, structural sections, and energy levels through digital signal processing, then selecting musically coherent cue points and crossfading between tracks with tempo matching.

<img width="316" height="267" alt="graphoftempo" src="https://github.com/user-attachments/assets/abdca8f8-504a-4505-9482-c70533c53629" />

## How It Works

The pipeline runs in six stages: audio is loaded and converted to a mel spectrogram (STFT, 40 mel bands) --> an onset detection function is computed and autocorrelated to estimate tempo and beat positions --> the track is segmented into sections and each is classified as high/low energy --> cue points are chosen based on the desired transition type (relaxed, energetic, or smashcut) --> the mix is rendered with beat-matched time-stretching, an equal-power crossfade, and tempo ramping.

Architecture is a hybrid Android app: UI and control logic in Java, with all DSP (spectrogram computation, beat tracking, segmentation, mixing) implemented in native C++ for performance, connected via JNI.

**Original Contribution** The reference paper this was based on assumes similar-BPM tracks so the tempo change after a transition is imperceptible. To generalize to any pair of songs, we added a  tempo-ramping function, a cubic easing curve that gradually returns the second track to its original BPM after the crossfade, instead of snapping back abruptly.

## Validation

Beat tracking was checked against external BPM references (Tunebat) across several genres:

| Track | Detected BPM | Reference BPM |
|---|---|---|
| The Chainsmokers & Coldplay – Something Just Like This | 103.36 | 103 |
| Rihanna – We Found Love | 127.60 | 128 |
| Skrillex – Reptile Theme | 109.96 | 110 |
| Avicii – Levels | 126.05 | 126 |

Crossfade and tempo-ramp behavior were validated with waveform visualizations and listening tests isolating each component's perceptual contribution.

<img width="340" height="186" alt="Mel_Spectrograms" src="https://github.com/user-attachments/assets/f14e9bae-b7f1-45ee-a4ba-995f4b4110b1" />

<img width="298" height="220" alt="Section_Energy" src="https://github.com/user-attachments/assets/55ef61a9-17f0-4d84-8e48-2a05d113cb7f" />

<img width="310" height="233" alt="sectionsoffinalmix" src="https://github.com/user-attachments/assets/0edbacd2-0a61-4a77-af1b-823e74e1a392" />



