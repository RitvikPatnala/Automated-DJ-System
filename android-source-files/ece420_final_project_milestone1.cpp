


#include <jni.h>
#include <string>
#include <vector>
#include <cstdint>
#include <fstream>
#include <cmath>
#include <cstring>
#include <android/log.h>
#include "kiss_fft/kiss_fft.h"


#define LOG_TAG "SPECTRO-NATIVE"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)


// -------------------- Global constants --------------------


static const int FRAME_SIZE = 1024;
static const int ZP_FACTOR = 2;
static const int FFT_SIZE  = FRAME_SIZE * ZP_FACTOR;


// We want 40 mel bands
static const int NUM_MEL  = 40;
static const int NUM_BINS  = NUM_MEL;  // must match MainActivity.NUM_BINS
static const int N_FFT_BINS = FFT_SIZE / 2 + 1; // we only use 0..Nyquist


// Global spectrogram storage: time-major [frame][mel_bin]
static int gNumFrames = 0;
static std::vector<float> gSpectrogram; // size = gNumFrames * NUM_BINS


static std::vector<float> gOnset;


// Mel filterbank: [NUM_MEL][N_FFT_BINS]
static std::vector<float> gMelFilterbank; // flat row-major
static int gMelFbSampleRate = 0;


// Global onset detection function Γ(m)
static std::vector<float> gOnsetCurve;
static int gOnsetLength = 0;


// For later tempo → beat-time mapping
static int gSampleRateGlobal = 0;
static const int gHopSamples = FRAME_SIZE / 2;  // same hop used in STFT


// ---- Beat tracking globals ----
static float gEstimatedBpm = 0.0f;
static std::vector<float> gBeatTimesSec;  // beat times in seconds


// ---- Section boundaries ----
static std::vector<float> gSectionBoundariesSec;  // times in seconds


static std::vector<int> gSectionEnergyLabels;

static std::vector<float> gMixBeatTimes;

// +1 = HIGH energy, -1 = LOW energy


static float gCuePointRelaxed = -1.f;
static float gCuePointEnergetic = -1.f;
static float gCuePointSmashcut = -1.f;
static float gMixTransitionEndSec = -1.f;



static bool gSmashcutFirstTrack = true;

static std::vector<float> gMixBpmTimeline;
static std::vector<float> gMixBpmTimeSec;
static float gEstimatedBpmA = 0.f;
static float gEstimatedBpmB = 0.f;
static int gCurrentTransitionType = 1; // default RELAXED





// Helpers: WAV loading
#include <algorithm> // for std::max, std::min


// onset: length N = gNumFrames
static void computeTempoAndBeatsFromOnset(const std::vector<float>& onset, int sampleRate, int hopSize) {
    LOGD("DEBUG: COMPUTING TEMPO AND BEATS FROM ONSET");
    gEstimatedBpm = 0.0f;
    gBeatTimesSec.clear();


    const int N = (int)onset.size();
    if (N < 4) return;


    // 1) Autocorrelation over a tempo range
    const float minBpm = 60.0f;
    const float maxBpm = 180.0f;


    // lag (in frames) for these BPMs:
    // lag = 60 * fs / (BPM * hopSize)
    int minLag = (int)std::round(60.0f * sampleRate / (maxBpm * hopSize));
    int maxLag = (int)std::round(60.0f * sampleRate / (minBpm * hopSize));


    minLag = std::max(minLag, 1);
    maxLag = std::min(maxLag, N - 1);
    if (minLag >= maxLag) return;


    std::vector<float> acf(maxLag + 1, 0.0f);


    for (int k = minLag; k <= maxLag; ++k) {
        float s = 0.0f;
        for (int t = k; t < N; ++t) {
            s += onset[t] * onset[t - k];
        }
        acf[k] = s;
    }


    // Find best lag by argmax of ACF in the range
    int bestLag = minLag;
    float bestVal = acf[minLag];
    for (int k = minLag + 1; k <= maxLag; ++k) {
        if (acf[k] > bestVal) {
            bestVal = acf[k];
            bestLag = k;
        }
    }


    if (bestLag <= 0) return;


    // Convert lag to BPM
    gEstimatedBpm = 60.0f * sampleRate / (bestLag * hopSize);
    if (gEstimatedBpm < 70.0f){
        gEstimatedBpm = 2*gEstimatedBpm;
    }


    // Phase scan for best alignment
    const float twoPiOverLag = 2.0f * M_PI / (float)bestLag;
    int bestPhi = 0;
    float bestScore = -1e30f;


    for (int phi = 0; phi < bestLag; ++phi) {
        float score = 0.0f;
        for (int t = 0; t < N; ++t) {
            float phase = twoPiOverLag * (float)(t - phi);
            score += onset[t] * cosf(phase);
        }
        if (score > bestScore) {
            bestScore = score;
            bestPhi = phi;
        }
    }


    // Beat times in seconds
    const float hopSeconds = (float)hopSize / (float)sampleRate;
    for (int n = 0;; ++n) {
        int idx = bestPhi + n * bestLag;
        if (idx >= N) break;
        float tSec = idx * hopSeconds;
        gBeatTimesSec.push_back(tSec);
    }


    LOGD("Beat tracking: BPM=%.2f, beats=%zu", gEstimatedBpm, gBeatTimesSec.size());
}


// Simple, robust section boundaries based on a fixed number of beats per phrase.
static void computeSectionBoundaries() {
    LOGD("DEBUG: COMPUTING SECTION BOUNDARIES");
    gSectionBoundariesSec.clear();


    const int numBeats = (int)gBeatTimesSec.size();
    if (numBeats < 8) {
        LOGD("SectionBoundaries: not enough beats (%d)", numBeats);
        return;
    }


    // Decide phrase length in beats.
    // For shorter songs, use 16; for longer ones, 32.
    int phraseBeats = 16;
    if (numBeats > 200) {
        phraseBeats = 32;
    }


    LOGD("SectionBoundaries: numBeats=%d, phraseBeats=%d", numBeats, phraseBeats);


    // Always include the start of the track as a boundary at t = 0
    gSectionBoundariesSec.push_back(0.0f);


    // Now add a boundary every 'phraseBeats' beats.
    for (int i = phraseBeats; i < numBeats; i += phraseBeats) {
        float tSec = gBeatTimesSec[i]; // beat times are already in seconds
        gSectionBoundariesSec.push_back(tSec);
    }


    // Optionally ensure we have at least 2 boundaries; if not, try halving phrase size
    if (gSectionBoundariesSec.size() < 2 && numBeats >= 8) {
        gSectionBoundariesSec.clear();
        gSectionBoundariesSec.push_back(0.0f);
        phraseBeats = 8;
        for (int i = phraseBeats; i < numBeats; i += phraseBeats) {
            float tSec = gBeatTimesSec[i];
            gSectionBoundariesSec.push_back(tSec);
        }
        LOGD("SectionBoundaries: fallback to phraseBeats=%d, count=%d",
             phraseBeats, (int)gSectionBoundariesSec.size());
    }


    LOGD("SectionBoundaries: final count=%d", (int)gSectionBoundariesSec.size());
}



static void computeSectionEnergyLabels(const std::vector<float>& sectionTimesSec) {
    LOGD("DEBUG: COMPUTING SECTION ENERGY LABELS");
    if (gNumFrames <= 0 || gSpectrogram.empty() || gSampleRateGlobal <= 0) {
        LOGD("SectionEnergy: no spectrogram or sample rate, skipping");
        return;
    }

    if (sectionTimesSec.size() < 2) {
        LOGD("SectionEnergy: need at least 2 boundaries, got %zu", sectionTimesSec.size());
        return;
    }

    // 1) Copy & clean bounds (sorted, unique)
    std::vector<float> bounds = sectionTimesSec;
    std::sort(bounds.begin(), bounds.end());
    bounds.erase(std::unique(bounds.begin(), bounds.end()), bounds.end());

    if (bounds.size() < 2) {
        LOGD("SectionEnergy: not enough unique bounds");
        return;
    }

    const int numSections = (int)bounds.size() - 1;
    LOGD("SectionEnergy: numSections=%d (pre-merge)", numSections);

    // 2) Collect per-section energies (medians of frame energies)
    std::vector<std::vector<float>> sectionEnergies(numSections);

    for (int frame = 0; frame < gNumFrames; ++frame) {
        float tSec = (float)(frame * gHopSamples) / (float)gSampleRateGlobal;

        // Find which [bounds[s], bounds[s+1]) we're in
        int s = 0;
        while (s + 1 < (int)bounds.size() && tSec >= bounds[s + 1]) {
            ++s;
        }
        if (s >= numSections) {
            break; // beyond last section
        }

        float e = 0.0f;
        int base = frame * NUM_BINS;
        for (int k = 0; k < NUM_BINS; ++k) {
            e += gSpectrogram[base + k];
        }
        e /= (float)NUM_BINS;

        sectionEnergies[s].push_back(e);
    }

    std::vector<float> scores(numSections, 0.0f);
    for (int i = 0; i < numSections; ++i) {
        auto &vec = sectionEnergies[i];
        if (vec.empty()) {
            scores[i] = 0.0f;
            LOGD("SectionEnergy: section %d [%.3f, %.3f] has no frames",
                 i, bounds[i], bounds[i + 1]);
            continue;
        }
        std::sort(vec.begin(), vec.end());
        float median = vec[vec.size() / 2];
        scores[i] = median;
    }

    // 3) Compute z-scores across original sections
    float sum = 0.0f;
    for (float v : scores) sum += v;
    float mu = sum / (float)numSections;

    float var = 0.0f;
    for (float v : scores) {
        float d = v - mu;
        var += d * d;
    }
    var /= (float)numSections;
    float sd = std::sqrt(var + 1e-9f);

    LOGD("SectionEnergy: mean=%.6f, std=%.6f", mu, sd);

    // 4) Original labels (+1 HIGH, -1 LOW) at 16-beat resolution
    std::vector<int> originalLabels(numSections, 0);
    for (int i = 0; i < numSections; ++i) {
        float z = (scores[i] - mu) / sd;
        originalLabels[i] = (z > 0.0f) ? +1 : -1;

        LOGD("Section (pre-merge) %d: [%.2f, %.2f] z=%.3f label=%s",
             i,
             bounds[i], bounds[i + 1],
             z,
             (originalLabels[i] == +1 ? "high" : "low"));
    }

    // Special case: only 1 section → nothing to merge
    if (numSections == 1) {
        gSectionBoundariesSec = bounds;            // [start, end]
        gSectionEnergyLabels.assign(1, originalLabels[0]);
        LOGD("SectionEnergy: only 1 section, label=%s",
             (originalLabels[0] == +1 ? "high" : "low"));
        return;
    }

    // 5) Merge consecutive sections with the same label
    std::vector<float> mergedBounds;
    std::vector<int>   mergedLabels;
    mergedBounds.reserve(bounds.size());
    mergedLabels.reserve(numSections);

    // Always start at the first boundary
    mergedBounds.push_back(bounds[0]);
    int currLabel = originalLabels[0];

    for (int i = 1; i < numSections; ++i) {
        if (originalLabels[i] != currLabel) {
            // label changed → keep this boundary, finalize previous merged section
            mergedBounds.push_back(bounds[i]);
            mergedLabels.push_back(currLabel);
            currLabel = originalLabels[i];
        } else {
            // same label → do NOT add a boundary here (we merge across it)
        }
    }

    // Final merged section: add last boundary and its label
    mergedBounds.push_back(bounds.back());
    mergedLabels.push_back(currLabel);

    // 6) Update globals so the rest of the pipeline sees merged regions
    gSectionBoundariesSec = mergedBounds;    // boundaries only at label changes
    gSectionEnergyLabels  = mergedLabels;    // exactly one label per final section

    int finalSections = (int)mergedLabels.size();
    LOGD("SectionEnergy: merged to %d sections (from %d)", finalSections, numSections);

    for (int i = 0; i < finalSections; ++i) {
        float t0 = mergedBounds[i];
        float t1 = mergedBounds[i + 1];
        const char* lbl = (mergedLabels[i] == +1 ? "high" : "low");
        LOGD("MergedSection %d: [%.2f, %.2f] label=%s",
             i, t0, t1, lbl);
    }
}

static void computeCuePoints() {
    gCuePointRelaxed   = -1.f;
    gCuePointEnergetic = -1.f;
    gCuePointSmashcut  = -1.f;


    int numSections = (int)gSectionEnergyLabels.size();
    int numBounds   = (int)gSectionBoundariesSec.size();
    if (numSections == 0 || numBounds == 0)
        return;


    // We will *never* use section 0 for cue points.
    // Section i uses boundary gSectionBoundariesSec[i].
    const int startIdx = 1;


    // --- RELAXED TRANSITION → first LOW section (but not section 0) ---
    for (int i = startIdx; i < numSections; ++i) {
        if (gSectionEnergyLabels[i] == -1) {
            gCuePointRelaxed = gSectionBoundariesSec[i];
            break;
        }
    }
    // Fallback: if everything is high or we didn't find any low,
    // use second boundary if it exists.
    if (gCuePointRelaxed < 0.f && numBounds > 1) {
        gCuePointRelaxed = gSectionBoundariesSec[1];
    }


    // --- ENERGETIC TRANSITION → first HIGH section (but not section 0) ---
    for (int i = startIdx; i < numSections; ++i) {
        if (gSectionEnergyLabels[i] == +1) {
            gCuePointEnergetic = gSectionBoundariesSec[i];
            break;
        }
    }
    // Fallback
    if (gCuePointEnergetic < 0.f && numBounds > 1) {
        gCuePointEnergetic = gSectionBoundariesSec[1];
    }


    // --- SMASHCUT TRANSITION (cross-track logic) ---
    // Track A: high section after 10 s
    // Track B: low section after 10 s
    const float minTime = 10.0f;


    if (gSmashcutFirstTrack) {
        // Treat this as Track A
        for (int i = startIdx; i < numSections; ++i) {
            float t = gSectionBoundariesSec[i];
            if (t > minTime && gSectionEnergyLabels[i] == +1) {
                gCuePointSmashcut = t;
                break;
            }
        }
        // If no high section after 10 s, fall back to first boundary > 0
        if (gCuePointSmashcut < 0.f && numBounds > 1) {
            gCuePointSmashcut = gSectionBoundariesSec[1];
        }


        gSmashcutFirstTrack = false; // next computeCuePoints() = Track B
    } else {
        // Treat this as Track B
        for (int i = startIdx; i < numSections; ++i) {
            float t = gSectionBoundariesSec[i];
            if (t > minTime && gSectionEnergyLabels[i] == -1) {
                gCuePointSmashcut = t;
                break;
            }
        }
        // Fallback: first non-zero boundary
        if (gCuePointSmashcut < 0.f && numBounds > 1) {
            gCuePointSmashcut = gSectionBoundariesSec[1];
        }


        gSmashcutFirstTrack = true; // ready for the next A/B pair
    }


    LOGD("CuePoints: relaxed=%.2f energetic=%.2f smashcut=%.2f (firstTrack=%d)",
         gCuePointRelaxed, gCuePointEnergetic, gCuePointSmashcut,
         (int)gSmashcutFirstTrack);
}

static bool loadWavMono16(const char* path, std::vector<float>& outSamples, int& sampleRate) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        LOGD("Failed to open WAV: %s", path);
        return false;
    }


    // Read RIFF header
    char riff[4];
    f.read(riff, 4);
    if (strncmp(riff, "RIFF", 4) != 0) {
        LOGD("Not RIFF");
        return false;
    }


    // chunk size (skip)
    uint32_t chunkSize = 0;
    f.read(reinterpret_cast<char*>(&chunkSize), 4);


    char wave[4];
    f.read(wave, 4);
    if (strncmp(wave, "WAVE", 4) != 0) {
        LOGD("Not WAVE");
        return false;
    }


    // Find "fmt " chunk
    char chunkId[4];
    uint32_t subchunkSize = 0;
    bool fmtFound = false;
    while (f.read(chunkId, 4)) {
        f.read(reinterpret_cast<char*>(&subchunkSize), 4);
        if (strncmp(chunkId, "fmt ", 4) == 0) {
            fmtFound = true;
            break;
        } else {
            f.seekg(subchunkSize, std::ios::cur);
        }
    }


    if (!fmtFound) {
        LOGD("No fmt chunk");
        return false;
    }


    uint16_t audioFormat = 0;
    uint16_t numChannels = 0;
    uint32_t sampleRateLocal = 0;
    uint32_t byteRate = 0;
    uint16_t blockAlign = 0;
    uint16_t bitsPerSample = 0;


    f.read(reinterpret_cast<char*>(&audioFormat), 2);
    f.read(reinterpret_cast<char*>(&numChannels), 2);
    f.read(reinterpret_cast<char*>(&sampleRateLocal), 4);
    f.read(reinterpret_cast<char*>(&byteRate), 4);
    f.read(reinterpret_cast<char*>(&blockAlign), 2);
    f.read(reinterpret_cast<char*>(&bitsPerSample), 2);


    // Skip any extra fmt bytes
    if (subchunkSize > 16) {
        f.seekg(subchunkSize - 16, std::ios::cur);
    }


    if (audioFormat != 1) {
        LOGD("Only PCM format supported (got %u)", audioFormat);
        return false;
    }
    if (bitsPerSample != 16) {
        LOGD("Only 16-bit PCM supported (got %u bits)", bitsPerSample);
        return false;
    }


    // Find "data" chunk
    bool dataFound = false;
    while (f.read(chunkId, 4)) {
        f.read(reinterpret_cast<char*>(&subchunkSize), 4);
        if (strncmp(chunkId, "data", 4) == 0) {
            dataFound = true;
            break;
        } else {
            f.seekg(subchunkSize, std::ios::cur);
        }
    }


    if (!dataFound) {
        LOGD("No data chunk");
        return false;
    }


    const size_t numSamples16 = subchunkSize / 2;
    std::vector<int16_t> buf(numSamples16);
    f.read(reinterpret_cast<char*>(buf.data()), subchunkSize);


    outSamples.clear();
    outSamples.reserve(numSamples16 / numChannels);


    const float scale = 1.0f / 32768.0f;


    if (numChannels == 1) {
        for (size_t i = 0; i < numSamples16; ++i) {
            outSamples.push_back(buf[i] * scale);
        }
    } else {
        for (size_t i = 0; i + (numChannels - 1) < numSamples16; i += numChannels) {
            float acc = 0.f;
            for (int ch = 0; ch < numChannels; ++ch) {
                acc += buf[i + ch] * scale;
            }
            outSamples.push_back(acc / numChannels);
        }
    }


    sampleRate = (int)sampleRateLocal;
    LOGD("Loaded WAV %s, samples=%zu, fs=%d, channels=%u",
         path, outSamples.size(), sampleRate, numChannels);


    return true;
}


//Mel filterbank


static float hzToMel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}


static float melToHz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}


// Build 40-band mel filterbank for this sampleRate
static void buildMelFilterbank(int sampleRate) {
    gMelFilterbank.assign(NUM_MEL * N_FFT_BINS, 0.0f);
    gMelFbSampleRate = sampleRate;


    const float fMin = 0.0f;
    const float fMax = sampleRate * 0.5f;


    const float melMin = hzToMel(fMin);
    const float melMax = hzToMel(fMax);


    std::vector<float> melPoints(NUM_MEL + 2);
    for (int i = 0; i < NUM_MEL + 2; ++i) {
        float frac = (float)i / (NUM_MEL + 1);
        melPoints[i] = melMin + frac * (melMax - melMin);
    }


    std::vector<float> hzPoints(NUM_MEL + 2);
    for (int i = 0; i < NUM_MEL + 2; ++i) {
        hzPoints[i] = melToHz(melPoints[i]);
    }


    std::vector<int> bin(NUM_MEL + 2);
    for (int i = 0; i < NUM_MEL + 2; ++i) {
        // mapping from Hz to FFT bin index
        bin[i] = (int)floorf((FFT_SIZE + 1) * hzPoints[i] / sampleRate);
        if (bin[i] < 0) bin[i] = 0;
        if (bin[i] > N_FFT_BINS - 1) bin[i] = N_FFT_BINS - 1;
    }


    // Triangular filters
    for (int m = 1; m <= NUM_MEL; ++m) {
        int start = bin[m - 1];
        int center = bin[m];
        int end = bin[m + 1];


        for (int k = start; k < center; ++k) {
            float w = (float)(k - start) / (float)(center - start);
            gMelFilterbank[(m - 1) * N_FFT_BINS + k] = w;
        }
        for (int k = center; k < end; ++k) {
            float w = (float)(end - k) / (float)(end - center);
            gMelFilterbank[(m - 1) * N_FFT_BINS + k] = w;
        }
    }


    LOGD("Mel filterbank built: sampleRate=%d, melBands=%d, fftBins=%d",
         sampleRate, NUM_MEL, N_FFT_BINS);
}


// Spectrogram computation


static void computeSpectrogramFromPcm(const std::vector<float>& pcm, int sampleRate) {
    gSpectrogram.clear();
    gNumFrames = 0;
    //store sample rate
    gSampleRateGlobal = sampleRate;


    if (pcm.size() < (size_t)FRAME_SIZE) {
        LOGD("PCM too short for one frame");
        return;
    }


    // Hamming window
    static float Hamm[FRAME_SIZE];
    static bool hammInit = false;
    if (!hammInit) {
        for (int n = 0; n < FRAME_SIZE; ++n) {
            Hamm[n] = 0.54f - 0.46f * cosf(2.0f * M_PI * n / (FRAME_SIZE - 1));
        }
        hammInit = true;
    }


    // Mel filterbank (rebuild if sample rate changed)
    if (gMelFilterbank.empty() || gMelFbSampleRate != sampleRate) {
        buildMelFilterbank(sampleRate);
    }


    kiss_fft_cfg cfg = kiss_fft_alloc(FFT_SIZE, 0, nullptr, nullptr);
    std::vector<kiss_fft_cpx> fIn(FFT_SIZE);
    std::vector<kiss_fft_cpx> fOut(FFT_SIZE);


    const int hop = FRAME_SIZE / 2; // 50% overlap
    const int totalSamples = (int)pcm.size();
    const int numFrames = (totalSamples - FRAME_SIZE) / hop + 1;


    gNumFrames = numFrames;
    gSpectrogram.resize(numFrames * NUM_BINS);


    LOGD("computeSpectrogramFromPcm: numFrames=%d", numFrames);


    // Temporary magnitude buffer for each frame (0..Nyquist)
    std::vector<float> mag2(N_FFT_BINS);


    for (int frame = 0; frame < numFrames; ++frame) {
        const int offset = frame * hop;


        // Window + zero-pad
        for (int i = 0; i < FRAME_SIZE; ++i) {
            float s = (offset + i < totalSamples) ? pcm[offset + i] : 0.f;
            float w = s * Hamm[i];
            fIn[i].r = w;
            fIn[i].i = 0.f;
        }
        for (int i = FRAME_SIZE; i < FFT_SIZE; ++i) {
            fIn[i].r = 0.f;
            fIn[i].i = 0.f;
        }


        kiss_fft(cfg, fIn.data(), fOut.data());


        // Magnitude squared for bins 0..N_FFT_BINS-1
        for (int k = 0; k < N_FFT_BINS; ++k) {
            float re = fOut[k].r;
            float im = fOut[k].i;
            mag2[k] = re * re + im * im;
        }


        // Apply mel filterbank → NUM_MEL band energies
        float melEnergies[NUM_MEL];
        for (int m = 0; m < NUM_MEL; ++m) {
            float sum = 0.f;
            const float* fbRow = &gMelFilterbank[m * N_FFT_BINS];
            for (int k = 0; k < N_FFT_BINS; ++k) {
                sum += fbRow[k] * mag2[k];
            }
            melEnergies[m] = sum;
        }


        // Log + per-frame normalize 0..1
//        float minVal = INFINITY;
//        float maxVal = -INFINITY;
//        float tmp[NUM_MEL];
//        for (int m = 0; m < NUM_MEL; ++m) {
//            float v = log10f(melEnergies[m] + 1e-10f);
//            tmp[m] = v;
//            if (v < minVal) minVal = v;
//            if (v > maxVal) maxVal = v;
//        }
//        float range = maxVal - minVal;
//        if (range <= 0.f) range = 1.f;
//
//        for (int m = 0; m < NUM_MEL; ++m) {
//            float norm = (tmp[m] - minVal) / range; // 0..1
//            gSpectrogram[frame * NUM_BINS + m] = norm;
//        }
        // Log mel energies, store directly into gSpectrogram
        for (int m = 0; m < NUM_MEL; ++m) {
            float v = log10f(melEnergies[m] + 1e-10f);
            gSpectrogram[frame * NUM_BINS + m] = v;
        }


    }


    // --- Global normalization over all frames/bands ---
    if (gNumFrames > 0) {
        float globalMin = INFINITY;
        float globalMax = -INFINITY;


        for (int f = 0; f < gNumFrames; ++f) {
            for (int m = 0; m < NUM_BINS; ++m) {
                float v = gSpectrogram[f * NUM_BINS + m];
                if (v < globalMin) globalMin = v;
                if (v > globalMax) globalMax = v;
            }
        }


        float range = globalMax - globalMin;
        if (range <= 0.f) range = 1.f;


        for (int i = 0; i < gNumFrames * NUM_BINS; i++) {
            float v = gSpectrogram[i];
            gSpectrogram[i] = (v - globalMin) / range;   // final 0–1
        }


//        float floorVal = globalMin + 0.05f * (globalMax - globalMin);
//        float ceilVal = globalMin + 0.95f * (globalMax - globalMin);
//
//        float range_1 = ceilVal - floorVal;
//        if (range_1 <= 0.f) range_1 = 1.f;
//
//        for (int i = 0; i < gNumFrames * NUM_BINS; i++) {
//            float v = gSpectrogram[i];
//
//            if (v < floorVal) v = floorVal;
//            if (v > ceilVal) v = ceilVal;
//
//            gSpectrogram[i] = (v - floorVal) / range_1;
//        }
    }




    // Build onset detection function
    gOnsetLength = gNumFrames;
    gOnsetCurve.assign(gOnsetLength, 0.0f);


    if (gOnsetLength > 1) {
        for (int m = 1; m < gOnsetLength; ++m) {
            const float* prev = &gSpectrogram[(m - 1) * NUM_BINS];
            const float* curr = &gSpectrogram[m * NUM_BINS];


            float sum = 0.0f;
            for (int k = 0; k < NUM_BINS; ++k) {
                float diff = curr[k] - prev[k];
                if (diff > 0.0f) {
                    sum += diff;
                }
            }
            gOnsetCurve[m] = sum;
        }


        // Optional: normalize
        float maxVal = 0.0f;
        for (int m = 0; m < gOnsetLength; ++m) {
            if (gOnsetCurve[m] > maxVal) maxVal = gOnsetCurve[m];
        }
        if (maxVal > 0.0f) {
            for (int m = 0; m < gOnsetLength; ++m) {
                gOnsetCurve[m] /= maxVal;
            }
        }
    }


    free(cfg);


    gOnset.assign(gNumFrames, 0.0f);


    float prevEnergy = 0.0f;
    for (int f = 0; f < gNumFrames; ++f) {
        float energy = 0.0f;


        // Sum all mel bands for this frame as a simple energy proxy
        int base = f * NUM_BINS;
        for (int k = 0; k < NUM_BINS; ++k) {
            energy += gSpectrogram[base + k];
        }


        float diff = energy - prevEnergy;
        if (diff > 0.0f) {
            gOnset[f] = diff;  // halfwave rectified
        } else {
            gOnset[f] = 0.0f;
        }


        prevEnergy = energy;
    }




    LOGD("Spectrogram ready (mel): frames=%d, bins=%d, onsetLen=%d",
         gNumFrames, NUM_BINS, gOnsetLength);


    const int hopSize = FRAME_SIZE / 2;  // half size overlap
    computeTempoAndBeatsFromOnset(gOnset, sampleRate, hopSize);
    computeSectionBoundaries();
    computeSectionEnergyLabels(gSectionBoundariesSec);
    computeCuePoints();
}

//////////////////////// MIXING LOGIC ////////////////////////

static inline float spb(float bpm) {
    return 60.0f / bpm;
}

//static std::vector<float> timeStretchWSOLA(
//        const std::vector<float>& in,
//        float rate,
//        int sampleRate
//) {
//    if (rate <= 0.f || in.empty())
//        return {};
//
//    const int win = 1024;          // window size
//    const int hopIn = win / 2;
//    const int hopOut = (int)(hopIn / rate);
//    const int search = hopIn;      // search radius
//
//    std::vector<float> out((size_t)(in.size() / rate) + win, 0.f);
//    std::vector<float> window(win);
//
//    // Hann window
//    for (int i = 0; i < win; ++i)
//        window[i] = 0.5f - 0.5f * cosf(2.f * M_PI * i / win);
//
//    int inPos = 0;
//    int outPos = 0;
//
//    while (inPos + win + search < (int)in.size()) {
//        int bestOffset = 0;
//        float bestCorr = -1e30f;
//
//        // Find best alignment via correlation
//        for (int d = -search; d <= search; ++d) {
//            float corr = 0.f;
//            for (int i = 0; i < win; ++i) {
//                corr += out[outPos + i] * in[inPos + d + i];
//            }
//            if (corr > bestCorr) {
//                bestCorr = corr;
//                bestOffset = d;
//            }
//        }
//
//        inPos += bestOffset;
//
//        // Overlap-add
//        for (int i = 0; i < win; ++i) {
//            out[outPos + i] += in[inPos + i] * window[i];
//        }
//
//        inPos += hopIn;
//        outPos += hopOut;
//    }
//
//    out.resize(outPos + win);
//    return out;
//}

//static std::vector<float> timeStretchWSOLA(
//        const std::vector<float>& in,
//        float rate,
//        int sampleRate
//) {
//    (void)sampleRate; // unused for now
//    if (rate <= 0.f || in.empty()) return {};
//
//    const int win    = 2048;
//    const int hopIn  = win / 2;
//    const int hopOut = std::max(1, (int)std::lround((float)hopIn / rate));
//    const int search = hopIn;
//
//    // Output big enough for nominal stretch + one window.
//    std::vector<float> out((size_t)(in.size() / rate) + win, 0.f);
//
//    // Hann window
//    std::vector<float> window(win);
//    for (int i = 0; i < win; ++i)
//        window[i] = 0.5f - 0.5f * cosf(2.f * (float)M_PI * (float)i / (float)win);
//
//    int inPos  = 0;
//    int outPos = 0;
//
//    // We must guarantee:
//    // - inPos..inPos+win-1 is valid
//    // - outPos..outPos+win-1 is valid
//    while (inPos + win <= (int)in.size() && outPos + win <= (int)out.size()) {
//
//        // Limit d so that (inPos + d) is within [0, in.size()-win]
//        const int dMin = std::max(-search, -inPos);
//        const int dMax = std::min(search, (int)in.size() - (inPos + win));
//
//        int bestOffset = 0;
//        float bestCorr = -1e30f;
//
//        // If out is still all zeros early on, corr values will be ~0.
//        // That's okay; we just need to stay in-bounds.
//        for (int d = dMin; d <= dMax; ++d) {
//            float corr = 0.f;
//            const int inBase = inPos + d;
//
//            for (int i = 0; i < win; ++i) {
//                corr += out[outPos + i] * in[inBase + i];
//            }
//            if (corr > bestCorr) {
//                bestCorr = corr;
//                bestOffset = d;
//            }
//        }
//
//        // Apply bestOffset and clamp to valid range
//        inPos = std::clamp(inPos + bestOffset, 0, (int)in.size() - win);
//
//        // Overlap-add
//        for (int i = 0; i < win; ++i) {
//            out[outPos + i] += in[inPos + i] * window[i];
//        }
//
//        inPos  += hopIn;
//        outPos += hopOut;
//    }
//
//    // Trim to what we actually filled (keep one window tail)
//    if (outPos + win < (int)out.size()) out.resize(outPos + win);
//    return out;
//}

static std::vector<float> timeStretchWSOLA(
        const std::vector<float>& y,
        float rate,
        int sampleRate
) {
    std::vector<float> out;

    if (rate <= 0.f || y.empty())
        return out;

    const int inLen = (int)y.size();
    const int outLen = (int)std::lround((float)inLen / rate);

    if (outLen <= 0)
        return out;

    out.resize(outLen);

    for (int i = 0; i < outLen; ++i) {
        float inPos = i * rate;

        int i0 = (int)std::floor(inPos);
        int i1 = std::min(i0 + 1, inLen - 1);

        float frac = inPos - (float)i0;

        float s0 = y[i0];
        float s1 = y[i1];

        out[i] = s0 + frac * (s1 - s0); // linear interpolation
    }

    return out;
}


static void applyCrossfade( std::vector<float>& A, std::vector<float>& B) {
    int L = std::min(A.size(), B.size());
    for (int i = 0; i < L; ++i) {
        float t = (float)i / (float)L;
        float fo = cosf(t * M_PI_2);
        float fi = sinf(t * M_PI_2);
        A[i] = A[i] * fo + B[i] * fi;
    }
}



//static std::vector<float> rampTimeStretch(
//        const std::vector<float>& y,
//        int sampleRate,
//        float rateStart,
//        float rateEnd,
//        float rampSeconds
//) {
//    if (rampSeconds <= 0.f) return {};
//
//    const float chunkOutSec = 0.05f;
//    int chunkOutN = (int)(chunkOutSec * sampleRate);
//    int numChunks = std::max(1, (int)(rampSeconds / chunkOutSec));
//
//    std::vector<float> out;
//    size_t used = 0;
//
//    for (int i = 0; i < numChunks && used < y.size(); ++i) {
//        float r = rateStart + (rateEnd - rateStart) * ((float)i / numChunks);
//        int needIn = (int)(chunkOutN * r);
//
//        if (used + needIn > y.size())
//            needIn = y.size() - used;
//
//        std::vector<float> segIn(y.begin() + used, y.begin() + used + needIn);
//        auto segOut = timeStretchWSOLA(segIn, r, sampleRate);
//
////        if (segOut.size() > chunkOutN)
////            segOut.resize(chunkOutN);
//
//        out.insert(out.end(), segOut.begin(), segOut.end());
//        used += needIn;
//    }
//
//    // Append remaining audio at native tempo
//    if (used < y.size()) {
//        out.insert(out.end(), y.begin() + used, y.end());
//    }
//
//    return out;
//}

static std::vector<float> rampTimeStretch(
        const std::vector<float>& y,
        int sampleRate,
        float rateStart,
        float rateEnd,
        float rampSeconds,
        int& samplesConsumed
) {


    std::vector<float> out;
    if (y.empty() || rampSeconds <= 0.f)
        return out;

    samplesConsumed = 0;

    const float perceptualStretch = 1.7f;
    const int rampSamples = (int)(rampSeconds * sampleRate * perceptualStretch);
    const int N = (int)y.size();

    float phase = 0.0f;
    int outSamples = 0;

    while (outSamples < rampSamples && phase + 1 < N) {

        float alpha = (float)outSamples / (float)rampSamples;

        // Perceptual easing (VERY important)
        alpha = alpha * alpha * (3.f - 2.f * alpha);
        alpha = alpha * alpha;

        float rate = rateStart + alpha * (rateEnd - rateStart);

        int i0 = (int)phase;
        int i1 = i0 + 1;
        float frac = phase - i0;

        float sample = (1.f - frac) * y[i0] + frac * y[i1];
        out.push_back(sample);

        phase += rate;

        samplesConsumed = (int)phase;   // <-- THIS IS THE KEY LINE

        outSamples++;
    }

    // Native tempo remainder
    while (phase + 1 < N) {
        int i0 = (int)phase;
        int i1 = i0 + 1;
        float frac = phase - i0;
        out.push_back((1.f - frac) * y[i0] + frac * y[i1]);
        phase += 1.f;
    }

    return out;
}


static std::vector<float> renderMix(
        const std::vector<float>& yA,
        const std::vector<float>& yB,
        float bpmA,
        float bpmB,
        float A_mix_time,
        float B_start_time,
        int sampleRate,
        int xfadeBeats = 16,
        int rampBeats = 32
) {
    LOGD("Starting Render Mix");
    gMixBeatTimes.clear();
    gMixBpmTimeline.clear();
    gMixBpmTimeSec.clear();

    float spbA = spb(bpmA);
    auto logBpm = [&](float tSec, float bpm) {
        if (gMixBpmTimeSec.empty() || tSec - gMixBpmTimeSec.back() >= 0.05f) {
            gMixBpmTimeSec.push_back(tSec);
            gMixBpmTimeline.push_back(bpm);
        }
    };

    int L = (int)(xfadeBeats * spbA * sampleRate);


    int beatIdx = 0;
    float tBeat = 0.f;

    while (tBeat < A_mix_time) {
        gMixBeatTimes.push_back(tBeat);
        logBpm(tBeat, bpmA);
        tBeat += spbA;
    }


    int A_start = (int)(A_mix_time * sampleRate);
    int A_end   = std::min(A_start + L, (int)yA.size());

    std::vector<float> A_pre(yA.begin(), yA.begin() + A_start);
    std::vector<float> A_xseg(yA.begin() + A_start, yA.begin() + A_end);

    float rateMatch = bpmA / bpmB;
    rateMatch = std::clamp(rateMatch, 0.85f, 1.15f);

    int B_start = (int)(B_start_time * sampleRate);
    std::vector<float> B_tail(yB.begin() + B_start, yB.end());

    int needIn = (int)(L * rateMatch);
    if ((int)B_tail.size() < needIn)
        B_tail.resize(needIn, 0.f);

    std::vector<float> B_xin(B_tail.begin(), B_tail.begin() + needIn);
    auto B_xout = timeStretchWSOLA(B_xin, rateMatch, sampleRate);
    B_xout.resize(L);

    float crossfadeFraction = 0.5f; // default


//// Choose crossfade behavior based on transition type
//    switch (gCurrentTransitionType) {
//        case 1: // RELAXED
//            crossfadeFraction = 0.75f;   // long, smooth blend
//            break;
//
//        case 2: // ENERGETIC
//            crossfadeFraction = 0.25f;   // short, punchy
//            break;
//
//        case 3: // SMASHCUT
//            crossfadeFraction = 0.05f;   // almost instant
//            break;
//
//        default:
//            crossfadeFraction = 0.5f;
//            break;
//    }

    applyCrossfade(A_xseg, B_xout);


    float xfadeDur = (float)L / sampleRate;

    float tX = A_mix_time;
    while (tX < A_mix_time + xfadeDur) {
        logBpm(tX, bpmA);
        tX += 0.05f;
    }


    float t = A_mix_time;
    for (int i = 0; i < xfadeBeats; i++) {
        t += spbA;
        gMixBeatTimes.push_back(t);
    }

    int rampConsumed = 0;

    auto B_ramp = rampTimeStretch(
            std::vector<float>(B_tail.begin() + needIn, B_tail.end()),
            sampleRate,
            rateMatch,
            1.0f,
            rampBeats * spbA,
            rampConsumed
    );


    float rampDur = rampBeats * spbA;

    float rampStart = A_mix_time + xfadeDur;
    float rampEnd   = rampStart + rampDur;
    gMixTransitionEndSec = rampEnd;


    float tRamp = rampStart;
    float rampDurSec = rampBeats * spbA;

    while (tRamp < rampEnd) {
        float alpha = (tRamp - rampStart) / rampDurSec;
        alpha = std::clamp(alpha, 0.f, 1.f);

        float bpmNow = bpmA + alpha * (bpmB - bpmA);
        float spbNow = spb(bpmNow);

        gMixBeatTimes.push_back(tRamp);
        logBpm(tRamp, bpmNow);

        tRamp += spbNow;
    }




    std::vector<float> mix;
    mix.reserve(A_pre.size() + A_xseg.size() + B_ramp.size());
    mix.insert(mix.end(), A_pre.begin(), A_pre.end());
    mix.insert(mix.end(), A_xseg.begin(), A_xseg.end());
    mix.insert(mix.end(), B_ramp.begin(), B_ramp.end());

    // Append remaining B audio at native tempo
    int B_rest_start = B_start + needIn + rampConsumed;

    if (B_rest_start < (int)yB.size()) {
        mix.insert(
                mix.end(),
                yB.begin() + B_rest_start,
                yB.end()
        );
    }


    float tB = rampEnd;
    float spbB = 60.f / bpmB;

// Continue beats until end of mix
    float mixDur = (float)mix.size() / sampleRate;

    while (tB < mixDur) {
        gMixBeatTimes.push_back(tB);
        logBpm(tB, bpmB);
        tB += spbB;
    }


    return mix;
}





// JNI calls


extern "C" JNIEXPORT jint JNICALL
Java_com_example_ece420_1final_1project_1milestone1_MainActivity_initFromWav(
        JNIEnv* env,
        jobject /* this */,
        jstring jpath) {


    const char* cpath = env->GetStringUTFChars(jpath, nullptr);
    LOGD("initFromWav: path=%s", cpath);


    std::vector<float> pcm;
    int sampleRate = 0;
    bool ok = loadWavMono16(cpath, pcm, sampleRate);


    env->ReleaseStringUTFChars(jpath, cpath);


    if (!ok) {
        gNumFrames = 0;
        gSpectrogram.clear();
        return 0;
    }


    computeSpectrogramFromPcm(pcm, sampleRate);
//    return gNumFrames;
// Store BPM depending on call order (A then B)
    static bool isTrackA = true;
    if (isTrackA) {
        gEstimatedBpmA = gEstimatedBpm;
    } else {
        gEstimatedBpmB = gEstimatedBpm;
    }
    isTrackA = !isTrackA;

    return gNumFrames;

}


extern "C" JNIEXPORT void JNICALL
Java_com_example_ece420_1final_1project_1milestone1_MainActivity_getSpectrogramColumn(
        JNIEnv* env,
        jobject /* this */,
        jint frameIndex,
        jfloatArray jbuffer) {


    if (gNumFrames <= 0 || frameIndex < 0 || frameIndex >= gNumFrames) {
        jsize len = env->GetArrayLength(jbuffer);
        std::vector<float> zeros(len, 0.f);
        env->SetFloatArrayRegion(jbuffer, 0, len, zeros.data());
        return;
    }


    jsize len = env->GetArrayLength(jbuffer);
    const int binsToCopy = (len < NUM_BINS) ? len : NUM_BINS;


    const float* src = &gSpectrogram[frameIndex * NUM_BINS];
    env->SetFloatArrayRegion(jbuffer, 0, binsToCopy, src);
}


extern "C" JNIEXPORT jint JNICALL
Java_com_example_ece420_1final_1project_1milestone1_MainActivity_getOnsetLength(
        JNIEnv* /*env*/,
        jobject /*this*/) {
    return gOnsetLength;
}


extern "C" JNIEXPORT void JNICALL
Java_com_example_ece420_1final_1project_1milestone1_MainActivity_getOnsetCurve(
        JNIEnv* env,
        jobject /*this*/,
        jfloatArray jbuffer) {


    jsize len = env->GetArrayLength(jbuffer);


    if (gOnsetLength <= 0) {
        std::vector<float> zeros(len, 0.0f);
        env->SetFloatArrayRegion(jbuffer, 0, len, zeros.data());
        return;
    }


    int copyCount = (len < gOnsetLength) ? len : gOnsetLength;
    env->SetFloatArrayRegion(jbuffer, 0, copyCount, gOnsetCurve.data());
}


extern "C" JNIEXPORT jfloat JNICALL
Java_com_example_ece420_1final_1project_1milestone1_MainActivity_getEstimatedBpm(
        JNIEnv* env,
        jobject /* this */) {
    return (jfloat) gEstimatedBpm;
}


extern "C" JNIEXPORT jint JNICALL
Java_com_example_ece420_1final_1project_1milestone1_MainActivity_getNumBeats(
        JNIEnv* env,
        jobject /* this */) {
    return (jint) gBeatTimesSec.size();
}


extern "C" JNIEXPORT void JNICALL
Java_com_example_ece420_1final_1project_1milestone1_MainActivity_getBeatTimes(
        JNIEnv* env,
        jobject /* this */,
        jfloatArray jbuffer) {


    jsize len = env->GetArrayLength(jbuffer);
    int n = std::min(len, (jsize) gBeatTimesSec.size());
    if (n > 0) {
        env->SetFloatArrayRegion(jbuffer, 0, n, gBeatTimesSec.data());
    }
}


extern "C" JNIEXPORT jint JNICALL
Java_com_example_ece420_1final_1project_1milestone1_MainActivity_getNumSections(
        JNIEnv*, jobject) {
    return gSectionBoundariesSec.size();
}


extern "C" JNIEXPORT void JNICALL
Java_com_example_ece420_1final_1project_1milestone1_MainActivity_getSectionBoundaries(
        JNIEnv* env, jobject, jfloatArray jbuf) {


    jsize len = env->GetArrayLength(jbuf);
    int n = std::min(len, (jsize)gSectionBoundariesSec.size());
    if (n > 0)
        env->SetFloatArrayRegion(jbuf, 0, n, gSectionBoundariesSec.data());
}


extern "C" JNIEXPORT void JNICALL
Java_com_example_ece420_1final_1project_1milestone1_MainActivity_getSectionEnergyLabels(
        JNIEnv* env,
        jobject /* this */,
        jintArray jbuffer) {


    jsize len = env->GetArrayLength(jbuffer);


    if (gSectionEnergyLabels.empty()) {
        // Return zeros if nothing computed
        std::vector<jint> zeros(len, 0);
        env->SetIntArrayRegion(jbuffer, 0, len, zeros.data());
        return;
    }


    int count = std::min((int)len, (int)gSectionEnergyLabels.size());


    env->SetIntArrayRegion(
            jbuffer,
            0,
            count,
            reinterpret_cast<const jint*>(gSectionEnergyLabels.data()));
}


extern "C" JNIEXPORT jfloat JNICALL
Java_com_example_ece420_1final_1project_1milestone1_MainActivity_getCuePoint(
        JNIEnv*, jobject, jint transitionType) {

    gCurrentTransitionType = transitionType;

    switch (transitionType) {
        case 1: return gCuePointRelaxed;
        case 2: return gCuePointEnergetic;
        case 3: return gCuePointSmashcut;
        default: return -1.f;
    }
}


extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_ece420_1final_1project_1milestone1_MainActivity_renderMix(
        JNIEnv* env,
        jobject /* this */,
        jstring pathA,
        jstring pathB,
        jfloat bpmA,
        jfloat bpmB,
        jfloat cueA,
        jfloat cueB
) {
    const char* cpathA = env->GetStringUTFChars(pathA, nullptr);
    const char* cpathB = env->GetStringUTFChars(pathB, nullptr);

    std::vector<float> yA, yB;
    int srA = 0, srB = 0;

    if (!loadWavMono16(cpathA, yA, srA) ||
        !loadWavMono16(cpathB, yB, srB)) {

        env->ReleaseStringUTFChars(pathA, cpathA);
        env->ReleaseStringUTFChars(pathB, cpathB);
        return nullptr;
    }

    env->ReleaseStringUTFChars(pathA, cpathA);
    env->ReleaseStringUTFChars(pathB, cpathB);

    // Resampling NOT implemented here — assume same SR for now
    if (srA != srB) {
        LOGD("Sample rate mismatch (%d vs %d)", srA, srB);
        return nullptr;
    }

    std::vector<float> mix = renderMix(
            yA,
            yB,
            gEstimatedBpmA,   // ✅ BPM A
            gEstimatedBpmB,   // ✅ BPM B
            cueA,
            cueB,
            srA
    );

    jfloatArray out = env->NewFloatArray(mix.size());
    env->SetFloatArrayRegion(out, 0, mix.size(), mix.data());

    return out;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_ece420_1final_1project_1milestone1_MainActivity_getMixBeatCount(
        JNIEnv*, jobject) {
    return (jint) gMixBeatTimes.size();
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_ece420_1final_1project_1milestone1_MainActivity_getMixBeatTimes(
        JNIEnv* env, jobject, jfloatArray outArray) {

    jfloat* out = env->GetFloatArrayElements(outArray, nullptr);
    for (size_t i = 0; i < gMixBeatTimes.size(); ++i)
        out[i] = gMixBeatTimes[i];
    env->ReleaseFloatArrayElements(outArray, out, 0);
}

extern "C"
JNIEXPORT jfloat JNICALL
Java_com_example_ece420_1final_1project_1milestone1_MainActivity_getMixTransitionEndSec(
        JNIEnv*, jobject) {
    return gMixTransitionEndSec;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_ece420_1final_1project_1milestone1_MainActivity_getMixBpmCount(
        JNIEnv*, jobject) {
    return (jint)gMixBpmTimeline.size();
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_ece420_1final_1project_1milestone1_MainActivity_getMixBpmTimeline(
        JNIEnv* env, jobject, jfloatArray jbuf) {

    jsize n = std::min(env->GetArrayLength(jbuf),
                       (jsize)gMixBpmTimeline.size());
    if (n > 0)
        env->SetFloatArrayRegion(jbuf, 0, n, gMixBpmTimeline.data());
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_ece420_1final_1project_1milestone1_MainActivity_getMixBpmTimes(
        JNIEnv* env, jobject, jfloatArray jbuf) {

    jsize n = std::min(env->GetArrayLength(jbuf),
                       (jsize)gMixBpmTimeSec.size());
    if (n > 0)
        env->SetFloatArrayRegion(jbuf, 0, n, gMixBpmTimeSec.data());
}







