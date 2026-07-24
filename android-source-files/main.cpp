Native-lib.cpp:

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
static const int NUM_BINS  = NUM_MEL; // must match MainActivity.NUM_BINS
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



// Helpers: WAV loading
#include <algorithm> // for std::max, std::min

// onset: length N = gNumFrames
static void computeTempoAndBeatsFromOnset(const std::vector<float>& onset,
int sampleRate,
                                         int hopSize) {
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


static bool loadWavMono16(const char* path,
                         std::vector<float>& outSamples,
                         int& sampleRate) {
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
   std::vector<kiss_fft_cpx> fIn(FFT_SIZE), fOut(FFT_SIZE);

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
       float minVal = INFINITY;
       float maxVal = -INFINITY;
       float tmp[NUM_MEL];
       for (int m = 0; m < NUM_MEL; ++m) {
           float v = log10f(melEnergies[m] + 1e-10f);
           tmp[m] = v;
           if (v < minVal) minVal = v;
           if (v > maxVal) maxVal = v;
       }
       float range = maxVal - minVal;
       if (range <= 0.f) range = 1.f;

       for (int m = 0; m < NUM_MEL; ++m) {
           float norm = (tmp[m] - minVal) / range; // 0..1
           gSpectrogram[frame * NUM_BINS + m] = norm;
       }
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
           gOnset[f] = diff;   // halfwave rectified
       } else {
           gOnset[f] = 0.0f;
       }

       prevEnergy = energy;
   }


   LOGD("Spectrogram ready (mel): frames=%d, bins=%d, onsetLen=%d",
        gNumFrames, NUM_BINS, gOnsetLength);

   const int hopSize = FRAME_SIZE / 2;   // half size overlap
   computeTempoAndBeatsFromOnset(gOnset, sampleRate, hopSize);
}




// JNI calls

extern "C" JNIEXPORT jint JNICALL
Java_com_example_autodjmilestone1_MainActivity_initFromWav(
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
   return gNumFrames;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_autodjmilestone1_MainActivity_getSpectrogramColumn(
       JNIEnv* env,
       jobject /* this */,
       jint frameIndex,
       jfloatArray jbuffer) {

   if (gNumFrames <= 0 ||
       frameIndex < 0 ||
       frameIndex >= gNumFrames) {
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
Java_com_example_autodjmilestone1_MainActivity_getOnsetLength(
       JNIEnv* /*env*/,
       jobject /*this*/) {
   return gOnsetLength;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_autodjmilestone1_MainActivity_getOnsetCurve(
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
Java_com_example_autodjmilestone1_MainActivity_getEstimatedBpm(
       JNIEnv* env,
       jobject /* this */) {
   return (jfloat) gEstimatedBpm;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_autodjmilestone1_MainActivity_getNumBeats(
       JNIEnv* env,
       jobject /* this */) {
   return (jint) gBeatTimesSec.size();
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_autodjmilestone1_MainActivity_getBeatTimes(
       JNIEnv* env,
       jobject /* this */,
       jfloatArray jbuffer) {

   jsize len = env->GetArrayLength(jbuffer);
   int n = std::min(len, (jsize) gBeatTimesSec.size());
   if (n > 0) {
       env->SetFloatArrayRegion(jbuffer, 0, n, gBeatTimesSec.data());
   }
}