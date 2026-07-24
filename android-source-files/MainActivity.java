
package com.example.ece420_final_project_milestone1;


import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Button;
import android.content.Intent;
import android.net.Uri;
import android.app.Activity;
import android.widget.Toast;
import android.provider.OpenableColumns;
import android.database.Cursor;
import android.widget.SeekBar;


import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;


import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;


public class MainActivity extends AppCompatActivity {

    // Global lock to serialize ALL native calls
    private static final Object JNI_LOCK = new Object();

    private static final int STORAGE_REQUEST = 1001;
    private static final int REQ_TRACK_A = 2001;
    private static final int REQ_TRACK_B = 2002;


    // MUST match native NUM_BINS (40 mel bands)
    private static final int NUM_BINS = 40;


    // UI Views
    private ImageView spectrogramViewA;
    private ImageView spectrogramViewB;
    private ImageView waveformViewA;
    private ImageView waveformViewB;
    private Button btnShowWaveforms;

    private ImageView bpmTimelineView;


    private Button btnLoadTrackA, btnLoadTrackB, btnShowBpm, btnShowSpectrogram;
    private TextView txtTrackAName, txtTrackBName, txtBpmValue;

    private Button btnStartMixing, btnAnalysis;
    private ImageView waveformMixView;
    private SeekBar seekMix;
    private float[] mixedAudio = null;

    private AudioTrack audioTrack;
    private boolean isPlaying = false;
    private int playbackSampleRate = 44100; // must match C++ output

    // Analysis cache flags
    private boolean trackAAnalyzed = false;
    private boolean trackBAnalyzed = false;

    // Cached cue points
    private float cachedCueA = -1f;
    private float cachedCueB = -1f;

    private String mixWavPath = null;
    private Bitmap mixWaveformBitmap = null;




    // --- Paths for each track (null = not loaded) ---
    private String trackAPath = null;


    private int currentTransitionMode = 0;
    private String trackBPath = null;


    // --- For spectrogram toggling when both tracks are loaded ---
    // 'A' means last drawn spectrogram was Track A, 'B' for Track B, 'U' for none yet
    private char lastSpectrogramTrack = 'U';

    private float[] mixBeatTimes = null;

    private float[] mixBpmValues = null;
    private float[] mixBpmTimes  = null;




    // JNI calls into native-lib.cpp
    public native int initFromWav(String path);
    public native void getSpectrogramColumn(int frameIndex, float[] column);
    public native int getOnsetLength();
    public native void getOnsetCurve(float[] buffer);


    public native float getEstimatedBpm();
    public native int getNumBeats();
    public native void getBeatTimes(float[] buffer);
    public native int getNumSections();
    public native void getSectionBoundaries(float[] buffer);
    public native void getSectionEnergyLabels(int[] buffer);
    public native float getCuePoint(int transitionType);

    public native int getMixBeatCount();
    public native void getMixBeatTimes(float[] buffer);

    public native float[] renderMix(
            String pathA, String pathB, float cueA, float cueB, float bpmA, float bpmB
    );

    public native float getMixTransitionEndSec();

    // ---- BPM over time for rendered mix ----
    public native int getMixBpmCount();
    public native void getMixBpmTimeline(float[] bpmValues);
    public native void getMixBpmTimes(float[] timeValues);




    static {
        System.loadLibrary("ece420_final_project_milestone1");
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        Spinner spinnerTransition = findViewById(R.id.spinnerTransition);


        spinnerTransition.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {
                currentTransitionMode = pos;  // 0=no transition, 1=relaxed, 2=energetic, 3=smashcut
            }


            @Override
            public void onNothingSelected(AdapterView<?> parent) {}
        });
        // --- Find all views from XML ---
        spectrogramViewA = findViewById(R.id.spectrogramViewA);
        spectrogramViewB = findViewById(R.id.spectrogramViewB);
        bpmTimelineView = findViewById(R.id.bpmTimelineView);


        btnLoadTrackA = findViewById(R.id.btnLoadTrackA);
        btnLoadTrackB = findViewById(R.id.btnLoadTrackB);
        btnShowBpm = findViewById(R.id.btnShowBpm);
        btnShowSpectrogram = findViewById(R.id.btnShowSpectrogram);
        txtTrackAName = findViewById(R.id.txtTrackAName);
        txtTrackBName = findViewById(R.id.txtTrackBName);
        txtBpmValue = findViewById(R.id.txtBpmValue);
        waveformViewA = findViewById(R.id.waveformViewA);
        waveformViewB = findViewById(R.id.waveformViewB);
        btnShowWaveforms = findViewById(R.id.btnShowWaveforms);

        btnStartMixing = findViewById(R.id.btnStartMixing);
        btnAnalysis    = findViewById(R.id.btnAnalysis);
        waveformMixView = findViewById(R.id.waveformMixView);

//        seekMix.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
//            @Override
//            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
//                if (fromUser && audioTrack != null) {
//                    int frame = progress;
//                    audioTrack.pause();
//                    audioTrack.flush();
//                    audioTrack.play();
//                }
//            }
//
//            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
//            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
//        });


        String neededPermission;
        if (Build.VERSION.SDK_INT >= 33) {
            neededPermission = Manifest.permission.READ_MEDIA_AUDIO;
        } else {
            neededPermission = Manifest.permission.READ_EXTERNAL_STORAGE;
        }


        if (checkSelfPermission(neededPermission) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{neededPermission}, STORAGE_REQUEST);
        }


        // Load Track A
        btnLoadTrackA.setOnClickListener(v -> openAudioFilePicker(REQ_TRACK_A));


        // Load Track B
        btnLoadTrackB.setOnClickListener(v -> openAudioFilePicker(REQ_TRACK_B));


        // Show BPM button: show BPM(s) for whichever tracks are loaded
        btnShowBpm.setOnClickListener(v -> onShowBpmClicked());


        // Show Spectrogram button: show spectrogram for whichever tracks are loaded,
        // toggling between A and B if both are present
        btnShowSpectrogram.setOnClickListener(v -> onShowSpectrogramClicked());


        btnShowWaveforms.setOnClickListener(v -> onShowWaveformsClicked());

        btnStartMixing.setOnClickListener(v -> {
            new Thread(this::runMixInBackground).start();
        });


        btnAnalysis.setOnClickListener(v -> {
            findViewById(R.id.analysisContainer).setVisibility(View.VISIBLE);
            findViewById(R.id.rowTrackA).setVisibility(View.GONE);
            findViewById(R.id.rowTrackB).setVisibility(View.GONE);
            findViewById(R.id.rowControls).setVisibility(View.GONE);
            findViewById(R.id.btnStartMixing).setVisibility(View.GONE);
            findViewById(R.id.mixOutputContainer).setVisibility(View.GONE);
        });

        Button btnBack = findViewById(R.id.btnBack);
        btnBack.setOnClickListener(v -> {
            findViewById(R.id.analysisContainer).setVisibility(View.GONE);
            findViewById(R.id.rowTrackA).setVisibility(View.VISIBLE);
            findViewById(R.id.rowTrackB).setVisibility(View.VISIBLE);
            findViewById(R.id.rowControls).setVisibility(View.VISIBLE);
            findViewById(R.id.btnStartMixing).setVisibility(View.VISIBLE);
            findViewById(R.id.mixOutputContainer).setVisibility(View.VISIBLE);
        });

        btnAnalysis.setVisibility(View.GONE);
        btnShowSpectrogram.setEnabled(false);


    }
    // Launch system file picker for an audio file
    private void openAudioFilePicker(int requestCode) {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("audio/*");  // allow any audio type (wav, mp3, etc.)
        startActivityForResult(intent, requestCode);
    }


    // Receive chosen file for Track A or Track B
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);


        if (resultCode != Activity.RESULT_OK || data == null) {
            return;
        }


        Uri uri = data.getData();
        if (uri == null) return;


        // Copy content from content:// URI into a local temp file with a real filesystem path
        String path = decodeUriToWavTempFile(uri);
        if (path == null) {
            Toast.makeText(this, "Failed to load file", Toast.LENGTH_SHORT).show();
            return;
        }


//        String displayName = uri.getLastPathSegment();
//        if (displayName == null) {
//            displayName = new File(path).getName();
//        }


        String displayName = null;


        // Try to get a nice human-readable name from the content resolver
        Cursor cursor = getContentResolver().query(uri, null, null, null, null);
        if (cursor != null) {
            try {
                int nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (nameIndex != -1 && cursor.moveToFirst()) {
                    displayName = cursor.getString(nameIndex);
                }
            } finally {
                cursor.close();
            }
        }


        if (displayName == null) {
            // Fallback to file name from the temp path
            displayName = new File(path).getName();
        }


        if (requestCode == REQ_TRACK_A) {
            trackAAnalyzed = false;
            cachedCueA = -1f;
            trackAPath = path;
            txtTrackAName.setText(displayName);
            Toast.makeText(this, "Track A loaded", Toast.LENGTH_SHORT).show();
        } else if (requestCode == REQ_TRACK_B) {
            trackBAnalyzed = false;
            cachedCueB = -1f;
            trackBPath = path;
            txtTrackBName.setText(displayName);
            Toast.makeText(this, "Track B loaded", Toast.LENGTH_SHORT).show();
        }


        Log.d("SPECTRO", "Selected file: " + displayName + " path=" + path);
    }


//    private String copyUriToTempFile(Uri uri) {
//        File outFile = null;
//        try (InputStream in = getContentResolver().openInputStream(uri)) {
//            if (in == null) return null;
//
//            outFile = File.createTempFile("track_", ".wav", getCacheDir());
//            try (FileOutputStream out = new FileOutputStream(outFile)) {
//                byte[] buffer = new byte[8192];
//                int len;
//                while ((len = in.read(buffer)) > 0) {
//                    out.write(buffer, 0, len);
//                }
//            }
//            return outFile.getAbsolutePath();
//        } catch (IOException e) {
//            Log.e("SPECTRO", "Error copying Uri to temp file", e);
//            if (outFile != null) {
//                //noinspection ResultOfMethodCallIgnored
//                outFile.delete();
//            }
//            return null;
//        }
//    }


    // NEW: Decode any audio (e.g. MP3) from Uri into a 16-bit PCM WAV temp file
// and return the path to that WAV file. C++ still thinks it's just a normal WAV.
    private String decodeUriToWavTempFile(Uri uri) {
        File outFile = null;
        try {
            // 1) Use MediaExtractor + MediaCodec to decode compressed audio to PCM
            android.media.MediaExtractor extractor = new android.media.MediaExtractor();
            extractor.setDataSource(this, uri, null);


            int audioTrackIndex = -1;
            for (int i = 0; i < extractor.getTrackCount(); i++) {
                android.media.MediaFormat format = extractor.getTrackFormat(i);
                String mime = format.getString(android.media.MediaFormat.KEY_MIME);
                if (mime != null && mime.startsWith("audio/")) {
                    audioTrackIndex = i;
                    break;
                }
            }


            if (audioTrackIndex < 0) {
                extractor.release();
                return null;
            }


            extractor.selectTrack(audioTrackIndex);
            android.media.MediaFormat format = extractor.getTrackFormat(audioTrackIndex);
            String mime = format.getString(android.media.MediaFormat.KEY_MIME);
            int sampleRate = format.getInteger(android.media.MediaFormat.KEY_SAMPLE_RATE);
            int channelCount = format.getInteger(android.media.MediaFormat.KEY_CHANNEL_COUNT);


            android.media.MediaCodec codec =
                    android.media.MediaCodec.createDecoderByType(mime);
            codec.configure(format, null, null, 0);
            codec.start();


            // 2) Decode to raw PCM bytes in memory
            java.io.ByteArrayOutputStream pcmOut = new java.io.ByteArrayOutputStream();


            boolean sawInputEof = false;
            boolean sawOutputEof = false;


            android.media.MediaCodec.BufferInfo info = new android.media.MediaCodec.BufferInfo();


            while (!sawOutputEof) {
                if (!sawInputEof) {
                    int inputBufferIndex = codec.dequeueInputBuffer(10000);
                    if (inputBufferIndex >= 0) {
                        java.nio.ByteBuffer inputBuffer = codec.getInputBuffer(inputBufferIndex);
                        if (inputBuffer != null) {
                            int sampleSize = extractor.readSampleData(inputBuffer, 0);
                            if (sampleSize < 0) {
                                codec.queueInputBuffer(inputBufferIndex, 0, 0, 0L,
                                        android.media.MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                                sawInputEof = true;
                            } else {
                                long presentationTimeUs = extractor.getSampleTime();
                                codec.queueInputBuffer(inputBufferIndex, 0, sampleSize,
                                        presentationTimeUs, 0);
                                extractor.advance();
                            }
                        }
                    }
                }


                int outputBufferIndex = codec.dequeueOutputBuffer(info, 10000);
                if (outputBufferIndex >= 0) {
                    java.nio.ByteBuffer outputBuffer = codec.getOutputBuffer(outputBufferIndex);
                    if (outputBuffer != null && info.size > 0) {
                        byte[] chunk = new byte[info.size];
                        outputBuffer.get(chunk);
                        outputBuffer.clear();
                        pcmOut.write(chunk);
                    }


                    codec.releaseOutputBuffer(outputBufferIndex, false);


                    if ((info.flags & android.media.MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                        sawOutputEof = true;
                    }
                }
            }


            codec.stop();
            codec.release();
            extractor.release();


            byte[] pcmBytes = pcmOut.toByteArray();


            // 3) Downmix to mono if needed and ensure 16-bit little-endian
            //    (Most decoders already give 16-bit PCM, so we'll just handle mono downmix.)
            int bytesPerSample = 2;
            int totalSamplesPerChannel = pcmBytes.length / (bytesPerSample * channelCount);
            short[] monoSamples = new short[totalSamplesPerChannel];


            java.nio.ByteBuffer bb = java.nio.ByteBuffer.wrap(pcmBytes)
                    .order(java.nio.ByteOrder.LITTLE_ENDIAN);


            for (int i = 0; i < totalSamplesPerChannel; i++) {
                int sum = 0;
                for (int ch = 0; ch < channelCount; ch++) {
                    short s = bb.getShort();
                    sum += s;
                }
                monoSamples[i] = (short)(sum / channelCount);
            }


            // 4) Write a proper WAV file header + monoSamples
            outFile = File.createTempFile("track_", ".wav", getCacheDir());
            try (FileOutputStream wavOut = new FileOutputStream(outFile)) {
                writeWavHeader(wavOut, sampleRate, 1, 16, monoSamples.length);
                // Write PCM data
                java.nio.ByteBuffer outBB = java.nio.ByteBuffer.allocate(monoSamples.length * 2)
                        .order(java.nio.ByteOrder.LITTLE_ENDIAN);
                for (short s : monoSamples) {
                    outBB.putShort(s);
                }
                wavOut.write(outBB.array());
            }


            return outFile.getAbsolutePath();


        } catch (Exception e) {
            Log.e("SPECTRO", "Error decoding MP3 to WAV", e);
            if (outFile != null) {
                //noinspection ResultOfMethodCallIgnored
                outFile.delete();
            }
            return null;
        }
    }


    // Write a minimal 16-bit PCM WAV header
    private void writeWavHeader(
            FileOutputStream out,
            int sampleRate,
            int channels,
            int bitsPerSample,
            int numSamples
    ) throws IOException {


        int byteRate = sampleRate * channels * bitsPerSample / 8;
        int blockAlign = channels * bitsPerSample / 8;
        int dataSize = numSamples * channels * bitsPerSample / 8;
        int chunkSize = 36 + dataSize;


        java.nio.ByteBuffer bb = java.nio.ByteBuffer.allocate(44)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);


        // RIFF header
        bb.put("RIFF".getBytes("US-ASCII"));
        bb.putInt(chunkSize);
        bb.put("WAVE".getBytes("US-ASCII"));


        // fmt chunk
        bb.put("fmt ".getBytes("US-ASCII"));
        bb.putInt(16);                       // Subchunk1Size for PCM
        bb.putShort((short) 1);              // AudioFormat = PCM
        bb.putShort((short) channels);
        bb.putInt(sampleRate);
        bb.putInt(byteRate);
        bb.putShort((short) blockAlign);
        bb.putShort((short) bitsPerSample);


        // data chunk
        bb.put("data".getBytes("US-ASCII"));
        bb.putInt(dataSize);


        out.write(bb.array());
    }






    @Override
    public void onRequestPermissionsResult(
            int requestCode,
            @NonNull String[] permissions,
            @NonNull int[] grantResults
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);


        if (requestCode == STORAGE_REQUEST) {
            if (grantResults.length > 0 &&
                    grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Permission granted. You can now load audio files.", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(this, "Storage permission is required to load tracks.", Toast.LENGTH_LONG).show();
            }
        }
    }
    // --- BUTTON LOGIC ---


    // When "Show BPM" is clicked:
    // - If no tracks loaded: do nothing, show toast
    // - If only A loaded: show BPM for A
    // - If only B loaded: show BPM for B
    // - If both loaded: show BPM for A and B
    private void onShowBpmClicked() {
        boolean hasA = (trackAPath != null);
        boolean hasB = (trackBPath != null);


        if (!hasA && !hasB) {
            Toast.makeText(this, "Please load Track A or B first", Toast.LENGTH_SHORT).show();
            txtBpmValue.setText("BPM: --");
            return;
        }


        StringBuilder sb = new StringBuilder();


        if (hasA) {
            float bpmA = analyzeTrackForBpm(trackAPath, "A");
            if (bpmA > 0) {
                if (sb.length() > 0) sb.append(" | ");
                sb.append(String.format("A: %.1f", bpmA));
            }
        }


        if (hasB) {
            float bpmB = analyzeTrackForBpm(trackBPath, "B");
            if (bpmB > 0) {
                if (sb.length() > 0) sb.append(" | ");
                sb.append(String.format("B: %.1f", bpmB));
            }
        }


        if (sb.length() == 0) {
            txtBpmValue.setText("BPM: --");
        } else {
            txtBpmValue.setText("BPM " + sb.toString());
        }
    }


    private void onShowSpectrogramClicked() {
        boolean hasA = (trackAPath != null);
        boolean hasB = (trackBPath != null);


        if (!hasA && !hasB) {
            Toast.makeText(this, "Please load Track A or B first", Toast.LENGTH_SHORT).show();
            spectrogramViewA.setImageBitmap(null);
            spectrogramViewB.setImageBitmap(null);
            return;
        }


        if (hasA) {
            Bitmap bmpA = renderSpectrogramBitmapForPath(trackAPath, "A");
            spectrogramViewA.setImageBitmap(bmpA);
        } else {
            spectrogramViewA.setImageBitmap(null);
        }


        if (hasB) {
            Bitmap bmpB = renderSpectrogramBitmapForPath(trackBPath, "B");
            spectrogramViewB.setImageBitmap(bmpB);
        } else {
            spectrogramViewB.setImageBitmap(null);
        }
    }


//    private void onShowWaveformsClicked() {
//        boolean hasA = (trackAPath != null);
//        boolean hasB = (trackBPath != null);
//
//
//        if (!hasA && !hasB) {
//            Toast.makeText(this, "Please load Track A or B first", Toast.LENGTH_SHORT).show();
//            waveformViewA.setImageBitmap(null);
//            waveformViewB.setImageBitmap(null);
//            return;
//        }
//
//
//        // Choose a reasonable display size
//        int widthPx = 1400;  // can tweak; Android will scale it to the ImageView
//        int heightPx = 350;
//
//
//        if (hasA) {
//            Bitmap wa = renderWaveformWithBeats(trackAPath, "A", widthPx, heightPx);
//            waveformViewA.setImageBitmap(wa);
//        } else {
//            waveformViewA.setImageBitmap(null);
//        }
//
//
//        if (hasB) {
//            Bitmap wb = renderWaveformWithBeats(trackBPath, "B", widthPx, heightPx);
//            waveformViewB.setImageBitmap(wb);
//        } else {
//            waveformViewB.setImageBitmap(null);
//        }
//    }

    private void onShowWaveformsClicked() {
        Toast.makeText(this,
                "Waveform analysis disabled for performance",
                Toast.LENGTH_SHORT).show();
    }




    // Analyze one track just enough to get its BPM (and log some info)
    private float analyzeTrackForBpm(String path, String tag) {
        File f = new File(path);
        Log.d("SPECTRO", "Track " + tag + " exists? " + f.exists() + " length=" + f.length());


        int numFrames = initFromWav(path);
        Log.d("SPECTRO", "Track " + tag + " numFrames = " + numFrames);


        if (numFrames <= 0) {
            Toast.makeText(this, "Failed to analyze Track " + tag, Toast.LENGTH_SHORT).show();
            return -1f;
        }


        float bpm = getEstimatedBpm();
        int numBeats = getNumBeats();
        Log.d("SPECTRO", "Track " + tag + " BPM = " + bpm + ", numBeats = " + numBeats);


        if (numBeats > 0) {
            float[] beatTimes = new float[numBeats];
            getBeatTimes(beatTimes);
            for (int i = 0; i < Math.min(5, numBeats); i++) {
                Log.d("SPECTRO", "Track " + tag + " Beat " + i + " at t = " + beatTimes[i] + " sec");
            }
        }


//        int numBeats = getNumBeats();
//        Log.d("SPECTRO", "Track " + tag + " numBeats = " + numBeats);
//
//        if (numBeats > 1) {
//            float[] beatTimes = new float[numBeats];
//            getBeatTimes(beatTimes);
//
//            // Log first few beats
//            int limit = Math.min(5, numBeats);
//            for (int i = 0; i < limit; i++) {
//                Log.d("SPECTRO", "Track " + tag + " Beat " + i + " at t = " + beatTimes[i] + " sec");
//            }
//
//            // Compute an approximate BPM from beat intervals as a sanity check
//            float sumDiff = 0f;
//            int count = 0;
//            for (int i = 1; i < numBeats; i++) {
//                float diff = beatTimes[i] - beatTimes[i - 1];
//                if (diff > 0.1f && diff < 2.0f) { // ignore crazy outliers
//                    sumDiff += diff;
//                    count++;
//                }
//            }
//            if (count > 0) {
//                float avgPeriod = sumDiff / count; // seconds per beat
//                float bpmFromBeats = 60f / avgPeriod;
//                Log.d("SPECTRO", "Track " + tag + " BPM (from beat intervals) = " + bpmFromBeats);
//            }
//        }




        // We're only returning BPM here. Spectrogram will be handled separately.
        return bpm;
    }


//    private void drawSpectrogramForPath(String wavPath, String tag) {
//        File f = new File(wavPath);
//        Log.d("SPECTRO", "Track " + tag + " exists? " + f.exists() + " length=" + f.length());
//
//        int numFrames = initFromWav(wavPath);     // C++: fill gSpectrogram, gOnset, gEstimatedBpm, etc.
//        Log.d("SPECTRO", "Track " + tag + " numFrames = " + numFrames);
//
//        int onsetLen = getOnsetLength();
//        Log.d("SPECTRO", "Track " + tag + " onsetLen = " + onsetLen);
//
//        // --- Query tempo + beats from native (same as before) ---
//        float bpm = getEstimatedBpm();
//        Log.d("SPECTRO", "Track " + tag + " Estimated BPM = " + bpm);
//
//        int numBeats = getNumBeats();
//        Log.d("SPECTRO", "Track " + tag + " Num beats = " + numBeats);
//
//        if (numBeats > 0) {
//            float[] beatTimes = new float[numBeats];
//            getBeatTimes(beatTimes);
//            int limit = Math.min(5, numBeats);
//            for (int i = 0; i < limit; i++) {
//                Log.d("SPECTRO", "Track " + tag + " Beat " + i + " at t = " + beatTimes[i] + " sec");
//            }
//        }
//
//        if (numFrames <= 0) {
//            // Native code failed to load WAV or compute spectrogram
//            Toast.makeText(this, "Failed to analyze Track " + tag, Toast.LENGTH_SHORT).show();
//            return;
//        }
//
//        // --- Exactly your original drawing code, but using numFrames and NUM_BINS ---
//
//        Bitmap bitmap = Bitmap.createBitmap(numFrames, NUM_BINS, Bitmap.Config.ARGB_8888);
//        Canvas canvas = new Canvas(bitmap);
//        Paint paint = new Paint();
//        float[] column = new float[NUM_BINS];
//
//        for (int frame = 0; frame < numFrames; frame++) {
//            getSpectrogramColumn(frame, column);  // C++: copy one column of gSpectrogram
//
//            for (int bin = 0; bin < NUM_BINS; bin++) {
//                float v = column[bin];
//                if (v < 0f) v = 0f;
//                if (v > 1f) v = 1f;
//
//                int level = (int) (v * 255f);
//                int color = Color.rgb(level, level, level);
//                paint.setColor(color);
//
//                // invert vertical axis: bin 0 at bottom
//                int y = NUM_BINS - 1 - bin;
//                canvas.drawPoint((float) frame, (float) y, paint);
//            }
//        }
//
//        spectrogramView.setImageBitmap(bitmap);
//        Toast.makeText(this, "Showing spectrogram for Track " + tag, Toast.LENGTH_SHORT).show();
//    }


    private Bitmap renderSpectrogramBitmapForPath(String wavPath, String tag) {
        File f = new File(wavPath);


        Log.d("SPECTRO", "Track " + tag + " exists? " + f.exists() + " length=" + f.length());


        int numFrames = initFromWav(wavPath); // C++: computes gSpectrogram, gOnset, BPM, beats
        Log.d("SPECTRO", "Track " + tag + " numFrames = " + numFrames);


        int onsetLen = getOnsetLength();
        Log.d("SPECTRO", "Track " + tag + " onsetLen = " + onsetLen);


        float bpm = getEstimatedBpm();
        Log.d("SPECTRO", "Track " + tag + " Estimated BPM = " + bpm);


        int numBeats = getNumBeats();
        Log.d("SPECTRO", "Track " + tag + " Num beats = " + numBeats);


        if (numBeats > 0) {
            float[] beatTimes = new float[numBeats];
            getBeatTimes(beatTimes);
            int limit = Math.min(5, numBeats);
            for (int i = 0; i < limit; i++) {
                Log.d("SPECTRO", "Track " + tag + " Beat " + i + " at t = " + beatTimes[i] + " sec");
            }
        }


        if (numFrames <= 0) {
            Log.d("SPECTRO", "Track " + tag + " numFrames <= 0, no spectrogram");
            return null;
        }


        // ---- Better scaling: limit width and resample frames ----
        int maxDisplayWidth = 800; // or tweak as you like
        int displayWidth = Math.min(numFrames, maxDisplayWidth);


        Bitmap bitmap = Bitmap.createBitmap(displayWidth, NUM_BINS, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bitmap);
        Paint paint = new Paint();
        float[] column = new float[NUM_BINS];


        for (int x = 0; x < displayWidth; x++) {
            // Map display column x → frame index in [0, numFrames-1]
            int frameIndex = (int) ((long) x * numFrames / displayWidth);
            getSpectrogramColumn(frameIndex, column);


            for (int bin = 0; bin < NUM_BINS; bin++) {
                float v = column[bin];
                if (v < 0f) v = 0f;
                if (v > 1f) v = 1f;


                // Slight gamma to emphasize mid-levels
                v = (float) Math.pow(v, 0.6);


                // Map v in [0,1] to a hue from 240 (blue) to 0 (red)
                float hue = (1f - v) * 240f;  // low energy = blue, high = red
                float[] hsv = new float[]{hue, 1f, v < 0.1f ? 0.1f : v}; // keep very dark values slightly visible




                int color = Color.HSVToColor(hsv);


                paint.setColor(color);


                int y = NUM_BINS - 1 - bin;
                canvas.drawPoint((float) x, (float) y, paint);
            }
        }




        return bitmap;
    }


    private Bitmap renderWaveformWithBeats(String wavPath, String tag,
                                           int widthPx, int heightPx) {


        try {
            int numFrames = initFromWav(wavPath);
            Log.d("INIT", "Initialized " + tag + " with " + numFrames + " frames");


            // ---------- 1) READ WAV HEADER (same as before) ----------
            java.io.RandomAccessFile raf = new java.io.RandomAccessFile(wavPath, "r");


            raf.seek(12); // skip "RIFF" + size + "WAVE"


            byte[] chunkId = new byte[4];
            int fmtSize;
            while (true) {
                int read = raf.read(chunkId);
                if (read < 4) { raf.close(); return null; }
                fmtSize = Integer.reverseBytes(raf.readInt());
                String id = new String(chunkId, "US-ASCII");
                if ("fmt ".equals(id)) break;
                raf.seek(raf.getFilePointer() + fmtSize);
            }


            short audioFormat   = Short.reverseBytes(raf.readShort());
            short numChannels   = Short.reverseBytes(raf.readShort());
            int   sampleRate    = Integer.reverseBytes(raf.readInt());
            int   byteRate      = Integer.reverseBytes(raf.readInt());
            short blockAlign    = Short.reverseBytes(raf.readShort());
            short bitsPerSample = Short.reverseBytes(raf.readShort());


            if (fmtSize > 16) {
                raf.seek(raf.getFilePointer() + (fmtSize - 16));
            }


            int dataSize;
            while (true) {
                int read = raf.read(chunkId);
                if (read < 4) { raf.close(); return null; }
                int subSize = Integer.reverseBytes(raf.readInt());
                String id = new String(chunkId, "US-ASCII");
                if ("data".equals(id)) {
                    dataSize = subSize;
                    break;
                }
                raf.seek(raf.getFilePointer() + subSize);
            }


            long dataStart       = raf.getFilePointer();
            int  bytesPerSample  = bitsPerSample / 8;
            int  totalSamples    = dataSize / bytesPerSample; // mono
            float durationSec    = (float) totalSamples / (float) sampleRate;


            // ---------- 2) DRAW WAVEFORM ----------
            Bitmap bmp = Bitmap.createBitmap(widthPx, heightPx, Bitmap.Config.ARGB_8888);
            Canvas canvas = new Canvas(bmp);
            canvas.drawColor(Color.BLACK);


            // reserve a bit of space at the bottom for labels:
            int labelHeightPx = 40;
            float waveTop     = 0f;
            float waveBottom  = heightPx - labelHeightPx;
            float midY        = (waveTop + waveBottom) / 2f;
            float halfAmpPix  = (waveBottom - waveTop) / 2f;


            Paint paintWave = new Paint();
            paintWave.setColor(Color.WHITE);
            paintWave.setStrokeWidth(1f);


            int samplesPerCol = Math.max(1, totalSamples / widthPx);
            byte[] buffer = new byte[samplesPerCol * bytesPerSample];
            java.nio.ByteBuffer bb = java.nio.ByteBuffer.wrap(buffer)
                    .order(java.nio.ByteOrder.LITTLE_ENDIAN);


            for (int x = 0; x < widthPx; x++) {
                int sampleIndexStart = x * samplesPerCol;
                long byteOffset = dataStart + (long) sampleIndexStart * bytesPerSample;
                if (byteOffset >= dataStart + dataSize) break;


                raf.seek(byteOffset);
                int toRead = Math.min(buffer.length,
                        (int) ((dataStart + dataSize) - byteOffset));
                int read = raf.read(buffer, 0, toRead);
                if (read <= 0) break;


                bb.rewind();
                float minVal = 1f;
                float maxVal = -1f;


                int samplesRead = read / bytesPerSample;
                for (int i = 0; i < samplesRead; i++) {
                    short s = bb.getShort();
                    float v = s / 32768f;
                    if (v < minVal) minVal = v;
                    if (v > maxVal) maxVal = v;
                }


                float y1 = midY - minVal * halfAmpPix;
                float y2 = midY - maxVal * halfAmpPix;
                canvas.drawLine(x, y1, x, y2, paintWave);
            }


            raf.close();


            // ---------- 3) CUE POINT (needs initFromWav already called) ----------
            float cue = getCuePoint(currentTransitionMode);
            Log.d("CUE", "Cue point for " + tag + " = " + cue);


            // ---------- 4) BEAT LINES ----------
            int numBeats = getNumBeats();
            if (numBeats > 0) {
                float[] beatTimes = new float[numBeats];
                getBeatTimes(beatTimes);


                Paint paintBeat = new Paint();
                paintBeat.setColor(Color.CYAN);
                paintBeat.setStrokeWidth(2f);


                for (int i = 0; i < numBeats; i++) {
                    float t = beatTimes[i];
                    if (t < 0 || t > durationSec) continue;
                    int x = (int) (t / durationSec * widthPx);
                    canvas.drawLine(x, waveTop, x, waveBottom, paintBeat);
                }
            }


            // ---------- 5) SECTION BOUNDARIES ----------
            int numBoundaries = getNumSections();           // = number of boundary times
            float[] sectionTimes = new float[numBoundaries];
            if (numBoundaries > 0) {
                getSectionBoundaries(sectionTimes);


                Paint paintSection = new Paint();
                paintSection.setColor(Color.MAGENTA);
                paintSection.setStrokeWidth(4f);


                for (int i = 0; i < numBoundaries; i++) {
                    float tSec = sectionTimes[i];
                    if (tSec < 0 || tSec > durationSec) continue;
                    int x = (int) (tSec / durationSec * widthPx);
                    canvas.drawLine(x, waveTop, x, waveBottom, paintSection);
                }
            }


            // ---------- 6) SECTION ENERGY LABELS (HIGH / LOW) ----------
            // There is one energy label per *section*, i.e. (numBoundaries - 1)
            if (numBoundaries > 1) {
                int numSections = numBoundaries - 1;
                int[] sectionEnergy = new int[numSections];
                getSectionEnergyLabels(sectionEnergy);  // fills +1 or -1


                Paint labelPaint = new Paint();
                labelPaint.setColor(Color.WHITE);
                labelPaint.setTextSize(24f);
                labelPaint.setAntiAlias(true);
                labelPaint.setTextAlign(Paint.Align.CENTER);


                float textY = heightPx - 10f;  // a little above the bottom


                for (int i = 0; i < numSections; i++) {
                    float tStart = sectionTimes[i];
                    float tEnd   = sectionTimes[i + 1];


                    // clamp to track duration
                    if (tStart >= durationSec || tEnd <= 0f) continue;


                    float tCenter = 0.5f * (tStart + tEnd);
                    if (tCenter < 0f) tCenter = 0f;
                    if (tCenter > durationSec) tCenter = durationSec;


                    int xCenter = (int) (tCenter / durationSec * widthPx);


                    String label = (sectionEnergy[i] == +1) ? "HIGH" : "LOW";
                    canvas.drawText(label, xCenter, textY, labelPaint);
                }
            }


            // ---------- 7) CUE POINT MARKER ----------
            if (currentTransitionMode > 0 && cue > 0 && cue <= durationSec) {
                int xCue = (int) (cue / durationSec * widthPx);
                Paint cuePaint = new Paint();
                cuePaint.setColor(Color.GREEN);
                cuePaint.setStrokeWidth(5f);
                canvas.drawLine(xCue, waveTop, xCue, heightPx, cuePaint);
            }


            return bmp;


        } catch (Exception e) {
            Log.e("WAVE", "Error rendering waveform for " + tag, e);
            return null;
        }
    }


//    private void onStartMixingClicked() {
//        if (trackAPath == null || trackBPath == null) {
//            Toast.makeText(this, "Load both tracks first", Toast.LENGTH_SHORT).show();
//            return;
//        }
//
//        // Ensure analysis has been run
//        initFromWav(trackAPath);
//        float cueA = getCuePoint(currentTransitionMode);
//
//        initFromWav(trackBPath);
//        float cueB = getCuePoint(currentTransitionMode);
//
//        if (cueA <= 0 || cueB <= 0) {
//            Toast.makeText(this, "Failed to compute cue points", Toast.LENGTH_SHORT).show();
//            return;
//        }
//
//        mixedAudio = renderMix(trackAPath, trackBPath, cueA, cueB);
//
//        if (mixedAudio == null || mixedAudio.length == 0) {
//            Toast.makeText(this, "Mixing failed", Toast.LENGTH_SHORT).show();
//            return;
//        }
//
//        Bitmap mixWave = drawWaveformFromFloatArray(mixedAudio, 1400, 300);
//        waveformMixView.setImageBitmap(mixWave);
//
//        seekMix.setMax(mixedAudio.length);
//        seekMix.setProgress(0);
//
//        Toast.makeText(this, "Mix ready", Toast.LENGTH_SHORT).show();
//    }

    private Bitmap drawWaveformFromFloatArray(float[] audio, int sampleRate, float [] beatTimesSec,float mixTransitionEndSec, int width, int height) {
        Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bmp);
        canvas.drawColor(Color.BLACK);

        Paint paint = new Paint();
        paint.setColor(Color.GREEN);
        paint.setStrokeWidth(1f);

        int samplesPerPixel = Math.max(1, audio.length / width);
        float midY = height / 2f;
        float scale = height / 2f;

        for (int x = 0; x < width; x++) {
            int start = x * samplesPerPixel;
            float min = 1f, max = -1f;

            for (int i = 0; i < samplesPerPixel && start + i < audio.length; i++) {
                float v = audio[start + i];
                min = Math.min(min, v);
                max = Math.max(max, v);
            }

            canvas.drawLine(
                    x,
                    midY - min * scale,
                    x,
                    midY - max * scale,
                    paint
            );
        }

        // ---------- BEAT MARKERS ----------
        if (beatTimesSec != null && beatTimesSec.length > 0) {

            // ---------- BEAT MARKERS (A vs B) ----------
            if (beatTimesSec != null && beatTimesSec.length > 0) {

                Paint paintA = new Paint();
                paintA.setColor(Color.CYAN);       // Track A beats
                paintA.setStrokeWidth(2f);
                paintA.setAlpha(200);

                Paint paintB = new Paint();
                paintB.setColor(Color.YELLOW);     // Track B beats
                paintB.setStrokeWidth(2f);
                paintB.setAlpha(200);

                float durationSec = (float) audio.length / (float) sampleRate;

                for (float t : beatTimesSec) {
                    if (t <= 0f || t > durationSec) continue;

                    int xBeat = (int) ((t / durationSec) * width);

                    if (mixTransitionEndSec > 0 && t <= mixTransitionEndSec) {
                        // Track A beat
                        canvas.drawLine(xBeat, 0, xBeat, height, paintA);
                    } else {
                        // Track B beat
                        canvas.drawLine(xBeat, 0, xBeat, height, paintB);
                    }
                }
            }

        }

        return bmp;
    }

//    private void runMixInBackground() {
//        if (trackAPath == null || trackBPath == null) {
//            runOnUiThread(() ->
//                    Toast.makeText(this, "Load both tracks first", Toast.LENGTH_SHORT).show()
//            );
//            return;
//        }
//
//        // ⚠️ Analyze ONCE
//        analyzeTrackOnce('A');
//        analyzeTrackOnce('B');
//
//        float cueA = cachedCueA;
//        float cueB = cachedCueB;
//
//
//        long t0 = System.currentTimeMillis();
//        float[] mix = renderMix(trackAPath, trackBPath, cueA, cueB);
//        long t1 = System.currentTimeMillis();
//
//        Log.d("MIX", "renderMix time = " + (t1 - t0) + " ms");
//
//        if (mix == null || mix.length == 0) {
//            runOnUiThread(() ->
//                    Toast.makeText(this, "Mixing failed", Toast.LENGTH_SHORT).show()
//            );
//            return;
//        }
//
//        runOnUiThread(() -> {
//            Bitmap bmp = drawWaveformFromFloatArray(mix, 800, 250);
//            waveformMixView.setImageBimap(bmp);
//            playMixedAudio(mix);
//            Toast.makeText(this, "Mix ready", Toast.LENGTH_SHORT).show();
//        });
//    }

    private void runMixInBackground() {

        // Disable UI actions that could trigger JNI calls
        runOnUiThread(() -> {
            btnShowBpm.setEnabled(false);
            btnShowSpectrogram.setEnabled(false);
            btnShowWaveforms.setEnabled(false);
            btnAnalysis.setEnabled(false);
            btnStartMixing.setEnabled(false);
        });

        new Thread(() -> {

            synchronized (JNI_LOCK) {   // 🔒 CRITICAL LINE

                try {
                    if (trackAPath == null || trackBPath == null) {
                        runOnUiThread(() ->
                                Toast.makeText(this, "Load both tracks first", Toast.LENGTH_SHORT).show()
                        );
                        return;
                    }

                    // ---- STRICT ORDER (DO NOT CHANGE) ----

                    // Analyze Track A
                    initFromWav(trackAPath);
                    float bpmA = getEstimatedBpm();
                    float cueA = getCuePoint(currentTransitionMode);

                    // Analyze Track B
                    initFromWav(trackBPath);
                    float bpmB = getEstimatedBpm();
                    float cueB = getCuePoint(currentTransitionMode);

                    if (cueA <= 0 || cueB <= 0) {
                        runOnUiThread(() ->
                                Toast.makeText(this, "Failed to compute cue points", Toast.LENGTH_SHORT).show()
                        );
                        return;
                    }

                    long t0 = System.currentTimeMillis();
                    float[] mix = renderMix(trackAPath, trackBPath, bpmA, bpmB, cueA, cueB);

                    // ---- FETCH BPM TIMELINE AFTER MIX ----
                    int bpmCount = getMixBpmCount();
                    if (bpmCount > 0) {
                        mixBpmValues = new float[bpmCount];
                        mixBpmTimes  = new float[bpmCount];
                        getMixBpmTimeline(mixBpmValues);
                        getMixBpmTimes(mixBpmTimes);
                    } else {
                        mixBpmValues = null;
                        mixBpmTimes  = null;
                    }


// ---- FETCH BEATS IMMEDIATELY AFTER MIX ----
                    int numBeats = getMixBeatCount();
                    if (numBeats > 0) {
                        mixBeatTimes = new float[numBeats];
                        getMixBeatTimes(mixBeatTimes);
                    } else {
                        mixBeatTimes = null;
                    }

                    float mixTransitionEndSec = getMixTransitionEndSec();

                    long t1 = System.currentTimeMillis();

                    Log.d("MIX", "renderMix time = " + (t1 - t0) + " ms");

                    if (mix == null || mix.length == 0) {
                        runOnUiThread(() ->
                                Toast.makeText(this, "Mixing failed", Toast.LENGTH_SHORT).show()
                        );
                        return;
                    }

                    String wavPath = writeMixToWavFile(mix, playbackSampleRate);

                    runOnUiThread(() -> {
                        if (wavPath != null) {
                            mixWavPath = wavPath;

                            // Draw waveform ONCE from float array
                            mixWaveformBitmap = drawWaveformFromFloatArray(
                                    mix,
                                    playbackSampleRate,
                                    mixBeatTimes,
                                    mixTransitionEndSec,
                                    1000,
                                    300
                            );

                            waveformMixView.setImageBitmap(mixWaveformBitmap);

                            Bitmap bpmBmp = drawBpmTimeline(
                                    mixBpmValues,
                                    mixBpmTimes,
                                    1000,
                                    200
                            );

                            if (bpmBmp != null) {
                                bpmTimelineView.setImageBitmap(bpmBmp);
                            }


                            // Play from PCM (still fine)
                            playMixedAudio(mix);

                            Toast.makeText(this, "Final mix ready", Toast.LENGTH_SHORT).show();
                        } else {
                            Toast.makeText(this, "Failed to save mix", Toast.LENGTH_SHORT).show();
                        }
                    });


                } catch (Exception e) {
                    Log.e("MIX", "Java-side exception", e);
                }

            } // 🔓 JNI_LOCK released here

            // Re-enable UI
            runOnUiThread(() -> {
                btnShowBpm.setEnabled(true);
                btnShowSpectrogram.setEnabled(true);
                btnShowWaveforms.setEnabled(true);
                btnAnalysis.setEnabled(true);
                btnStartMixing.setEnabled(true);
            });

        }).start();
    }


    private short[] floatToPcm16(float[] audio) {
        short[] pcm = new short[audio.length];
        for (int i = 0; i < audio.length; i++) {
            float v = Math.max(-1f, Math.min(1f, audio[i]));
            pcm[i] = (short) (v * 32767);
        }
        return pcm;
    }


    private void playMixedAudio(float[] mix) {
        if (mix == null || mix.length == 0) return;

        short[] pcm = floatToPcm16(mix);

        int bufferSize = AudioTrack.getMinBufferSize(
                playbackSampleRate,
                AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_16BIT
        );

        audioTrack = new AudioTrack(
                AudioManager.STREAM_MUSIC,
                playbackSampleRate,
                AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize,
                AudioTrack.MODE_STREAM
        );

        audioTrack.play();
        isPlaying = true;

        // Write audio on a background thread
        new Thread(() -> {
            audioTrack.write(pcm, 0, pcm.length);
            audioTrack.stop();
            audioTrack.release();
            isPlaying = false;
        }).start();
    }

    private void analyzeTrackOnce(char which) {
        if (which == 'A' && trackAPath != null && !trackAAnalyzed) {
            initFromWav(trackAPath);
            cachedCueA = getCuePoint(currentTransitionMode);
            trackAAnalyzed = true;
        }
        if (which == 'B' && trackBPath != null && !trackBAnalyzed) {
            initFromWav(trackBPath);
            cachedCueB = getCuePoint(currentTransitionMode);
            trackBAnalyzed = true;
        }
    }

    private String writeMixToWavFile(float[] mix, int sampleRate) {
        try {
            File outFile = File.createTempFile("final_mix_", ".wav", getCacheDir());

            try (FileOutputStream fos = new FileOutputStream(outFile)) {
                // Write WAV header
                writeWavHeader(fos, sampleRate, 1, 16, mix.length);

                // Convert float → PCM16
                ByteBuffer bb = ByteBuffer.allocate(mix.length * 2)
                        .order(ByteOrder.LITTLE_ENDIAN);

                for (float v : mix) {
                    float clamped = Math.max(-1f, Math.min(1f, v));
                    bb.putShort((short)(clamped * 32767));
                }

                fos.write(bb.array());
            }

            return outFile.getAbsolutePath();

        } catch (IOException e) {
            Log.e("WAV", "Failed to write mix WAV", e);
            return null;
        }
    }

    private Bitmap drawBpmTimeline(
            float[] bpmValues,
            float[] timeSec,
            int width,
            int height
    ) {
        if (bpmValues == null || timeSec == null || bpmValues.length == 0)
            return null;

        Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bmp);
        canvas.drawColor(Color.BLACK);

        Paint axisPaint = new Paint();
        axisPaint.setColor(Color.GRAY);
        axisPaint.setStrokeWidth(2f);

        Paint linePaint = new Paint();
        linePaint.setColor(Color.GREEN);
        linePaint.setStrokeWidth(3f);
        linePaint.setAntiAlias(true);

        // Time range
        float tMin = timeSec[0];
        float tMax = timeSec[timeSec.length - 1];

        // BPM range
        float bpmMin = bpmValues[0];
        float bpmMax = bpmValues[0];
        for (float v : bpmValues) {
            bpmMin = Math.min(bpmMin, v);
            bpmMax = Math.max(bpmMax, v);
        }

        // Padding so line doesn’t touch edges
        float pad = 0.1f * (bpmMax - bpmMin + 1e-3f);
        bpmMin -= pad;
        bpmMax += pad;

        // Axes
        canvas.drawLine(0, height - 1, width, height - 1, axisPaint);
        canvas.drawLine(0, 0, 0, height, axisPaint);

        // Plot BPM curve
        for (int i = 1; i < bpmValues.length; i++) {
            float x0 = (timeSec[i - 1] - tMin) / (tMax - tMin) * width;
            float y0 = height - (bpmValues[i - 1] - bpmMin) / (bpmMax - bpmMin) * height;

            float x1 = (timeSec[i] - tMin) / (tMax - tMin) * width;
            float y1 = height - (bpmValues[i] - bpmMin) / (bpmMax - bpmMin) * height;

            canvas.drawLine(x0, y0, x1, y1, linePaint);
        }

        return bmp;
    }









}

