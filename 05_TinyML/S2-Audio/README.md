# Audio Feature Extraction for TinyML

A comprehensive guide to extracting audio features for machine learning on embedded devices.

---

## Overview

Audio Feature Extraction is the process of converting raw audio signals into compact, meaningful representations that machine learning models can efficiently process. This is especially critical for **TinyML** applications where computational resources are severely limited.

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE BIG PICTURE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   🎤 Microphone → 📊 Features → 🧠 ML Model → 🎯 Prediction     │
│                                                                 │
│   "Hey Siri"   →   MFCCs    →   Neural Net →   "Wake Word!"    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why Feature Extraction?

### The Problem

| Metric | Raw Audio | MFCCs |
|--------|-----------|-------|
| Data per second | 16,000 values | ~1,300 values |
| Memory needed | 64 KB | 5 KB |
| Useful for ML? | ❌ Redundant info | ✅ Meaningful features |

### The Solution

Feature extraction reduces data size while **preserving information relevant to human speech perception**.

---

## The Pipeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE PIPELINE                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────┐   ┌─────────┐   ┌─────┐   ┌───────────┐   ┌─────┐   ┌──────┐ │
│  │Raw Audio│ → │ Framing │ → │ FFT │ → │Spectrogram│ → │ Mel │ → │MFCCs │ │
│  └─────────┘   └─────────┘   └─────┘   └───────────┘   └─────┘   └──────┘ │
│                                                                            │
│   [16,000]  →  [98 × 400] → [98×257] →  [98 × 257]  → [98×40] → [98×13]  │
│                                                                            │
│                      📉 92% Data Reduction                                 │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Raw Waveform

### What
Converting continuous sound waves into discrete digital samples.

### Why
Computers can only process numbers, not continuous analog signals.

### How
Sample the audio at regular intervals (typically 16,000 times per second).

```
Analog Wave              Digital Samples
    ╭──╮                  • • •
   ╱    ╲      ────►     •     •
──╱      ╲──              •   •

Sample Rate: 16,000 Hz
1 second = 16,000 values
```

### Key Parameters
| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| Sample Rate | 16,000 Hz | Samples per second |
| Bit Depth | 16-bit | Resolution per sample |
| Duration | 1 second | Length of audio clip |

---

## Phase 2: Framing

### What
Splitting the audio into small overlapping segments (frames).

### Why
Sound characteristics change over time. Small frames (~25ms) are "quasi-stationary" and can be analyzed independently.

### How

```
Full Signal: [════════════════════════════════════════]

Frames with 50% overlap:
           [████████]
               [████████]
                   [████████]
                       [████████]
```

### Key Parameters
| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| Frame Length | 25 ms (400 samples) | Size of each frame |
| Hop Length | 10 ms (160 samples) | Step between frames |
| Overlap | 60% | Frame overlap percentage |

### Windowing
Apply a window function (Hamming) to avoid spectral leakage:

```
Before:  │████████│  (sharp edges = artifacts)
After:   ╱▄▄▄▄▄▄▄▄╲  (smooth edges = clean spectrum)
```

---

## Phase 3: Fourier Transform (FFT)

### What
Converting each frame from time domain to frequency domain.

### Why
Humans recognize sounds by their frequency content, not by the raw waveform shape.

### How

```
TIME DOMAIN              FREQUENCY DOMAIN
(When?)                  (What frequencies?)

   ╭─╮╭──╮                    █
 ─╯   ╰──╰─      FFT         █ █
                ────►        █ █ █
[400 samples]              [257 bins]
```

### Key Concepts
- **FFT**: Fast Fourier Transform - efficient algorithm for DFT
- **Magnitude**: Energy at each frequency
- **Nyquist Frequency**: Maximum representable frequency = Sample Rate / 2

### Output
Each frequency bin represents energy at a specific frequency:
```
Bin 0   → 0 Hz (DC component)
Bin 1   → 31.25 Hz
Bin 2   → 62.5 Hz
...
Bin 256 → 8000 Hz (Nyquist)
```

---

## Phase 4: Spectrogram

### What
Stacking all FFT results into a 2D representation.

### Why
Visualize how frequencies change over time - like a "fingerprint" of the audio.

### How

```
Freq ▲  ░░░▒▒▒░░░░░░
(Hz) │  ░▒▓▓▓▒░░░░░░
8000 │  ▒▓███▓▒░░▒▒░
     │  ▓████▓▓▒▒▓▓▓
 500 │  ██████████████
   0 └──────────────────► Time

X-axis: Time (frame number)
Y-axis: Frequency (Hz)
Color:  Energy (darker = louder)
```

### Types
| Type | Formula | Use Case |
|------|---------|----------|
| Magnitude | \|FFT\| | General visualization |
| Power | \|FFT\|² | Energy analysis |
| Log Power | 10 × log₁₀(\|FFT\|²) | ML features (most common) |

---

## Phase 5: Mel Scale

### What
Converting linear frequency scale to perceptual (Mel) scale.

### Why
Human hearing is non-linear - we're more sensitive to differences in low frequencies than high frequencies.

### How

```
LINEAR SCALE              MEL SCALE
(Equal Hz)                (Equal perception)

8000 ├────────            ├────  High
6000 ├────────            ├───
4000 ├────────     →      ├──
2000 ├────────            ├─
1000 ├────────            ├    More detail
 500 ├────────            ├    at low freq!
   0 └────────            └    Low
```

### Formula
```
mel = 2595 × log₁₀(1 + f/700)
```

### Mel Filterbank
Apply triangular filters to group frequencies:

```
     /\      /\      /\        /\          /\
    /  \    /  \    /  \      /  \        /  \
   /    \  /    \  /    \    /    \      /    \
  /      \/      \/      \  /      \    /      \
──────────────────────────────────────────────────►
0Hz      1000Hz    2000Hz    4000Hz      8000Hz

Dense filters          →          Sparse filters
(more detail)                     (less detail)
```

### Data Reduction
```
257 frequency bins → 40 Mel bands (84% reduction)
```

---

## Phase 6: MFCCs

### What
Mel-Frequency Cepstral Coefficients - the final compact representation.

### Why
- Decorrelates Mel features
- Compresses information further
- Matches how humans perceive speech

### How

```
Mel Spectrogram    Log      DCT       MFCCs
   [98 × 40]    →  log  →  DCT  →  [98 × 13]
```

### Steps
1. **Log**: Take logarithm of Mel energies
2. **DCT**: Apply Discrete Cosine Transform
3. **Truncate**: Keep first 13 coefficients

### MFCC Interpretation
| Coefficient | Meaning |
|-------------|---------|
| MFCC 0 | Overall energy/loudness |
| MFCC 1 | Spectral slope (brightness) |
| MFCC 2-12 | Finer spectral details |

### Optional: Deltas
```
Static MFCCs:      13 coefficients (what)
Delta MFCCs:       13 coefficients (velocity - how it changes)
Delta-Delta MFCCs: 13 coefficients (acceleration)
─────────────────────────────────────────────
Total:             39 features per frame
```

## TinyML Benefits

### Memory Savings

```
┌────────────────────────────────────────────┐
│           MEMORY COMPARISON                │
├────────────────────────────────────────────┤
│  Raw Audio:  16,000 × 4 bytes = 64 KB     │
│  MFCCs:       1,274 × 4 bytes =  5 KB     │
│                                            │
│  💾 Savings: 92%                           │
└────────────────────────────────────────────┘
```

### Computation Savings

| Processing | Raw Audio | MFCCs |
|------------|-----------|-------|
| Input neurons | 16,000 | 1,274 |
| Operations | Millions | Thousands |
| Speed | Slow | Fast |

### Better Features

- MFCCs capture **perceptually relevant** information
- Raw audio contains **redundant** information
- Result: **Better accuracy with less data**


---

## Project Structure

```
audio-feature-extraction/
├── README.md
├── requirements.txt
├── extract_features.py
├── utils/
│   ├── __init__.py
│   ├── audio_utils.py
│   └── visualization.py
├── examples/
│   ├── basic_extraction.py
│   ├── compare_words.py
│   └── tinyml_pipeline.py
└── notebooks/
    └── tutorial.ipynb
```

---

## Quick Reference

### Pipeline Summary

| Phase | Input | Output | Purpose |
|-------|-------|--------|---------|
| 1. Raw Audio | Analog | [16,000] | Digitize |
| 2. Framing | [16,000] | [98, 400] | Segment |
| 3. FFT | [98, 400] | [98, 257] | Time→Freq |
| 4. Spectrogram | [98, 257] | [98, 257] | Visualize |
| 5. Mel Scale | [98, 257] | [98, 40] | Perception |
| 6. MFCCs | [98, 40] | [98, 13] | Compress |

### Common Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sample Rate | 16,000 Hz | Audio sampling rate |
| Frame Length | 25 ms | Window size |
| Hop Length | 10 ms | Step size |
| n_fft | 512 | FFT size |
| n_mels | 40 | Mel bands |
| n_mfcc | 13 | MFCC coefficients |

---
