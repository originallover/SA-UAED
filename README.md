# SA-UAED
# SA-UAED Data Simulation Pipeline

This repository provides the data simulation pipeline used for **SA-UAED**: *Speaker-Attributed Unified Audio Event Detection*. The pipeline generates simulated multi-speaker audio mixtures with frame-level labels for:

* background sound events,
* speaker speech activities,
* speaker-attributed laughter,
* speaker-attributed coughing.

The generated data can be used to train and evaluate unified audio event detection models that jointly perform sound event detection, speaker activity detection, and speaker-attributed paralinguistic event detection.

The simulated dataset **LibriPara** is available at:

```text
https://huggingface.co/datasets/originalover/Libripara
```

---

## Pipeline Overview

The simulation pipeline consists of three main steps:

1. **Generate conversation definitions**
   `generate_conversation_definitions.py` creates JSON metadata files for each simulated conversation. Each file defines the complete event timeline, including speech segments, speaker identities, background events, foreground sound events, and speaker-attributed paralinguistic events.

2. **Synthesize conversation audio**
   `synthesize_conversations.py` reads the generated JSON metadata and mixes speech, sound events, and paralinguistic events into 32-second waveform files.

3. **Generate frame-level labels**
   `generate_conversation_labels.py` converts the JSON metadata into frame-level multi-label annotations.

The full process can be launched with:

```bash
bash generate_conversation.sh
```

---

## Required Data

Before running the pipeline, please prepare the following resources.

### 1. Kaldi-style LibriSpeech data

The speech data should be prepared in Kaldi format. Each split should contain at least:

```text
wav.scp
utt2spk
```

The default script expects the following directories:

```text
./Kaldidatadir/alltrain_persession
./Kaldidatadir/dev_clean_persession
```

You may modify `generate_conversation.sh` if your Kaldi data is stored elsewhere.

### 2. Sound event index

The pipeline expects a sound event index file:

```text
./conversation_data/sound_events_index.json
```

This file should contain paths to sound event audio files. The current code uses two groups:

```text
background
foreground
```

The supported sound event classes are:

```text
Alarm_bell_ringing
Blender
Cat
Dishes
Dog
Electric_shaver_toothbrush
Frying
Running_water
Vacuum_cleaner
```

### 3. Paralinguistic audio clips

The pipeline supports speaker-specific laughter and coughing clips. The default directory is:

```text
./conversation_data/paralinguistic_filtered
```

The expected directory structure is:

```text
paralinguistic_filtered/
├── train-clean-100/
│   └── <speaker_id>/
│       ├── laugh/
│       │   └── *.wav
│       └── cough/
│           └── *.wav
├── train-clean-360/
├── train-other-500/
└── dev-clean/
```

Each speaker may have separate `laugh` and `cough` folders. During simulation, laughter and coughing events are inserted according to the real speaker identity of each speech segment.

---

## Default Configuration

The default configuration in `generate_conversation.sh` is:

```bash
TRAIN_HOURS=500
EVAL_HOURS=5
TEST_HOURS=5

CONV_DURATION=32.0
NUM_SPEAKERS=3
SAMPLING_RATE=16000
FRAME_RATE=50

LAUGH_PROB=0.3
COUGH_PROB=0.2
```

This means that each generated audio clip is 32 seconds long, sampled at 16 kHz, and labeled at 50 Hz. Each clip contains up to 3 speakers. Laughter and coughing are inserted with probabilities of 0.3 and 0.2 per speech segment, respectively.

You can modify these values directly in `generate_conversation.sh`.

---

## Usage

### 1. Clone the repository

```bash
git clone https://github.com/originallover/SA-UAED.git
cd SA-UAED
```

### 2. Prepare the input data

Please make sure the following paths exist or modify them in `generate_conversation.sh`:

```bash
OUTPUT_DIR="./conversation_data"
DESED_DIR="./conversation_data/synthetic/audio/eval/soundbank"
SOUND_events_dir="./conversation_data/sound_events_index.json"
PARALINGUISTIC_DIR="./conversation_data/paralinguistic_filtered"
```

The Kaldi-style speech data should be placed at:

```bash
./Kaldidatadir/alltrain_persession
./Kaldidatadir/dev_clean_persession
```

### 3. Run the full simulation pipeline

```bash
bash generate_conversation.sh
```

This script will run the following stages:

```text
Step 1: Generate conversation definitions
Step 2: Synthesize conversation audio
Step 3: Generate frame-level labels
```

---

## Running Each Step Separately

You can also run each step manually.

---

## Output Structure

After running the pipeline, the output directory will look like this:

```text
conversation_data/
├── train0/
│   ├── conversations/
│   │   ├── conversations.list
│   │   ├── conv_000000.json
│   │   ├── conv_000001.json
│   │   └── ...
│   ├── audio/
│   │   ├── conv_000000.wav
│   │   ├── conv_000001.wav
│   │   └── ...
│   └── labels/
│       ├── conv_000000.json
│       ├── conv_000001.json
│       └── ...
├── eval0/
└── test0/
```

Each conversation has three corresponding files:

```text
conversations/conv_xxxxxx.json   # event timeline and metadata
audio/conv_xxxxxx.wav            # synthesized audio mixture
labels/conv_xxxxxx.json          # frame-level labels
```





---

## Notes

* The current pipeline assumes 32-second audio clips by default.
* Speech overlap is controlled inside `generate_conversation_definitions.py`.
* Laughter and coughing are inserted around or inside speech segments according to the selected speaker identity.
* Foreground events are mixed with an SNR sampled from 0 to 5 dB.
* Background events are mixed with an SNR sampled from 10 to 15 dB.
* Paralinguistic events are mixed with an SNR sampled from 3 to 8 dB.
* The final waveform is normalized to avoid clipping, and light white noise is added during synthesis.

---

## Citation

If you use this simulation pipeline or the LibriPara dataset, please cite:

```bibtex
@inproceedings{lan2026sauaed,
  title     = {SA-UAED: Joint Frame-Level Detection of Audio Events, Speaker Activities, and Speaker-Attributed Paralinguistic Events},
  author    = {Zekun Lan and Wangyou Zhang and Yanmin Qian},
  booktitle = {Proc. Interspeech},
  year      = {2026}
}
```
