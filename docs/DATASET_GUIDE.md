# Dataset Guide

## Where to put audio

Add files under:

`emotion_voice_system/backend/dataset/raw/`

Inside emotion folders:

- `happy/`
- `sad/`
- `angry/`
- `fear/`
- `neutral/`

## Audio requirements (recommended)

- `.wav`
- 16kHz sample rate (backend will resample if needed)
- mono
- 3–8 seconds
- clear voice, minimal noise

## Naming convention

Examples:

- `happy_01.wav`
- `sad_02.wav`
- `angry_03.wav`

## Minimum

For a usable baseline:

- ~20 samples per class (minimum)
- 100+ total for better performance

