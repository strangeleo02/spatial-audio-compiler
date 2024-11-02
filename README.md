# spatial-audio-compiler
./spatial_mixer output.wav vocals.wav bass.wav drums.wav instrumental.wav
gcc spatial_mixer.c -o spatial_mixer $(pkg-config --cflags --libs gtk+-3.0 sndfile portaudio-2.0)