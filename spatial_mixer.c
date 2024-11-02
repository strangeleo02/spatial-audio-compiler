#include <stdio.h>
#include <stdlib.h>
#include <sndfile.h>
#include <gtk/gtk.h>
#include <portaudio.h>

#define FRAMES_PER_BUFFER 512
#define NUM_TRACKS 4

typedef struct {
    SNDFILE *file;
    float left_volume;
    float right_volume;
    int is_playing;
} AudioTrack;

AudioTrack tracks[NUM_TRACKS];

static int audioCallback(const void *inputBuffer, void *outputBuffer, unsigned long framesPerBuffer,
                         const PaStreamCallbackTimeInfo *timeInfo, PaStreamCallbackFlags statusFlags,
                         void *userData) {
    float *out = (float *)outputBuffer;

    // Clear the output buffer
    for (unsigned long i = 0; i < framesPerBuffer; i++) {
        *out++ = 0; // Left channel
        *out++ = 0; // Right channel
    }

    for (int track_index = 0; track_index < NUM_TRACKS; track_index++) {
        if (tracks[track_index].is_playing) {
            float buffer[FRAMES_PER_BUFFER];
            int readcount = sf_readf_float(tracks[track_index].file, buffer, framesPerBuffer);

            for (int i = 0; i < readcount; i++) {
                out[-2] += buffer[i] * tracks[track_index].left_volume;  // Left channel
                out[-1] += buffer[i] * tracks[track_index].right_volume; // Right channel
                out += 2; // Move to the next sample
            }
        }
    }
    return paContinue; // Keep the stream open
}

void merge_audio_files(const char *input_files[]);
void start_playback();
void spatial_adjust(GtkWidget *widget, gpointer data);

void merge_audio_files(const char *input_files[]) {
    SF_INFO sfinfo;
    for (int i = 0; i < NUM_TRACKS; i++) {
        sfinfo.format = 0; // Format will be set when opening the file
        tracks[i].file = sf_open(input_files[i], SFM_READ, &sfinfo);
        if (!tracks[i].file) {
            fprintf(stderr, "Error opening input file %s\n", input_files[i]);
            exit(1);
        }
        tracks[i].left_volume = 1.0;  // Initial left volume
        tracks[i].right_volume = 1.0; // Initial right volume
        tracks[i].is_playing = 1;     // Mark track as playing
    }
}

void start_playback() {
    // Placeholder for any start playback logic if needed
}

void spatial_adjust(GtkWidget *widget, gpointer data) {
    int track_index = GPOINTER_TO_INT(data);
    tracks[track_index].left_volume = gtk_range_get_value(GTK_RANGE(widget));
    tracks[track_index].right_volume = 1.0 - tracks[track_index].left_volume;
}

int main(int argc, char *argv[]) {
    if (argc < 6) {
        fprintf(stderr, "Usage: %s output.wav input1.wav input2.wav input3.wav input4.wav\n", argv[0]);
        return 1;
    }

    gtk_init(&argc, &argv);
    Pa_Initialize();

    merge_audio_files((const char**)&argv[2]);

    GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "Spatial Audio Mixer");
    gtk_container_set_border_width(GTK_CONTAINER(window), 10);
    GtkWidget *grid = gtk_grid_new();
    gtk_container_add(GTK_CONTAINER(window), grid);

    for (int i = 0; i < NUM_TRACKS; i++) {
        GtkWidget *label = gtk_label_new("Track Position");
        GtkWidget *scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 0.0, 1.0, 0.01);
        gtk_grid_attach(GTK_GRID(grid), label, 0, i, 1, 1); // Attach label to grid
        gtk_grid_attach(GTK_GRID(grid), scale, 1, i, 1, 1); // Attach scale to grid
        g_signal_connect(scale, "value-changed", G_CALLBACK(spatial_adjust), GINT_TO_POINTER(i));
    }

    g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);
    gtk_widget_show_all(window);
    
    // Start playback
    start_playback();

    // Open audio stream
    PaStream *stream;
    Pa_OpenDefaultStream(&stream, 0, 2, paFloat32, 44100, FRAMES_PER_BUFFER, audioCallback, NULL); // Pass NULL for userData
    Pa_StartStream(stream);
    
    gtk_main();

    // Stop and close audio stream
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    for (int i = 0; i < NUM_TRACKS; i++) {
        sf_close(tracks[i].file);
    }

    return 0;
}
