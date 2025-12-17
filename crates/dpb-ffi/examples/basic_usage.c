/**
 * @file basic_usage.c
 * @brief Basic usage example of the DPB FFI library
 *
 * Demonstrates:
 * - Creating a time series from synthetic data
 * - Using a level-crossing encoder
 * - Inspecting spike train results
 * - Proper memory management
 *
 * Compile with:
 *   gcc -o basic_usage basic_usage.c -I../include -L../../target/release -ldpb_ffi -lm
 *
 * Run with:
 *   LD_LIBRARY_PATH=../../target/release ./basic_usage
 */

#include "../include/dpb.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(void) {
    printf("=== DPB FFI Basic Usage Example ===\n\n");

    // Print version
    printf("DPB Version: %s\n\n", dpb_version());

    // Create a synthetic sine wave signal
    const size_t num_samples = 1000;
    const size_t num_channels = 1;
    const double sample_rate = 1000.0;  // Hz
    const double frequency = 10.0;      // Hz

    printf("Generating %zu samples of %g Hz sine wave at %g Hz sample rate\n",
           num_samples, frequency, sample_rate);

    float* data = (float*)malloc(num_samples * sizeof(float));
    if (!data) {
        fprintf(stderr, "Error: Failed to allocate memory for data\n");
        return 1;
    }

    for (size_t i = 0; i < num_samples; i++) {
        data[i] = (float)sin(2.0 * M_PI * frequency * i / sample_rate);
    }

    // Create time series
    DpbTimeSeries* ts = dpb_timeseries_new(data, num_samples, num_channels, sample_rate);
    free(data);  // Data is copied, so we can free our buffer

    if (!ts) {
        fprintf(stderr, "Error creating time series: %s\n", dpb_last_error());
        return 1;
    }

    // Print time series properties
    printf("\nTime Series Properties:\n");
    printf("  Duration: %f seconds\n", dpb_timeseries_duration(ts));
    printf("  Samples: %zu\n", dpb_timeseries_num_samples(ts));
    printf("  Channels: %zu\n", dpb_timeseries_num_channels(ts));
    printf("  Sample Rate: %f Hz\n", dpb_timeseries_sample_rate(ts));

    // Create a level-crossing encoder with threshold 0.5
    const double threshold = 0.5;
    printf("\nCreating level-crossing encoder with threshold %g\n", threshold);

    DpbEncoder* encoder = dpb_encoder_level_crossing_new(threshold);
    if (!encoder) {
        fprintf(stderr, "Error creating encoder: %s\n", dpb_last_error());
        dpb_timeseries_free(ts);
        return 1;
    }

    // Encode the signal
    printf("Encoding signal to spikes...\n");
    DpbSpikeTrain* spikes = dpb_encoder_encode(encoder, ts);
    if (!spikes) {
        fprintf(stderr, "Error encoding signal: %s\n", dpb_last_error());
        dpb_encoder_free(encoder);
        dpb_timeseries_free(ts);
        return 1;
    }

    // Print spike train properties
    size_t num_spikes = dpb_spike_train_len(spikes);
    printf("\nSpike Train Properties:\n");
    printf("  Total events: %zu\n", num_spikes);
    printf("  Channels: %u\n", dpb_spike_train_num_channels(spikes));

    if (num_spikes > 0) {
        double spike_rate = num_spikes / dpb_timeseries_duration(ts);
        printf("  Average spike rate: %f Hz\n", spike_rate);
    }

    // Print first 10 spike events
    printf("\nFirst %zu spike events:\n", num_spikes < 10 ? num_spikes : 10);
    printf("  Index | Timestamp (s) | Channel | Polarity | Magnitude\n");
    printf("  ------|---------------|---------|----------|----------\n");

    for (size_t i = 0; i < num_spikes && i < 10; i++) {
        double timestamp;
        uint32_t channel;
        int8_t polarity;
        float magnitude;

        int result = dpb_spike_train_get_event(spikes, i, &timestamp, &channel,
                                                &polarity, &magnitude);
        if (result != DPB_SUCCESS) {
            fprintf(stderr, "Error getting spike event %zu: %s\n", i, dpb_last_error());
            continue;
        }

        printf("  %5zu | %13.6f | %7u | %8d | %9.3f\n",
               i, timestamp, channel, polarity, magnitude);
    }

    if (num_spikes > 10) {
        printf("  ... (%zu more events)\n", num_spikes - 10);
    }

    // Calculate inter-spike intervals for first few events
    if (num_spikes >= 2) {
        printf("\nInter-Spike Intervals (first %zu):\n", num_spikes < 6 ? num_spikes - 1 : 5);
        double prev_time = 0.0;
        dpb_spike_train_get_event(spikes, 0, &prev_time, NULL, NULL, NULL);

        for (size_t i = 1; i < num_spikes && i < 6; i++) {
            double curr_time;
            dpb_spike_train_get_event(spikes, i, &curr_time, NULL, NULL, NULL);
            double isi = curr_time - prev_time;
            printf("  ISI %zu->%zu: %f ms\n", i-1, i, isi * 1000.0);
            prev_time = curr_time;
        }
    }

    // Clean up
    printf("\nCleaning up...\n");
    dpb_spike_train_free(spikes);
    dpb_encoder_free(encoder);
    dpb_timeseries_free(ts);

    printf("Done!\n");
    return 0;
}
