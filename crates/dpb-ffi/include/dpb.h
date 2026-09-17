/**
 * @file dpb.h
 * @brief DPB Framework - C API Header
 *
 * C-compatible foreign function interface for the Delta-Predictive Biosensing Framework.
 * This header provides access to time series processing, spike encoding, and neuromorphic
 * signal processing capabilities.
 *
 * @author AuraSense Tech Corporation
 * @version 0.1.0
 *
 * ## Memory Management
 *
 * - Objects created with `*_new()` functions must be freed with corresponding `*_free()` functions
 * - Do not free pointers returned by `dpb_version()` or `dpb_last_error()`
 * - Do not free data pointers returned by `dpb_timeseries_get_data()` - they are owned by the TimeSeries
 *
 * ## Error Handling
 *
 * Most functions return error codes or NULL on failure. Always check return values.
 * Use `dpb_last_error()` to get detailed error messages.
 *
 * ## Thread Safety
 *
 * Error messages are stored in thread-local storage. Each thread has its own error state.
 */

#ifndef DPB_H
#define DPB_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Opaque Handle Types
 * ============================================================================ */

/**
 * @brief Opaque handle to a time series object.
 *
 * Represents multi-channel time series data with a specific sample rate.
 */
typedef struct DpbTimeSeries DpbTimeSeries;

/**
 * @brief Opaque handle to a spike train object.
 *
 * Represents a collection of discrete spike events.
 */
typedef struct DpbSpikeTrain DpbSpikeTrain;

/**
 * @brief Opaque handle to an encoder object.
 *
 * Encodes continuous signals into discrete spike events.
 */
typedef struct DpbEncoder DpbEncoder;

/* ============================================================================
 * Error Codes
 * ============================================================================ */

/**
 * @brief Error codes returned by DPB functions.
 */
typedef enum {
    DPB_SUCCESS = 0,              /**< Operation succeeded */
    DPB_ERROR_NULL_POINTER = 1,   /**< Null pointer passed as argument */
    DPB_ERROR_INVALID_PARAM = 2,  /**< Invalid parameter value */
    DPB_ERROR_ALLOC_FAILED = 3,   /**< Memory allocation failed */
    DPB_ERROR_INVALID_DIMS = 4,   /**< Invalid dimensions */
    DPB_ERROR_ENCODING_FAILED = 5,/**< Encoding operation failed */
    DPB_ERROR_UNKNOWN = 99        /**< Unknown error */
} DpbErrorCode;

/* ============================================================================
 * Version Information
 * ============================================================================ */

/**
 * @brief Returns the DPB framework version string.
 *
 * @return Pointer to a null-terminated version string (do NOT free)
 *
 * Example: "DPB Framework v0.1.0"
 */
const char* dpb_version(void);

/* ============================================================================
 * Error Handling
 * ============================================================================ */

/**
 * @brief Returns the last error message for the current thread.
 *
 * @return Pointer to error message, or NULL if no error (do NOT free)
 *
 * The returned pointer is valid until the next DPB function call on the same thread.
 */
const char* dpb_last_error(void);

/**
 * @brief Clears the last error message for the current thread.
 */
void dpb_clear_error(void);

/* ============================================================================
 * Memory Management
 * ============================================================================ */

/**
 * @brief Frees a string allocated by the DPB library.
 *
 * @param s String to free
 *
 * Do NOT use this on strings returned by `dpb_version()` or `dpb_last_error()`.
 */
void dpb_free_string(char* s);

/**
 * @brief Frees an array allocated by the DPB library.
 *
 * @param ptr Array pointer to free
 */
void dpb_free_array(float* ptr);

/* ============================================================================
 * TimeSeries Functions
 * ============================================================================ */

/**
 * @brief Creates a new time series object.
 *
 * @param data Pointer to sample data (will be copied)
 * @param num_samples Total number of samples
 * @param num_channels Number of channels
 * @param sample_rate Sampling rate in Hz
 * @return Pointer to new TimeSeries, or NULL on error
 *
 * The data is stored in interleaved format:
 * [ch0_s0, ch1_s0, ..., chN_s0, ch0_s1, ch1_s1, ...]
 *
 * Must be freed with `dpb_timeseries_free()`.
 */
DpbTimeSeries* dpb_timeseries_new(
    const float* data,
    size_t num_samples,
    size_t num_channels,
    double sample_rate
);

/**
 * @brief Frees a time series object.
 *
 * @param ts TimeSeries to free
 *
 * Do not use the pointer after calling this function.
 */
void dpb_timeseries_free(DpbTimeSeries* ts);

/**
 * @brief Returns the duration of the time series in seconds.
 *
 * @param ts TimeSeries object
 * @return Duration in seconds, or 0.0 on error
 */
double dpb_timeseries_duration(const DpbTimeSeries* ts);

/**
 * @brief Returns the number of samples in the time series.
 *
 * @param ts TimeSeries object
 * @return Number of samples, or 0 on error
 */
size_t dpb_timeseries_num_samples(const DpbTimeSeries* ts);

/**
 * @brief Returns the number of channels in the time series.
 *
 * @param ts TimeSeries object
 * @return Number of channels, or 0 on error
 */
size_t dpb_timeseries_num_channels(const DpbTimeSeries* ts);

/**
 * @brief Returns the sample rate of the time series.
 *
 * @param ts TimeSeries object
 * @return Sample rate in Hz, or 0.0 on error
 */
double dpb_timeseries_sample_rate(const DpbTimeSeries* ts);

/**
 * @brief Returns a pointer to the raw sample data.
 *
 * @param ts TimeSeries object
 * @return Pointer to sample data, or NULL on error (do NOT free)
 *
 * The data is in interleaved format and remains valid until the TimeSeries is freed.
 */
const float* dpb_timeseries_get_data(const DpbTimeSeries* ts);

/* ============================================================================
 * SpikeTrain Functions
 * ============================================================================ */

/**
 * @brief Creates a new spike train object.
 *
 * @param num_channels Number of channels
 * @return Pointer to new SpikeTrain, or NULL on error
 *
 * Must be freed with `dpb_spike_train_free()`.
 */
DpbSpikeTrain* dpb_spike_train_new(uint32_t num_channels);

/**
 * @brief Frees a spike train object.
 *
 * @param st SpikeTrain to free
 *
 * Do not use the pointer after calling this function.
 */
void dpb_spike_train_free(DpbSpikeTrain* st);

/**
 * @brief Adds a spike event to the spike train.
 *
 * @param st SpikeTrain object
 * @param timestamp Time of spike in seconds
 * @param channel Channel/neuron index
 * @param polarity Spike polarity (-1 or +1)
 * @return DPB_SUCCESS on success, error code otherwise
 */
int dpb_spike_train_add_event(
    DpbSpikeTrain* st,
    double timestamp,
    uint32_t channel,
    int8_t polarity
);

/**
 * @brief Returns the number of spike events in the train.
 *
 * @param st SpikeTrain object
 * @return Number of events, or 0 on error
 */
size_t dpb_spike_train_len(const DpbSpikeTrain* st);

/**
 * @brief Returns the number of channels in the spike train.
 *
 * @param st SpikeTrain object
 * @return Number of channels, or 0 on error
 */
uint32_t dpb_spike_train_num_channels(const DpbSpikeTrain* st);

/**
 * @brief Gets a spike event from the train by index.
 *
 * @param st SpikeTrain object
 * @param index Event index (0-based)
 * @param timestamp Output for timestamp (can be NULL)
 * @param channel Output for channel (can be NULL)
 * @param polarity Output for polarity (can be NULL)
 * @param magnitude Output for magnitude (can be NULL)
 * @return DPB_SUCCESS on success, error code otherwise
 */
int dpb_spike_train_get_event(
    const DpbSpikeTrain* st,
    size_t index,
    double* timestamp,
    uint32_t* channel,
    int8_t* polarity,
    float* magnitude
);

/* ============================================================================
 * Encoder Functions
 * ============================================================================ */

/**
 * @brief Creates a new level-crossing encoder.
 *
 * @param threshold Delta-mode quantum: how far the signal must move from the
 *        last emitted level before another event fires, and so the bound on
 *        reconstruction error. Must be finite and positive once narrowed to
 *        float; zero, negative, NaN and values that overflow a float are
 *        rejected, because each of them would otherwise encode to an empty
 *        spike train with no error.
 * @return Pointer to new Encoder, or NULL on error (see dpb_last_error()).
 *
 * Earlier builds ignored this argument and always encoded with 0.5.
 *
 * Must be freed with `dpb_encoder_free()`.
 */
DpbEncoder* dpb_encoder_level_crossing_new(double threshold);

/**
 * @brief Frees an encoder object.
 *
 * @param enc Encoder to free
 *
 * Do not use the pointer after calling this function.
 */
void dpb_encoder_free(DpbEncoder* enc);

/**
 * @brief Encodes a time series into a spike train.
 *
 * @param enc Encoder object
 * @param ts TimeSeries to encode
 * @return Pointer to new SpikeTrain, or NULL on error
 *
 * The returned SpikeTrain must be freed with `dpb_spike_train_free()`.
 */
DpbSpikeTrain* dpb_encoder_encode(
    const DpbEncoder* enc,
    const DpbTimeSeries* ts
);

#ifdef __cplusplus
}
#endif

#endif /* DPB_H */
