#' Level Crossing Encoder
#'
#' @description
#' Encodes continuous signals into spike trains using level crossing detection.
#' A spike is generated whenever the signal crosses a quantization level defined
#' by the threshold parameter.
#'
#' @details
#' Level crossing encoding is efficient for signals with smooth variations.
#' The encoder tracks quantization levels and emits:
#' - Positive spikes (+1) when crossing upward
#' - Negative spikes (-1) when crossing downward
#'
#' @export
#' @examples
#' # Create encoder with 0.1 threshold
#' encoder <- LevelCrossingEncoder$new(threshold = 0.1)
#'
#' # Create test signal
#' t <- seq(0, 1, length.out = 1000)
#' signal <- sin(2 * pi * 5 * t)
#' ts <- TimeSeries$new(signal, sample_rate = 1000)
#'
#' # Encode
#' spikes <- encoder$encode(ts)
#' print(spikes)
LevelCrossingEncoder <- R6::R6Class("LevelCrossingEncoder",
  cloneable = FALSE,

  private = list(
    ptr = NULL,
    .threshold = NULL,

    finalize = function() {
      if (!is.null(private$ptr)) {
        .Call(C_dpb_encoder_free, private$ptr)
        private$ptr <- NULL
      }
    }
  ),

  public = list(
    #' @description Create a new Level Crossing Encoder
    #' @param threshold Threshold value for level crossing detection (must be positive)
    #' @return A new LevelCrossingEncoder object
    initialize = function(threshold = 0.1) {
      if (!is.numeric(threshold) || length(threshold) != 1 || threshold <= 0) {
        stop("Threshold must be a positive number")
      }

      private$.threshold <- threshold
      private$ptr <- .Call(C_dpb_encoder_level_crossing_new, as.double(threshold))

      if (is.null(private$ptr)) {
        err <- dpb_last_error()
        stop(paste("Failed to create encoder:", ifelse(is.null(err), "Unknown error", err)))
      }

      invisible(self)
    },

    #' @description Encode a TimeSeries into a SpikeTrain
    #' @param timeseries A TimeSeries object to encode
    #' @return A SpikeTrain object containing the encoded spikes
    encode = function(timeseries) {
      if (!inherits(timeseries, "TimeSeries")) {
        stop("Input must be a TimeSeries object")
      }

      if (!timeseries$is_valid()) {
        stop("TimeSeries object is not valid")
      }

      if (is.null(private$ptr)) {
        stop("Encoder object is not valid")
      }

      st_ptr <- .Call(C_dpb_encoder_encode, private$ptr, timeseries$get_ptr())

      if (is.null(st_ptr)) {
        err <- dpb_last_error()
        stop(paste("Encoding failed:", ifelse(is.null(err), "Unknown error", err)))
      }

      # Create SpikeTrain wrapper
      st <- SpikeTrain$new(timeseries$num_channels)
      # Replace the pointer
      .Call(C_dpb_spike_train_free, st$.__enclos_env__$private$ptr)
      st$.__enclos_env__$private$ptr <- st_ptr

      st
    },

    #' @description Print method
    print = function() {
      cat("LevelCrossingEncoder\n")
      cat("  Threshold:", self$threshold, "\n")
      invisible(self)
    }
  ),

  active = list(
    #' @field threshold The level crossing threshold value
    threshold = function() {
      private$.threshold
    }
  )
)

#' Delta Encoder
#'
#' @description
#' Encodes continuous signals into spike trains using delta modulation.
#' A spike is generated when the change from a reference value exceeds the threshold.
#'
#' @details
#' Delta encoding tracks a reference value and emits spikes when the signal
#' deviates significantly. After each spike, the reference is updated to the

' current value. This is efficient for signals with varying activity levels.
#'
#' @export
#' @examples
#' # Create delta encoder
#' encoder <- DeltaEncoder$new(threshold = 0.05)
#'
#' # Encode ECG-like signal
#' t <- seq(0, 5, length.out = 5000)
#' ecg <- sin(2 * pi * 1.2 * t) + 0.3 * sin(2 * pi * 2.4 * t)
#' ts <- TimeSeries$new(ecg, sample_rate = 1000)
#' spikes <- encoder$encode(ts)
DeltaEncoder <- R6::R6Class("DeltaEncoder",
  cloneable = FALSE,

  private = list(
    ptr = NULL,
    .threshold = NULL,

    finalize = function() {
      if (!is.null(private$ptr)) {
        .Call(C_dpb_encoder_free, private$ptr)
        private$ptr <- NULL
      }
    }
  ),

  public = list(
    #' @description Create a new Delta Encoder
    #' @param threshold Threshold for delta detection (must be positive)
    #' @return A new DeltaEncoder object
    initialize = function(threshold = 0.1) {
      if (!is.numeric(threshold) || length(threshold) != 1 || threshold <= 0) {
        stop("Threshold must be a positive number")
      }

      private$.threshold <- threshold
      private$ptr <- .Call(C_dpb_encoder_delta_new, as.double(threshold))

      if (is.null(private$ptr)) {
        err <- dpb_last_error()
        stop(paste("Failed to create encoder:", ifelse(is.null(err), "Unknown error", err)))
      }

      invisible(self)
    },

    #' @description Encode a TimeSeries into a SpikeTrain
    #' @param timeseries A TimeSeries object to encode
    #' @return A SpikeTrain object containing the encoded spikes
    encode = function(timeseries) {
      if (!inherits(timeseries, "TimeSeries")) {
        stop("Input must be a TimeSeries object")
      }

      if (!timeseries$is_valid()) {
        stop("TimeSeries object is not valid")
      }

      if (is.null(private$ptr)) {
        stop("Encoder object is not valid")
      }

      st_ptr <- .Call(C_dpb_encoder_encode, private$ptr, timeseries$get_ptr())

      if (is.null(st_ptr)) {
        err <- dpb_last_error()
        stop(paste("Encoding failed:", ifelse(is.null(err), "Unknown error", err)))
      }

      st <- SpikeTrain$new(timeseries$num_channels)
      .Call(C_dpb_spike_train_free, st$.__enclos_env__$private$ptr)
      st$.__enclos_env__$private$ptr <- st_ptr

      st
    },

    #' @description Print method
    print = function() {
      cat("DeltaEncoder\n")
      cat("  Threshold:", self$threshold, "\n")
      invisible(self)
    }
  ),

  active = list(
    #' @field threshold The delta threshold value
    threshold = function() {
      private$.threshold
    }
  )
)
