#' TimeSeries Class
#'
#' @description
#' R6 class representing multi-channel time series data for biosignal processing.
#'
#' @details
#' TimeSeries objects store continuous biosignal data with a specified sample rate.
#' Data can be single-channel (vector) or multi-channel (matrix with columns as channels).
#' The underlying data is stored in single-precision floating point format.
#'
#' @export
#' @examples
#' # Create a single-channel sine wave
#' t <- seq(0, 1, length.out = 1000)
#' data <- sin(2 * pi * 10 * t)
#' ts <- TimeSeries$new(data, sample_rate = 1000)
#' print(ts$duration)  # 1.0 second
#'
#' # Create multi-channel data
#' data <- matrix(rnorm(3000), ncol = 3)
#' ts <- TimeSeries$new(data, sample_rate = 500)
#' print(ts$num_channels)  # 3
TimeSeries <- R6::R6Class("TimeSeries",
  cloneable = FALSE,

  private = list(
    ptr = NULL,
    .num_samples = NULL,
    .num_channels = NULL,
    .sample_rate = NULL,

    finalize = function() {
      if (!is.null(private$ptr)) {
        .Call(C_dpb_timeseries_free, private$ptr)
        private$ptr <- NULL
      }
    }
  ),

  public = list(
    #' @description Create a new TimeSeries object
    #' @param data Numeric vector or matrix (rows = samples, columns = channels)
    #' @param sample_rate Sampling rate in Hz (must be positive)
    #' @return A new TimeSeries object
    initialize = function(data, sample_rate) {
      # Input validation
      if (is.vector(data)) {
        data <- matrix(data, ncol = 1)
      }

      if (!is.matrix(data) || !is.numeric(data)) {
        stop("Data must be a numeric vector or matrix")
      }

      if (!is.numeric(sample_rate) || length(sample_rate) != 1 || sample_rate <= 0) {
        stop("Sample rate must be a positive number")
      }

      num_samples <- nrow(data)
      num_channels <- ncol(data)

      if (num_samples == 0) {
        stop("Data cannot be empty")
      }

      # Store metadata
      private$.num_samples <- num_samples
      private$.num_channels <- num_channels
      private$.sample_rate <- sample_rate

      # Convert to single precision for FFI
      # R stores in column-major order, same as Rust expects
      storage.mode(data) <- "double"

      # Create the native object
      private$ptr <- .Call(
        C_dpb_timeseries_new,
        as.vector(data),  # Flatten column-major
        as.integer(num_samples),
        as.integer(num_channels),
        as.double(sample_rate)
      )

      if (is.null(private$ptr)) {
        err <- dpb_last_error()
        stop(paste("Failed to create TimeSeries:", ifelse(is.null(err), "Unknown error", err)))
      }

      invisible(self)
    },

    #' @description Print method for TimeSeries
    print = function() {
      cat("TimeSeries object\n")
      cat("  Samples:", self$num_samples, "\n")
      cat("  Channels:", self$num_channels, "\n")
      cat("  Sample rate:", self$sample_rate, "Hz\n")
      cat("  Duration:", round(self$duration, 4), "seconds\n")
      invisible(self)
    },

    #' @description Get the internal pointer (for advanced use)
    #' @return External pointer to native TimeSeries object
    get_ptr = function() {
      private$ptr
    },

    #' @description Check if the object is valid
    #' @return Logical indicating if the native object exists
    is_valid = function() {
      !is.null(private$ptr)
    }
  ),

  active = list(
    #' @field duration Duration of the time series in seconds
    duration = function() {
      if (is.null(private$ptr)) return(NA_real_)
      .Call(C_dpb_timeseries_duration, private$ptr)
    },

    #' @field num_samples Number of samples in the time series
    num_samples = function() {
      private$.num_samples
    },

    #' @field num_channels Number of channels in the time series
    num_channels = function() {
      private$.num_channels
    },

    #' @field sample_rate Sample rate in Hz
    sample_rate = function() {
      private$.sample_rate
    },

    #' @field data Raw data as a matrix (samples x channels)
    data = function() {
      if (is.null(private$ptr)) return(NULL)
      result <- .Call(C_dpb_timeseries_get_data, private$ptr)
      if (is.null(result)) return(NULL)
      matrix(result, nrow = private$.num_samples, ncol = private$.num_channels)
    }
  )
)

#' Create TimeSeries from file
#'
#' @description
#' Load time series data from a CSV or similar file format.
#'
#' @param file Path to the data file
#' @param sample_rate Sample rate in Hz
#' @param header Whether the file has a header row
#' @param sep Field separator
#' @return A TimeSeries object
#' @export
#' @examples
#' \dontrun{
#' ts <- timeseries_from_file("ecg_data.csv", sample_rate = 360)
#' }
timeseries_from_file <- function(file, sample_rate, header = TRUE, sep = ",") {
  data <- read.csv(file, header = header, sep = sep)
  data <- as.matrix(data)
  TimeSeries$new(data, sample_rate)
}
