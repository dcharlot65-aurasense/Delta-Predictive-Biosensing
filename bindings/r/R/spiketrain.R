#' SpikeTrain Class
#'
#' @description
#' R6 class representing a collection of discrete spike events from encoded biosignals.
#'
#' @details
#' A SpikeTrain contains temporal spike events with timestamps, channel indices,
#' polarities (+1 for upward crossing, -1 for downward), and optional magnitudes.
#' This is the output format of spike encoders in the DPB framework.
#'
#' @export
#' @examples
#' # Create an empty spike train
#' st <- SpikeTrain$new(num_channels = 4)
#'
#' # Add events manually
#' st$add_event(timestamp = 0.1, channel = 0, polarity = 1)
#' st$add_event(timestamp = 0.2, channel = 1, polarity = -1)
#'
#' # Get all events as a data frame
#' events <- st$get_events()
SpikeTrain <- R6::R6Class("SpikeTrain",
  cloneable = FALSE,

  private = list(
    ptr = NULL,
    .num_channels = NULL,

    finalize = function() {
      if (!is.null(private$ptr)) {
        .Call(C_dpb_spike_train_free, private$ptr)
        private$ptr <- NULL
      }
    }
  ),

  public = list(
    #' @description Create a new SpikeTrain object
    #' @param num_channels Number of channels/neurons in the spike train
    #' @return A new SpikeTrain object
    initialize = function(num_channels = 1L) {
      if (!is.numeric(num_channels) || num_channels < 1) {
        stop("num_channels must be a positive integer")
      }

      private$.num_channels <- as.integer(num_channels)
      private$ptr <- .Call(C_dpb_spike_train_new, as.integer(num_channels))

      if (is.null(private$ptr)) {
        err <- dpb_last_error()
        stop(paste("Failed to create SpikeTrain:", ifelse(is.null(err), "Unknown error", err)))
      }

      invisible(self)
    },

    #' @description Create SpikeTrain from an external pointer (internal use)
    #' @param ptr External pointer from encoding operation
    #' @param num_channels Number of channels
    #' @return A new SpikeTrain object wrapping the pointer
    new_from_ptr = function(ptr, num_channels) {
      obj <- SpikeTrain$new(num_channels)
      # Free the empty one we just created
      if (!is.null(obj$.__enclos_env__$private$ptr)) {
        .Call(C_dpb_spike_train_free, obj$.__enclos_env__$private$ptr)
      }
      obj$.__enclos_env__$private$ptr <- ptr
      obj
    },

    #' @description Add a spike event to the train
    #' @param timestamp Time of the spike in seconds
    #' @param channel Channel index (0-based)
    #' @param polarity Spike polarity (+1 or -1)
    #' @param magnitude Optional spike magnitude (default 1.0)
    #' @return Self (invisibly), for method chaining
    add_event = function(timestamp, channel, polarity, magnitude = 1.0) {
      if (is.null(private$ptr)) {
        stop("SpikeTrain object is not valid")
      }

      result <- .Call(
        C_dpb_spike_train_add_event,
        private$ptr,
        as.double(timestamp),
        as.integer(channel),
        as.integer(polarity),
        as.double(magnitude)
      )

      if (result != 0L) {
        err <- dpb_last_error()
        stop(paste("Failed to add event:", ifelse(is.null(err), "Unknown error", err)))
      }

      invisible(self)
    },

    #' @description Get all spike events as a data frame
    #' @return Data frame with columns: timestamp, channel, polarity, magnitude
    get_events = function() {
      if (is.null(private$ptr)) {
        return(data.frame(
          timestamp = double(),
          channel = integer(),
          polarity = integer(),
          magnitude = double()
        ))
      }

      n <- self$length
      if (n == 0) {
        return(data.frame(
          timestamp = double(),
          channel = integer(),
          polarity = integer(),
          magnitude = double()
        ))
      }

      # Pre-allocate vectors
      timestamps <- double(n)
      channels <- integer(n)
      polarities <- integer(n)
      magnitudes <- double(n)

      for (i in seq_len(n)) {
        event <- .Call(C_dpb_spike_train_get_event, private$ptr, as.integer(i - 1L))
        if (!is.null(event)) {
          timestamps[i] <- event$timestamp
          channels[i] <- event$channel
          polarities[i] <- event$polarity
          magnitudes[i] <- event$magnitude
        }
      }

      data.frame(
        timestamp = timestamps,
        channel = channels,
        polarity = polarities,
        magnitude = magnitudes
      )
    },

    #' @description Print method for SpikeTrain
    print = function() {
      cat("SpikeTrain object\n")
      cat("  Events:", self$length, "\n")
      cat("  Channels:", self$num_channels, "\n")
      if (self$length > 0) {
        events <- self$get_events()
        cat("  Time range:", round(min(events$timestamp), 4), "-",
            round(max(events$timestamp), 4), "seconds\n")
        cat("  Spike rate:", round(self$length / diff(range(events$timestamp)), 2),
            "spikes/sec\n")
      }
      invisible(self)
    },

    #' @description Get the internal pointer (for advanced use)
    get_ptr = function() {
      private$ptr
    }
  ),

  active = list(
    #' @field length Number of spike events in the train
    length = function() {
      if (is.null(private$ptr)) return(0L)
      .Call(C_dpb_spike_train_len, private$ptr)
    },

    #' @field num_channels Number of channels in the spike train
    num_channels = function() {
      private$.num_channels
    }
  )
)
