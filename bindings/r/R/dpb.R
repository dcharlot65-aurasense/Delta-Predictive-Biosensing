#' @useDynLib dpb, .registration = TRUE
#' @importFrom R6 R6Class
NULL

#' DPB Package
#'
#' @description
#' R interface to the Delta-Predictive Biosensing (DPB) framework for
#' neuromorphic signal processing and spike encoding.
#'
#' @docType package
#' @name dpb-package
#' @aliases dpb
NULL

#' Get DPB Framework Version
#'
#' @description
#' Returns the version string of the DPB framework library.
#'
#' @return Character string with version information
#' @export
#' @examples
#' dpb_version()
dpb_version <- function() {
  .Call(C_dpb_version)
}

#' Get Last Error Message
#'
#' @description
#' Returns the last error message from a DPB operation on the current thread.
#'
#' @return Character string with error message, or NULL if no error
#' @export
#' @examples
#' # After a failed operation
#' err <- dpb_last_error()
#' if (!is.null(err)) {
#'   message("Error: ", err)
#' }
dpb_last_error <- function() {
  .Call(C_dpb_last_error)
}

#' Clear Last Error
#'
#' @description
#' Clears the last error message for the current thread.
#'
#' @return NULL (invisibly)
#' @export
dpb_clear_error <- function() {
  invisible(.Call(C_dpb_clear_error))
}

# Package load hook
.onLoad <- function(libname, pkgname) {
  # Find and load the DPB FFI library
  lib_path <- system.file("libs", package = pkgname)

  # Platform-specific library name
  lib_name <- switch(
    Sys.info()["sysname"],
    "Windows" = "dpb_ffi.dll",
    "Darwin" = "libdpb_ffi.dylib",
    "libdpb_ffi.so"
  )

  full_path <- file.path(lib_path, lib_name)

  if (!file.exists(full_path)) {
    packageStartupMessage(
      "Note: DPB FFI library not found at ", full_path, "\n",
      "You may need to build it with: cargo build --release -p dpb-ffi"
    )
  }
}

# Package unload hook
.onUnload <- function(libpath) {
  library.dynam.unload("dpb", libpath)
}
