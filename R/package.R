# globals
.globals <- new.env(parent = emptyenv())

#' Handle to the `tensorflow_probability` module
#'
#' @export
#' @return Module(tensorflow_probability)
tfp <- NULL

.onLoad <- function(libname, pkgname) {
  Sys.setenv(TF_USE_LEGACY_KERAS = 1L)
  reticulate::py_require(c(
    "tensorflow!=2.20.0",
    "numpy<2",
    "tf-keras",
    "tensorflow-probability"
  ))
  tfp <<- reticulate::import("tensorflow_probability", delay_load = list(

    priority = 20,

    environment = "r-tensorflow"

    # on_load = function() {
    #   if (!grepl("tensorflow", keras::backend()))
    #     stop("TensorFlow Probability has to be used with the TensorFlow Keras backend.")
    #   if (!grepl("tensorflow", keras::implementation()))
    #     stop("TensorFlow Probability has to be used with the TensorFlow Keras implementation.")
    # }
    # ,
    #
    # on_error = function(e) {
    #   stop(e$message, call. = FALSE)
    # }

  ))

}

#' TensorFlow Probability Version
#'
#' @return the Python TFP version
#'
#' @export
tfp_version <- function() {
  version <- (tfp$`__version__` %>% strsplit(".", fixed = TRUE))[[1]]
  pkg_version <- package_version(paste(version[[1]], version[[2]], sep = "."))
  pkg_version
}
