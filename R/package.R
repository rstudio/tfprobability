# globals
.globals <- new.env(parent = emptyenv())

#' @export
tfp <- NULL

.onLoad <- function(libname, pkgname) {

  tfp <<- import("tensorflow_probability", delay_load = list(

    priority = 5,

    environment = "r-tensorflow",

    on_load = function() {
      if (!grepl("tensorflow", backend()))
        stop("TensorFlow Probability has to be used with the TensorFlow Keras backend.")
      if (!grepl("tensorflow.python.keras", implementation()))
        stop("TensorFlow Probability has to be used with the TensorFlow Keras implementation.")
    }
    ,

    on_error = function(e) {
      stop(e$message, call. = FALSE)
    }

  ))

}

tfp_version <- function() {
  version <- (tfp$`__version__` %>% strsplit(".", fixed = TRUE))[[1]]
  pkg_version <- package_version(paste(version[[1]], version[[2]], sep = "."))
  pkg_version
}




