#' Installs TensorFlow Probability
#'
#'
#' @inheritParams keras::install_keras
#' @return invisible
#' @export
install_tfprobability <- function (method = c("auto", "virtualenv", "conda"),
                                   conda = "auto",
                                   version = "default", tensorflow = "default",
                                   extra_packages = NULL,
                                   ...) {


  if (version  == "default" || is.null(version))
    package <- "tensorflow-probability"
  else if (version == "nightly")
    package <- "tfp-nightly"
  else
    package <- paste0("tensorflow-probability==", version)

  extra_packages <- unique(c(package, extra_packages))

  tensorflow::install_tensorflow(method = method, conda = conda, version = tensorflow,
                     extra_packages = extra_packages, pip_ignore_installed = FALSE,
                     ...)
}
