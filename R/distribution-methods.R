

#' @export
sample <-
  function(distribution,
           sample_shape = list(),
           seed = NULL,
           name = "sample") {
    UseMethod("sample")
  }

#' @export
sample.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution,
           sample_shape = list(),
           seed = NULL,
           name = "sample") {
    distribution$sample(as.integer(sample_shape), seed, name)
  }

#' @export
log_prob <- function(distribution, value, name = "log_prob") {
  UseMethod("log_prob")
}

#' @export
log_prob.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, value, name = "log_prob") {
    distribution$log_prob(value, name)
  }

#' @export
prob <- function(distribution, value, name = "prob") {
  UseMethod("prob")
}

#' @export
prob.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, value, name = "prob") {
    distribution$prob(value, name)
  }

#' @export
log_cdf <- function(distribution, value, name = "log_cdf") {
  UseMethod("log_cdf")
}

#' @export
log_cdf.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, value, name = "log_cdf") {
    distribution$log_cdf(value, name)
  }

#' @export
cdf <- function(distribution, value, name = "cdf") {
  UseMethod("cdf")
}

#' @export
cdf.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, value, name = "cdf") {
    distribution$cdf(value, name)
  }

#' @export
log_survival_function <- function(distribution, value, name = "log_survival_function") {
  UseMethod("log_survival_function")
}

#' @export
log_survival_function.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, value, name = "log_survival_function") {
    distribution$log_survival_function(value, name)
  }

#' @export
survival_function <- function(distribution, value, name = "survival_function") {
  UseMethod("survival_function")
}

#' @export
survival_function.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, value, name = "survival_function") {
    distribution$survival_function(value, name)
  }

#' @export
entropy <- function(distribution, name = "entropy") {
  UseMethod("entropy")
}

#' @export
entropy.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, name = "entropy") {
    distribution$entropy(name)
  }

#' @export
mean <- function(distribution, name = "mean") {
  UseMethod("mean")
}

#' @export
mean.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, name = "mean") {
    distribution$mean(name)
  }

#' @export
quantile <- function(distribution, value, name = "quantile") {
  UseMethod("quantile")
}

#' @export
quantile.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, value, name = "quantile") {
    distribution$quantile(value, name)
  }

#' @export
variance <- function(distribution, name = "variance") {
  UseMethod("variance")
}

#' @export
variance.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, name = "variance") {
    distribution$variance(name)
  }

#' @export
stddev <- function(distribution, name = "stddev") {
  UseMethod("stddev")
}

#' @export
stddev.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, name = "stddev") {
    distribution$stddev(name)
  }

#' @export
covariance <- function(distribution, name = "covariance") {
  UseMethod("covariance")
}

#' @export
covariance.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, name = "covariance") {
    distribution$covariance(name)
  }

#' @export
mode <- function(distribution, name = "mode") {
  UseMethod("mode")
}

#' @export
mode.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, name = "mode") {
    distribution$mode(name)
  }

#' @export
cross_entropy <- function(distribution, other, name = "cross_entropy") {
  UseMethod("cross_entropy")
}

#' @export
cross_entropy.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, other, name = "cross_entropy") {
    distribution$cross_entropy(other, name)
  }

#' @export
kl_divergence <- function(distribution, other, name = "kl_divergence") {
  UseMethod("kl_divergence")
}

#' @export
kl_divergence.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, other, name = "kl_divergence") {
    distribution$kl_divergence(other, name)
  }

