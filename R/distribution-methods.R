

#' @export
sample <-
  function(distribution,
           sample_shape = list(),
           seed = NULL,
           name = "sample") {
    UseMethod("sample")
  }


#' Generate samples of the specified shape.
#'
#' Note that a call to `sample()` without arguments will generate a single sample.
#'
#' @param distribution The distribution being used.
#' @param sample_shape 0D or 1D `int32` `Tensor`. Shape of the generated samples.
#' @param seed integer seed for RNG
#' @param name name to give to the op.
#'
#' @return a `Tensor` with prepended dimensions `sample_shape`.
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

#' Log probability density/mass function.
#'
#' @param distribution The distribution being used.
#' @param value `float` or `double` `Tensor`.
#' @param name String prepended to names of ops created by this function.
#'
#' @return a `Tensor` of shape `sample_shape(x) + self$batch_shape` with values of type `self$dtype`.
#' @export

log_prob.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, value, name = "log_prob") {
    distribution$log_prob(value, name)
  }

#' @export
prob <- function(distribution, value, name = "prob") {
  UseMethod("prob")
}


#' Probability density/mass function.
#'
#' @inherit log_prob.tensorflow_probability.python.distributions.distribution.Distribution return params
#' @export

prob.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, value, name = "prob") {
    distribution$prob(value, name)
  }


#' @export
log_cdf <- function(distribution, value, name = "log_cdf") {
  UseMethod("log_cdf")
}

#' Log cumulative distribution function.
#'
#' Given random variable `X`, the cumulative distribution function `cdf` is:
#' `log_cdf(x) := Log[ P[X <= x] ]``
#' Often, a numerical approximation can be used for `log_cdf(x)` that yields
#' a more accurate answer than simply taking the logarithm of the `cdf` when `x << -1`.
#'
#' @inherit log_prob.tensorflow_probability.python.distributions.distribution.Distribution return params
#' @export
#'
log_cdf.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, value, name = "log_cdf") {
    distribution$log_cdf(value, name)
  }

#' @export
cdf <- function(distribution, value, name = "cdf") {
  UseMethod("cdf")
}

#' Cumulative distribution function.
#' Given random variable `X`, the cumulative distribution function `cdf` is:
#' `cdf(x) := P[X <= x]``
#'
#' @inherit log_prob.tensorflow_probability.python.distributions.distribution.Distribution return params
#' @export
cdf.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, value, name = "cdf") {
    distribution$cdf(value, name)
  }

#' @export
log_survival_function <- function(distribution, value, name = "log_survival_function") {
  UseMethod("log_survival_function")
}

#' Log survival function.
#'
#' Given random variable `X`, the survival function is defined:
#' `log_survival_function(x) = Log[ P[X > x] ] = Log[ 1 - P[X <= x] ] = Log[ 1 - cdf(x) ]`
#'
#' Typically, different numerical approximations can be used for the log survival function,
#'  which are more accurate than `1 - cdf(x)` when `x >> 1`.
#'
#' @inherit log_prob.tensorflow_probability.python.distributions.distribution.Distribution return params
#' @export
log_survival_function.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, value, name = "log_survival_function") {
    distribution$log_survival_function(value, name)
  }

#' @export
survival_function <- function(distribution, value, name = "survival_function") {
  UseMethod("survival_function")
}

#' Survival function.
#'
#' Given random variable `X`, the survival function is defined:
#' `survival_function(x) = P[X > x] = 1 - P[X <= x] = 1 - cdf(x).``
#'
#' @inherit log_prob.tensorflow_probability.python.distributions.distribution.Distribution return params
#' @export
survival_function.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, value, name = "survival_function") {
    distribution$survival_function(value, name)
  }

#' @export
entropy <- function(distribution, name = "entropy") {
  UseMethod("entropy")
}

#' Shannon entropy in nats.
#'
#' @export
entropy.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, name = "entropy") {
    distribution$entropy(name)
  }

#' Mean.
#'
#' @inherit variance.tensorflow_probability.python.distributions.distribution.Distribution return params
#' @export
#'
mean.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, name = "mean") {
    distribution$mean(name)
  }

#' Quantile function. Aka "inverse cdf" or "percent point function".
#'
#' Given random variable `X` and `p in [0, 1]`, the `quantile` is:
#' `quantile(p) := x such that P[X <= x] == p`
#'
#' @inherit log_prob.tensorflow_probability.python.distributions.distribution.Distribution return params
#'
#' @export
quantile.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, value, name = "quantile") {
    distribution$quantile(value, name)
  }

#' @export
variance <- function(distribution, name = "variance") {
  UseMethod("variance")
}

#' Variance.
#'
#' Variance is defined as, `Var = E[(X - E[X])**2]`
#' where `X` is the random variable associated with this distribution, `E` denotes expectation,
#' and `Var$shape = batch_shape + event_shape`.
#'
#' @param name String prepended to names of ops created by this function.
#'
#' @return a `Tensor` of shape `sample_shape(x) + self$batch_shape` with values of type `self$dtype`.
#'
#' @export
variance.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, name = "variance") {
    distribution$variance(name)
  }

#' @export
stddev <- function(distribution, name = "stddev") {
  UseMethod("stddev")
}

#' Standard deviation.
#'
#' Standard deviation is defined as, `stddev = E[(X - E[X])**2]**0.5``
#' #' where `X` is the random variable associated with this distribution, `E` denotes expectation,
#' and `Var$shape = batch_shape + event_shape`.
#'
#' @inherit variance.tensorflow_probability.python.distributions.distribution.Distribution return params
#' @export
stddev.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, name = "stddev") {
    distribution$stddev(name)
  }

#' @export
covariance <- function(distribution, name = "covariance") {
  UseMethod("covariance")
}

#' Covariance.
#'
#' Covariance is (possibly) defined only for non-scalar-event distributions.
#' For example, for a length-`k`, vector-valued distribution, it is calculated as,
#' `Cov[i, j] = Covariance(X_i, X_j) = E[(X_i - E[X_i]) (X_j - E[X_j])]`
#' where `Cov` is a (batch of) `k x k` matrix, `0 <= (i, j) < k`, and `E` denotes expectation.
#'
#' Alternatively, for non-vector, multivariate distributions (e.g., matrix-valued, Wishart),
#' `Covariance` shall return a (batch of) matrices under some vectorization of the events, i.e.,
#' `Cov[i, j] = Covariance(Vec(X)_i, Vec(X)_j) = [as above]`
#' where `Cov` is a (batch of) `k' x k'` matrices, `0 <= (i, j) < k' = reduce_prod(event_shape)`,
#' and `Vec` is some function mapping indices of this distribution's event dimensions to indices of a
#' length-`k'` vector.
#'
#' @inherit variance.tensorflow_probability.python.distributions.distribution.Distribution params
#'
#' @return Floating-point `Tensor` with shape `[B1, ..., Bn, k', k']` where the first `n` dimensions
#' are batch coordinates and `k' = reduce_prod(self.event_shape)`.
#' @export
covariance.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, name = "covariance") {
    distribution$covariance(name)
  }

#' @export
mode <- function(distribution, name = "mode") {
  UseMethod("mode")
}

#' Mean.
#'
#' @inherit variance.tensorflow_probability.python.distributions.distribution.Distribution return params
#' @export
mode.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, name = "mode") {
    distribution$mode(name)
  }

#' @export
cross_entropy <- function(distribution, other, name = "cross_entropy") {
  UseMethod("cross_entropy")
}

#' Computes the (Shannon) cross entropy.
#'
#' Denote this distribution (`self`) by `P` and the `other` distribution by `Q`.
#' Assuming `P, Q` are absolutely continuous with respect to one another and permit densities
#' `p(x) dr(x)` and `q(x) dr(x)`, (Shannon) cross entropy is defined as:
#' `H[P, Q] = E_p[-log q(X)] = -int_F p(x) log q(x) dr(x)`
#' where `F` denotes the support of the random variable `X ~ P`.
#'
#' @param distribution The distribution being used.
#' @param other `tfp$distributions$Distribution` instance.
#' @param name String prepended to names of ops created by this function.
#'
#' @return cross_entropy: `self.dtype` `Tensor` with shape `[B1, ..., Bn]` representing `n` different calculations of (Shannon) cross entropy.
#'
#' @export
cross_entropy.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, other, name = "cross_entropy") {
    distribution$cross_entropy(other, name)
  }

#' @export
kl_divergence <- function(distribution, other, name = "kl_divergence") {
  UseMethod("kl_divergence")
}

#' Computes the Kullback--Leibler divergence.
#'
#' Denote this distribution (`self`) by `p` and the `other` distribution by `q`.
#' Assuming `p, q` are absolutely continuous with respect to reference measure `r`,
#' the KL divergence is defined as:
#' `KL[p, q] = E_p[log(p(X)/q(X))] = -int_F p(x) log q(x) dr(x) + int_F p(x) log p(x) dr(x) = H[p, q] - H[p]`
#' where `F` denotes the support of the random variable `X ~ p`, `H[., .]`
#' denotes (Shannon) cross entropy, and `H[.]` denotes (Shannon) entropy.
#'
#' @param distribution The distribution being used.
#' @param other `tfp$distributions$Distribution` instance.
#' @param name String prepended to names of ops created by this function.
#'
#' @return `self$dtype` `Tensor` with shape `[B1, ..., Bn]` representing `n` different calculations
#'  of the Kullback-Leibler divergence.
#'
#' @export
kl_divergence.tensorflow_probability.python.distributions.distribution.Distribution <-
  function(distribution, other, name = "kl_divergence") {
    distribution$kl_divergence(other, name)
  }

