#' Generate samples of the specified shape.
#'
#' Note that a call to `tfd_sample()` without arguments will generate a single sample.
#'
#' @param distribution The distribution being used.
#' @param sample_shape 0D or 1D int32 Tensor. Shape of the generated samples.
#' @param ... Additional parameters passed to Python.
#'
#' @return a Tensor with prepended dimensions sample_shape.
#' @examples
#' \donttest{
#'   d <- tfd_normal(loc = c(1, 2), scale = c(1, 0.5))
#'   d %>% tfd_sample()
#' }
#' @family distribution_methods
#' @export
tfd_sample <- function(distribution,
                       sample_shape = list(),
                       ...) {

  distribution$sample(normalize_shape(sample_shape), ...)
}

#' Log probability density/mass function.
#'
#' @param distribution The distribution being used.
#' @param value float or double Tensor.
#' @param ... Additional parameters passed to Python.
#'
#' @return a Tensor of shape `sample_shape(x) + self$batch_shape` with values of type `self$dtype`.
#' @examples
#' \donttest{
#'   d <- tfd_normal(loc = c(1, 2), scale = c(1, 0.5))
#'   x <- d %>% tfd_sample()
#'   d %>% tfd_log_prob(x)
#' }
#' @family distribution_methods
#' @export
tfd_log_prob <- function(distribution, value, ...) {
  distribution$log_prob(value, ...)
}

#' Probability density/mass function.
#'
#' @param distribution The distribution being used.
#' @param value float or double Tensor.
#' @inherit tfd_log_prob return params
#' @family distribution_methods
#' @examples
#' \donttest{
#'   d <- tfd_normal(loc = c(1, 2), scale = c(1, 0.5))
#'   x <- d %>% tfd_sample()
#'   d %>% tfd_prob(x)
#' }
#' @export
tfd_prob <- function(distribution, value, ...) {
  distribution$prob(value, ...)
}

#' Log cumulative distribution function.
#'
#' Given random variable X, the cumulative distribution function cdf is:
#' `tfd_log_cdf(x) := Log[ P[X <= x] ]`
#' Often, a numerical approximation can be used for `tfd_log_cdf(x)` that yields
#' a more accurate answer than simply taking the logarithm of the cdf when x << -1.
#'
#' @inherit tfd_prob return params
#' @family distribution_methods
#' @examples
#' \donttest{
#'   d <- tfd_normal(loc = c(1, 2), scale = c(1, 0.5))
#'   x <- d %>% tfd_sample()
#'   d %>% tfd_log_cdf(x)
#' }
#' @export
tfd_log_cdf <- function(distribution, value, ...) {
  distribution$log_cdf(value, ...)
}

#' Cumulative distribution function.
#' Given random variable X, the cumulative distribution function cdf is:
#' `cdf(x) := P[X <= x]`
#'
#' @inherit tfd_prob return params
#' @family distribution_methods
#' @examples
#' \donttest{
#'   d <- tfd_normal(loc = c(1, 2), scale = c(1, 0.5))
#'   x <- d %>% tfd_sample()
#'   d %>% tfd_cdf(x)
#' }
#' @export
tfd_cdf <- function(distribution, value, ...) {
  distribution$cdf(value, ...)
}

#' Log survival function.
#'
#' Given random variable X, the survival function is defined:
#' `tfd_log_survival_function(x) = Log[ P[X > x] ] = Log[ 1 - P[X <= x] ] = Log[ 1 - cdf(x) ]`
#'
#' Typically, different numerical approximations can be used for the log survival function,
#'  which are more accurate than 1 - cdf(x) when x >> 1.
#'
#' @inherit tfd_prob return params
#' @examples
#' \donttest{
#'   d <- tfd_normal(loc = c(1, 2), scale = c(1, 0.5))
#'   x <- d %>% tfd_sample()
#'   d %>% tfd_log_survival_function(x)
#' }
#' @family distribution_methods
#' @export
tfd_log_survival_function <- function(distribution, value, ...) {
  distribution$log_survival_function(value, ...)
}

#' Survival function.
#'
#' Given random variable X, the survival function is defined:
#' `tfd_survival_function(x) = P[X > x] = 1 - P[X <= x] = 1 - cdf(x)`.
#'
#' @inherit tfd_prob return params
#' @family distribution_methods
#' @examples
#' \donttest{
#'   d <- tfd_normal(loc = c(1, 2), scale = c(1, 0.5))
#'   x <- d %>% tfd_sample()
#'   d %>% tfd_survival_function(x)
#' }
#' @export
tfd_survival_function <- function(distribution, value, ...) {
  distribution$survival_function(value, ...)
}

#' Shannon entropy in nats.
#'
#' @inherit tfd_prob return params
#' @family distribution_methods
#' @examples
#' \donttest{
#'   d <- tfd_normal(loc = c(1, 2), scale = c(1, 0.5))
#'   d %>% tfd_entropy()
#' }
#' @export
tfd_entropy <- function(distribution, ...) {
  distribution$entropy(...)
}

#' Mean.
#'
#' @inherit tfd_prob return params
#' @family distribution_methods
#' @examples
#' \donttest{
#'   d <- tfd_normal(loc = c(1, 2), scale = c(1, 0.5))
#'   d %>% tfd_mean()
#' }
#' @export
tfd_mean <- function(distribution, ...) {
  distribution$mean(...)
}

#' Quantile function. Aka "inverse cdf" or "percent point function".
#'
#' Given random variable X and p in `[0, 1]`, the quantile is:
#' `tfd_quantile(p) := x` such that `P[X <= x] == p`
#'
#' @inherit tfd_prob return params
#' @examples
#' \donttest{
#'   d <- tfd_normal(loc = c(1, 2), scale = c(1, 0.5))
#'   d %>% tfd_quantile(0.5)
#' }
#' @family distribution_methods
#' @export
tfd_quantile <- function(distribution, value, ...) {
  distribution$quantile(value, ...)
}

#' Variance.
#'
#' Variance is defined as, `Var = E[(X - E[X])**2]`
#' where X is the random variable associated with this distribution, E denotes expectation,
#' and `Var$shape = batch_shape + event_shape`.
#'
#' @inherit tfd_prob return params
#' @return a Tensor of shape `sample_shape(x) + self$batch_shape` with values of type `self$dtype`.
#' @examples
#' \donttest{
#'   d <- tfd_normal(loc = c(1, 2), scale = c(1, 0.5))
#'   d %>% tfd_variance()
#' }
#' @family distribution_methods
#' @export
tfd_variance <- function(distribution, ...) {
  distribution$variance(...)
}

#' Standard deviation.
#'
#' Standard deviation is defined as, stddev = `E[(X - E[X])**2]**0.5`
#' #' where X is the random variable associated with this distribution, E denotes expectation,
#' and `Var$shape = batch_shape + event_shape`.
#'
#' @inherit tfd_prob return params
#' @family distribution_methods
#' @examples
#' \donttest{
#'   d <- tfd_normal(loc = c(1, 2), scale = c(1, 0.5))
#'   d %>% tfd_stddev()
#' }
#' @export
tfd_stddev <- function(distribution, ...) {
  distribution$stddev(...)
}

#' Covariance.
#'
#' Covariance is (possibly) defined only for non-scalar-event distributions.
#' For example, for a length-k, vector-valued distribution, it is calculated as,
#' `Cov[i, j] = Covariance(X_i, X_j) = E[(X_i - E[X_i]) (X_j - E[X_j])]`
#' where Cov is a (batch of) k x k matrix, 0 <= (i, j) < k, and E denotes expectation.
#'
#' Alternatively, for non-vector, multivariate distributions (e.g., matrix-valued, Wishart),
#' Covariance shall return a (batch of) matrices under some vectorization of the events, i.e.,
#' `Cov[i, j] = Covariance(Vec(X)_i, Vec(X)_j) = [as above]`
#' where Cov is a (batch of) k x k matrices, 0 <= (i, j) < k = reduce_prod(event_shape),
#' and Vec is some function mapping indices of this distribution's event dimensions to indices of a
#' length-k vector.
#'
#' @inherit tfd_prob return params
#'
#' @return Floating-point Tensor with shape `[B1, ..., Bn, k, k]` where the first n dimensions
#' are batch coordinates and `k = reduce_prod(self.event_shape)`.
#' @examples
#' \donttest{
#' d <- tfd_normal(loc = c(1, 2), scale = c(1, 0.5))
#' d %>% tfd_variance()
#' }
#' @family distribution_methods
#' @export
tfd_covariance <- function(distribution, ...) {
  distribution$covariance(...)
}

#' Mode.
#'
#' @inherit tfd_prob return params
#' @family distribution_methods
#' @examples
#' \donttest{
#'   d <- tfd_normal(loc = c(1, 2), scale = c(1, 0.5))
#'   d %>% tfd_mode()
#' }
#' @export
tfd_mode <- function(distribution, ...) {
  distribution$mode(...)
}

#' Computes the (Shannon) cross entropy.
#'
#' Denote this distribution (self) by P and the other distribution by Q.
#' Assuming P, Q are absolutely continuous with respect to one another and permit densities
#' p(x) dr(x) and q(x) dr(x), (Shannon) cross entropy is defined as:
#' `H[P, Q] = E_p[-log q(X)] = -int_F p(x) log q(x) dr(x)`
#' where F denotes the support of the random variable `X ~ P`.
#'
#' @param distribution The distribution being used.
#' @param other `tfp$distributions$Distribution` instance.
#' @param name String prepended to names of ops created by this function.
#'
#' @return cross_entropy: self.dtype Tensor with shape `[B1, ..., Bn]` representing n different calculations of (Shannon) cross entropy.
#' @examples
#' \donttest{
#'   d1 <- tfd_normal(loc = 1, scale = 1)
#'   d2 <- tfd_normal(loc = 2, scale = 1)
#'   d1 %>% tfd_cross_entropy(d2)
#' }
#' @family distribution_methods
#' @export
tfd_cross_entropy <- function(distribution, other, name = "cross_entropy") {
  distribution$cross_entropy(other, name)
}

#' Computes the Kullback--Leibler divergence.
#'
#' Denote this distribution by p and the other distribution by q.
#' Assuming p, q are absolutely continuous with respect to reference measure r,
#' the KL divergence is defined as:
#' `KL[p, q] = E_p[log(p(X)/q(X))] = -int_F p(x) log q(x) dr(x) + int_F p(x) log p(x) dr(x) = H[p, q] - H[p]`
#' where F denotes the support of the random variable `X ~ p`, `H[., .]`
#' denotes (Shannon) cross entropy, and `H[.]` denotes (Shannon) entropy.
#'
#' @param distribution The distribution being used.
#' @param other `tfp$distributions$Distribution` instance.
#' @param name String prepended to names of ops created by this function.
#'
#' @return self$dtype Tensor with shape `[B1, ..., Bn]` representing n different calculations
#'  of the Kullback-Leibler divergence.
#' @examples
#' \donttest{
#'   d1 <- tfd_normal(loc = c(1, 2), scale = c(1, 0.5))
#'   d2 <- tfd_normal(loc = c(1.5, 2), scale = c(1, 0.5))
#'   d1 %>% tfd_kl_divergence(d2)
#' }
#' @family distribution_methods
#' @export
tfd_kl_divergence <- function(distribution, other, name = "kl_divergence") {
  distribution$kl_divergence(other, name)
}
