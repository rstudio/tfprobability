#' The Normal distribution with `loc` and `scale` parameters
#'
#' @param loc Floating point tensor; the means of the distribution(s).
#' @param scale loating point tensor; the stddevs of the distribution(s).
#'  Must contain only positive values.
#' @param validate_args Logical, default `FALSE`. When `TRUE` distribution parameters are checked
#'  for validity despite possibly degrading runtime performance. When `FALSE` invalid inputs may
#'  silently render incorrect outputs. Default value: `FALSE`.
#' @param allow_nan_stats Logical, default `TRUE`. When `TRUE`, tatistics (e.g., mean, mode, variance)
#'  use the value `NaN` to indicate the result is undefined. When `FALSE`, an exception is raised if
#'  one or more of the statistic's batch members are undefined.
#' @param name name prefixed to Ops created by this class.
#'
#' @family distributions
#' @return A normal distribution
#' @export
tfd_normal <- function(loc,
                       scale,
                       validate_args = FALSE,
                       allow_nan_stats = TRUE,
                       name = "Normal") {
  args <- list(
    loc = loc,
    scale = scale,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Normal, args)
}


#' Independent distribution from batch of distributions
#'
#' @param distribution The base distribution instance to transform. Typically an  instance of `Distribution`
#' @param reinterpreted_batch_ndims Scalar, integer number of rightmost batch dims  which
#'  will be regarded as event dims. When NULL all but the first batch axis (batch axis 0)
#'  will be transferred to event dimensions (analogous to `tf$layers$flatten`).
#' @param validate_args Logical, default `FALSE`. When `TRUE` distribution parameters are checked
#'  for validity despite possibly degrading runtime performance. When `FALSE` invalid inputs may
#'  silently render incorrect outputs. Default value: `FALSE`.
#' @param name The name for ops managed by the distribution.  Default value: `Independent + distribution.name`.
#'
#' @family distributions
#' @export
tfd_independent <- function(distribution,
                            reinterpreted_batch_ndims,
                            validate_args = FALSE,
                            name = paste0("Independent", distribution$name)) {
  args <- list(
    distribution = distribution,
    reinterpreted_batch_ndims = as.integer(reinterpreted_batch_ndims),
    validate_args = validate_args,
    name = name
  )

  do.call(tfp$distributions$Independent, args)
}



#' The Bernoulli distribution class.
#'
#' @inheritParams tfd_normal
#'
#' @param logits An N-D `Tensor` representing the log-odds of a `1` event. Each entry in the `Tensor`
#'  parametrizes an independent Bernoulli distribution where the probability of an event
#'  is sigmoid(logits). Only one of `logits` or `probs` should be passed in.
#' @param probs An N-D `Tensor` representing the probability of a `1` event. Each entry in the `Tensor`
#'  parameterizes an independent Bernoulli distribution. Only one of `logits` or `probs`
#'  should be passed in.
#' @param dtype The type of the event samples. Default: `int32`.
#'
#' @export
tfd_bernoulli <- function(logits = NULL,
                          probs = NULL,
                          dtype = tf$int32,
                          validate_args = FALSE,
                          allow_nan_stats = TRUE,
                          name = "Bernoulli") {
  args <- list(
    logits = logits,
    probs = probs,
    dtype = dtype,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Bernoulli, args)
}

#' Title
#'
#' @inheritParams tfd_normal
#'
#' @param loc Floating-point `Tensor`. If this is set to `NULL`, `loc` is implicitly `0`.
#' When specified, may have shape `[B1, ..., Bb, k]` where `b >= 0` and `k` is the event size.
#' @param scale_diag Non-zero, floating-point `Tensor` representing a diagonal matrix added to `scale`.
#'  May have shape `[B1, ..., Bb, k]`, `b >= 0`, and characterizes `b`-batches of `k x k` diagonal matrices
#'  added to `scale`. When both `scale_identity_multiplier` and `scale_diag` are `None` then `scale`
#'  is the `Identity`.
#' @param scale_identity_multiplier Non-zero, floating-point `Tensor` representing a scaled-identity-matrix
#'  added to `scale`. May have shape `[B1, ..., Bb]`, `b >= 0`, and characterizes `b`-batches of scaled
#'  `k x k` identity matrices added to `scale`. When both `scale_identity_multiplier` and `scale_diag`
#'   are `None` then `scale` is the `Identity`.
#'
#' @export
tfd_multivariate_normal_diag <- function(loc = NULL,
                                         scale_diag = NULL,
                                         scale_identity_multiplier = NULL,
                                         validate_args = FALSE,
                                         allow_nan_stats = TRUE,
                                         name = "MultivariateNormalDiag") {
  args <- list(
    loc = loc,
    scale_diag = scale_diag,
    scale_identity_multiplier = scale_identity_multiplier,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$MultivariateNormalDiag, args)
}

#' OneHotCategorical distribution.
#'
#' The categorical distribution is parameterized by the log-probabilities of a set of classes.
#'   The difference between OneHotCategorical and Categorical distributions is that OneHotCategorical
#'   is a discrete distribution over one-hot bit vectors whereas Categorical is a discrete distribution
#'   over positive integers. OneHotCategorical is equivalent to Categorical except Categorical has
#'   event_dim=() while OneHotCategorical has event_dim=K, where K is the number of classes.
#'
#' This class provides methods to create indexed batches of OneHotCategorical distributions.
#'   If the provided logits or probs is rank 2 or higher, for every fixed set of leading dimensions,
#'   the last dimension represents one single OneHotCategorical distribution. When calling distribution
#'   functions (e.g. dist.prob(x)), logits and x are broadcast to the same shape (if possible).
#'   In all cases, the last dimension of logits, x represents single OneHotCategorical distributions.
#' @inheritParams tfd_normal
#' @param logits An N-D Tensor, N >= 1, representing the log probabilities of a set of Categorical distributions.
#'  The first N - 1 dimensions index into a batch of independent distributions and the last dimension represents
#'  a vector of logits for each class. Only one of logits or probs should be passed in.
#' @param probs An N-D Tensor, N >= 1, representing the probabilities of a set of Categorical distributions. The
#'   first N - 1 dimensions index into a batch of independent distributions and the last dimension represents a
#'   vector of probabilities for each class. Only one of logits or probs should be passed in.
#' @param dtype The type of the event samples (default: int32).
#'
#' @export
tfd_one_hot_categorical <- function(logits = NULL,
                                    probs = NULL,
                                    dtype = tf$int32,
                                    validate_args = FALSE,
                                    allow_nan_stats = TRUE,
                                    name = "OneHotCategorical") {
  args <- list(
    logits = logits,
    probs = probs,
    dtype = dtype,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$OneHotCategorical, args)
}

#' RelaxedOneHotCategorical distribution with temperature and logits.
#'
#' @param temperature An 0-D Tensor, representing the temperature of a set of RelaxedOneHotCategorical distributions.
#'  The temperature should be positive.
#' @param logits An N-D Tensor, N >= 1, representing the log probabilities of a set of RelaxedOneHotCategorical
#'  distributions. The first N - 1 dimensions index into a batch of independent distributions and the last dimension
#'  represents a vector of logits for each class. Only one of logits or probs should be passed in.
#' @param probs An N-D Tensor, N >= 1, representing the probabilities of a set of RelaxedOneHotCategorical distributions.
#'   The first N - 1 dimensions index into a batch of independent distributions and the last dimension represents a vector
#'   of probabilities for each class. Only one of logits or probs should be passed in.
#'
#' @inheritParams tfd_normal
#'
#' @family distributions
#' @export
tfd_relaxed_one_hot_categorical <- function(temperature,
                                            logits = NULL,
                                            probs = NULL,
                                            validate_args = FALSE,
                                            allow_nan_stats = TRUE,
                                            name = "RelaxedOneHotCategorical") {
  args <- list(
    temperature = temperature,
    logits = logits,
    probs = probs,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$RelaxedOneHotCategorical, args)
}

#' A Transformed Distribution.
#'
#' A `TransformedDistribution` models `p(y)` given a base distribution `p(x)`,
#' and a deterministic, invertible, differentiable transform, `Y = g(X)`. The
#' transform is typically an instance of the `Bijector` class and the base
#' distribution is typically an instance of the `Distribution` class.
#'
#' @param distribution The base distribution instance to transform. Typically an instance of `Distribution`.
#' @param bijector The object responsible for calculating the transformation. Typically an instance of `Bijector`.
#' @param batch_shape `integer` vector `Tensor` which overrides `distribution` `batch_shape`;
#' valid only if `distribution.is_scalar_batch()`.
#' @param event_shape `integer` vector `Tensor` which overrides `distribution` `event_shape`;
#' valid only if `distribution.is_scalar_event()`.
#' @param validate_args Logical, default `FALSE`. When `TRUE` distribution parameters are checked
#'  for validity despite possibly degrading runtime performance. When `FALSE` invalid inputs may
#'  silently render incorrect outputs. Default value: `FALSE`.
#' @param name The name for ops managed by the distribution.  Default value: `bijector.name + distribution.name`.
#' @export
tfd_transformed <- function(distribution,
                            bijector,
                            batch_shape = NULL,
                            event_shape = NULL,
                            validate_args = FALSE,
                            name = NULL) {
  args <- list(
    distribution = distribution,
    bijector = bijector,
    batch_shape = batch_shape,
    event_shape = event_shape, # wrap in tf$TensorShape?
    validate_args = validate_args,
    name = name
  )

  do.call(tfp$distributions$TransformedDistribution, args)
}
