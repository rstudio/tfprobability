
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
distribution_normal <- function(loc,
                                scale,
                                validate_args = FALSE,
                                allow_nan_stats = TRUE,
                                name = "Normal") {
  args <- list(loc = loc,
               scale = scale,
               validate_args = validate_args,
               allow_nan_stats = allow_nan_stats,
               name = name)

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
#' @return
#' @export

distribution_independent <- function(distribution,
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
#' @inheritParams distribution_normal
#'
#' @param logits An N-D `Tensor` representing the log-odds of a `1` event. Each entry in the `Tensor`
#'  parametrizes an independent Bernoulli distribution where the probability of an event
#'  is sigmoid(logits). Only one of `logits` or `probs` should be passed in.
#' @param probs An N-D `Tensor` representing the probability of a `1` event. Each entry in the `Tensor`
#'  parameterizes an independent Bernoulli distribution. Only one of `logits` or `probs`
#'  should be passed in.
#' @param dtype The type of the event samples. Default: `int32`.
#'
#' @return
#' @export
distribution_bernoulli <- function(logits = NULL,
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
#' @inheritParams distribution_normal
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
#' @return
#' @export
distribution_multivariate_normal_diag <- function(loc = NULL,
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

