#' The Normal distribution with loc and scale parameters
#'
#' @param loc Floating point tensor; the means of the distribution(s).
#' @param scale loating point tensor; the stddevs of the distribution(s).
#'  Must contain only positive values.
#' @param validate_args Logical, default FALSE. When TRUE distribution parameters are checked
#'  for validity despite possibly degrading runtime performance. When FALSE invalid inputs may
#'  silently render incorrect outputs. Default value: FALSE.
#' @param allow_nan_stats Logical, default TRUE. When TRUE, tatistics (e.g., mean, mode, variance)
#'  use the value NaN to indicate the result is undefined. When FALSE, an exception is raised if
#'  one or more of the statistic's batch members are undefined.
#' @param name name prefixed to Ops created by this class.
#'
#' @family distributions
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
#' @param distribution The base distribution instance to transform. Typically an  instance of Distribution
#' @param reinterpreted_batch_ndims Scalar, integer number of rightmost batch dims  which
#'  will be regarded as event dims. When NULL all but the first batch axis (batch axis 0)
#'  will be transferred to event dimensions (analogous to tf$layers$flatten).
#' @param validate_args Logical, default FALSE. When TRUE distribution parameters are checked
#'  for validity despite possibly degrading runtime performance. When FALSE invalid inputs may
#'  silently render incorrect outputs. Default value: FALSE.
#' @param name The name for ops managed by the distribution.  Default value: Independent + distribution.name.
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
#' @param logits An N-D Tensor representing the log-odds of a 1 event. Each entry in the Tensor
#'  parametrizes an independent Bernoulli distribution where the probability of an event
#'  is sigmoid(logits). Only one of logits or probs should be passed in.
#' @param probs An N-D Tensor representing the probability of a 1 event. Each entry in the Tensor
#'  parameterizes an independent Bernoulli distribution. Only one of logits or probs
#'  should be passed in.
#' @param dtype The type of the event samples. Default: int32.
#'
#' @family distributions
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

#' The multivariate normal distribution on `R^k`.
#'
#' The Multivariate Normal distribution is defined over R^k and parameterized
#' by a (batch of) length-k loc vector (aka "mu") and a (batch of) `k x k`
#' scale matrix; `covariance = scale @ scale.T` where `@` denotes
#' matrix-multiplication.
#'
#' @inheritParams tfd_normal
#'
#' @param loc Floating-point Tensor. If this is set to NULL, loc is implicitly 0.
#' When specified, may have shape `[B1, ..., Bb, k]` where b >= 0 and k is the event size.
#' @param scale_diag Non-zero, floating-point Tensor representing a diagonal matrix added to scale.
#'  May have shape `[B1, ..., Bb, k]`, b >= 0, and characterizes b-batches of `k x k` diagonal matrices
#'  added to scale. When both scale_identity_multiplier and scale_diag are NULL then scale
#'  is the Identity.
#' @param scale_identity_multiplier Non-zero, floating-point Tensor representing a scaled-identity-matrix
#'  added to scale. May have shape `[B1, ..., Bb]`, b >= 0, and characterizes b-batches of scaled
#'  `k x k` identity matrices added to scale. When both scale_identity_multiplier and scale_diag
#'   are NULL then scale is the Identity.
#' @family distributions
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
#' @family distributions
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

#' RelaxedBernoulli distribution with temperature and logits parameters.
#'
#' The RelaxedBernoulli is a distribution over the unit interval (0,1), which continuously approximates a Bernoulli.
#'   The degree of approximation is controlled by a temperature: as the temperature goes to 0 the RelaxedBernoulli
#'   becomes discrete with a distribution described by the logits or probs parameters, as the temperature goes to
#'   infinity the RelaxedBernoulli becomes the constant distribution that is identically 0.5.
#'
#' @param temperature An 0-D Tensor, representing the temperature of a set of RelaxedBernoulli distributions.
#'   The temperature should be positive.
#' @param logits  An N-D Tensor representing the log-odds of a positive event. Each entry in the Tensor
#'   parametrizes an independent RelaxedBernoulli distribution where the probability of an event is sigmoid(logits).
#'   Only one of logits or probs should be passed in.
#' @param probs AAn N-D Tensor representing the probability of a positive event. Each entry in the Tensor parameterizes
#'  an independent Bernoulli distribution. Only one of logits or probs should be passed in.
#'
#' @inheritParams tfd_normal
#'
#' @family distributions
#' @export
tfd_relaxed_bernoulli <- function(temperature,
                                  logits = NULL,
                                  probs = NULL,
                                  validate_args = FALSE,
                                  allow_nan_stats = TRUE,
                                  name = "RelaxedBernoulli") {
  args <- list(
    temperature = temperature,
    logits = logits,
    probs = probs,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$RelaxedBernoulli, args)
}

#' A Transformed Distribution.
#'
#' A TransformedDistribution models `p(y)` given a base distribution `p(x)`,
#' and a deterministic, invertible, differentiable transform,`Y = g(X)`. The
#' transform is typically an instance of the Bijector class and the base
#' distribution is typically an instance of the Distribution class.
#'
#' @param distribution The base distribution instance to transform. Typically an instance of Distribution.
#' @param bijector The object responsible for calculating the transformation. Typically an instance of Bijector.
#' @param batch_shape integer vector Tensor which overrides distribution batch_shape;
#' valid only if distribution.is_scalar_batch().
#' @param event_shape integer vector Tensor which overrides distribution event_shape;
#' valid only if distribution.is_scalar_event().
#' @param validate_args Logical, default FALSE. When TRUE distribution parameters are checked
#'  for validity despite possibly degrading runtime performance. When FALSE invalid inputs may
#'  silently render incorrect outputs. Default value: FALSE.
#' @param name The name for ops managed by the distribution.  Default value: bijector.name + distribution.name.
#' @family distributions
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

#' Zipf distribution.
#'
#' The Zipf distribution is parameterized by a power parameter.
#'
#' @param power Float like Tensor representing the power parameter. Must be
#' strictly greater than 1.
#' @param dtype The dtype of Tensor returned by sample. Default value: tf$int32.
#' @param interpolate_nondiscrete Logical. When FALSE, log_prob returns
#' -inf (and prob returns 0) for non-integer inputs. When TRUE,
#' log_prob evaluates the continuous function `-power log(k) -   log(zeta(power))` ,
#' which matches the Zipf pmf at integer arguments k
#' (note that this function is not itself a normalized probability  log-density).
#' Default value: TRUE.
#' @param sample_maximum_iterations Maximum number of iterations of allowable
#' iterations in sample. When validate_args=TRUE, samples which fail to
#' reach convergence (subject to this cap) are masked out with
#' `self$dtype$min` or nan depending on `self$dtype$is_integer`.
#' Default value: 100.
#'
#' @param allow_nan_stats Default value: FALSE.
#'
#' @inheritParams tfd_normal
#'
#' @family distributions
#' @export
tfd_zipf <- function(power,
                     dtype = tf$int32,
                     interpolate_nondiscrete = TRUE,
                     sample_maximum_iterations = 100,
                     validate_args = FALSE,
                     allow_nan_stats = FALSE,
                     name = "Zipf") {
  args <- list(
    power = power,
    dtype = dtype,
    interpolate_nondiscrete = interpolate_nondiscrete,
    sample_maximum_iterations = sample_maximum_iterations,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Zipf, args)
}

#' The matrix Wishart distribution on positive definite matrices.
#'
#' This distribution is defined by a scalar number of degrees of freedom df and
#' an instance of LinearOperator, which provides matrix-free access to a
#' symmetric positive definite operator, which defines the scale matrix.
#' @param df float or double tensor, the degrees of freedom of the
#' distribution(s). df must be greater than or equal to k.
#' @param scale float or double Tensor. The symmetric positive definite
#' scale matrix of the distribution. Exactly one of scale and 'scale_tril must be passed.
#' @param scale_tril float or double Tensor. The Cholesky factorization
#' of the symmetric positive definite scale matrix of the distribution.
#' Exactly one of scale and 'scale_tril must be passed.
#' @param input_output_cholesky Logical. If TRUE, functions whose input or
#' output have the semantics of samples assume inputs are in Cholesky form
#' and return outputs in Cholesky form. In particular, if this flag is
#' TRUE, input to log_prob is presumed of Cholesky form and output from
#' sample, mean, and mode are of Cholesky form.  Setting this
#' argument to TRUE is purely a computational optimization and does not
#' change the underlying distribution; for instance, mean returns the
#' Cholesky of the mean, not the mean of Cholesky factors. The variance
#' and stddev methods are unaffected by this flag.
#' Default value: FALSE (i.e., input/output does not have Cholesky semantics).
#'
#' @inheritParams tfd_normal
#'
#' @family distributions
#' @export
tfd_wishart <- function(df,
                        scale = NULL,
                        scale_tril = NULL,
                        input_output_cholesky = FALSE,
                        validate_args = FALSE,
                        allow_nan_stats = TRUE,
                        name = "Wishart") {
  args <- list(
    df = df,
    scale = scale,
    scale_tril = scale_tril,
    input_output_cholesky = input_output_cholesky,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Wishart, args)
}

#' The von Mises-Fisher distribution over unit vectors on `S^{n-1}`.
#'
#' The von Mises-Fisher distribution is a directional distribution over vectors
#' on the unit hypersphere `S^{n-1}` embedded in n dimensions `(R^n)`.
#'
#' NOTE: Currently only n in {2, 3, 4, 5} are supported. For n=5 some numerical
#' instability can occur for low concentrations (<.01).
#' @param mean_direction Floating-point Tensor with shape `[B1, ... Bn, D]`.
#' A unit vector indicating the mode of the distribution, or the
#' unit-normalized direction of the mean. (This is *not* in general the
#' mean of the distribution; the mean is not generally in the support of
#' the distribution.) NOTE: D is currently restricted to <= 5.
#' @param concentration Floating-point Tensor having batch shape `[B1, ... Bn]`
#' broadcastable with mean_direction. The level of concentration of
#' samples around the mean_direction. concentration=0 indicates a
#' uniform distribution over the unit hypersphere, and concentration=+inf
#' indicates a Deterministic distribution (delta function) at mean_direction.
#' @inheritParams tfd_normal
#'
#' @family distributions
#' @export
tfd_von_mises_fisher <- function(df,
                        mean_direction,
                        concentration,
                        validate_args = FALSE,
                        allow_nan_stats = TRUE,
                        name = "VonMisesFisher") {
  args <- list(
    mean_direction = mean_direction,
    concentration = concentration,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$VonMisesFisher, args)
}

#' The von Mises distribution over angles.
#'
#' The von Mises distribution is a univariate directional distribution.
#' Similarly to Normal distribution, it is a maximum entropy distribution.
#' The samples of this distribution are angles, measured in radians.
#' They are 2 pi-periodic: x = 0 and x = 2pi are equivalent.
#' This means that the density is also 2 pi-periodic.
#' The generated samples, however, are guaranteed to be in `[-pi, pi)` range.
#' When concentration = 0, this distribution becomes a Uniform distribuion on
#' the `[-pi, pi)` domain.
#'
#' The von Mises distribution is a special case of von Mises-Fisher distribution
#' for n=2. However, the TFP's VonMisesFisher implementation represents the
#' samples and location as (x, y) points on a circle, while VonMises represents
#' them as scalar angles.
#'
#' The parameters loc and concentration must be shaped in a way that
#' supports broadcasting (e.g. loc + concentration is a valid operation).

#' @param loc Floating point tensor, the circular means of the distribution(s).
#' @param concentration Floating point tensor, the level of concentration of the
#' distribution(s) around loc. Must take non-negative values.
#' concentration = 0 defines a Uniform distribution, while
#' concentration = +inf indicates a Deterministic distribution at loc.
#' @inheritParams tfd_normal
#'
#' @family distributions
#' @export
tfd_von_mises <- function(loc,
                          concentration,
                          validate_args = FALSE,
                          allow_nan_stats = TRUE,
                          name = "VonMises") {
  args <- list(
    loc = loc,
    concentration = concentration,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$VonMises, args)
}

#' The (diagonal) SinhArcsinh transformation of a distribution on `R^k`.
#'
#' This distribution models a random vector `Y = (Y1,...,Yk)`, making use of
#' a SinhArcsinh transformation (which has adjustable tailweight and skew),
#' a rescaling, and a shift.
#' The SinhArcsinh transformation of the Normal is described in great depth in
#' [Sinh-arcsinh distributions](https://www.jstor.org/stable/27798865).
#' Here we use a slightly different parameterization, in terms of tailweight
#' and skewness.  Additionally we allow for distributions other than Normal,
#' and control over scale as well as a "shift" parameter loc.
#'
#' @param loc Floating-point Tensor. If this is set to NULL, loc is
#' implicitly 0. When specified, may have shape `[B1, ..., Bb, k]` where
#' b >= 0 and k is the event size.
#' @param scale_diag Non-zero, floating-point Tensor representing a diagonal
#' matrix added to scale. May have shape `[B1, ..., Bb, k]`, b >= 0,
#' and characterizes b-batches of k x k diagonal matrices added to
#' scale. When both scale_identity_multiplier and scale_diag are
#' NULL then scale is the Identity.
#' @param scale_identity_multiplier Non-zero, floating-point Tensor representing
#' a scale-identity-matrix added to scale. May have shape
#' `[B1, ..., Bb]`, b >= 0, and characterizes b-batches of scale
#' k x k identity matrices added to scale. When both
#' scale_identity_multiplier and scale_diag are NULL then scale
#' is the Identity.
#' @param skewness  Skewness parameter.  floating-point Tensor with shape
#' broadcastable with event_shape.
#' @param tailweight  Tailweight parameter.  floating-point Tensor with shape
#' broadcastable with event_shape.
#' @param distribution `tf$distributions$Distribution`-like instance. Distribution from which k
#' iid samples are used as input to transformation F.  Default is
#' tfd_normal(loc = 0, scale = 1).
#' Must be a scalar-batch, scalar-event distribution.  Typically
#' distribution$reparameterization_type = FULLY_REPARAMETERIZED or it is
#' a function of non-trainable parameters. WARNING: If you backprop through
#' a VectorSinhArcsinhDiag sample and distribution is not
#' FULLY_REPARAMETERIZED yet is a function of trainable variables, then
#' the gradient will be incorrect!
#' @inheritParams tfd_normal
#'
#' @family distributions
#' @export
tfd_vector_sinh_arcsinh_diag <- function(loc = NULL,
                          scale_diag = NULL,
                          scale_identity_multiplier = NULL,
                          skewness = NULL,
                          tailweight = NULL,
                          distribution = NULL,
                          validate_args = FALSE,
                          allow_nan_stats = TRUE,
                          name = "VectorSinhArcsinhDiag") {
  args <- list(
    loc = loc,
    scale_diag = scale_diag,
    scale_identity_multiplier = scale_identity_multiplier,
    skewness = skewness,
    tailweight = tailweight,
    distribution = distribution,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$VectorSinhArcsinhDiag, args)
}

#' The vectorization of the Laplace distribution on `R^k`.
#'
#' The vector laplace distribution is defined over `R^k`, and parameterized by
#' a (batch of) length-k loc vector (the means) and a (batch of) k x k
#' scale matrix:  `covariance = 2 * scale @ scale.T`, where `@` denotes
#' matrix-multiplication.
#'
#' About VectorLaplace and Vector distributions in TensorFlow.
#' The VectorLaplace is a non-standard distribution that has useful properties.
#' The marginals Y_1, ..., Y_k are *not* Laplace random variables, due to
#' the fact that the sum of Laplace random variables is not Laplace.
#' Instead, Y is a vector whose components are linear combinations of Laplace
#' random variables.  Thus, Y lives in the vector space generated by vectors
#' of Laplace distributions.  This allows the user to decide the mean and
#' covariance (by setting loc and scale), while preserving some properties of
#' the Laplace distribution.  In particular, the tails of Y_i will be (up to
#' polynomial factors) exponentially decaying.
#' To see this last statement, note that the pdf of Y_i is the convolution of
#' the pdf of k independent Laplace random variables.  One can then show by
#' induction that distributions with exponential (up to polynomial factors) tails
#' are closed under convolution.
#'
#' The batch_shape is the broadcast shape between loc and scale
#' arguments.
#' The event_shape is given by last dimension of the matrix implied by
#' scale. The last dimension of loc (if provided) must broadcast with this.
#' Recall that `covariance = 2 * scale @ scale.T`.
#' Additional leading dimensions (if any) will index batches.
#'
#' @param  loc Floating-point Tensor. If this is set to NULL, loc is
#' implicitly 0. When specified, may have shape `[B1, ..., Bb, k]` where
#' b >= 0 and k is the event size.
#' @param scale Instance of LinearOperator with same dtype as loc and shape
#' `[B1, ..., Bb, k, k]`.

#' @inheritParams tfd_normal
#'
#' @family distributions
#' @export
tfd_vector_laplace_linear_operator <- function(loc = NULL,
                                               scale = NULL,
                                               validate_args = FALSE,
                                               allow_nan_stats = TRUE,
                                               name = "VectorLaplaceLinearOperator") {
  args <- list(
    loc = loc,
    scale = scale,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$vector_laplace_linear_operator$VectorLaplaceLinearOperator, args)
}

#' The vectorization of the Laplace distribution on `R^k`.
#'
#' The vector laplace distribution is defined over `R^k`, and parameterized by
#' a (batch of) length-k loc vector (the means) and a (batch of) k x k
#' scale matrix:  `covariance = 2 * scale @ scale.T`, where @ denotes
#' matrix-multiplication.
#'
#' About VectorLaplace and Vector distributions in TensorFlow.
#' The VectorLaplace is a non-standard distribution that has useful properties.
#' The marginals Y_1, ..., Y_k are *not* Laplace random variables, due to
#' the fact that the sum of Laplace random variables is not Laplace.
#' Instead, Y is a vector whose components are linear combinations of Laplace
#' random variables.  Thus, Y lives in the vector space generated by vectors
#' of Laplace distributions.  This allows the user to decide the mean and
#' covariance (by setting loc and scale), while preserving some properties of
#' the Laplace distribution.  In particular, the tails of Y_i will be (up to
#' polynomial factors) exponentially decaying.
#' To see this last statement, note that the pdf of Y_i is the convolution of
#' the pdf of k independent Laplace random variables.  One can then show by
#' induction that distributions with exponential (up to polynomial factors) tails
#' are closed under convolution.
#'
#' @param loc Floating-point Tensor. If this is set to NULL, loc is
#' implicitly 0. When specified, may have shape `[B1, ..., Bb, k]` where
#' b >= 0 and k is the event size.
#' @param scale_diag Non-zero, floating-point Tensor representing a diagonal
#' matrix added to scale. May have shape `[B1, ..., Bb, k]`, b >= 0,
#' and characterizes b-batches of k x k diagonal matrices added to
#' scale. When both scale_identity_multiplier and scale_diag are
#' NULL then scale is the Identity.
#' @param scale_identity_multiplier Non-zero, floating-point Tensor representing
#' a scaled-identity-matrix added to scale. May have shape
#' [B1, ..., Bb], b >= 0, and characterizes b-batches of scaled
#' k x k identity matrices added to scale. When both
#' scale_identity_multiplier and scale_diag are NULL then scale is
#' the Identity.

#' @inheritParams tfd_normal
#'
#' @family distributions
#' @export
tfd_vector_laplace_diag <- function(loc = NULL,
                                    scale_diag = NULL,
                                    scale_identity_multiplier = NULL,
                                    validate_args = FALSE,
                                    allow_nan_stats = TRUE,
                                    name = "VectorLaplaceDiag") {
  args <- list(
    loc = loc,
    scale_diag = scale_diag,
    scale_identity_multiplier = scale_identity_multiplier,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$VectorLaplaceDiag, args)
}

#' @inheritParams tfd_normal
#'
#' @family distributions
#' @export
tfd_vector_laplace_diag <- function(loc = NULL,
                                    scale_diag = NULL,
                                    scale_identity_multiplier = NULL,
                                    validate_args = FALSE,
                                    allow_nan_stats = TRUE,
                                    name = "VectorLaplaceDiag") {
  args <- list(
    loc = loc,
    scale_diag = scale_diag,
    scale_identity_multiplier = scale_identity_multiplier,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$VectorLaplaceDiag, args)
}

#' The vectorization of the Exponential distribution on `R^k`.
#'
#' The vector exponential distribution is defined over a subset of `R^k`, and
#' parameterized by a (batch of) length-`k` `loc` vector and a (batch of) `k x k`
#' `scale` matrix:  `covariance = scale @ scale.T`, where `@` denotes
#' matrix-multiplication.
#'
#' The `VectorExponential` is a non-standard distribution that has useful
#' properties.
#' The marginals `Y_1, ..., Y_k` are *not* Exponential random variables, due to
#' the fact that the sum of Exponential random variables is not Exponential.
#' Instead, `Y` is a vector whose components are linear combinations of
#' Exponential random variables.  Thus, `Y` lives in the vector space generated
#' by `vectors` of Exponential distributions.  This allows the user to decide the
#' mean and covariance (by setting `loc` and `scale`), while preserving some
#' properties of the Exponential distribution.  In particular, the tails of `Y_i`
#' will be (up to polynomial factors) exponentially decaying.
#' To see this last statement, note that the pdf of `Y_i` is the convolution of
#' the pdf of `k` independent Exponential random variables.  One can then show by
#' induction that distributions with exponential (up to polynomial factors) tails
#' are closed under convolution.
#'
#' The batch_shape is the broadcast shape between loc and scale
#' arguments.
#' The event_shape is given by last dimension of the matrix implied by
#' scale. The last dimension of loc (if provided) must broadcast with this.
#' Recall that `covariance = 2 * scale @ scale.T`.
#' Additional leading dimensions (if any) will index batches.
#' If both `scale_diag` and `scale_identity_multiplier` are `NULL`, then
#' `scale` is the Identity matrix.

#'
#' @param loc Floating-point Tensor. If this is set to NULL, loc is
#' implicitly 0. When specified, may have shape `[B1, ..., Bb, k]` where
#' b >= 0 and k is the event size.
#' @param scale_diag Non-zero, floating-point Tensor representing a diagonal
#' matrix added to scale. May have shape `[B1, ..., Bb, k]`, b >= 0,
#' and characterizes b-batches of k x k diagonal matrices added to
#' scale. When both scale_identity_multiplier and scale_diag are
#' NULL then scale is the Identity.
#' @param scale_identity_multiplier Non-zero, floating-point Tensor representing
#' a scaled-identity-matrix added to scale. May have shape
#' [B1, ..., Bb], b >= 0, and characterizes b-batches of scaled
#' k x k identity matrices added to scale. When both
#' scale_identity_multiplier and scale_diag are NULL then scale is
#' the Identity.

#' @inheritParams tfd_normal
#'
#' @family distributions
#' @export
tfd_vector_exponential_diag <- function(loc = NULL,
                                        scale_diag = NULL,
                                        scale_identity_multiplier = NULL,
                                        validate_args = FALSE,
                                        allow_nan_stats = TRUE,
                                        name = "VectorExponentialDiag") {
  args <- list(
    loc = loc,
    scale_diag = scale_diag,
    scale_identity_multiplier = scale_identity_multiplier,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$VectorExponentialDiag, args)
}

#' The vectorization of the Exponential distribution on `R^k`.
#'
#' The vector exponential distribution is defined over a subset of `R^k`, and
#' parameterized by a (batch of) length-`k` `loc` vector and a (batch of) `k x k`
#' `scale` matrix:  `covariance = scale @ scale.T`, where `@` denotes
#' matrix-multiplication.
#'
#' The `VectorExponential` is a non-standard distribution that has useful
#' properties.
#' The marginals `Y_1, ..., Y_k` are *not* Exponential random variables, due to
#' the fact that the sum of Exponential random variables is not Exponential.
#' Instead, `Y` is a vector whose components are linear combinations of
#' Exponential random variables.  Thus, `Y` lives in the vector space generated
#' by `vectors` of Exponential distributions.  This allows the user to decide the
#' mean and covariance (by setting `loc` and `scale`), while preserving some
#' properties of the Exponential distribution.  In particular, the tails of `Y_i`
#' will be (up to polynomial factors) exponentially decaying.
#' To see this last statement, note that the pdf of `Y_i` is the convolution of
#' the pdf of `k` independent Exponential random variables.  One can then show by
#' induction that distributions with exponential (up to polynomial factors) tails
#' are closed under convolution.
#'
#' The batch_shape is the broadcast shape between loc and scale
#' arguments.
#' The event_shape is given by last dimension of the matrix implied by
#' scale. The last dimension of loc (if provided) must broadcast with this.
#' Recall that `covariance = 2 * scale @ scale.T`.
#' Additional leading dimensions (if any) will index batches.
#'
#' #' @param  loc Floating-point Tensor. If this is set to NULL, loc is
#' implicitly 0. When specified, may have shape `[B1, ..., Bb, k]` where
#' b >= 0 and k is the event size.
#' @param scale Instance of LinearOperator with same dtype as loc and shape
#' `[B1, ..., Bb, k, k]`.

#' @inheritParams tfd_normal
#'
#' @family distributions
#' @export
tfd_vector_exponential_linear_operator <- function(loc = NULL,
                                                   scale = NULL,
                                                   validate_args = FALSE,
                                                   allow_nan_stats = TRUE,
                                                   name = "VectorExponentialLinearOperator") {
  args <- list(
    loc = loc,
    scale = scale,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$vector_exponential_linear_operator$VectorExponentialLinearOperator, args)
}


#' VectorDiffeomixture distribution.
#'
#' A vector diffeomixture (VDM) is a distribution parameterized by a convex
#' combination of `K` component `loc` vectors, `loc[k], k = 0,...,K-1`, and `K`
#' `scale` matrices `scale[k], k = 0,..., K-1`.  It approximates the following
#' [compound distribution]
#' (https://en.wikipedia.org/wiki/Compound_probability_distribution)
#' `p(x) = int p(x | z) p(z) dz`, where z is in the K-simplex, and
#' `p(x | z) := p(x | loc=sum_k z[k] loc[k], scale=sum_k z[k] scale[k])`
#'
#' The integral `int p(x | z) p(z) dz` is approximated with a quadrature scheme
#' adapted to the mixture density `p(z)`.  The `N` quadrature points `z_{N, n}`
#' and weights `w_{N, n}` (which are non-negative and sum to 1) are chosen such that
#' `q_N(x) := sum_{n=1}^N w_{n, N} p(x | z_{N, n}) --> p(x)` as `N --> infinity`.
#'
#' Since `q_N(x)` is in fact a mixture (of `N` points), we may sample from
#' `q_N` exactly.  It is important to note that the VDM is *defined* as `q_N`
#' above, and *not* `p(x)`.  Therefore, sampling and pdf may be implemented as
#' exact (up to floating point error) methods.
#'
#' A common choice for the conditional `p(x | z)` is a multivariate Normal.
#' The implemented marginal `p(z)` is the `SoftmaxNormal`, which is a
#' `K-1` dimensional Normal transformed by a `SoftmaxCentered` bijector, making
#' it a density on the `K`-simplex.  That is,
#' `Z = SoftmaxCentered(X)`, `X = Normal(mix_loc / temperature, 1 / temperature)`
#'
#' The default quadrature scheme chooses `z_{N, n}` as `N` midpoints of
#' the quantiles of `p(z)` (generalized quantiles if `K > 2`).
#' See [Dillon and Langmore (2018)][1] for more details.
#'
#' About `Vector` distributions in TensorFlow.
#'
#' The `VectorDiffeomixture` is a non-standard distribution that has properties
#' particularly useful in [variational Bayesian methods](https://en.wikipedia.org/wiki/Variational_Bayesian_methods).
#' Conditioned on a draw from the SoftmaxNormal, `X|z` is a vector whose
#' components are linear combinations of affine transformations, thus is itself
#' an affine transformation.
#'
#' Note: The marginals `X_1|v, ..., X_d|v` are *not* generally identical to some
#' parameterization of `distribution`.  This is due to the fact that the sum of
#' draws from `distribution` are not generally itself the same `distribution`.
#'
#' About `Diffeomixture`s and reparameterization.
#'
#' The `VectorDiffeomixture` is designed to be reparameterized, i.e., its
#' parameters are only used to transform samples from a distribution which has no
#' trainable parameters. This property is important because backprop stops at
#' sources of stochasticity. That is, as long as the parameters are used *after*
#' the underlying source of stochasticity, the computed gradient is accurate.
#' Reparametrization means that we can use gradient-descent (via backprop) to
#' optimize Monte-Carlo objectives. Such objectives are a finite-sample
#' approximation of an expectation and arise throughout scientific computing.
#'
#' WARNING: If you backprop through a VectorDiffeomixture sample and the "base"
#' distribution is both: not `FULLY_REPARAMETERIZED` and a function of trainable
#' variables, then the gradient is not guaranteed correct!

#' References
#' [1]: Joshua Dillon and Ian Langmore.
#' Quadrature Compound: An approximating family of distributions.
#' _arXiv preprint arXiv:1801.03080_, 2018. https://arxiv.org/abs/1801.03080

#' @param mix_loc: `float`-like `Tensor` with shape `[b1, ..., bB, K-1]`.
#' In terms of samples, larger `mix_loc[..., k]` ==>
#'   `Z` is more likely to put more weight on its `kth` component.
#' @param temperature `float`-like `Tensor`. Broadcastable with `mix_loc`.
#' In terms of samples, smaller `temperature` means one component is more
#' likely to dominate.  I.e., smaller `temperature` makes the VDM look more
#' like a standard mixture of `K` components.
#' @param distribution `tfp$distributions$Distribution`-like instance. Distribution
#' from which `d` iid samples are used as input to the selected affine
#' transformation. Must be a scalar-batch, scalar-event distribution.
#' Typically `distribution$reparameterization_type = FULLY_REPARAMETERIZED`
#' or it is a function of non-trainable parameters. WARNING: If you
#' backprop through a VectorDiffeomixture sample and the `distribution`
#' is not `FULLY_REPARAMETERIZED` yet is a function of trainable variables,
#' then the gradient will be incorrect!
#' @param loc Length-`K` list of `float`-type `Tensor`s. The `k`-th element
#' represents the `shift` used for the `k`-th affine transformation.  If
#' the `k`-th item is `NULL`, `loc` is implicitly `0`.  When specified,
#' must have shape `[B1, ..., Bb, d]` where `b >= 0` and `d` is the event
#' size.
#' @param scale: Length-`K` list of `LinearOperator`s. Each should be
#' positive-definite and operate on a `d`-dimensional vector space. The
#' `k`-th element represents the `scale` used for the `k`-th affine
#' transformation. `LinearOperator`s must have shape `[B1, ..., Bb, d, d]`,
#' `b >= 0`, i.e., characterizes `b`-batches of `d x d` matrices
#' @param quadrature_size: `integer` scalar representing number of
#' quadrature points.  Larger `quadrature_size` means `q_N(x)` better
#' approximates `p(x)`.
#' @param quadrature_fn: Function taking `normal_loc`, `normal_scale`,
#' `quadrature_size`, `validate_args` and returning `tuple(grid, probs)`
#' representing the SoftmaxNormal grid and corresponding normalized weight.
#' normalized) weight.
#' Default value: `quadrature_scheme_softmaxnormal_quantiles`.
#' @inheritParams tfd_normal
#'
#' @family distributions
#' @export
tfd_vector_diffeomixture <- function(mix_loc,
                                     temperature,
                                     distribution,
                                     loc = NULL,
                                     scale = NULL,
                                     quadrature_size = 8,
                                     quadrature_fn = tfp$distributions$quadrature_scheme_softmaxnormal_quantiles,
                                     validate_args = FALSE,
                                     allow_nan_stats = TRUE,
                                     name = "VectorDiffeomixture") {
  args <- list(
    mix_loc = mix_loc,
    temperature = temperature,
    distribution = distribution,
    loc = loc,
    scale = scale,
    quadrature_size = as.integer(quadrature_size),
    quadrature_fn = quadrature_fn,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$VectorDiffeomixture, args)
}

#' Posterior predictive of a variational Gaussian process.
#'
#' This distribution implements the variational Gaussian process (VGP), as
#' described in [Titsias, 2009][1] and [Hensman, 2013][2]. The VGP is an
#' inducing point-based approximation of an exact GP posterior.
#' Ultimately, this Distribution class represents a marginal distribution over function values at a
#' collection of `index_points`. It is parameterized by
#' - a kernel function,
#' - a mean function,
#' - the (scalar) observation noise variance of the normal likelihood,
#' - a set of index points,
#' - a set of inducing index points, and
#' - the parameters of the (full-rank, Gaussian) variational posterior
#' distribution over function values at the inducing points, conditional on some observations.
#'
#' A VGP is "trained" by selecting any kernel parameters, the locations of the
#' inducing index points, and the variational parameters. [Titsias, 2009][1] and
#' [Hensman, 2013][2] describe a variational lower bound on the marginal log
#' likelihood of observed data, which this class offers through the
#' `variational_loss` method (this is the negative lower bound, for convenience
#' when plugging into a TF Optimizer's `minimize` function).
#' Training may be done in minibatches.
#'
#' [Titsias, 2009][1] describes a closed form for the optimal variational
#' parameters, in the case of sufficiently small observational data (ie,
#' small enough to fit in memory but big enough to warrant approximating the GP
#' posterior). A method to compute these optimal parameters in terms of the full
#' observational data set is provided as a staticmethod,
#' `optimal_variational_posterior`. It returns a
#' `MultivariateNormalLinearOperator` instance with optimal location and scale parameters.
#'
#' # References
#' [1]: Titsias, M. "Variational Model Selection for Sparse Gaussian Process Regression", 2009.
#' http://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf
#' [2]: Hensman, J., Lawrence, N. "Gaussian Processes for Big Data", 2013. https://arxiv.org/abs/1309.6835
#' [3]: Carl Rasmussen, Chris Williams. Gaussian Processes For Machine Learning, 2006. http://www.gaussianprocess.org/gpml/

#' @param  kernel `PositiveSemidefiniteKernel`-like instance representing the
#' GP's covariance function.
#' @param index_points `float` `Tensor` representing finite (batch of) vector(s) of
#' points in the index set over which the VGP is defined. Shape has the
#' form `[b1, ..., bB, e1, f1, ..., fF]` where `F` is the number of feature
#' dimensions and must equal `kernel$feature_ndims` and `e1` is the number
#' (size) of index points in each batch (we denote it `e1` to distinguish
#' it from the numer of inducing index points, denoted `e2` below).
#' Ultimately the VariationalGaussianProcess distribution corresponds to an
#' `e1`-dimensional multivariate normal. The batch shape must be
#' broadcastable with `kernel$batch_shape`, the batch shape of
#' `inducing_index_points`, and any batch dims yielded by `mean_fn`.
#' @param inducing_index_points `float` `Tensor` of locations of inducing points in
#' the index set. Shape has the form `[b1, ..., bB, e2, f1, ..., fF]`, just
#' like `index_points`. The batch shape components needn't be identical to
#' those of `index_points`, but must be broadcast compatible with them.
#' @param variational_inducing_observations_loc `float` `Tensor`; the mean of the
#' (full-rank Gaussian) variational posterior over function values at the
#' inducing points, conditional on observed data. Shape has the form `[b1, ..., bB, e2]`,
#' where `b1, ..., bB` is broadcast compatible with other
#' parameters' batch shapes, and `e2` is the number of inducing points.
#' @param variational_inducing_observations_scale `float` `Tensor`; the scale
#' matrix of the (full-rank Gaussian) variational posterior over function
#' values at the inducing points, conditional on observed data. Shape has
#' the form `[b1, ..., bB, e2, e2]`, where `b1, ..., bB` is broadcast
#' compatible with other parameters and `e2` is the number of inducing points.
#' @param mean_fn function that acts on index points to produce a (batch
#' of) vector(s) of mean values at those index points. Takes a `Tensor` of
#' shape `[b1, ..., bB, f1, ..., fF]` and returns a `Tensor` whose shape is
#' (broadcastable with) `[b1, ..., bB]`. Default value: `NULL` implies constant zero function.
#' @param observation_noise_variance `float` `Tensor` representing the variance
#' of the noise in the Normal likelihood distribution of the model. May be
#' batched, in which case the batch shape must be broadcastable with the
#' shapes of all other batched parameters (`kernel$batch_shape`, `index_points`, etc.).
#' Default value: `0.`
#' @param predictive_noise_variance `float` `Tensor` representing additional
#' variance in the posterior predictive model. If `NULL`, we simply re-use
#' `observation_noise_variance` for the posterior predictive noise. If set
#' explicitly, however, we use the given value. This allows us, for
#' example, to omit predictive noise variance (by setting this to zero) to
#' obtain noiseless posterior predictions of function values, conditioned
#' on noisy observations.
#' @param jitter `float` scalar `Tensor` added to the diagonal of the covariance
#' matrix to ensure positive definiteness of the covariance matrix. Default value: `1e-6`.

#' @inheritParams tfd_normal
#' @family distributions
#' @export
tfd_variational_gaussian_process <- function(kernel,
                                             index_points,
                                             inducing_index_points,
                                             variational_inducing_observations_loc,
                                             variational_inducing_observations_scale,
                                             mean_fn = NULL,
                                             observation_noise_variance = 0,
                                             predictive_noise_variance = 0,
                                             jitter = 1e-6,
                                             validate_args = FALSE,
                                             allow_nan_stats = FALSE,
                                             name = "VariationalGaussianProcess") {
  args <- list(
    kernel = kernel,
    index_points = index_points,
    inducing_index_points = inducing_index_points,
    variational_inducing_observations_loc = variational_inducing_observations_loc,
    variational_inducing_observations_scale = variational_inducing_observations_scale,
    mean_fn = mean_fn,
    observation_noise_variance = observation_noise_variance,
    predictive_noise_variance = predictive_noise_variance,
    jitter = jitter,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$VariationalGaussianProcess, args)
}

#' Uniform distribution with `low` and `high` parameters.
#'
#' The parameters `low` and `high` must be shaped in a way that supports
#' broadcasting (e.g., `high - low` is a valid operation).
#'
#' @param low Floating point tensor, lower boundary of the output interval. Must
#' have `low < high`.
#' @param high Floating point tensor, upper boundary of the output interval. Must
#' have `low < high`.
#' @inheritParams tfd_normal
#'
#' @family distributions
#' @export
tfd_uniform <- function(low = 0,
                        high = 1,
                        validate_args = FALSE,
                        allow_nan_stats = TRUE,
                        name = "Uniform") {
  args <- list(
    low = low,
    high = high,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Uniform, args)
}

#' The Truncated Normal distribution.
#'
#' The truncated normal is a normal distribution bounded between `low`
#' and `high` (the pdf is 0 outside these bounds and renormalized).
#' Samples from this distribution are differentiable with respect to `loc`,
#' `scale` as well as the bounds, `low` and `high`, i.e., this
#' implementation is fully reparameterizeable.
#' For more details, see [here](https://en.wikipedia.org/wiki/Truncated_normal_distribution).
#'
#' This is a scalar distribution so the event shape is always scalar and the
#' dimensions of the parameters defined the batch_shape.
#'
#' @param  low `float` `Tensor` representing lower bound of the distribution's
#' support. Must be such that `low < high`.
#' @param high `float` `Tensor` representing upper bound of the distribution's
#' support. Must be such that `low < high`.
#' @inheritParams tfd_normal
#' @family distributions
#' @export
tfd_truncated_normal <- function(loc,
                                 scale,
                                 low,
                                 high,
                                 validate_args = FALSE,
                                 allow_nan_stats = TRUE,
                                 name = "TruncatedNormal") {
  args <- list(
    loc = loc,
    scale = scale,
    low = low,
    high = high,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$TruncatedNormal, args)
}

#' Triangular distribution with `low`, `high` and `peak` parameters.
#'
#' The parameters `low`, `high` and `peak` must be shaped in a way that supports
#' broadcasting (e.g., `high - low` is a valid operation).
#'
#' @param low Floating point tensor, lower boundary of the output interval. Must
#' have `low < high`. Default value: `0`.
#' @param high Floating point tensor, upper boundary of the output interval. Must
#' have `low < high`. Default value: `1`.
#' @param peak Floating point tensor, mode of the output interval. Must have
#' `low <= peak` and `peak <= high`. Default value: `0.5`.
#'
#' @inheritParams tfd_normal
#' @family distributions
#' @export
tfd_triangular <- function(low = 0,
                           high = 1,
                           peak = 0.5,
                           validate_args = FALSE,
                           allow_nan_stats = TRUE,
                           name = "Triangular") {
  args <- list(
    low = low,
    high = high,
    peak = peak,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Triangular, args)
}

#' Student's t-distribution.
#'
#' This distribution has parameters: degree of freedom `df`, location `loc`, and `scale`.
#' Notice that `scale` has semantics more similar to standard deviation than
#' variance. However it is not actually the std. deviation; the Student's
#' t-distribution std. dev. is `scale sqrt(df / (df - 2))` when `df > 2`.
#'
#' Samples of this distribution are reparameterized (pathwise differentiable).
#' The derivatives are computed using the approach described in the paper
#' [Michael Figurnov, Shakir Mohamed, Andriy Mnih. Implicit Reparameterization Gradients, 2018](https://arxiv.org/abs/1805.08498)
#'
#' @param df Floating-point `Tensor`. The degrees of freedom of the
#' distribution(s). `df` must contain only positive values.
#' @param loc Floating-point `Tensor`. The mean(s) of the distribution(s).
#' @param scale Floating-point `Tensor`. The scaling factor(s) for the
#' distribution(s). Note that `scale` is not technically the standard
#' deviation of this distribution but has semantics more similar to
#' standard deviation than variance.
#' @inheritParams tfd_normal
#' @family distributions
#' @export
tfd_student_t <- function(df,
                          loc,
                          scale,
                          validate_args = FALSE,
                          allow_nan_stats = TRUE,
                          name = "StudentT") {
  args <- list(
    df = df,
    loc = loc,
    scale = scale,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$StudentT, args)
}


