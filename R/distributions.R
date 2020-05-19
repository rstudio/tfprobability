#' Normal distribution with loc and scale parameters
#'
#' Mathematical details
#'
#' The probability density function (pdf) is,
#' ```
#' pdf(x; mu, sigma) = exp(-0.5 (x - mu)**2 / sigma**2) / Z
#' Z = (2 pi sigma**2)**0.5
#' ```
#' where `loc = mu` is the mean, `scale = sigma` is the std. deviation, and, `Z`
#' is the normalization constant.
#' The Normal distribution is a member of the [location-scale family](
#'  https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
#'  constructed as,
#'  ```
#'  X ~ Normal(loc=0, scale=1)
#'  Y = loc + scale * X
#'  ```
#'
#' @param loc Floating point tensor; the means of the distribution(s).
#' @param scale loating point tensor; the stddevs of the distribution(s).
#'  Must contain only positive values.
#' @param validate_args Logical, default FALSE. When TRUE distribution parameters are checked
#'  for validity despite possibly degrading runtime performance. When FALSE invalid inputs may
#'  silently render incorrect outputs. Default value: FALSE.
#' @param allow_nan_stats Logical, default TRUE. When TRUE, statistics (e.g., mean, mode, variance)
#'  use the value NaN to indicate the result is undefined. When FALSE, an exception is raised if
#'  one or more of the statistic's batch members are undefined.
#' @param name name prefixed to Ops created by this class.
#' @return a distribution instance.
#'
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
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
#' This distribution is useful for regarding a collection of independent,
#' non-identical distributions as a single random variable. For example, the
#' `Independent` distribution composed of a collection of `Bernoulli`
#' distributions might define a distribution over an image (where each
#' `Bernoulli` is a distribution over each pixel).
#'
#' More precisely, a collection of `B` (independent) `E`-variate random variables
#' (rv) `{X_1, ..., X_B}`, can be regarded as a `[B, E]`-variate random variable
#' `(X_1, ..., X_B)` with probability
#' `p(x_1, ..., x_B) = p_1(x_1) * ... * p_B(x_B)` where `p_b(X_b)` is the
#' probability of the `b`-th rv. More generally `B, E` can be arbitrary shapes.
#' Similarly, the `Independent` distribution specifies a distribution over
#' `[B, E]`-shaped events. It operates by reinterpreting the rightmost batch dims as
#' part of the event dimensions. The `reinterpreted_batch_ndims` parameter
#' controls the number of batch dims which are absorbed as event dims;
#' `reinterpreted_batch_ndims <= len(batch_shape)`.  For example, the `log_prob`
#' function entails a `reduce_sum` over the rightmost `reinterpreted_batch_ndims`
#' after calling the base distribution's `log_prob`.  In other words, since the
#'  batch dimension(s) index independent distributions, the resultant multivariate
#'  will have independent components.
#'
#'  Mathematical Details
#'
#'  The probability function is,
#'  ```
#'  prob(x; reinterpreted_batch_ndims) =
#'   tf.reduce_prod(dist.prob(x), axis=-1-range(reinterpreted_batch_ndims))
#' ```
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
#' @inherit tfd_normal return
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_independent <- function(distribution,
                            reinterpreted_batch_ndims = NULL,
                            validate_args = FALSE,
                            name = paste0("Independent", distribution$name)) {
  args <- list(
    distribution = distribution,
    reinterpreted_batch_ndims = as_nullable_integer(reinterpreted_batch_ndims),
    validate_args = validate_args,
    name = name
  )

  do.call(tfp$distributions$Independent, args)
}



#' Bernoulli distribution
#'
#' The Bernoulli distribution with `probs` parameter, i.e., the probability of a
#' `1` outcome (vs a `0` outcome).
#'
#' @inherit tfd_normal return params
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
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
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

#' Multivariate normal distribution on `R^k`
#'
#' The Multivariate Normal distribution is defined over `R^k`` and parameterized
#' by a (batch of) length-k loc vector (aka "mu") and a (batch of) `k x k`
#' scale matrix; `covariance = scale @ scale.T` where `@` denotes
#' matrix-multiplication.
#'
#' Mathematical Details
#'
#' The probability density function (pdf) is,
#' ```
#' pdf(x; loc, scale) = exp(-0.5 ||y||**2) / Z
#' y = inv(scale) @ (x - loc)
#' Z = (2 pi)**(0.5 k) |det(scale)|
#' ```
#' where:
#' * `loc` is a vector in `R^k`,
#' * `scale` is a linear operator in `R^{k x k}`, `cov = scale @ scale.T`,
#' * `Z` denotes the normalization constant, and,
#' * `||y||**2` denotes the squared Euclidean norm of `y`.
#'
#' A (non-batch) `scale` matrix is:
#' ```
#' scale = diag(scale_diag + scale_identity_multiplier * ones(k))
#' ```
#' where:
#' * `scale_diag.shape = [k]`, and,
#' * `scale_identity_multiplier.shape = []`.#'
#'
#' Additional leading dimensions (if any) will index batches.
#'
#' If both `scale_diag` and `scale_identity_multiplier` are `NULL`, then
#' `scale` is the Identity matrix.
#' The MultivariateNormal distribution is a member of the
#' [location-scale family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
#' constructed as,
#' ```
#' X ~ MultivariateNormal(loc=0, scale=1)   # Identity scale, zero shift.
#' Y = scale @ X + loc
#' ```
#'
#' @inherit tfd_normal return params
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
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
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

#' OneHotCategorical distribution
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
#' @inherit tfd_normal return params
#' @param logits An N-D Tensor, N >= 1, representing the log probabilities of a set of Categorical distributions.
#'  The first N - 1 dimensions index into a batch of independent distributions and the last dimension represents
#'  a vector of logits for each class. Only one of logits or probs should be passed in.
#' @param probs An N-D Tensor, N >= 1, representing the probabilities of a set of Categorical distributions. The
#'   first N - 1 dimensions index into a batch of independent distributions and the last dimension represents a
#'   vector of probabilities for each class. Only one of logits or probs should be passed in.
#' @param dtype The type of the event samples (default: int32).
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
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

#' RelaxedOneHotCategorical distribution with temperature and logits
#'
#' The RelaxedOneHotCategorical is a distribution over random probability
#' vectors, vectors of positive real values that sum to one, which continuously
#' approximates a OneHotCategorical. The degree of approximation is controlled by
#' a temperature: as the temperature goes to 0 the RelaxedOneHotCategorical
#' becomes discrete with a distribution described by the `logits` or `probs`
#' parameters, as the temperature goes to infinity the RelaxedOneHotCategorical
#' becomes the constant distribution that is identically the constant vector of
#' (1/event_size, ..., 1/event_size).
#' The RelaxedOneHotCategorical distribution was concurrently introduced as the
#' Gumbel-Softmax (Jang et al., 2016) and Concrete (Maddison et al., 2016)
#' distributions for use as a reparameterized continuous approximation to the
#' `Categorical` one-hot distribution. If you use this distribution, please cite
#' both papers.
#'
#' @section References:
#' - Eric Jang, Shixiang Gu, and Ben Poole. Categorical Reparameterization with
#' Gumbel-Softmax. 2016.
#' - Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. The Concrete Distribution:
#'  A Continuous Relaxation of Discrete Random Variables. 2016.
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
#' @inherit tfd_normal return params
#'
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
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

#' RelaxedBernoulli distribution with temperature and logits parameters
#'
#' The RelaxedBernoulli is a distribution over the unit interval (0,1), which continuously approximates a Bernoulli.
#' The degree of approximation is controlled by a temperature: as the temperature goes to 0 the RelaxedBernoulli
#' becomes discrete with a distribution described by the logits or probs parameters, as the temperature goes to
#' infinity the RelaxedBernoulli becomes the constant distribution that is identically 0.5.
#'
#' The RelaxedBernoulli distribution is a reparameterized continuous
#' distribution that is the binary special case of the RelaxedOneHotCategorical
#' distribution (Maddison et al., 2016; Jang et al., 2016). For details on the
#' binary special case see the appendix of Maddison et al. (2016) where it is
#' referred to as BinConcrete. If you use this distribution, please cite both papers.
#'
#' Some care needs to be taken for loss functions that depend on the
#' log-probability of RelaxedBernoullis, because computing log-probabilities of
#' the RelaxedBernoulli can suffer from underflow issues. In many case loss
#' functions such as these are invariant under invertible transformations of
#' the random variables. The KL divergence, found in the variational autoencoder
#' loss, is an example. Because RelaxedBernoullis are sampled by a Logistic
#' random variable followed by a `tf$sigmoid` op, one solution is to treat
#' the Logistic as the random variable and `tf$sigmoid` as downstream. The
#' KL divergences of two Logistics, which are always followed by a `tf.sigmoid`
#' op, is equivalent to evaluating KL divergences of RelaxedBernoulli samples.
#' See Maddison et al., 2016 for more details where this distribution is called
#' the BinConcrete.
#' An alternative approach is to evaluate Bernoulli log probability or KL
#' directly on relaxed samples, as done in Jang et al., 2016. In this case,
#' guarantees on the loss are usually violated. For instance, using a Bernoulli
#' KL in a relaxed ELBO is no longer a lower bound on the log marginal
#' probability of the observation. Thus care and early stopping are important.
#'
#' @param temperature An 0-D Tensor, representing the temperature of a set of RelaxedBernoulli distributions.
#'   The temperature should be positive.
#' @param logits  An N-D Tensor representing the log-odds of a positive event. Each entry in the Tensor
#'   parametrizes an independent RelaxedBernoulli distribution where the probability of an event is sigmoid(logits).
#'   Only one of logits or probs should be passed in.
#' @param probs AAn N-D Tensor representing the probability of a positive event. Each entry in the Tensor parameterizes
#'  an independent Bernoulli distribution. Only one of logits or probs should be passed in.
#'
#' @inherit tfd_normal return params
#'
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
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

#' A Transformed Distribution
#'
#' A TransformedDistribution models `p(y)` given a base distribution `p(x)`,
#' and a deterministic, invertible, differentiable transform,`Y = g(X)`. The
#' transform is typically an instance of the Bijector class and the base
#' distribution is typically an instance of the Distribution class.
#'
#' A `Bijector` is expected to implement the following functions:
#' - `forward`,
#' - `inverse`,
#' - `inverse_log_det_jacobian`.
#'
#' The semantics of these functions are outlined in the `Bijector` documentation.
#'
#' We now describe how a `TransformedDistribution` alters the input/outputs of a
#' `Distribution` associated with a random variable (rv) `X`.
#' Write `cdf(Y=y)` for an absolutely continuous cumulative distribution function
#' of random variable `Y`; write the probability density function
#' `pdf(Y=y) := d^k / (dy_1,...,dy_k) cdf(Y=y)` for its derivative wrt to `Y` evaluated at
#' `y`. Assume that `Y = g(X)` where `g` is a deterministic diffeomorphism,
#' i.e., a non-random, continuous, differentiable, and invertible function.
#' Write the inverse of `g` as `X = g^{-1}(Y)` and `(J o g)(x)` for the Jacobian
#' of `g` evaluated at `x`.
#'
#' A `TransformedDistribution` implements the following operations:
#' * `sample`
#' Mathematically:   `Y = g(X)`
#' Programmatically: `bijector.forward(distribution.sample(...))`
#' * `log_prob`
#' Mathematically:   `(log o pdf)(Y=y) = (log o pdf o g^{-1})(y) + (log o abs o det o J o g^{-1})(y)`
#' Programmatically: `(distribution.log_prob(bijector.inverse(y)) + bijector.inverse_log_det_jacobian(y))`
#' * `log_cdf`
#' Mathematically:   `(log o cdf)(Y=y) = (log o cdf o g^{-1})(y)`
#' Programmatically: `distribution.log_cdf(bijector.inverse(x))`
#' * and similarly for: `cdf`, `prob`, `log_survival_function`, `survival_function`.
#'
#' @param distribution The base distribution instance to transform. Typically an instance of Distribution.
#' @param bijector The object responsible for calculating the transformation. Typically an instance of Bijector.
#' @param batch_shape integer vector Tensor which overrides distribution batch_shape;
#' valid only if distribution.is_scalar_batch().
#' @param event_shape integer vector Tensor which overrides distribution event_shape;
#' valid only if distribution.is_scalar_event().
#' @param kwargs_split_fn Python `callable` which takes a kwargs `dict` and returns
#' a tuple of kwargs `dict`s for each of the `distribution` and `bijector`
#' parameters respectively. Default value: `_default_kwargs_split_fn` (i.e.,
#' `lambda kwargs: (kwargs.get('distribution_kwargs', {}), kwargs.get('bijector_kwargs', {}))`)
#' @param validate_args Logical, default FALSE. When TRUE distribution parameters are checked
#'  for validity despite possibly degrading runtime performance. When FALSE invalid inputs may
#'  silently render incorrect outputs. Default value: FALSE.
#' @param parameters Locals dict captured by subclass constructor, to be used for
#' copy/slice re-instantiation operations.
#' @param name The name for ops managed by the distribution.  Default value: bijector.name + distribution.name.
#' @family distributions
#' @inherit tfd_normal return
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_transformed_distribution <- function(distribution,
                                         bijector,
                                         batch_shape = NULL,
                                         event_shape = NULL,
                                         kwargs_split_fn = NULL,
                                         validate_args = FALSE,
                                         parameters = NULL,
                                         name = NULL) {
  kwargs_split_fn <-
    if (is.null(kwargs_split_fn))
      tfp$distributions$transformed_distribution$`_default_kwargs_split_fn`
  else
    kwargs_split_fn

  args <- list(
    distribution = distribution,
    bijector = bijector,
    batch_shape = normalize_shape(batch_shape),
    event_shape = normalize_shape(event_shape),
    kwargs_split_fn = kwargs_split_fn,
    validate_args = validate_args,
    parameters = parameters,
    name = name
  )

  do.call(tfp$distributions$TransformedDistribution, args)
}

#' Zipf distribution
#'
#' The Zipf distribution is parameterized by a power parameter.
#'
#' Mathematical Details
#' The probability mass function (pmf) is,
#' ```
#' pmf(k; alpha, k >= 0) = (k^(-alpha)) / Z
#' Z = zeta(alpha).
#' ```
#' where `power = alpha` and Z is the normalization constant.
#' `zeta` is the [Riemann zeta function](
#' https://en.wikipedia.org/wiki/Riemann_zeta_function).
#' Note that gradients with respect to the `power` parameter are not
#' supported in the current implementation.
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
#' @inherit tfd_normal return params
#'
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
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

#' The matrix Wishart distribution on positive definite matrices
#'
#' This distribution is defined by a scalar number of degrees of freedom df and
#' an instance of LinearOperator, which provides matrix-free access to a
#' symmetric positive definite operator, which defines the scale matrix.
#'
#' Mathematical Details
#'
#' The probability density function (pdf) is,
#' ```
#' pdf(X; df, scale) = det(X)**(0.5 (df-k-1)) exp(-0.5 tr[inv(scale) X]) / Z
#' Z = 2**(0.5 df k) |det(scale)|**(0.5 df) Gamma_k(0.5 df)
#' ```
#' where:
#' * `df >= k` denotes the degrees of freedom,
#' * `scale` is a symmetric, positive definite, `k x k` matrix,
#' * `Z` is the normalizing constant, and,
#' * `Gamma_k` is the [multivariate Gamma function](
#'  https://en.wikipedia.org/wiki/Multivariate_gamma_function).
#'
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
#' @inherit tfd_normal return params
#'
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
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

#' The von Mises-Fisher distribution over unit vectors on `S^{n-1}`
#'
#' The von Mises-Fisher distribution is a directional distribution over vectors
#' on the unit hypersphere `S^{n-1}` embedded in n dimensions `(R^n)`.
#'
#' Mathematical details
#' The probability density function (pdf) is,
#' ```
#' pdf(x; mu, kappa) = C(kappa) exp(kappa * mu^T x)
#' where,
#' C(kappa) = (2 pi)^{-n/2} kappa^{n/2-1} / I_{n/2-1}(kappa),
#' I_v(z) being the modified Bessel function of the first kind of order v
#' ```
#' where:
#' * `mean_direction = mu`; a unit vector in `R^k`,
#' * `concentration = kappa`; scalar real >= 0, concentration of samples around
#' `mean_direction`, where 0 pertains to the uniform distribution on the
#' hypersphere, and `inf` indicates a delta function at `mean_direction`.
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
#' @inherit tfd_normal return params
#'
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_von_mises_fisher <- function(mean_direction,
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

#' The von Mises distribution over angles
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
#' Mathematical details
#' The probability density function (pdf) of this distribution is,
#' ```
#' pdf(x; loc, concentration) = exp(concentration cos(x - loc)) / Z
#' Z = 2 * pi * I_0 (concentration)
#' ```
#' where:
#' * `I_0 (concentration)` is the modified Bessel function of order zero;
#' * `loc` the circular mean of the distribution, a scalar. It can take arbitrary
#' values, but it is 2pi-periodic: loc and loc + 2pi result in the same
#' distribution.
#' * `concentration >= 0` parameter is the concentration parameter. When
#' `concentration = 0`,
#' this distribution becomes a Uniform distribution on [-pi, pi).
#'
#' The parameters loc and concentration must be shaped in a way that
#' supports broadcasting (e.g. loc + concentration is a valid operation).
#' @param loc Floating point tensor, the circular means of the distribution(s).
#' @param concentration Floating point tensor, the level of concentration of the
#' distribution(s) around loc. Must take non-negative values.
#' concentration = 0 defines a Uniform distribution, while
#' concentration = +inf indicates a Deterministic distribution at loc.
#' @inherit tfd_normal return params
#'
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
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

#' The (diagonal) SinhArcsinh transformation of a distribution on `R^k`
#'
#' This distribution models a random vector `Y = (Y1,...,Yk)`, making use of
#' a SinhArcsinh transformation (which has adjustable tailweight and skew),
#' a rescaling, and a shift.
#' The SinhArcsinh transformation of the Normal is described in great depth in
#' [Sinh-arcsinh distributions](https://oro.open.ac.uk/22510/).
#' Here we use a slightly different parameterization, in terms of tailweight
#' and skewness.  Additionally we allow for distributions other than Normal,
#' and control over scale as well as a "shift" parameter loc.
#'
#' Mathematical Details
#'
#' Given iid random vector `Z = (Z1,...,Zk)`, we define the VectorSinhArcsinhDiag
#' transformation of `Z`, `Y`, parameterized by
#' `(loc, scale, skewness, tailweight)`, via the relation (with `@` denoting matrix multiplication):
#' ```
#' Y := loc + scale @ F(Z) * (2 / F_0(2))
#' F(Z) := Sinh( (Arcsinh(Z) + skewness) * tailweight )
#' F_0(Z) := Sinh( Arcsinh(Z) * tailweight )
#' ```
#'
#' This distribution is similar to the location-scale transformation
#' `L(Z) := loc + scale @ Z` in the following ways:
#' * If `skewness = 0` and `tailweight = 1` (the defaults), `F(Z) = Z`, and then
#' `Y = L(Z)` exactly.
#' * `loc` is used in both to shift the result by a constant factor.
#' * The multiplication of `scale` by `2 / F_0(2)` ensures that if `skewness = 0`
#' `P[Y - loc <= 2 * scale] = P[L(Z) - loc <= 2 * scale]`.
#' Thus it can be said that the weights in the tails of `Y` and `L(Z)` beyond
#' `loc + 2 * scale` are the same.
#' This distribution is different than `loc + scale @ Z` due to the
#' reshaping done by `F`:
#'   * Positive (negative) `skewness` leads to positive (negative) skew.
#'   * positive skew means, the mode of `F(Z)` is "tilted" to the right.
#'   * positive skew means positive values of `F(Z)` become more likely, and
#'   negative values become less likely.
#'   * Larger (smaller) `tailweight` leads to fatter (thinner) tails.
#'   * Fatter tails mean larger values of `|F(Z)|` become more likely.
#'   * `tailweight < 1` leads to a distribution that is "flat" around `Y = loc`,
#'   and a very steep drop-off in the tails.
#'   * `tailweight > 1` leads to a distribution more peaked at the mode with
#'   heavier tails.
#'   To see the argument about the tails, note that for `|Z| >> 1` and
#'   `|Z| >> (|skewness| * tailweight)**tailweight`, we have
#'   `Y approx 0.5 Z**tailweight e**(sign(Z) skewness * tailweight)`.
#'   To see the argument regarding multiplying `scale` by `2 / F_0(2)`,
#'   ```
#'   P[(Y - loc) / scale <= 2] = P[F(Z) * (2 / F_0(2)) <= 2]
#'   = P[F(Z) <= F_0(2)]
#'   = P[Z <= 2]  (if F = F_0).
#'   ```
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
#' @inherit tfd_normal return params
#'
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
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

#' The vectorization of the Laplace distribution on `R^k`
#'
#' The vector laplace distribution is defined over `R^k`, and parameterized by
#' a (batch of) length-k loc vector (the means) and a (batch of) k x k
#' scale matrix:  `covariance = 2 * scale @ scale.T`, where `@` denotes
#' matrix-multiplication.
#'
#' Mathematical Details
#' The probability density function (pdf) is,
#' ```
#' pdf(x; loc, scale) = exp(-||y||_1) / Z,
#' y = inv(scale) @ (x - loc),
#' Z = 2**k |det(scale)|,
#' ```
#' where:
#' * `loc` is a vector in `R^k`,
#' * `scale` is a linear operator in `R^{k x k}`, `cov = scale @ scale.T`,
#' * `Z` denotes the normalization constant, and,
#' * `||y||_1` denotes the `l1` norm of `y`, `sum_i |y_i|.
#'
#' The VectorLaplace distribution is a member of the [location-scale
#' family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
#' constructed as,
#' ```
#' X = (X_1, ..., X_k), each X_i ~ Laplace(loc=0, scale=1)
#' Y = (Y_1, ...,Y_k) = scale @ X + loc
#' ```
#'
#' About VectorLaplace and Vector distributions in TensorFlow
#'
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
#' @inherit tfd_normal return params
#'
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
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

  do.call(
    tfp$distributions$vector_laplace_linear_operator$VectorLaplaceLinearOperator,
    args
  )
}

#' The vectorization of the Laplace distribution on `R^k`
#'
#' The vector laplace distribution is defined over `R^k`, and parameterized by
#' a (batch of) length-k loc vector (the means) and a (batch of) k x k
#' scale matrix:  `covariance = 2 * scale @ scale.T`, where @ denotes
#' matrix-multiplication.
#'
#' Mathematical Details
#' The probability density function (pdf) is,
#' ```
#' pdf(x; loc, scale) = exp(-||y||_1) / Z
#' y = inv(scale) @ (x - loc)
#' Z = 2**k |det(scale)|
#' ```
#' where:
#' * `loc` is a vector in `R^k`,
#' * `scale` is a linear operator in `R^{k x k}`, `cov = scale @ scale.T`,
#' * `Z` denotes the normalization constant, and,
#' * `||y||_1` denotes the `l1` norm of `y`, `sum_i |y_i|.
#'
#' A (non-batch) `scale` matrix is:
#' ```
#' scale = diag(scale_diag + scale_identity_multiplier * ones(k))
#' ```
#' where:
#'  * `scale_diag.shape = [k]`, and,
#'  * `scale_identity_multiplier.shape = []`.
#'  Additional leading dimensions (if any) will index batches.
#'  If both `scale_diag` and `scale_identity_multiplier` are `NULL`, then
#'  `scale` is the Identity matrix.
#'
#' About VectorLaplace and Vector distributions in TensorFlow
#'
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
#' `[B1, ..., Bb]`, b >= 0, and characterizes b-batches of scaled
#' k x k identity matrices added to scale. When both
#' scale_identity_multiplier and scale_diag are NULL then scale is
#' the Identity.
#' @inherit tfd_normal return params
#'
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
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

#' The vectorization of the Exponential distribution on `R^k`
#'
#' The vector exponential distribution is defined over a subset of `R^k`, and
#' parameterized by a (batch of) length-`k` `loc` vector and a (batch of) `k x k`
#' `scale` matrix:  `covariance = scale @ scale.T`, where `@` denotes
#' matrix-multiplication.
#'
#' Mathematical Details
#' The probability density function (pdf) is defined over the image of the
#' `scale` matrix + `loc`, applied to the positive half-space:
#' `Supp = {loc + scale @ x : x in R^k, x_1 > 0, ..., x_k > 0}`.  On this set,
#' ```
#' pdf(y; loc, scale) = exp(-||x||_1) / Z,  for y in Supp
#' x = inv(scale) @ (y - loc),
#' Z = |det(scale)|,
#' ```
#' where:
#' * `loc` is a vector in `R^k`,
#' * `scale` is a linear operator in `R^{k x k}`, `cov = scale @ scale.T`,
#' * `Z` denotes the normalization constant, and,
#' * `||x||_1` denotes the `l1` norm of `x`, `sum_i |x_i|`.
#' The VectorExponential distribution is a member of the [location-scale
#' family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
#' constructed as,
#' ```
#' X = (X_1, ..., X_k), each X_i ~ Exponential(rate=1)
#' Y = (Y_1, ...,Y_k) = scale @ X + loc
#' ```
#' About `VectorExponential` and `Vector` distributions in TensorFlow.
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
#' `[B1, ..., Bb]`, b >= 0, and characterizes b-batches of scaled
#' k x k identity matrices added to scale. When both
#' scale_identity_multiplier and scale_diag are NULL then scale is
#' the Identity.
#' @inherit tfd_normal return params
#'
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
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

#' The vectorization of the Exponential distribution on `R^k`
#'
#' The vector exponential distribution is defined over a subset of `R^k`, and
#' parameterized by a (batch of) length-`k` `loc` vector and a (batch of) `k x k`
#' `scale` matrix:  `covariance = scale @ scale.T`, where `@` denotes
#' matrix-multiplication.
#'
#' Mathematical Details
#' The probability density function (pdf) is
#' ```
#' pdf(y; loc, scale) = exp(-||x||_1) / Z,  for y in S(loc, scale),
#' x = inv(scale) @ (y - loc),
#' Z = |det(scale)|,
#' ```
#' where:
#' * `loc` is a vector in `R^k`,
#' * `scale` is a linear operator in `R^{k x k}`, `cov = scale @ scale.T`,
#' * `S = {loc + scale @ x : x in R^k, x_1 > 0, ..., x_k > 0}`, is an image of
#' the positive half-space,
#' * `||x||_1` denotes the `l1` norm of `x`, `sum_i |x_i|`,
#' * `Z` denotes the normalization constant.
#'
#' The VectorExponential distribution is a member of the [location-scale
#' family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
#' constructed as,
#' ```
#' X = (X_1, ..., X_k), each X_i ~ Exponential(rate=1)
#' Y = (Y_1, ...,Y_k) = scale @ X + loc
#' ```
#' About `VectorExponential` and `Vector` distributions in TensorFlow.
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
#' @inherit tfd_normal return params
#'
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
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

  do.call(
    tfp$distributions$vector_exponential_linear_operator$VectorExponentialLinearOperator,
    args
  )
}


#' VectorDiffeomixture distribution
#'
#' A vector diffeomixture (VDM) is a distribution parameterized by a convex
#' combination of `K` component `loc` vectors, `loc[k], k = 0,...,K-1`, and `K`
#' `scale` matrices `scale[k], k = 0,..., K-1`.  It approximates the following
#' [compound distribution](https://en.wikipedia.org/wiki/Compound_probability_distribution)
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
#' See Dillon and Langmore (2018) for more details.
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
#'
#' @section References:
#' - [Joshua Dillon and Ian Langmore. Quadrature Compound: An approximating family of distributions.
#' _arXiv preprint arXiv:1801.03080_, 2018.](https://arxiv.org/abs/1801.03080)
#'
#' @param mix_loc `float`-like `Tensor` with shape `[b1, ..., bB, K-1]`.
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
#' @param scale Length-`K` list of `LinearOperator`s. Each should be
#' positive-definite and operate on a `d`-dimensional vector space. The
#' `k`-th element represents the `scale` used for the `k`-th affine
#' transformation. `LinearOperator`s must have shape `[B1, ..., Bb, d, d]`,
#' `b >= 0`, i.e., characterizes `b`-batches of `d x d` matrices
#' @param quadrature_size `integer` scalar representing number of
#' quadrature points.  Larger `quadrature_size` means `q_N(x)` better
#' approximates `p(x)`.
#' @param quadrature_fn Function taking `normal_loc`, `normal_scale`,
#' `quadrature_size`, `validate_args` and returning `tuple(grid, probs)`
#' representing the SoftmaxNormal grid and corresponding normalized weight.
#' normalized) weight.
#' Default value: `quadrature_scheme_softmaxnormal_quantiles`.
#' @inherit tfd_normal return params
#'
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
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

#' Posterior predictive of a variational Gaussian process
#'
#' This distribution implements the variational Gaussian process (VGP), as
#' described in Titsias (2009) and Hensman (2013). The VGP is an
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
#' inducing index points, and the variational parameters. Titsias (2009) and
#' Hensman (2013) describe a variational lower bound on the marginal log
#' likelihood of observed data, which this class offers through the
#' `variational_loss` method (this is the negative lower bound, for convenience
#' when plugging into a TF Optimizer's `minimize` function).
#' Training may be done in minibatches.
#'
#' Titsias (2009) describes a closed form for the optimal variational
#' parameters, in the case of sufficiently small observational data (ie,
#' small enough to fit in memory but big enough to warrant approximating the GP
#' posterior). A method to compute these optimal parameters in terms of the full
#' observational data set is provided as a staticmethod,
#' `optimal_variational_posterior`. It returns a
#' `MultivariateNormalLinearOperator` instance with optimal location and scale parameters.
#'
#'
#' Mathematical Details
#'
#' Notation
#' We will in general be concerned about three collections of index points, and
#' it'll be good to give them names:
#'  * `x[1], ..., x[N]`: observation index points -- locations of our observed data.
#'  * `z[1], ..., z[M]`: inducing index points  -- locations of the
#'        "summarizing" inducing points
#'  * `t[1], ..., t[P]`: predictive index points -- locations where we are
#'  making posterior predictions based on observations and the variational
#'  parameters.
#'
#'  To lighten notation, we'll use `X, Z, T` to denote the above collections.
#'  Similarly, we'll denote by `f(X)` the collection of function values at each of
#'  the `x[i]`, and by `Y`, the collection of (noisy) observed data at each `x[i]`.
#'  We'll denote kernel matrices generated from pairs of index points as `K_tt`,
#'  `K_xt`, `K_tz`, etc, e.g.,
#'
#'  ```
#'  K_tz =
#'  | k(t[1], z[1])    k(t[1], z[2])  ...  k(t[1], z[M]) |
#'  | k(t[2], z[1])    k(t[2], z[2])  ...  k(t[2], z[M]) |
#'  |      ...              ...                 ...      |
#'  | k(t[P], z[1])    k(t[P], z[2])  ...  k(t[P], z[M]) |
#'
#'  ```
#'
#'  Preliminaries
#'  A Gaussian process is an indexed collection of random variables, any finite
#'  collection of which are jointly Gaussian. Typically, the index set is some
#'  finite-dimensional, real vector space, and indeed we make this assumption in
#'  what follows. The GP may then be thought of as a distribution over functions
#'  on the index set. Samples from the GP are functions *on the whole index set*;
#'  these can't be represented in finite compute memory, so one typically works
#'  with the marginals at a finite collection of index points. The properties of
#'  the GP are entirely determined by its mean function `m` and covariance
#'  function `k`. The generative process, assuming a mean-zero normal likelihood
#'  with stddev `sigma`, is
#'
#'  ```
#'  f ~ GP(m, k)
#'  Y | f(X) ~ Normal(f(X), sigma),   i = 1, ... , N
#'  ```
#'
#'  In finite terms (ie, marginalizing out all but a finite number of f(X), sigma),
#'  we can write
#'  ```
#'  f(X) ~ MVN(loc=m(X), cov=K_xx)
#'  Y | f(X) ~ Normal(f(X), sigma),   i = 1, ... , N
#'  ```
#'
#'  Posterior inference is possible in analytical closed form but becomes
#'  intractible as data sizes get large. See Rasmussen (2006) for details.
#'
#'  The VGP
#'
#'  The VGP is an inducing point-based approximation of an exact GP posterior,
#'  where two approximating assumptions have been made:
#'  1. function values at non-inducing points are mutually independent
#'  conditioned on function values at the inducing points,
#'  2. the (expensive) posterior over function values at inducing points
#'  conditional on obseravtions is replaced with an arbitrary (learnable)
#'  full-rank Gaussian distribution,
#'
#'  ```
#'  q(f(Z)) = MVN(loc=m, scale=S),
#'  ```
#'  where `m` and `S` are parameters to be chosen by optimizing an evidence
#'  lower bound (ELBO).
#'  The posterior predictive distribution becomes
#'  ```
#'  q(f(T)) = integral df(Z) p(f(T) | f(Z)) q(f(Z)) = MVN(loc = A @ m, scale = B^(1/2))
#'  ```
#'  where
#'  ```
#'  A = K_tz @ K_zz^-1
#'  B = K_tt - A @ (K_zz - S S^T) A^T
#'  ```
#'
#'  The approximate posterior predictive distribution `q(f(T))` is what the
#'  `VariationalGaussianProcess` class represents.
#'
#'  Model selection in this framework entails choosing the kernel parameters,
#'  inducing point locations, and variational parameters. We do this by optimizing
#'  a variational lower bound on the marginal log likelihood of observed data. The
#'  lower bound takes the following form (see Titsias (2009) and
#'  Hensman (2013) for details on the derivation):
#'  ```
#'  L(Z, m, S, Y) = MVN(loc=
#'  (K_zx @ K_zz^-1) @ m, scale_diag=sigma).log_prob(Y) -
#'  (Tr(K_xx - K_zx @ K_zz^-1 @ K_xz) +
#'  Tr(S @ S^T @ K_zz^1 @ K_zx @ K_xz @ K_zz^-1)) / (2 * sigma^2) -
#'  KL(q(f(Z)) || p(f(Z))))
#'  ```
#'
#'  where in the final KL term, `p(f(Z))` is the GP prior on inducing point
#'  function values. This variational lower bound can be computed on minibatches
#'  of the full data set `(X, Y)`. A method to compute the *negative* variational
#'  lower bound is implemented as `VariationalGaussianProcess$variational_loss`.
#'
#'  Optimal variational parameters
#'
#'  As described in Titsias (2009), a closed form optimum for the variational
#'  location and scale parameters, `m` and `S`, can be computed when the
#'  observational data are not prohibitively voluminous. The
#'  `optimal_variational_posterior` function to computes the optimal variational
#'  posterior distribution over inducing point function values in terms of the GP
#'  parameters (mean and kernel functions), inducing point locations, observation
#'  index points, and observations. Note that the inducing index point locations
#'  must still be optimized even when these parameters are known functions of the
#'  inducing index points. The optimal parameters are computed as follows:
#'
#'  ```
#'  C = sigma^-2 (K_zz + K_zx @ K_xz)^-1
#'  optimal Gaussian covariance: K_zz @ C @ K_zz
#'  optimal Gaussian location: sigma^-2 K_zz @ C @ K_zx @ Y
#'  ```
#'
#' @section References:
#' - [Titsias, M. "Variational Model Selection for Sparse Gaussian Process Regression", 2009.](http://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf)
#' - [Hensman, J., Lawrence, N. "Gaussian Processes for Big Data", 2013.](https://arxiv.org/abs/1309.6835)
#' - [Carl Rasmussen, Chris Williams. Gaussian Processes For Machine Learning, 2006.](http://www.gaussianprocess.org/gpml/)
#'
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
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
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

#' Uniform distribution with `low` and `high` parameters
#'
#' Mathematical Details
#'
#' The probability density function (pdf) is,
#' ```
#' pdf(x; a, b) = I[a <= x < b] / Z
#' Z = b - a
#' ```
#' where
#' - `low = a`,
#' - `high = b`,
#' - `Z` is the normalizing constant, and
#' - `I[predicate]` is the [indicator function](
#'  https://en.wikipedia.org/wiki/Indicator_function) for `predicate`.
#'
#' The parameters `low` and `high` must be shaped in a way that supports
#' broadcasting (e.g., `high - low` is a valid operation).
#'
#' @param low Floating point tensor, lower boundary of the output interval. Must
#' have `low < high`.
#' @param high Floating point tensor, upper boundary of the output interval. Must
#' have `low < high`.
#' @inherit tfd_normal return params
#'
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
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

#' Truncated Normal distribution
#'
#' The truncated normal is a normal distribution bounded between `low`
#' and `high` (the pdf is 0 outside these bounds and renormalized).
#' Samples from this distribution are differentiable with respect to `loc`,
#' `scale` as well as the bounds, `low` and `high`, i.e., this
#' implementation is fully reparameterizeable.
#' For more details, see [here](https://en.wikipedia.org/wiki/Truncated_normal_distribution).
#'
#' Mathematical Details
#'
#' The probability density function (pdf) of this distribution is:
#' ```
#' pdf(x; loc, scale, low, high) =
#'   { (2 pi)**(-0.5) exp(-0.5 y**2) / (scale * z)} for low <= x <= high
#'   { 0 }                                  otherwise
#' y = (x - loc)/scale
#' z = NormalCDF((high - loc) / scale) - NormalCDF((lower - loc) / scale)
#' ```
#' where:
#' * `NormalCDF` is the cumulative density function of the Normal distribution
#' with 0 mean and unit variance.
#'
#' This is a scalar distribution so the event shape is always scalar and the
#' dimensions of the parameters defined the batch_shape.
#'
#' @param  low `float` `Tensor` representing lower bound of the distribution's
#' support. Must be such that `low < high`.
#' @param high `float` `Tensor` representing upper bound of the distribution's
#' support. Must be such that `low < high`.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
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

#' Triangular distribution with `low`, `high` and `peak` parameters
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
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
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

#' Student's t-distribution
#'
#' This distribution has parameters: degree of freedom `df`, location `loc`, and `scale`.
#'
#' Mathematical details
#'
#' The probability density function (pdf) is,
#' ```
#' pdf(x; df, mu, sigma) = (1 + y**2 / df)**(-0.5 (df + 1)) / Z
#' where,
#' y = (x - mu) / sigma
#' Z = abs(sigma) sqrt(df pi) Gamma(0.5 df) / Gamma(0.5 (df + 1))
#' ```
#' where:
#' * `loc = mu`,
#' * `scale = sigma`, and,
#' * `Z` is the normalization constant, and,
#' * `Gamma` is the [gamma function](
#'   https://en.wikipedia.org/wiki/Gamma_function).
#'   The StudentT distribution is a member of the [location-scale family](
#'     https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
#'     constructed as,
#' ```
#' X ~ StudentT(df, loc=0, scale=1)
#' Y = loc + scale * X
#' ```
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
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
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

#' Marginal distribution of a Student's T process at finitely many points
#'
#' A Student's T process (TP) is an indexed collection of random variables, any
#' finite collection of which are jointly Multivariate Student's T. While this
#' definition applies to finite index sets, it is typically implicit that the
#' index set is infinite; in applications, it is often some finite dimensional
#' real or complex vector space. In such cases, the TP may be thought of as a
#' distribution over (real- or complex-valued) functions defined over the index set.
#'
#' Just as Student's T distributions are fully specified by their degrees of
#' freedom, location and scale, a Student's T process can be completely specified
#' by a degrees of freedom parameter, mean function and covariance function.
#'
#' Let `S` denote the index set and `K` the space in which each indexed random variable
#'  takes its values (again, often R or C).
#'  The mean function is then a map `m: S -> K`, and the covariance function,
#'  or kernel, is a positive-definite function `k: (S x S) -> K`. The properties
#'  of functions drawn from a TP are entirely dictated (up to translation) by
#'  the form of the kernel function.
#'
#' This `Distribution` represents the marginal joint distribution over function
#' values at a given finite collection of points `[x[1], ..., x[N]]` from the
#' index set `S`. By definition, this marginal distribution is just a
#' multivariate Student's T distribution, whose mean is given by the vector
#' `[ m(x[1]), ..., m(x[N]) ]` and whose covariance matrix is constructed from
#' pairwise applications of the kernel function to the given inputs:
#'
#' ```
#' | k(x[1], x[1])    k(x[1], x[2])  ...  k(x[1], x[N]) |
#' | k(x[2], x[1])    k(x[2], x[2])  ...  k(x[2], x[N]) |
#' |      ...              ...                 ...      |
#' | k(x[N], x[1])    k(x[N], x[2])  ...  k(x[N], x[N]) |
#'
#' ```
#' For this to be a valid covariance matrix, it must be symmetric and positive
#' definite; hence the requirement that `k` be a positive definite function
#' (which, by definition, says that the above procedure will yield PD matrices).
#' Note also we use a parameterization as suggested in Shat et al. (2014), which requires `df`
#' to be greater than 2. This allows for the covariance for any finite
#' dimensional marginal of the TP (a multivariate Student's T distribution) to
#' just be the PD matrix generated by the kernel.
#'
#' Mathematical Details
#'
#' The probability density function (pdf) is a multivariate Student's T whose
#' parameters are derived from the TP's properties:
#'
#' ```
#' pdf(x; df, index_points, mean_fn, kernel) = MultivariateStudentT(df, loc, K)
#' K = (df - 2) / df  * (kernel.matrix(index_points, index_points) + jitter * eye(N))
#' loc = (x - mean_fn(index_points))^T @ K @ (x - mean_fn(index_points))
#' ```
#' where:
#' * `df` is the degrees of freedom parameter for the TP.
#' * `index_points` are points in the index set over which the TP is defined,
#' * `mean_fn` is a callable mapping the index set to the TP's mean values,
#' * `kernel` is `PositiveSemidefiniteKernel`-like and represents the covariance
#'  function of the TP,
#' * `jitter` is added to the diagonal to ensure positive definiteness up to
#' machine precision (otherwise Cholesky-decomposition is prone to failure),
#' * `eye(N)` is an N-by-N identity matrix.
#'
#' @section References:
#' - [Amar Shah, Andrew Gordon Wilson, and Zoubin Ghahramani. Student-t Processes as Alternatives to Gaussian Processes. In _Artificial Intelligence and Statistics_, 2014.](https://www.cs.cmu.edu/~andrewgw/tprocess.pdf)
#'
#' @param df Positive Floating-point `Tensor` representing the degrees of freedom.
#' Must be greater than 2.
#' @param kernel `PositiveSemidefiniteKernel`-like instance representing the
#' TP's covariance function.
#' @param index_points `float` `Tensor` representing finite (batch of) vector(s) of
#' points in the index set over which the TP is defined. Shape has the form
#' `[b1, ..., bB, e, f1, ..., fF]` where `F` is the number of feature
#' dimensions and must equal `kernel.feature_ndims` and `e` is the number
#' (size) of index points in each batch. Ultimately this distribution
#' corresponds to a `e`-dimensional multivariate Student's T. The batch
#' shape must be broadcastable with `kernel.batch_shape` and any batch dims
#' yielded by `mean_fn`.
#' @param mean_fn Function that acts on `index_points` to produce a (batch
#' of) vector(s) of mean values at `index_points`. Takes a `Tensor` of
#' shape `[b1, ..., bB, f1, ..., fF]` and returns a `Tensor` whose shape is
#' broadcastable with `[b1, ..., bB]`. Default value: `NULL` implies
#' constant zero function.
#' @param jitter `float` scalar `Tensor` added to the diagonal of the covariance
#' matrix to ensure positive definiteness of the covariance matrix. Default value: `1e-6`.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_student_t_process <- function(df,
                                  kernel,
                                  index_points,
                                  mean_fn = NULL,
                                  jitter = 1e-6,
                                  validate_args = FALSE,
                                  allow_nan_stats = FALSE,
                                  name = "StudentTProcess") {
  args <- list(
    df = df,
    kernel = kernel,
    index_points = index_points,
    mean_fn = mean_fn,
    jitter = jitter,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$StudentTProcess, args)
}

#' The SinhArcsinh transformation of a distribution on `(-inf, inf)`
#'
#' This distribution models a random variable, making use of
#' a `SinhArcsinh` transformation (which has adjustable tailweight and skew),
#' a rescaling, and a shift.
#' The `SinhArcsinh` transformation of the Normal is described in great depth in
#' [Sinh-arcsinh distributions](https://oro.open.ac.uk/22510/).
#' Here we use a slightly different parameterization, in terms of `tailweight`
#' and `skewness`.  Additionally we allow for distributions other than Normal,
#' and control over `scale` as well as a "shift" parameter `loc`.
#'
#' Mathematical Details
#'
#' Given random variable `Z`, we define the SinhArcsinh
#' transformation of `Z`, `Y`, parameterized by
#' `(loc, scale, skewness, tailweight)`, via the relation:
#' ```
#' Y := loc + scale * F(Z) * (2 / F_0(2))
#' F(Z) := Sinh( (Arcsinh(Z) + skewness) * tailweight )
#' F_0(Z) := Sinh( Arcsinh(Z) * tailweight )
#' ```
#'
#' This distribution is similar to the location-scale transformation
#' `L(Z) := loc + scale * Z` in the following ways:
#' * If `skewness = 0` and `tailweight = 1` (the defaults), `F(Z) = Z`, and then
#' `Y = L(Z)` exactly.
#'
#' * `loc` is used in both to shift the result by a constant factor.
#' * The multiplication of `scale` by `2 / F_0(2)` ensures that if `skewness = 0`
#' `P[Y - loc <= 2 * scale] = P[L(Z) - loc <= 2 * scale]`.
#' Thus it can be said that the weights in the tails of `Y` and `L(Z)` beyond
#' `loc + 2 * scale` are the same.
#'
#' This distribution is different than `loc + scale * Z` due to the
#' reshaping done by `F`:
#'
#' * Positive (negative) `skewness` leads to positive (negative) skew.
#' * positive skew means, the mode of `F(Z)` is "tilted" to the right.
#' * positive skew means positive values of `F(Z)` become more likely, and
#' negative values become less likely.
#' * Larger (smaller) `tailweight` leads to fatter (thinner) tails.
#' * Fatter tails mean larger values of `|F(Z)|` become more likely.
#' * `tailweight < 1` leads to a distribution that is "flat" around `Y = loc`,
#' and a very steep drop-off in the tails.
#' * `tailweight > 1` leads to a distribution more peaked at the mode with
#' heavier tails.
#'
#' To see the argument about the tails, note that for `|Z| >> 1` and
#' `|Z| >> (|skewness| * tailweight)**tailweight`, we have
#' `Y approx 0.5 Z**tailweight e**(sign(Z) skewness * tailweight)`.
#'
#' To see the argument regarding multiplying `scale` by `2 / F_0(2)`,
#' ```
#' P[(Y - loc) / scale <= 2] = P[F(Z) * (2 / F_0(2)) <= 2]
#'                           = P[F(Z) <= F_0(2)]
#'                           = P[Z <= 2]  (if F = F_0).
#' ```
#'
#' @param loc Floating-point `Tensor`.
#' @param scale  `Tensor` of same `dtype` as `loc`.
#' @param skewness  Skewness parameter.  Default is `0.0` (no skew).
#' @param tailweight  Tailweight parameter. Default is `1.0` (unchanged tailweight)
#' @param distribution `tf$distributions$Distribution`-like instance. Distribution that is
#'  transformed to produce this distribution. Default is `tfd_normal(0, 1)`.
#'  Must be a scalar-batch, scalar-event distribution.  Typically
#'  `distribution$reparameterization_type = FULLY_REPARAMETERIZED` or it is
#'  a function of non-trainable parameters. WARNING: If you backprop through
#'  a `SinhArcsinh` sample and `distribution` is not
#'  `FULLY_REPARAMETERIZED` yet is a function of trainable variables, then
#'  the gradient will be incorrect!
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_sinh_arcsinh <- function(loc,
                             scale,
                             skewness = NULL,
                             tailweight = NULL,
                             distribution = NULL,
                             validate_args = FALSE,
                             allow_nan_stats = TRUE,
                             name = "SinhArcsinh") {
  args <- list(
    loc = loc,
    scale = scale,
    skewness = skewness,
    tailweight = tailweight,
    distribution = distribution,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$SinhArcsinh, args)
}


#' Distribution representing the quantization `Y = ceiling(X)`
#'
#' Definition in Terms of Sampling
#'
#' ```
#' 1. Draw X
#' 2. Set Y <-- ceiling(X)
#' 3. If Y < low, reset Y <-- low
#' 4. If Y > high, reset Y <-- high
#' 5. Return Y
#' ```
#'
#' Definition in Terms of the Probability Mass Function
#'
#' Given scalar random variable `X`, we define a discrete random variable `Y`
#' supported on the integers as follows:
#'
#'  ```
#'  P[Y = j] := P[X <= low],  if j == low,
#'           := P[X > high - 1],  j == high,
#'           := 0, if j < low or j > high,
#'           := P[j - 1 < X <= j],  all other j.
#'  ```
#'
#' Conceptually, without cutoffs, the quantization process partitions the real
#' line `R` into half open intervals, and identifies an integer `j` with the
#' right endpoints:
#'
#'  ```
#'  R = ... (-2, -1](-1, 0](0, 1](1, 2](2, 3](3, 4] ...
#'  j = ...      -1      0     1     2     3     4  ...
#'  ```
#'
#'  `P[Y = j]` is the mass of `X` within the `jth` interval.
#'  If `low = 0`, and `high = 2`, then the intervals are redrawn
#'  and `j` is re-assigned:
#'
#'  ```
#'  R = (-infty, 0](0, 1](1, infty)
#'  j =          0     1     2
#'  ```
#'
#'  `P[Y = j]` is still the mass of `X` within the `jth` interval.
#'
#'  @section References:
#'  - [Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P. Kingma. PixelCNN++: Improving the PixelCNN with discretized logistic mixture likelihood and other modifications. International Conference on Learning Representations_, 2017.](https://arxiv.org/abs/1701.05517)
#'  - [Aaron van den Oord et al. Parallel WaveNet: Fast High-Fidelity Speech Synthesis. _arXiv preprint arXiv:1711.10433_, 2017.](https://arxiv.org/abs/1711.10433)
#'
#' @param distribution  The base distribution class to transform. Typically an
#' instance of `Distribution`.
#' @param low `Tensor` with same `dtype` as this distribution and shape
#' able to be added to samples. Should be a whole number. Default `NULL`.
#' If provided, base distribution's `prob` should be defined at `low`.
#' @param high `Tensor` with same `dtype` as this distribution and shape
#' able to be added to samples. Should be a whole number. Default `NULL`.
#' If provided, base distribution's `prob` should be defined at `high - 1`.
#' `high` must be strictly greater than `low`.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_quantized <- function(distribution,
                          low = NULL,
                          high = NULL,
                          validate_args = FALSE,
                          name = "QuantizedDistribution") {
  args <- list(
    distribution = distribution,
    low = low,
    high = high,
    validate_args = validate_args,
    name = name
  )

  do.call(tfp$distributions$QuantizedDistribution, args)
}

#' Poisson distribution
#'
#' The Poisson distribution is parameterized by an event `rate` parameter.
#'
#' Mathematical Details
#'
#' The probability mass function (pmf) is,
#' ```
#' pmf(k; lambda, k >= 0) = (lambda^k / k!) / Z
#' Z = exp(lambda).
#' ```
#' where `rate = lambda` and `Z` is the normalizing constant.
#'
#' @param rate Floating point tensor, the rate parameter. `rate` must be positive.
#' Must specify exactly one of `rate` and `log_rate`.
#' @param log_rate Floating point tensor, the log of the rate parameter.
#' Must specify exactly one of `rate` and `log_rate`.
#' @param interpolate_nondiscrete Logical. When `FALSE`,
#' `log_prob` returns `-inf` (and `prob` returns `0`) for non-integer
#' inputs. When `TRUE`, `log_prob` evaluates the continuous function
#' `k * log_rate - lgamma(k+1) - rate`, which matches the Poisson pmf
#' at integer arguments `k` (note that this function is not itself
#' a normalized probability log-density). Default value: `TRUE`.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_poisson <- function(rate = NULL,
                        log_rate = NULL,
                        interpolate_nondiscrete = TRUE,
                        validate_args = FALSE,
                        allow_nan_stats = TRUE,
                        name = "Poisson") {
  args <- list(
    rate = rate,
    log_rate = log_rate,
    interpolate_nondiscrete = interpolate_nondiscrete,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Poisson, args)
}

#' `PoissonLogNormalQuadratureCompound` distribution
#'
#' The `PoissonLogNormalQuadratureCompound` is an approximation to a
#' Poisson-LogNormal [compound distribution](https://en.wikipedia.org/wiki/Compound_probability_distribution), i.e.,
#' ```
#' p(k|loc, scale) = int_{R_+} dl LogNormal(l | loc, scale) Poisson(k | l)
#' approx= sum{ prob[d] Poisson(k | lambda(grid[d])) : d=0, ..., deg-1 }
#' ```
#'
#' By default, the `grid` is chosen as quantiles of the `LogNormal` distribution
#' parameterized by `loc`, `scale` and the `prob` vector is
#' `[1. / quadrature_size]*quadrature_size`.
#'
#' In the non-approximation case, a draw from the LogNormal prior represents the
#' Poisson rate parameter. Unfortunately, the non-approximate distribution lacks
#' an analytical probability density function (pdf). Therefore the
#' `PoissonLogNormalQuadratureCompound` class implements an approximation based
#' on [quadrature](https://en.wikipedia.org/wiki/Numerical_integration).
#' Note: although the `PoissonLogNormalQuadratureCompound` is approximately the
#' Poisson-LogNormal compound distribution, it is itself a valid distribution.
#' Viz., it possesses a `sample`, `log_prob`, `mean`, `variance`, etc. which are
#' all mutually consistent.
#'
#' Mathematical Details
#'
#' The `PoissonLogNormalQuadratureCompound` approximates a Poisson-LogNormal
#' [compound distribution](https://en.wikipedia.org/wiki/Compound_probability_distribution).
#' Using variable-substitution and [numerical quadrature](
#' https://en.wikipedia.org/wiki/Numerical_integration) (default:
#' based on `LogNormal` quantiles) we can redefine the distribution to be a
#' parameter-less convex combination of `deg` different Poisson samples.
#' That is, defined over positive integers, this distribution is parameterized
#' by a (batch of) `loc` and `scale` scalars.
#'
#' The probability density function (pdf) is,
#' ```
#' pdf(k | loc, scale, deg) = sum{ prob[d] Poisson(k | lambda=exp(grid[d])) : d=0, ..., deg-1 }
#' ```
#' Note: `probs` returned by (optional) `quadrature_fn` are presumed to be
#' either a length-`quadrature_size` vector or a batch of vectors in 1-to-1
#' correspondence with the returned `grid`. (I.e., broadcasting is only partially supported.)
#'
#' @param loc `float`-like (batch of) scalar `Tensor`; the location parameter of
#' the LogNormal prior.
#' @param scale `float`-like (batch of) scalar `Tensor`; the scale parameter of
#' the LogNormal prior.
#' @param quadrature_size  `integer` scalar representing the number of quadrature
#' points.
#' @param  quadrature_fn Function taking `loc`, `scale`,
#' `quadrature_size`, `validate_args` and returning `tuple(grid, probs)`
#' representing the LogNormal grid and corresponding normalized weight.
#' Default value: `quadrature_scheme_lognormal_quantiles`.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_poisson_log_normal_quadrature_compound <- function(loc,
                                                       scale,
                                                       quadrature_size = 8,
                                                       quadrature_fn = tfp$distributions$quadrature_scheme_lognormal_quantiles,
                                                       validate_args = FALSE,
                                                       allow_nan_stats = TRUE,
                                                       name = "PoissonLogNormalQuadratureCompound") {
  args <- list(
    loc = loc,
    scale = scale,
    quadrature_size = as.integer(quadrature_size),
    quadrature_fn = quadrature_fn,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$PoissonLogNormalQuadratureCompound,
          args)
}

#' Pareto distribution
#'
#' The Pareto distribution is parameterized by a `scale` and a
#' `concentration` parameter.
#'
#' Mathematical Details
#'
#' The probability density function (pdf) is,
#'
#' ```
#' pdf(x; alpha, scale, x >= scale) = alpha * scale ** alpha / x ** (alpha + 1)
#' ```#'
#' where `concentration = alpha`.
#'
#' Note that `scale` acts as a scaling parameter, since
#' `Pareto(c, scale).pdf(x) == Pareto(c, 1.).pdf(x / scale)`.
#' The support of the distribution is defined on `[scale, infinity)`.
#'
#' @param concentration Floating point tensor. Must contain only positive values.
#' @param scale Floating point tensor, equivalent to `mode`. `scale` also
#' restricts the domain of this distribution to be in `[scale, inf)`.
#' Must contain only positive values. Default value: `1`.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_pareto <- function(concentration,
                       scale = 1,
                       validate_args = FALSE,
                       allow_nan_stats = TRUE,
                       name = "Pareto") {
  args <- list(
    concentration = concentration,
    scale = scale,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Pareto,
          args)
}

#' NegativeBinomial distribution
#'
#' The NegativeBinomial distribution is related to the experiment of performing
#' Bernoulli trials in sequence. Given a Bernoulli trial with probability `p` of
#' success, the NegativeBinomial distribution represents the distribution over
#' the number of successes `s` that occur until we observe `f` failures.
#'
#' The probability mass function (pmf) is,
#' ```
#' pmf(s; f, p) = p**s (1 - p)**f / Z
#' Z = s! (f - 1)! / (s + f - 1)!
#' ```
#'
#' where:
#' * `total_count = f`,
#' * `probs = p`,
#' * `Z` is the normalizaing constant, and,
#' * `n!` is the factorial of `n`.
#'
#' @param total_count Non-negative floating-point `Tensor` with shape
#' broadcastable to `[B1,..., Bb]` with `b >= 0` and the same dtype as
#' `probs` or `logits`. Defines this as a batch of `N1 x ... x Nm`
#' different Negative Binomial distributions. In practice, this represents
#' the number of negative Bernoulli trials to stop at (the `total_count`
#' of failures), but this is still a valid distribution when
#' `total_count` is a non-integer.
#' @param logits Floating-point `Tensor` with shape broadcastable to
#' `[B1, ..., Bb]` where `b >= 0` indicates the number of batch dimensions.
#' Each entry represents logits for the probability of success for
#' independent Negative Binomial distributions and must be in the open
#' interval `(-inf, inf)`. Only one of `logits` or `probs` should be
#' specified.
#' @param probs Positive floating-point `Tensor` with shape broadcastable to
#' `[B1, ..., Bb]` where `b >= 0` indicates the number of batch dimensions.
#' Each entry represents the probability of success for independent
#' Negative Binomial distributions and must be in the open interval
#' `(0, 1)`. Only one of `logits` or `probs` should be specified.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_negative_binomial <- function(total_count,
                                  logits = NULL,
                                  probs = NULL,
                                  validate_args = FALSE,
                                  allow_nan_stats = TRUE,
                                  name = "NegativeBinomial") {
  args <- list(
    total_count = total_count,
    logits = logits,
    probs = probs,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$NegativeBinomial,
          args)
}

#' The multivariate normal distribution on `R^k`
#'
#' The Multivariate Normal distribution is defined over `R^k`` and parameterized
#' by a (batch of) length-k loc vector (aka "mu") and a (batch of) `k x k`
#' scale matrix; `covariance = scale @ scale.T` where `@` denotes
#' matrix-multiplication.
#'
#' Mathematical Details
#'
#' The probability density function (pdf) is,
#' ```
#' pdf(x; loc, scale) = exp(-0.5 ||y||**2) / Z
#' y = inv(scale) @ (x - loc)
#' Z = (2 pi)**(0.5 k) |det(scale)|
#' ```
#' where:
#' * `loc` is a vector in `R^k`,
#' * `scale` is a linear operator in `R^{k x k}`, `cov = scale @ scale.T`,
#' * `Z` denotes the normalization constant, and,
#' * `||y||**2` denotes the squared Euclidean norm of `y`.
#'
#' A (non-batch) `scale` matrix is:
#' ```
#' scale = scale_tril
#' ```
#'
#' where `scale_tril` is lower-triangular `k x k` matrix with non-zero diagonal,
#' i.e., `tf$diag_part(scale_tril) != 0`.
#' Additional leading dimensions (if any) will index batches.
#'
#' The MultivariateNormal distribution is a member of the
#' [location-scale family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
#' constructed as,
#' ```
#' X ~ MultivariateNormal(loc=0, scale=1)   # Identity scale, zero shift.
#' Y = scale @ X + loc
#' ```
#'
#' @param loc Floating-point `Tensor`. If this is set to `NULL`, `loc` is
#' implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
#' `b >= 0` and `k` is the event size.
#' @param scale_tril Floating-point, lower-triangular `Tensor` with non-zero
#' diagonal elements. `scale_tril` has shape `[B1, ..., Bb, k, k]` where
#' `b >= 0` and `k` is the event size.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_multivariate_normal_tri_l <- function(loc = NULL,
                                          scale_tril = NULL,
                                          validate_args = FALSE,
                                          allow_nan_stats = TRUE,
                                          name = "MultivariateNormalTriL") {
  args <- list(
    loc = loc,
    scale_tril = scale_tril,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$MultivariateNormalTriL, args)
}

#' The multivariate normal distribution on `R^k`
#'
#' The Multivariate Normal distribution is defined over `R^k`` and parameterized
#' by a (batch of) length-k loc vector (aka "mu") and a (batch of) `k x k`
#' scale matrix; `covariance = scale @ scale.T` where `@` denotes
#' matrix-multiplication.
#'
#' Mathematical Details
#'
#' The probability density function (pdf) is,
#' ```
#' pdf(x; loc, scale) = exp(-0.5 ||y||**2) / Z
#' y = inv(scale) @ (x - loc)
#' Z = (2 pi)**(0.5 k) |det(scale)|
#' ```
#' where:
#' * `loc` is a vector in `R^k`,
#' * `scale` is a linear operator in `R^{k x k}`, `cov = scale @ scale.T`,
#' * `Z` denotes the normalization constant, and,
#' * `||y||**2` denotes the squared Euclidean norm of `y`.
#'
#' The MultivariateNormal distribution is a member of the
#' [location-scale family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
#' constructed as,
#' ```
#' X ~ MultivariateNormal(loc=0, scale=1)   # Identity scale, zero shift.
#' Y = scale @ X + loc
#' ```
#'
#' The `batch_shape` is the broadcast shape between `loc` and `scale`
#' arguments.
#' The `event_shape` is given by last dimension of the matrix implied by
#' `scale`. The last dimension of `loc` (if provided) must broadcast with this.
#' Recall that `covariance = scale @ scale.T`.
#' Additional leading dimensions (if any) will index batches.
#' @param loc Floating-point `Tensor`. If this is set to `NULL`, `loc` is
#' implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
#' `b >= 0` and `k` is the event size.
#' @param scale Instance of `LinearOperator` with same `dtype` as `loc` and shape
#' `[B1, ..., Bb, k, k]`.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_multivariate_normal_linear_operator <- function(loc = NULL,
                                                    scale = NULL,
                                                    validate_args = FALSE,
                                                    allow_nan_stats = TRUE,
                                                    name = "MultivariateNormalLinearOperator") {
  args <- list(
    loc = loc,
    scale = scale,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$MultivariateNormalLinearOperator, args)
}

#' Multivariate normal distribution on `R^k`
#'
#' The Multivariate Normal distribution is defined over `R^k`` and parameterized
#' by a (batch of) length-k loc vector (aka "mu") and a (batch of) `k x k`
#' scale matrix; `covariance = scale @ scale.T` where `@` denotes
#' matrix-multiplication.
#'
#' Mathematical Details
#'
#' The probability density function (pdf) is,
#' ```
#' pdf(x; loc, scale) = exp(-0.5 ||y||**2) / Z
#' y = inv(scale) @ (x - loc)
#' Z = (2 pi)**(0.5 k) |det(scale)|
#' ```
#' where:
#' * `loc` is a vector in `R^k`,
#' * `scale` is a linear operator in `R^{k x k}`, `cov = scale @ scale.T`,
#' * `Z` denotes the normalization constant, and,
#' * `||y||**2` denotes the squared Euclidean norm of `y`.
#'
#' The MultivariateNormal distribution is a member of the
#' [location-scale family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
#' constructed as,
#' ```
#' X ~ MultivariateNormal(loc=0, scale=1)   # Identity scale, zero shift.
#' Y = scale @ X + loc
#' ```
#'
#' The `batch_shape` is the broadcast shape between `loc` and
#' `covariance_matrix` arguments.
#' The `event_shape` is given by last dimension of the matrix implied by
#' `covariance_matrix`. The last dimension of `loc` (if provided) must
#' broadcast with this.
#' A non-batch `covariance_matrix` matrix is a `k x k` symmetric positive
#' definite matrix.  In other words it is (real) symmetric with all eigenvalues
#' strictly positive.
#' Additional leading dimensions (if any) will index batches.
#' @param loc Floating-point `Tensor`. If this is set to `NULL`, `loc` is
#' implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
#' `b >= 0` and `k` is the event size.
#' @param covariance_matrix Floating-point, symmetric positive definite `Tensor` of
#' same `dtype` as `loc`.  The strict upper triangle of `covariance_matrix`
#' is ignored, so if `covariance_matrix` is not symmetric no error will be
#' raised (unless `validate_args is TRUE`).  `covariance_matrix` has shape
#' `[B1, ..., Bb, k, k]` where `b >= 0` and `k` is the event size.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_multivariate_normal_full_covariance <- function(loc = NULL,
                                                    covariance_matrix = NULL,
                                                    validate_args = FALSE,
                                                    allow_nan_stats = TRUE,
                                                    name = "MultivariateNormalFullCovariance") {
  args <- list(
    loc = loc,
    covariance_matrix = covariance_matrix,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$MultivariateNormalFullCovariance, args)
}

#' Multivariate normal distribution on `R^k`
#'
#' The Multivariate Normal distribution is defined over `R^k`` and parameterized
#' by a (batch of) length-k loc vector (aka "mu") and a (batch of) `k x k`
#' scale matrix; `covariance = scale @ scale.T` where `@` denotes
#' matrix-multiplication.
#'
#' Mathematical Details
#'
#' The probability density function (pdf) is,
#' ```
#' pdf(x; loc, scale) = exp(-0.5 ||y||**2) / Z
#' y = inv(scale) @ (x - loc)
#' Z = (2 pi)**(0.5 k) |det(scale)|
#' ```
#' where:
#' * `loc` is a vector in `R^k`,
#' * `scale` is a linear operator in `R^{k x k}`, `cov = scale @ scale.T`,
#' * `Z` denotes the normalization constant, and,
#' * `||y||**2` denotes the squared Euclidean norm of `y`.
#'
#' A (non-batch) `scale` matrix is:
#' ```
#' scale = diag(scale_diag + scale_identity_multiplier ones(k)) +
#' scale_perturb_factor @ diag(scale_perturb_diag) @ scale_perturb_factor.T
#' ```
#'
#' where:
#' * `scale_diag.shape = [k]`,
#' * `scale_identity_multiplier.shape = []`,
#' * `scale_perturb_factor.shape = [k, r]`, typically `k >> r`, and,
#' * `scale_perturb_diag.shape = [r]`.
#'
#' Additional leading dimensions (if any) will index batches.
#' If both `scale_diag` and `scale_identity_multiplier` are `NULL`, then
#' `scale` is the Identity matrix.

#' The MultivariateNormal distribution is a member of the
#' [location-scale family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
#' constructed as,
#' ```
#' X ~ MultivariateNormal(loc=0, scale=1)   # Identity scale, zero shift.
#' Y = scale @ X + loc
#' ```
#'
#' @inherit tfd_normal return params
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
#' @param  scale_perturb_factor Floating-point `Tensor` representing a rank-`r`
#' perturbation added to `scale`. May have shape `[B1, ..., Bb, k, r]`,
#' `b >= 0`, and characterizes `b`-batches of rank-`r` updates to `scale`.
#' When `NULL`, no rank-`r` update is added to `scale`.#'
#' @param scale_perturb_diag Floating-point `Tensor` representing a diagonal matrix
#' inside the rank-`r` perturbation added to `scale`. May have shape
#' `[B1, ..., Bb, r]`, `b >= 0`, and characterizes `b`-batches of `r` x `r`
#' diagonal matrices inside the perturbation added to `scale`. When
#' `NULL`, an identity matrix is used inside the perturbation. Can only be
#' specified if `scale_perturb_factor` is also specified.
#'
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_multivariate_normal_diag_plus_low_rank <- function(loc = NULL,
                                                       scale_diag = NULL,
                                                       scale_identity_multiplier = NULL,
                                                       scale_perturb_factor = NULL,
                                                       scale_perturb_diag = NULL,
                                                       validate_args = FALSE,
                                                       allow_nan_stats = TRUE,
                                                       name = "MultivariateNormalDiagPlusLowRank") {
  args <- list(
    loc = loc,
    scale_diag = scale_diag,
    scale_identity_multiplier = scale_identity_multiplier,
    scale_perturb_factor = scale_perturb_factor,
    scale_perturb_diag = scale_perturb_diag,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$MultivariateNormalDiagPlusLowRank, args)
}

#' Multivariate Student's t-distribution on `R^k`
#'
#' Mathematical Details
#'
#' The probability density function (pdf) is,
#' ```
#' pdf(x; df, loc, Sigma) = (1 + ||y||**2 / df)**(-0.5 (df + k)) / Z
#' where,
#' y = inv(Sigma) (x - loc)
#' Z = abs(det(Sigma)) sqrt(df pi)**k Gamma(0.5 df) / Gamma(0.5 (df + k))
#' ```
#'
#' where:
#' * `df` is a positive scalar.
#' * `loc` is a vector in `R^k`,
#' * `Sigma` is a positive definite `shape` matrix in `R^{k x k}`, parameterized
#' as `scale @ scale.T` in this class,
#' * `Z` denotes the normalization constant, and,
#' * `||y||**2` denotes the squared Euclidean norm of `y`.
#'
#' The Multivariate Student's t-distribution distribution is a member of the
#' [location-scale family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
#' constructed as,
#' ```
#' X ~ MultivariateT(loc=0, scale=1)   # Identity scale, zero shift.
#' Y = scale @ X + loc
#' ```
#' @param df A positive floating-point `Tensor`. Has shape `[B1, ..., Bb]` where `b >= 0`.
#' @param loc Floating-point `Tensor`. Has shape `[B1, ..., Bb, k]` where `k` is
#' the event size.
#' @param scale Instance of `LinearOperator` with a floating `dtype` and shape
#' `[B1, ..., Bb, k, k]`.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_multivariate_student_t_linear_operator <- function(df,
                                                       loc,
                                                       scale,
                                                       validate_args = FALSE,
                                                       allow_nan_stats = TRUE,
                                                       name = "MultivariateStudentTLinearOperator") {
  args <- list(
    df = df,
    loc = loc,
    scale = scale,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$MultivariateStudentTLinearOperator,
          args)
}

#' Multinomial distribution
#'
#' This Multinomial distribution is parameterized by `probs`, a (batch of)
#' length-`K` `prob` (probability) vectors (`K > 1`) such that
#' `tf.reduce_sum(probs, -1) = 1`, and a `total_count` number of trials, i.e.,
#' the number of trials per draw from the Multinomial. It is defined over a
#' (batch of) length-`K` vector `counts` such that
#' `tf$reduce_sum(counts, -1) = total_count`. The Multinomial is identically the
#' Binomial distribution when `K = 2`.
#'
#' Mathematical Details
#'
#' The Multinomial is a distribution over `K`-class counts, i.e., a length-`K`
#' vector of non-negative integer `counts = n = [n_0, ..., n_{K-1}]`.
#' The probability mass function (pmf) is,
#'
#' ```
#' pmf(n; pi, N) = prod_j (pi_j)**n_j / Z
#' Z = (prod_j n_j!) / N!
#' ```
#' where:
#' * `probs = pi = [pi_0, ..., pi_{K-1}]`, `pi_j > 0`, `sum_j pi_j = 1`,
#' * `total_count = N`, `N` a positive integer,
#' * `Z` is the normalization constant, and,
#' * `N!` denotes `N` factorial.
#'
#' Distribution parameters are automatically broadcast in all functions; see
#' examples for details.
#'
#' Pitfalls
#'
#' The number of classes, `K`, must not exceed:
#' - the largest integer representable by `self$dtype`, i.e.,
#' `2**(mantissa_bits+1)` (IEE754),
#' - the maximum `Tensor` index, i.e., `2**31-1`.
#'
#' Note: This condition is validated only when `validate_args = TRUE`.
#' @param total_count Non-negative floating point tensor with shape broadcastable
#' to `[N1,..., Nm]` with `m >= 0`. Defines this as a batch of
#' `N1 x ... x Nm` different Multinomial distributions. Its components
#' should be equal to integer values.
#' @param logits Floating point tensor representing unnormalized log-probabilities
#' of a positive event with shape broadcastable to
#' `[N1,..., Nm, K]` `m >= 0`, and the same dtype as `total_count`. Defines
#' this as a batch of `N1 x ... x Nm` different `K` class Multinomial
#' distributions. Only one of `logits` or `probs` should be passed in.
#' @param probs Positive floating point tensor with shape broadcastable to
#' `[N1,..., Nm, K]` `m >= 0` and same dtype as `total_count`. Defines
#' this as a batch of `N1 x ... x Nm` different `K` class Multinomial
#' distributions. `probs`'s components in the last portion of its shape
#' should sum to `1`. Only one of `logits` or `probs` should be passed in.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_multinomial <- function(total_count,
                            logits = NULL,
                            probs = NULL,
                            validate_args = FALSE,
                            allow_nan_stats = TRUE,
                            name = "Multinomial") {
  args <- list(
    total_count = total_count,
    logits = logits,
    probs = probs,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Multinomial,
          args)
}

#' Mixture distribution
#'
#' The `Mixture` object implements batched mixture distributions.
#' The mixture model is defined by a `Categorical` distribution (the mixture)
#' and a list of `Distribution` objects.
#'
#' Methods supported include `tfd_log_prob`, `tfd_prob`, `tfd_mean`, `tfd_sample`,
#' and `entropy_lower_bound`.
#' @param cat A `Categorical` distribution instance, representing the probabilities
#' of `distributions`.
#' @param components A list or tuple of `Distribution` instances.
#' Each instance must have the same type, be defined on the same domain,
#' and have matching `event_shape` and `batch_shape`.
#' @param use_static_graph Calls to `sample` will not rely on dynamic tensor
#' indexing, allowing for some static graph compilation optimizations, but
#' at the expense of sampling all underlying distributions in the mixture.
#' (Possibly useful when running on TPUs). Default value: `FALSE` (i.e., use dynamic indexing).
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_mixture <- function(cat,
                        components,
                        validate_args = FALSE,
                        allow_nan_stats = TRUE,
                        use_static_graph = FALSE,
                        name = "Mixture") {
  args <- list(
    cat = cat,
    components = components,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    use_static_graph = use_static_graph,
    name = name
  )

  do.call(tfp$distributions$Mixture,
          args)
}

#' Categorical distribution over integers
#'
#' The Categorical distribution is parameterized by either probabilities or
#' log-probabilities of a set of `K` classes. It is defined over the integers
#' `{0, 1, ..., K-1}`.
#'
#' The Categorical distribution is closely related to the `OneHotCategorical` and
#' `Multinomial` distributions.  The Categorical distribution can be intuited as
#' generating samples according to `argmax{ OneHotCategorical(probs) }` itself
#' being identical to `argmax{ Multinomial(probs, total_count=1) }`.
#'
#' Mathematical Details
#'
#' The probability mass function (pmf) is,
#' ```
#' pmf(k; pi) = prod_j pi_j**[k == j]
#' ```
#' Pitfalls
#'
#' The number of classes, `K`, must not exceed:
#'  - the largest integer representable by `self$dtype`, i.e.,
#'  `2**(mantissa_bits+1)` (IEEE 754),
#'  - the maximum `Tensor` index, i.e., `2**31-1`.
#'
#'  Note: This condition is validated only when `validate_args = TRUE`.
#'
#' @param logits An N-D `Tensor`, `N >= 1`, representing the log probabilities
#' of a set of Categorical distributions. The first `N - 1` dimensions
#' index into a batch of independent distributions and the last dimension
#' represents a vector of logits for each class. Only one of `logits` or
#' `probs` should be passed in.
#' @param probs An N-D `Tensor`, `N >= 1`, representing the probabilities
#' of a set of Categorical distributions. The first `N - 1` dimensions
#' index into a batch of independent distributions and the last dimension
#' represents a vector of probabilities for each class. Only one of
#' `logits` or `probs` should be passed in.
#' @param dtype The type of the event samples (default: int32).
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_categorical <- function(logits = NULL,
                            probs = NULL,
                            dtype = tf$int32,
                            validate_args = FALSE,
                            allow_nan_stats = TRUE,
                            name = "Categorical") {
  args <- list(
    logits = logits,
    probs = probs,
    dtype = dtype,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Categorical,
          args)
}

#' Mixture (same-family) distribution
#'
#' The `MixtureSameFamily` distribution implements a (batch of) mixture
#' distribution where all components are from different parameterizations of the
#' same distribution type. It is parameterized by a `Categorical` "selecting
#' distribution" (over `k` components) and a components distribution, i.e., a
#' `Distribution` with a rightmost batch shape (equal to `[k]`) which indexes
#' each (batch of) component.
#'
#' @param mixture_distribution `tfp$distributions$Categorical`-like instance.
#' Manages the probability of selecting components. The number of
#' categories must match the rightmost batch dimension of the
#' `components_distribution`. Must have either scalar `batch_shape` or
#' `batch_shape` matching `components_distribution$batch_shape[:-1]`.
#' @param components_distribution `tfp$distributions$Distribution`-like instance.
#' Right-most batch dimension indexes components.
#' @param reparameterize Logical, default `FALSE`. Whether to reparameterize
#' samples of the distribution using implicit reparameterization gradients
#' (Figurnov et al., 2018). The gradients for the mixture logits are
#' equivalent to the ones described by (Graves, 2016). The gradients
#' for the components parameters are also computed using implicit
#' reparameterization (as opposed to ancestral sampling), meaning that
#' all components are updated every step.
#' Only works when:
#' (1) components_distribution is fully reparameterized;
#' (2) components_distribution is either a scalar distribution or
#' fully factorized (tfd.Independent applied to a scalar distribution);
#' (3) batch shape has a known rank.
#' Experimental, may be slow and produce infs/NaNs.
#'
#' @section References:
#' - [Michael Figurnov, Shakir Mohamed and Andriy Mnih. Implicit reparameterization gradients. In _Neural Information Processing Systems_, 2018. ](https://arxiv.org/abs/1805.08498)
#' - [Alex Graves. Stochastic Backpropagation through Mixture Density Distributions. _arXiv_, 2016.](https://arxiv.org/abs/1607.05690)
#'
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_mixture_same_family <- function(mixture_distribution,
                                    components_distribution,
                                    reparameterize = FALSE,
                                    validate_args = FALSE,
                                    allow_nan_stats = TRUE,
                                    name = "MixtureSameFamily") {
  args <- list(
    mixture_distribution = mixture_distribution,
    components_distribution = components_distribution,
    reparameterize = reparameterize,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$MixtureSameFamily,
          args)
}

#' Log-normal distribution
#'
#' The LogNormal distribution models positive-valued random variables
#' whose logarithm is normally distributed with mean `loc` and
#' standard deviation `scale`. It is constructed as the exponential
#' transformation of a Normal distribution.
#' @param loc Floating-point `Tensor`; the means of the underlying
#' Normal distribution(s).
#' @param scale Floating-point `Tensor`; the stddevs of the underlying
#' Normal distribution(s).
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_log_normal <- function(loc = NULL,
                           scale = NULL,
                           validate_args = FALSE,
                           allow_nan_stats = TRUE,
                           name = "LogNormal") {
  args <- list(
    loc = loc,
    scale = scale,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$LogNormal,
          args)
}

#' Logistic distribution with location `loc` and `scale` parameters
#'
#' Mathematical details
#'
#' The cumulative density function of this distribution is:
#' ```
#' cdf(x; mu, sigma) = 1 / (1 + exp(-(x - mu) / sigma))
#' ```
#' where `loc = mu` and `scale = sigma`.
#'
#' The Logistic distribution is a member of the [location-scale family](
#' https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
#' constructed as,
#'
#' ```
#' X ~ Logistic(loc=0, scale=1)
#' Y = loc + scale * X
#' ```
#' @param loc Floating point tensor, the means of the distribution(s).
#' @param scale Floating point tensor, the scales of the distribution(s). Must
#' contain only positive values.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_logistic <- function(loc,
                         scale ,
                         validate_args = FALSE,
                         allow_nan_stats = TRUE,
                         name = "Logistic") {
  args <- list(
    loc = loc,
    scale = scale,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Logistic,
          args)
}

#' LKJ distribution on correlation matrices
#'
#' This is a one-parameter  of distributions on correlation matrices.  The
#' probability density is proportional to the determinant raised to the power of
#' the parameter: `pdf(X; eta) = Z(eta) * det(X) ** (eta - 1)`, where `Z(eta)` is
#' a normalization constant.  The uniform distribution on correlation matrices is
#' the special case `eta = 1`.
#'
#' The distribution is named after Lewandowski, Kurowicka, and Joe, who gave a
#' sampler for the distribution in Lewandowski, Kurowicka, Joe, 2009.

#' @param dimension  `integer`. The dimension of the correlation matrices
#' to sample.
#' @param concentration `float` or `double` `Tensor`. The positive concentration
#' parameter of the LKJ distributions. The pdf of a sample matrix `X` is
#' proportional to `det(X) ** (concentration - 1)`.
#' @param input_output_cholesky `Logical`. If `TRUE`, functions whose input or
#' output have the semantics of samples assume inputs are in Cholesky form
#' and return outputs in Cholesky form. In particular, if this flag is
#' `TRUE`, input to `log_prob` is presumed of Cholesky form and output from
#' `sample` is of Cholesky form.  Setting this argument to `TRUE` is purely
#' a computational optimization and does not change the underlying
#' distribution. Additionally, validation checks which are only defined on
#' the multiplied-out form are omitted, even if `validate_args` is `TRUE`.
#' Default value: `FALSE` (i.e., input/output does not have Cholesky semantics).
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_lkj <- function(dimension,
                    concentration,
                    input_output_cholesky = FALSE,
                    validate_args = FALSE,
                    allow_nan_stats = TRUE,
                    name = "LKJ") {
  args <- list(
    dimension = as.integer(dimension),
    concentration = concentration,
    input_output_cholesky = input_output_cholesky,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

 do.call(tfp$distributions$LKJ,
          args)
}

#' The CholeskyLKJ distribution on cholesky factors of correlation matrices
#'
#' This is a one-parameter family of distributions on cholesky factors of
#' correlation matrices.
#' In other words, if If `X ~ CholeskyLKJ(c)`, then `X @ X^T ~ LKJ(c)`.
#' For more details on the LKJ distribution, see `tfd_lkj`.

#' @param dimension  `integer`. The dimension of the correlation matrices
#' to sample.
#' @param concentration `float` or `double` `Tensor`. The positive concentration
#' parameter of the CholeskyLKJ distributions.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_cholesky_lkj <- function(dimension,
                             concentration,
                             validate_args = FALSE,
                             allow_nan_stats = TRUE,
                             name = "CholeskyLKJ") {
  args <- list(
    dimension = as.integer(dimension),
    concentration = concentration,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$CholeskyLKJ,
          args)
}

#' Observation distribution from a linear Gaussian state space model
#'
#' The state space model, sometimes called a Kalman filter, posits a
#' latent state vector `z_t` of dimension `latent_size` that evolves
#' over time following linear Gaussian transitions,
#' ```z_{t+1} = F * z_t + N(b; Q)```
#' for transition matrix `F`, bias `b` and covariance matrix
#' `Q`. At each timestep, we observe a noisy projection of the
#' latent state `x_t = H * z_t + N(c; R)`. The transition and
#' observation models may be fixed or may vary between timesteps.
#'
#' This Distribution represents the marginal distribution on
#' observations, `p(x)`. The marginal `log_prob` is computed by
#' Kalman filtering, and `sample` by an efficient forward
#' recursion. Both operations require time linear in `T`, the total
#' number of timesteps.
#'
#' Shapes
#'
#' The event shape is `[num_timesteps, observation_size]`, where
#' `observation_size` is the dimension of each observation `x_t`.
#' The observation and transition models must return consistent
#' shapes.
#' This implementation supports vectorized computation over a batch of
#' models. All of the parameters (prior distribution, transition and
#' observation operators and noise models) must have a consistent
#' batch shape.
#'
#' Time-varying processes
#'
#' Any of the model-defining parameters (prior distribution, transition
#' and observation operators and noise models) may be specified as a
#' callable taking an integer timestep `t` and returning a
#' time-dependent value. The dimensionality (`latent_size` and
#' `observation_size`) must be the same at all timesteps.
#'
#' Importantly, the timestep is passed as a `Tensor`, not a Python
#' integer, so any conditional behavior must occur *inside* the
#' TensorFlow graph. For example, suppose we want to use a different
#' transition model on even days than odd days. It does *not* work to
#' write
#'
#' ```
#' transition_matrix <- function(t) {
#' if(t %% 2 == 0) even_day_matrix else odd_day_matrix
#' }
#' ```
#'
#' since the value of `t` is not fixed at graph-construction
#' time. Instead we need to write
#'
#' ```
#' transition_matrix <- function(t) {
#' tf$cond(tf$equal(tf$mod(t, 2), 0), function() even_day_matrix, function() odd_day_matrix)
#' }
#' ```
#'
#' so that TensorFlow can switch between operators appropriately at runtime.
#' @param num_timesteps Integer `Tensor` total number of timesteps.
#' @param transition_matrix A transition operator, represented by a Tensor or
#' LinearOperator of shape `[latent_size, latent_size]`, or by a
#' callable taking as argument a scalar integer Tensor `t` and
#' returning a Tensor or LinearOperator representing the transition
#' operator from latent state at time `t` to time `t + 1`.
#' @param transition_noise An instance of
#' `tfd$MultivariateNormalLinearOperator` with event shape
#' `[latent_size]`, representing the mean and covariance of the
#' transition noise model, or a callable taking as argument a
#' scalar integer Tensor `t` and returning such a distribution
#' representing the noise in the transition from time `t` to time `t + 1`.
#' @param observation_matrix An observation operator, represented by a Tensor
#' or LinearOperator of shape `[observation_size, latent_size]`,
#' or by a callable taking as argument a scalar integer Tensor
#' `t` and returning a timestep-specific Tensor or LinearOperator.
#' @param observation_noise An instance of `tfd.MultivariateNormalLinearOperator`
#' with event shape `[observation_size]`, representing the mean and covariance of
#' the observation noise model, or a callable taking as argument
#' a scalar integer Tensor `t` and returning a timestep-specific
#' noise model.
#' @param initial_state_prior An instance of `MultivariateNormalLinearOperator`
#' representing the prior distribution on latent states; must
#' have event shape `[latent_size]`.
#' @param initial_step optional `integer` specifying the time of the first
#' modeled timestep.  This is added as an offset when passing
#' timesteps `t` to (optional) callables specifying
#' timestep-specific transition and observation models.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_linear_gaussian_state_space_model <- function(num_timesteps,
                                                  transition_matrix,
                                                  transition_noise,
                                                  observation_matrix,
                                                  observation_noise,
                                                  initial_state_prior,
                                                  initial_step = 0L,
                                                  validate_args = FALSE,
                                                  allow_nan_stats = TRUE,
                                                  name = "LinearGaussianStateSpaceModel") {
  args <- list(
    num_timesteps = as.integer(num_timesteps),
    transition_matrix = transition_matrix,
    transition_noise = transition_noise,
    observation_matrix = observation_matrix,
    observation_noise = observation_noise,
    initial_state_prior = initial_state_prior,
    initial_step = as.integer(initial_step),
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$LinearGaussianStateSpaceModel,
          args)
}

#' Laplace distribution with location `loc` and `scale` parameters
#'
#' Mathematical details
#'
#' The probability density function (pdf) of this distribution is,
#' ```
#' pdf(x; mu, sigma) = exp(-|x - mu| / sigma) / Z
#' Z = 2 sigma
#' ```
#'
#' where `loc = mu`, `scale = sigma`, and `Z` is the normalization constant.
#'
#' Note that the Laplace distribution can be thought of two exponential
#' distributions spliced together "back-to-back."
#' The Laplace distribution is a member of the [location-scale family](
#' https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
#' constructed as,
#' ```
#' X ~ Laplace(loc=0, scale=1)
#' Y = loc + scale * X
#' ```
#' @param loc Floating point tensor which characterizes the location (center)
#' of the distribution.
#' @param scale Positive floating point tensor which characterizes the spread of
#' the distribution.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_laplace <- function(loc,
                        scale ,
                        validate_args = FALSE,
                        allow_nan_stats = TRUE,
                        name = "Laplace") {
  args <- list(
    loc = loc,
    scale = scale,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Laplace,
          args)
}

#' Kumaraswamy distribution
#'
#' The Kumaraswamy distribution is defined over the `(0, 1)` interval using
#' parameters `concentration1` (aka "alpha") and `concentration0` (aka "beta").  It has a
#' shape similar to the Beta distribution, but is easier to reparameterize.
#'
#' Mathematical Details
#'
#' The probability density function (pdf) is,
#'
#' ```
#' pdf(x; alpha, beta) = alpha * beta * x**(alpha - 1) * (1 - x**alpha)**(beta - 1)
#' ```
#' where:
#' * `concentration1 = alpha`,
#' * `concentration0 = beta`,
#' Distribution parameters are automatically broadcast in all functions.
#' @param concentration1 Positive floating-point `Tensor` indicating mean
#' number of successes; aka "alpha". Implies `self$dtype` and
#' `self$batch_shape`, i.e.,
#' `concentration1$shape = [N1, N2, ..., Nm] = self$batch_shape`.
#' @param concentration0 Positive floating-point `Tensor` indicating mean
#' number of failures; aka "beta". Otherwise has same semantics as
#' `concentration1`.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_kumaraswamy <- function(concentration1 = 1,
                            concentration0 = 1,
                            validate_args = FALSE,
                            allow_nan_stats = TRUE,
                            name = "Kumaraswamy") {
  args <- list(
    concentration1 = concentration1,
    concentration0 = concentration0,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Kumaraswamy,
          args)
}

#' Joint distribution parameterized by distribution-making functions
#'
#' This distribution enables both sampling and joint probability computation from
#' a single model specification.
#'
#' A joint distribution is a collection of possibly interdependent distributions.
#' Like `tf$keras$Sequential`, the `JointDistributionSequential` can be specified
#' via a `list` of functions (each responsible for making a
#' `tfp$distributions$Distribution`-like instance).  Unlike
#' `tf$keras$Sequential`, each function can depend on the output of all previous
#' elements rather than only the immediately previous.
#'
#' Mathematical Details
#'
#' The `JointDistributionSequential` implements the chain rule of probability.
#'
#' That is, the probability function of a length-`d` vector `x` is,
#' ```
#' p(x) = prod{ p(x[i] | x[:i]) : i = 0, ..., (d - 1) }
#' ```
#'
#' The `JointDistributionSequential` is parameterized by a `list` comprised of
#' either:
#' 1. `tfp$distributions$Distribution`-like instances or,
#' 2. `callable`s which return a `tfp$distributions$Distribution`-like instance.
#' Each `list` element implements the `i`-th *full conditional distribution*,
#' `p(x[i] | x[:i])`. The "conditioned on" elements are represented by the
#' `callable`'s required arguments. Directly providing a `Distribution`-like
#'  nstance is a convenience and is semantically identical a zero argument
#' `callable`.
#' Denote the `i`-th `callable`s non-default arguments as `args[i]`. Since the
#' `callable` is the conditional manifest, `0 <= len(args[i]) <= i - 1`. When
#' `len(args[i]) < i - 1`, the `callable` only depends on a subset of the
#' previous distributions, specifically those at indexes:
#' `range(i - 1, i - 1 - num_args[i], -1)`.
#'
#' **Name resolution**: `The names of `JointDistributionSequential` components
#' are defined by explicit `name` arguments passed to distributions
#' (`tfd.Normal(0., 1., name='x')`) and/or by the argument names in
#' distribution-making functions (`lambda x: tfd.Normal(x., 1.)`). Both
#' approaches may be used in the same distribution, as long as they are
#' consistent; referring to a single component by multiple names will raise a
#' `ValueError`. Unnamed components will be assigned a dummy name.
#'
#' @param  model  list of either `tfp$distributions$Distribution` instances and/or
#' functions which take the `k` previous distributions and returns a
#' new `tfp$distributions$Distribution` instance.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_joint_distribution_sequential <- function(model,
                                              validate_args = FALSE,
                                              name = NULL) {

  model <- Map(
    function(d) if (is.function(d)) reticulate::py_func(d) else d,
    model
  )
  args <- list(
    model = model,
    validate_args = validate_args,
    name = name
  )

  do.call(tfp$distributions$JointDistributionSequential,
          args)
}

#' Joint distribution parameterized by named distribution-making functions.
#'
#' This distribution enables both sampling and joint probability computation from
#' a single model specification.
#' A joint distribution is a collection of possibly interdependent distributions.
#' Like `JointDistributionSequential`, `JointDistributionNamed` is parameterized
#' by several distribution-making functions. Unlike `JointDistributionNamed`,
#' each distribution-making function must have its own key. Additionally every
#' distribution-making function's arguments must refer to only specified keys.
#'
#' Mathematical Details
#'
#' Internally `JointDistributionNamed` implements the chain rule of probability.
#' That is, the probability function of a length-`d` vector `x` is,
#'
#' ```
#' p(x) = prod{ p(x[i] | x[:i]) : i = 0, ..., (d - 1) }
#' ```
#'
#' The `JointDistributionNamed` is parameterized by a `dict` (or `namedtuple`)
#' composed of either:
#' 1. `tfp$distributions$Distribution`-like instances or,
#' 2. functions which return a `tfp$distributions$Distribution`-like instance.
#' The "conditioned on" elements are represented by the function's required
#' arguments; every argument must correspond to a key in the named
#' distribution-making functions. Distribution-makers which are directly a
#' `Distribution`-like instance are allowed for convenience and semantically
#' identical a zero argument function. When the maker takes no arguments it is
#' preferable to directly provide the distribution instance.
#'
#' @param model named list of distribution-making functions each
#' with required args corresponding only to other keys in the named list.
#' @param name The name for ops managed by the distribution. Default value: `"JointDistributionNamed"`.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_joint_distribution_named <- function(model,
                                         validate_args = FALSE,
                                         name = NULL) {
  model <- Map(
    function(d) if (is.function(d)) reticulate::py_func(d) else d,
    model
  )

  args <- list(
    model = model,
    validate_args = validate_args,
    name = name
  )

  do.call(tfp$distributions$JointDistributionNamed,
          args)
}

#' Exponential distribution
#'
#' The Exponential distribution is parameterized by an event `rate` parameter.
#'
#' Mathematical Details
#'
#' The probability density function (pdf) is,
#' ```
#' pdf(x; lambda, x > 0) = exp(-lambda x) / Z
#' Z = 1 / lambda
#' ```
#' where `rate = lambda` and `Z` is the normalizing constant.
#'
#' The Exponential distribution is a special case of the Gamma distribution,
#' i.e.,
#' ```
#' Exponential(rate) = Gamma(concentration=1., rate)
#' ```
#'
#' The Exponential distribution uses a `rate` parameter, or "inverse scale",
#' which can be intuited as,
#' ```
#' X ~ Exponential(rate=1)
#' Y = X / rate
#' ```
#'
#' @param rate Floating point tensor, equivalent to `1 / mean`. Must contain only
#' positive values.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_exponential <- function(rate,
                            validate_args = FALSE,
                            allow_nan_stats = TRUE,
                            name = "Exponential") {
  args <- list(
    rate = rate,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Exponential,
          args)
}

#' Gamma distribution
#'
#' The Gamma distribution is defined over positive real numbers using
#' parameters `concentration` (aka "alpha") and `rate` (aka "beta").
#'
#' Mathematical Details
#'
#' The probability density function (pdf) is,
#' ```
#' pdf(x; alpha, beta, x > 0) = x**(alpha - 1) exp(-x beta) / Z
#' Z = Gamma(alpha) beta**(-alpha)
#' ```
#'
#' where
#' * `concentration = alpha`, `alpha > 0`,
#' * `rate = beta`, `beta > 0`,
#' * `Z` is the normalizing constant, and,
#' * `Gamma` is the [gamma function](https://en.wikipedia.org/wiki/Gamma_function).
#'
#' The cumulative density function (cdf) is,
#' ```
#' cdf(x; alpha, beta, x > 0) = GammaInc(alpha, beta x) / Gamma(alpha)
#' ```
#'
#' where `GammaInc` is the [lower incomplete Gamma function](https://en.wikipedia.org/wiki/Incomplete_gamma_function).
#' The parameters can be intuited via their relationship to mean and stddev,
#' ```
#' concentration = alpha = (mean / stddev)**2
#' rate = beta = mean / stddev**2 = concentration / mean
#' ```
#'
#' Distribution parameters are automatically broadcast in all functions; see
#' examples for details.
#'
#' Warning: The samples of this distribution are always non-negative. However,
#' the samples that are smaller than `np$finfo(dtype)$tiny` are rounded
#' to this value, so it appears more often than it should.
#' This should only be noticeable when the `concentration` is very small, or the
#' `rate` is very large. See note in `tf$random_gamma` docstring.
#' Samples of this distribution are reparameterized (pathwise differentiable).
#' The derivatives are computed using the approach described in the paper
#' [Michael Figurnov, Shakir Mohamed, Andriy Mnih. Implicit Reparameterization Gradients, 2018](https://arxiv.org/abs/1805.08498)
#' @param concentration Floating point tensor, the concentration params of the
#' distribution(s). Must contain only positive values.
#' @param rate Floating point tensor, the inverse scale params of the
#' distribution(s). Must contain only positive values.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_gamma <- function(concentration,
                      rate,
                      validate_args = FALSE,
                      allow_nan_stats = TRUE,
                      name = "Gamma") {
  args <- list(
    concentration = concentration,
    rate = rate,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Gamma,
          args)
}

#' Inverse Gaussian distribution
#'
#' The [inverse Gaussian distribution](https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution)
#' is parameterized by a `loc` and a `concentration` parameter. It's also known
#' as the Wald distribution. Some, e.g., the Python scipy package, refer to the
#' special case when `loc` is 1 as the Wald distribution.
#'
#' The "inverse" in the name does not refer to the distribution associated to
#' the multiplicative inverse of a random variable. Rather, the cumulant
#' generating function of this distribution is the inverse to that of a Gaussian
#' random variable.
#'
#' Mathematical Details
#'
#' The probability density function (pdf) is,
#'
#' ```
#' pdf(x; mu, lambda) = [lambda / (2 pi x ** 3)] ** 0.5
#' exp{-lambda(x - mu) ** 2 / (2 mu ** 2 x)}
#' ```
#'
#' where
#' * `loc = mu`
#' * `concentration = lambda`.
#'
#' The support of the distribution is defined on `(0, infinity)`.
#' Mapping to R and Python scipy's parameterization:
#'
#' * R: statmod::invgauss
#' - mean = loc
#' - shape = concentration
#' - dispersion = 1 / concentration. Used only if shape is NULL.
#' * Python: scipy.stats.invgauss
#' - mu = loc / concentration
#' - scale = concentration
#' @param loc Floating-point `Tensor`, the loc params. Must contain only positive values.
#' @param concentration Floating-point `Tensor`, the concentration params. Must contain only positive values.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_inverse_gaussian <- function(loc,
                                 concentration,
                                 validate_args = FALSE,
                                 allow_nan_stats = TRUE,
                                 name = "InverseGaussian") {
  args <- list(
    loc = loc,
    concentration = concentration,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$InverseGaussian,
          args)
}

#' InverseGamma distribution
#'
#' The `InverseGamma` distribution is defined over positive real numbers using
#' parameters `concentration` (aka "alpha") and `scale` (aka "beta").
#'
#' Mathematical Details
#'
#' The probability density function (pdf) is,
#' ```
#' pdf(x; alpha, beta, x > 0) = x**(-alpha - 1) exp(-beta / x) / Z
#' Z = Gamma(alpha) beta**-alpha
#' ```
#'
#' where:
#' * `concentration = alpha`,
#' * `scale = beta`,
#' * `Z` is the normalizing constant, and,
#' * `Gamma` is the [gamma function](https://en.wikipedia.org/wiki/Gamma_function).
#'
#' The cumulative density function (cdf) is,
#' ```
#' cdf(x; alpha, beta, x > 0) = GammaInc(alpha, beta / x) / Gamma(alpha)#' ```
#'
#' where `GammaInc` is the [upper incomplete Gamma function](https://en.wikipedia.org/wiki/Incomplete_gamma_function).
#' The parameters can be intuited via their relationship to mean and variance
#' when these moments exist,
#' ```
#' mean = beta / (alpha - 1) when alpha > 1
#' variance = beta**2 / (alpha - 1)**2 / (alpha - 2)   when alpha > 2
#' ```
#' i.e., under the same conditions:
#' ```
#' alpha = mean**2 / variance + 2
#' beta = mean * (mean**2 / variance + 1)
#' ```
#'
#' Distribution parameters are automatically broadcast in all functions; see
#' examples for details.
#' Samples of this distribution are reparameterized (pathwise differentiable).
#' The derivatives are computed using the approach described in the paper
#' [Michael Figurnov, Shakir Mohamed, Andriy Mnih. Implicit Reparameterization Gradients, 2018](https://arxiv.org/abs/1805.08498)
#' @param concentration Floating point tensor, the concentration params of the
#' distribution(s). Must contain only positive values.
#' @param scale Floating point tensor, the scale params of the distribution(s).
#' Must contain only positive values. This parameter was called `rate` before release 0.8.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_inverse_gamma <- function(concentration,
                              scale,
                              validate_args = FALSE,
                              allow_nan_stats = TRUE,
                              name = "InverseGamma") {
  args <- list(
    concentration = concentration,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )
  if (tfp_version() <= "0.7") args$rate <- scale
  if (tfp_version() > "0.7") args$scale <- scale
  do.call(tfp$distributions$InverseGamma,
          args)
}

#' Horseshoe distribution
#'
#' The so-called 'horseshoe' distribution is a Cauchy-Normal scale mixture,
#' proposed as a sparsity-inducing prior for Bayesian regression. It is
#' symmetric around zero, has heavy (Cauchy-like) tails, so that large
#' coefficients face relatively little shrinkage, but an infinitely tall spike at
#' 0, which pushes small coefficients towards zero. It is parameterized by a
#' positive scalar `scale` parameter: higher values yield a weaker
#' sparsity-inducing effect.
#'
#' Mathematical details
#'
#' The Horseshoe distribution is centered at zero, with scale parameter $lambda$.
#' It is defined by:
#' ```
#'  horseshoe(scale = lambda) ~ Normal(0, lamda * sigma)
#' ```
#' where `sigma ~ half_cauchy(0, 1)`
#'
#'
#' @section References:
#' - [Carvalho, Polson, Scott. Handling Sparsity via the Horseshoe (2008)](http://faculty.chicagobooth.edu/nicholas.polson/research/papers/Horse.pdf).
#' - [Barry, Parlange, Li. Approximation for the exponential integral (2000)](https://doi.org/10.1016/S0022-1694(99)00184-5).
#'
#' @param scale Floating point tensor; the scales of the distribution(s). Must contain only positive values.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_horseshoe <- function(scale,
                          validate_args = FALSE,
                          allow_nan_stats = TRUE,
                          name = "Horseshoe") {
  args <- list(
    scale = scale,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Horseshoe,
          args)
}

#' Hidden Markov model distribution
#'
#' The `HiddenMarkovModel` distribution implements a (batch of) hidden
#' Markov models where the initial states, transition probabilities
#' and observed states are all given by user-provided distributions.
#'
#' This model assumes that the transition matrices are fixed over time.
#' In this model, there is a sequence of integer-valued hidden states:
#' `z[0], z[1], ..., z[num_steps - 1]` and a sequence of observed states:
#' `x[0], ..., x[num_steps - 1]`.
#'
#' The distribution of `z[0]` is given by `initial_distribution`.
#' The conditional probability of `z[i  +  1]` given `z[i]` is described by
#' the batch of distributions in `transition_distribution`.
#' For a batch of hidden Markov models, the coordinates before the rightmost one
#' of the `transition_distribution` batch correspond to indices into the hidden
#' Markov model batch. The rightmost coordinate of the batch is used to select
#' which distribution `z[i + 1]` is drawn from.  The distributions corresponding
#' to the probability of `z[i + 1]` conditional on `z[i] == k` is given by the
#' elements of the batch whose rightmost coordinate is `k`.
#'
#' Similarly, the conditional distribution of `z[i]` given `x[i]` is given by
#' the batch of `observation_distribution`.
#' When the rightmost coordinate of `observation_distribution` is `k` it
#' gives the conditional probabilities of `x[i]` given `z[i] == k`.
#' The probability distribution associated with the `HiddenMarkovModel`
#' distribution is the marginal distribution of `x[0],...,x[num_steps - 1]`.
#'
#' @param initial_distribution A `Categorical`-like instance.
#' Determines probability of first hidden state in Markov chain.
#' The number of categories must match the number of categories of
#' `transition_distribution` as well as both the rightmost batch
#' dimension of `transition_distribution` and the rightmost batch
#' dimension of `observation_distribution`.
#' @param transition_distribution A `Categorical`-like instance.
#' The rightmost batch dimension indexes the probability distribution
#' of each hidden state conditioned on the previous hidden state.
#' @param observation_distribution A `tfp$distributions$Distribution`-like
#' instance.  The rightmost batch dimension indexes the distribution
#' of each observation conditioned on the corresponding hidden state.
#' @param num_steps The number of steps taken in Markov chain. An `integer`.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_hidden_markov_model <- function(initial_distribution,
                                    transition_distribution,
                                    observation_distribution,
                                    num_steps,
                                    validate_args = FALSE,
                                    allow_nan_stats = TRUE,
                                    name = "HiddenMarkovModel") {
  args <- list(
    initial_distribution = initial_distribution,
    transition_distribution = transition_distribution,
    observation_distribution = observation_distribution,
    num_steps = as.integer(num_steps),
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$HiddenMarkovModel,
          args)
}

#' Half-Normal distribution with scale `scale`
#'
#' Mathematical details
#'
#' The half normal is a transformation of a centered normal distribution.
#' If some random variable `X` has normal distribution,
#' ```
#' X ~ Normal(0.0, scale)
#' Y = |X|
#' ```
#'
#' Then `Y` will have half normal distribution. The probability density
#' function (pdf) is:
#'
#' ```
#' pdf(x; scale, x > 0) = sqrt(2) / (scale * sqrt(pi)) * exp(- 1/2 * (x / scale) ** 2))
#' ```
#'
#' Where `scale = sigma` is the standard deviation of the underlying normal
#' distribution.
#' @param scale Floating point tensor; the scales of the distribution(s).
#' Must contain only positive values.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_half_normal <- function(scale,
                            validate_args = FALSE,
                            allow_nan_stats = TRUE,
                            name = "HalfNormal") {
  args <- list(
    scale = scale,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$HalfNormal,
          args)
}

#' Half-Cauchy distribution
#'
#' The half-Cauchy distribution is parameterized by a `loc` and a
#' `scale` parameter. It represents the right half of the two symmetric halves in
#' a [Cauchy distribution](https://en.wikipedia.org/wiki/Cauchy_distribution).
#'
#' Mathematical Details
#'
#' The probability density function (pdf) for the half-Cauchy distribution
#' is given by
#' ```
#' pdf(x; loc, scale) = 2 / (pi scale (1 + z**2))
#' z = (x - loc) / scale
#' ```
#'
#' where `loc` is a scalar in `R` and `scale` is a positive scalar in `R`.
#' The support of the distribution is given by the interval `[loc, infinity)`.
#' @param  loc Floating-point `Tensor`; the location(s) of the distribution(s).
#' @param scale Floating-point `Tensor`; the scale(s) of the distribution(s).
#' Must contain only positive values.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_half_cauchy <- function(loc,
                            scale,
                            validate_args = FALSE,
                            allow_nan_stats = TRUE,
                            name = "HalfCauchy") {
  args <- list(
    loc = loc,
    scale = scale,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$HalfCauchy,
          args)
}

#' Beta distribution
#'
#' The Beta distribution is defined over the `(0, 1)` interval using parameters
#' `concentration1` (aka "alpha") and `concentration0` (aka "beta").
#'
#' Mathematical Details
#'
#' The probability density function (pdf) is,
#'
#' ```
#' pdf(x; alpha, beta) = x**(alpha - 1) (1 - x)**(beta - 1) / Z
#' Z = Gamma(alpha) Gamma(beta) / Gamma(alpha + beta)
#' ```
#'
#' where:
#' * `concentration1 = alpha`,
#' * `concentration0 = beta`,
#' * `Z` is the normalization constant, and,
#' * `Gamma` is the [gamma function](https://en.wikipedia.org/wiki/Gamma_function).
#' The concentration parameters represent mean total counts of a `1` or a `0`,
#' i.e.,
#' ```
#' concentration1 = alpha = mean * total_concentration
#' concentration0 = beta  = (1. - mean) * total_concentration
#' ```
#'
#' where `mean` in `(0, 1)` and `total_concentration` is a positive real number
#' representing a mean `total_count = concentration1 + concentration0`.
#' Distribution parameters are automatically broadcast in all functions; see
#' examples for details.
#' Warning: The samples can be zero due to finite precision.
#' This happens more often when some of the concentrations are very small.
#' Make sure to round the samples to `np$finfo(dtype)$tiny` before computing the density.
#' Samples of this distribution are reparameterized (pathwise differentiable).
#' The derivatives are computed using the approach described in the paper
#' [Michael Figurnov, Shakir Mohamed, Andriy Mnih. Implicit Reparameterization Gradients, 2018](https://arxiv.org/abs/1805.08498)
#' @param concentration1 Positive floating-point `Tensor` indicating mean
#' number of successes; aka "alpha". Implies `self$dtype` and `self$batch_shape`, i.e.,
#' `concentration1$shape = [N1, N2, ..., Nm] = self$batch_shape`.
#' @param concentration0 Positive floating-point `Tensor` indicating mean
#' number of failures; aka "beta". Otherwise has same semantics as `concentration1`.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_beta <- function(concentration1 = NULL,
                     concentration0 = NULL,
                     validate_args = FALSE,
                     allow_nan_stats = TRUE,
                     name = "Beta") {
  args <- list(
    concentration1 = concentration1,
    concentration0 = concentration0,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Beta,
          args)
}

#' Binomial distribution
#'
#' This distribution is parameterized by `probs`, a (batch of) probabilities for
#' drawing a `1` and `total_count`, the number of trials per draw from the
#' Binomial.
#'
#' Mathematical Details
#'
#' The Binomial is a distribution over the number of `1`'s in `total_count`
#' independent trials, with each trial having the same probability of `1`, i.e.,
#' `probs`.
#'
#' The probability mass function (pmf) is,
#' ```
#' pmf(k; n, p) = p**k (1 - p)**(n - k) / Z
#' Z = k! (n - k)! / n!
#' ```
#'
#' where:
#' * `total_count = n`,
#' * `probs = p`,
#' * `Z` is the normalizing constant, and,
#' * `n!` is the factorial of `n`.
#' @param total_count Non-negative floating point tensor with shape broadcastable
#' to `[N1,..., Nm]` with `m >= 0` and the same dtype as `probs` or
#' `logits`. Defines this as a batch of `N1 x ...  x Nm` different Binomial
#' distributions. Its components should be equal to integer values.
#' @param logits Floating point tensor representing the log-odds of a
#' positive event with shape broadcastable to `[N1,..., Nm]` `m >= 0`, and
#' the same dtype as `total_count`. Each entry represents logits for the
#' probability of success for independent Binomial distributions. Only one
#' of `logits` or `probs` should be passed in.
#' @param probs Positive floating point tensor with shape broadcastable to
#' `[N1,..., Nm]` `m >= 0`, `probs in [0, 1]`. Each entry represents the
#' probability of success for independent Binomial distributions. Only one
#' of `logits` or `probs` should be passed in.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_binomial <- function(total_count,
                         logits = NULL,
                         probs = NULL,
                         validate_args = FALSE,
                         allow_nan_stats = TRUE,
                         name = "Beta") {
  args <- list(
    total_count = total_count,
    logits = logits,
    probs = probs,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Binomial,
          args)
}

#' Cauchy distribution with location `loc` and scale `scale`
#'
#' Mathematical details
#'
#' The probability density function (pdf) is,
#' ```
#' pdf(x; loc, scale) = 1 / (pi scale (1 + z**2))
#' z = (x - loc) / scale
#' ```
#'
#' where `loc` is the location, and `scale` is the scale.
#' The Cauchy distribution is a member of the [location-scale family](https://en.wikipedia.org/wiki/Location-scale_family), i.e.
#' `Y ~ Cauchy(loc, scale)` is equivalent to,
#' ```
#' X ~ Cauchy(loc=0, scale=1)
#' Y = loc + scale * X
#' ```
#' @param loc Floating point tensor; the modes of the distribution(s).
#' @param scale Floating point tensor; the locations of the distribution(s).
#' Must contain only positive values.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_cauchy <- function(loc,
                       scale,
                       validate_args = FALSE,
                       allow_nan_stats = TRUE,
                       name = "Cauchy") {
  args <- list(
    loc = loc,
    scale = scale,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Cauchy,
          args)
}

#' Gamma-Gamma distribution
#'
#' Gamma-Gamma is a [compound distribution](https://en.wikipedia.org/wiki/Compound_probability_distribution)
#' defined over positive real numbers using parameters `concentration`,
#' `mixing_concentration` and `mixing_rate`.
#'
#' This distribution is also referred to as the beta of the second kind (B2), and
#' can be useful for transaction value modeling, as in Fader and Hardi, 2013.
#'
#'  Mathematical Details
#'
#'  It is derived from the following Gamma-Gamma hierarchical model by integrating
#'  out the random variable `beta`.
#'
#'  ```
#'  beta ~ Gamma(alpha0, beta0)
#'  X | beta ~ Gamma(alpha, beta)
#'  ```
#'
#'  where
#'  * `concentration = alpha`
#'  * `mixing_concentration = alpha0`
#'  * `mixing_rate = beta0`
#'
#'  The probability density function (pdf) is
#'  ```
#'  x**(alpha - 1)
#'  pdf(x; alpha, alpha0, beta0) =  Z * (x + beta0)**(alpha + alpha0)
#'  ```
#'
#'  where the normalizing constant `Z = Beta(alpha, alpha0) * beta0**(-alpha0)`.
#'  Samples of this distribution are reparameterized as samples of the Gamma
#'  distribution are reparameterized using the technique described in
#'  (Figurnov et al., 2018).
#'
#'  @section References:
#'  - [Peter S. Fader, Bruce G. S. Hardi. The Gamma-Gamma Model of Monetary Value. _Technical Report_, 2013.](http://www.brucehardie.com/notes/025/gamma_gamma.pdf)
#'  - [Michael Figurnov, Shakir Mohamed, Andriy Mnih. Implicit Reparameterization Gradients. _arXiv preprint arXiv:1805.08498_, 2018](https://arxiv.org/abs/1805.08498)
#' @param concentration Floating point tensor, the concentration params of the
#' distribution(s). Must contain only positive values.
#' @param mixing_concentration Floating point tensor, the concentration params of
#' the mixing Gamma distribution(s). Must contain only positive values.
#' @param mixing_rate Floating point tensor, the rate params of the mixing Gamma
#' distribution(s). Must contain only positive values.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_gamma_gamma <- function(concentration,
                            mixing_concentration,
                            mixing_rate,
                            validate_args = FALSE,
                            allow_nan_stats = TRUE,
                            name = "GammaGamma") {
  args <- list(
    concentration = concentration,
    mixing_concentration = mixing_concentration,
    mixing_rate = mixing_rate,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$GammaGamma,
          args)
}

#' Chi distribution
#'
#' The Chi distribution is defined over nonnegative real numbers and uses a
#' degrees of freedom ("df") parameter.
#'
#' Mathematical Details
#'
#' The probability density function (pdf) is,
#' ```
#' pdf(x; df, x >= 0) = x**(df - 1) exp(-0.5 x**2) / Z
#' Z = 2**(0.5 df - 1) Gamma(0.5 df)
#' ```
#'
#' where:
#' * `df` denotes the degrees of freedom,
#' * `Z` is the normalization constant, and,
#' * `Gamma` is the [gamma function](https://en.wikipedia.org/wiki/Gamma_function).
#'
#' The Chi distribution is a transformation of the Chi2 distribution; it is the
#' distribution of the positive square root of a variable obeying a Chi
#' distribution.
#' @param  df Floating point tensor, the degrees of freedom of the distribution(s).
#'  `df` must contain only positive values.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_chi <- function(df,
                    validate_args = FALSE,
                    allow_nan_stats = TRUE,
                    name = "Chi") {
  args <- list(
    df = df,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Chi,
          args)
}

#' Chi Square distribution
#'
#' The Chi2 distribution is defined over positive real numbers using a degrees of
#' freedom ("df") parameter.
#'
#' Mathematical Details
#'
#' The probability density function (pdf) is,
#'
#' ```
#' pdf(x; df, x > 0) = x**(0.5 df - 1) exp(-0.5 x) / Z
#' Z = 2**(0.5 df) Gamma(0.5 df)
#' ```
#' where
#' * `df` denotes the degrees of freedom,
#' * `Z` is the normalization constant, and,
#' * `Gamma` is the [gamma function](https://en.wikipedia.org/wiki/Gamma_function).
#' The Chi2 distribution is a special case of the Gamma distribution, i.e.,
#' ```
#' Chi2(df) = Gamma(concentration=0.5 * df, rate=0.5)
#' ```
#' @param df Floating point tensor, the degrees of freedom of the
#' distribution(s). `df` must contain only positive values.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_chi2 <- function(df,
                     validate_args = FALSE,
                     allow_nan_stats = TRUE,
                     name = "Chi2") {
  args <- list(
    df = df,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Chi2,
          args)
}

#' Scalar Gumbel distribution with location `loc` and `scale` parameters
#'
#' Mathematical details
#'
#' The probability density function (pdf) of this distribution is,
#' ```
#' pdf(x; mu, sigma) = exp(-(x - mu) / sigma - exp(-(x - mu) / sigma)) / sigma
#' ```
#' where `loc = mu` and `scale = sigma`.
#'
#' The cumulative density function of this distribution is,
#' ```cdf(x; mu, sigma) = exp(-exp(-(x - mu) / sigma))```
#'
#' The Gumbel distribution is a member of the [location-scale family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
#' constructed as,
#'
#' ```
#' X ~ Gumbel(loc=0, scale=1)
#' Y = loc + scale * X
#' ```
#' @param  loc Floating point tensor, the means of the distribution(s).
#' @param scale Floating point tensor, the scales of the distribution(s). `scale`` must contain only positive values.
#' @inherit tfd_normal return params
#' @family distributions
#' @export
tfd_gumbel <- function(loc,
                       scale,
                       validate_args = FALSE,
                       allow_nan_stats = TRUE,
                       name = "Gumbel") {
  args <- list(
    loc = loc,
    scale = scale,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Gumbel,
          args)
}

#' Geometric distribution
#'
#' The Geometric distribution is parameterized by p, the probability of a
#' positive event. It represents the probability that in k + 1 Bernoulli trials,
#' the first k trials failed, before seeing a success.
#' The pmf of this distribution is:
#'
#' Mathematical Details
#'
#' ```
#' pmf(k; p) = (1 - p)**k * p
#' ```
#'
#' where:
#' * `p` is the success probability, `0 < p <= 1`, and,
#' * `k` is a non-negative integer.
#' @param logits Floating-point `Tensor` with shape `[B1, ..., Bb]` where `b >= 0`
#' indicates the number of batch dimensions. Each entry represents logits
#' for the probability of success for independent Geometric distributions
#' and must be in the range `(-inf, inf]`. Only one of `logits` or `probs`
#' should be specified.
#' @param  probs Positive floating-point `Tensor` with shape `[B1, ..., Bb]`
#' where `b >= 0` indicates the number of batch dimensions. Each entry
#' represents the probability of success for independent Geometric
#' distributions and must be in the range `(0, 1]`. Only one of `logits`
#' or `probs` should be specified.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_geometric <- function(logits = NULL,
                          probs = NULL,
                          validate_args = FALSE,
                          allow_nan_stats = TRUE,
                          name = "Geometric") {
  args <- list(
    logits = logits,
    probs = probs,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Geometric,
          args)
}

#' Dirichlet distribution
#'
#' The Dirichlet distribution is defined over the
#' [`(k-1)`-simplex](https://en.wikipedia.org/wiki/Simplex) using a positive,
#' length-`k` vector `concentration` (`k > 1`). The Dirichlet is identically the
#' Beta distribution when `k = 2`.
#'
#' Mathematical Details
#'
#' The Dirichlet is a distribution over the open `(k-1)`-simplex, i.e.,
#' ```
#' S^{k-1} = { (x_0, ..., x_{k-1}) in R^k : sum_j x_j = 1 and all_j x_j > 0 }.
#' ```
#'
#' The probability density function (pdf) is,
#' ```
#' pdf(x; alpha) = prod_j x_j**(alpha_j - 1) / Z
#' Z = prod_j Gamma(alpha_j) / Gamma(sum_j alpha_j)
#' ```
#'
#' where:
#'  * `x in S^{k-1}`, i.e., the `(k-1)`-simplex,
#'  * `concentration = alpha = [alpha_0, ..., alpha_{k-1}]`, `alpha_j > 0`,
#'  * `Z` is the normalization constant aka the [multivariate beta function]( https://en.wikipedia.org/wiki/Beta_function#Multivariate_beta_function),
#'  and,
#'  * `Gamma` is the [gamma function](https://en.wikipedia.org/wiki/Gamma_function).
#'
#' The `concentration` represents mean total counts of class occurrence, i.e.,
#' ```
#' concentration = alpha = mean * total_concentration
#'  ```
#' where `mean` in `S^{k-1}` and `total_concentration` is a positive real number
#' representing a mean total count.
#' Distribution parameters are automatically broadcast in all functions; see
#' examples for details.
#' Warning: Some components of the samples can be zero due to finite precision.
#' This happens more often when some of the concentrations are very small.
#' Make sure to round the samples to `np$finfo(dtype)$tiny` before computing the density.
#' Samples of this distribution are reparameterized (pathwise differentiable).
#' The derivatives are computed using the approach described in the paper
#' [Michael Figurnov, Shakir Mohamed, Andriy Mnih. Implicit Reparameterization Gradients, 2018](https://arxiv.org/abs/1805.08498)
#'
#' @param concentration Positive floating-point `Tensor` indicating mean number
#' of class occurrences; aka "alpha". Implies `self$dtype`, and
#' `self$batch_shape`, `self$event_shape`, i.e., if
#' `concentration$shape = [N1, N2, ..., Nm, k]` then
#' `batch_shape = [N1, N2, ..., Nm]` and `event_shape = [k]`.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_dirichlet <- function(concentration,
                          validate_args = FALSE,
                          allow_nan_stats = TRUE,
                          name = "Dirichlet") {
  args <- list(
    concentration = concentration,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Dirichlet,
          args)
}

#' Dirichlet-Multinomial compound distribution
#'
#' The Dirichlet-Multinomial distribution is parameterized by a (batch of)
#' length-`K` `concentration` vectors (`K > 1`) and a `total_count` number of
#' trials, i.e., the number of trials per draw from the DirichletMultinomial. It
#' is defined over a (batch of) length-`K` vector `counts` such that
#' `tf$reduce_sum(counts, -1) = total_count`. The Dirichlet-Multinomial is
#' identically the Beta-Binomial distribution when `K = 2`.
#'
#' Mathematical Details
#'
#' The Dirichlet-Multinomial is a distribution over `K`-class counts, i.e., a
#' length-`K` vector of non-negative integer `counts = n = [n_0, ..., n_{K-1}]`.
#'
#' The probability mass function (pmf) is,
#'
#' ```
#' pmf(n; alpha, N) = Beta(alpha + n) / (prod_j n_j!) / Z
#' Z = Beta(alpha) / N!
#' ```
#'
#' where:
#'
#'  * `concentration = alpha = [alpha_0, ..., alpha_{K-1}]`, `alpha_j > 0`,
#'  * `total_count = N`, `N` a positive integer,
#'  * `N!` is `N` factorial, and,
#'  * `Beta(x) = prod_j Gamma(x_j) / Gamma(sum_j x_j)` is the
#'  [multivariate beta function](https://en.wikipedia.org/wiki/Beta_function#Multivariate_beta_function),
#'  and,
#'  * `Gamma` is the [gamma function](https://en.wikipedia.org/wiki/Gamma_function).
#'
#'  Dirichlet-Multinomial is a [compound distribution](https://en.wikipedia.org/wiki/Compound_probability_distribution), i.e., its
#'  samples are generated as follows.
#'
#'  1. Choose class probabilities:
#'  `probs = [p_0,...,p_{K-1}] ~ Dir(concentration)`
#'  2. Draw integers:
#'  `counts = [n_0,...,n_{K-1}] ~ Multinomial(total_count, probs)`
#'
#'  The last `concentration` dimension parametrizes a single Dirichlet-Multinomial
#'  distribution. When calling distribution functions (e.g., `dist$prob(counts)`),
#'  `concentration`, `total_count` and `counts` are broadcast to the same shape.
#'  The last dimension of `counts` corresponds single Dirichlet-Multinomial distributions.
#'  Distribution parameters are automatically broadcast in all functions; see examples for details.
#'
#'  Pitfalls
#'  The number of classes, `K`, must not exceed:
#'  - the largest integer representable by `self$dtype`, i.e.,
#'    `2**(mantissa_bits+1)` (IEE754),
#'  - the maximum `Tensor` index, i.e., `2**31-1`.
#'
#' Note: This condition is validated only when `validate_args = TRUE`.
#'
#' @param total_count  Non-negative floating point tensor, whose dtype is the same
#' as `concentration`. The shape is broadcastable to `[N1,..., Nm]` with
#' `m >= 0`. Defines this as a batch of `N1 x ... x Nm` different
#' Dirichlet multinomial distributions. Its components should be equal to
#' integer values.
#' @param concentration Positive floating point tensor, whose dtype is the
#' same as `n` with shape broadcastable to `[N1,..., Nm, K]` `m >= 0`.
#' Defines this as a batch of `N1 x ... x Nm` different `K` class Dirichlet
#' multinomial distributions.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_dirichlet_multinomial <- function(total_count,
                                      concentration,
                                      validate_args = FALSE,
                                      allow_nan_stats = TRUE,
                                      name = "DirichletMultinomial") {
  args <- list(
    total_count = total_count,
    concentration = concentration,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$DirichletMultinomial,
          args)
}

#' Scalar `Deterministic` distribution on the real line
#'
#' The scalar `Deterministic` distribution is parameterized by a (batch) point
#' `loc` on the real line.  The distribution is supported at this point only,
#' and corresponds to a random variable that is constant, equal to `loc`.
#' See [Degenerate rv](https://en.wikipedia.org/wiki/Degenerate_distribution).
#'
#' Mathematical Details
#'
#' The probability mass function (pmf) and cumulative distribution function (cdf) are
#' ```
#' pmf(x; loc) = 1, if x == loc, else 0
#' cdf(x; loc) = 1, if x >= loc, else 0
#' ```
#' @param loc Numeric `Tensor` of shape `[B1, ..., Bb]`, with `b >= 0`.
#' The point (or batch of points) on which this distribution is supported.
#' @param atol  Non-negative `Tensor` of same `dtype` as `loc` and broadcastable
#' shape.  The absolute tolerance for comparing closeness to `loc`.
#' Default is `0`.
#' @param rtol  Non-negative `Tensor` of same `dtype` as `loc` and broadcastable
#' shape.  The relative tolerance for comparing closeness to `loc`.
#' Default is `0`.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_deterministic <- function(loc,
                              atol = NULL,
                              rtol = NULL,
                              validate_args = FALSE,
                              allow_nan_stats = TRUE,
                              name = "Deterministic") {
  args <- list(
    loc = loc,
    atol = atol,
    rtol = rtol,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Deterministic,
          args)
}

#' Empirical distribution
#'
#' The Empirical distribution is parameterized by a (batch) multiset of samples.
#' It describes the empirical measure (observations) of a variable.
#' Note: some methods (log_prob, prob, cdf, mode, entropy) are not differentiable
#' with regard to samples.
#'
#' Mathematical Details
#'
#' The probability mass function (pmf) and cumulative distribution function (cdf) are
#' ```
#' pmf(k; s1, ..., sn) = sum_i I(k)^{k == si} / n
#' I(k)^{k == si} == 1, if k == si, else 0.
#' cdf(k; s1, ..., sn) = sum_i I(k)^{k >= si} / n
#' I(k)^{k >= si} == 1, if k >= si, else 0.
#' ```
#' @param samples Numeric `Tensor` of shape `[B1, ..., Bk, S, E1, ..., En]`,
#' `k, n >= 0`. Samples or batches of samples on which the distribution
#' is based. The first `k` dimensions index into a batch of independent
#' distributions. Length of `S` dimension determines number of samples
#' in each multiset. The last `n` dimension represents samples for each
#' distribution. n is specified by argument event_ndims.
#' @param event_ndims `int32`, default `0`. number of dimensions for each
#' event. When `0` this distribution has scalar samples. When `1` this
#' distribution has vector-like samples.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_empirical <- function(samples,
                          event_ndims = 0,
                          validate_args = FALSE,
                          allow_nan_stats = TRUE,
                          name = "Empirical") {
  args <- list(
    samples = samples,
    event_ndims = as.integer(event_ndims),
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Empirical,
          args)
}

#' Batch-Reshaping distribution
#'
#' This "meta-distribution" reshapes the batch dimensions of another distribution.
#' @param distribution The base distribution instance to reshape. Typically an
#' instance of `Distribution`.
#' @param batch_shape Positive `integer`-like vector-shaped `Tensor` representing
#' the new shape of the batch dimensions. Up to one dimension may contain
#' `-1`, meaning the remainder of the batch size.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_batch_reshape <- function(distribution,
                              batch_shape,
                              validate_args = FALSE,
                              allow_nan_stats = TRUE,
                              name = NULL) {
  args <- list(
    distribution = distribution,
    batch_shape = normalize_shape(batch_shape),
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$BatchReshape,
          args)
}

#' Autoregressive distribution
#'
#' The Autoregressive distribution enables learning (often) richer multivariate
#' distributions by repeatedly applying a [diffeomorphic](https://en.wikipedia.org/wiki/Diffeomorphism)
#' transformation (such as implemented by `Bijector`s).
#'
#' Regarding terminology,
#' "Autoregressive models decompose the joint density as a product of
#' conditionals, and model each conditional in turn. Normalizing flows
#' transform a base density (e.g. a standard Gaussian) into the target density
#' by an invertible transformation with tractable Jacobian." (Papamakarios et al., 2016)
#'
#' In other words, the "autoregressive property" is equivalent to the
#' decomposition, `p(x) = prod{ p(x[i] | x[0:i]) : i=0, ..., d }`. The provided
#' `shift_and_log_scale_fn`, `tfb_masked_autoregressive_default_template`, achieves
#' this property by zeroing out weights in its `masked_dense` layers.
#' Practically speaking the autoregressive property means that there exists a
#' permutation of the event coordinates such that each coordinate is a
#' diffeomorphic function of only preceding coordinates
#' (van den Oord et al., 2016).
#'
#' Mathematical Details
#'
#' The probability function is
#' ```
#' prob(x; fn, n) = fn(x).prob(x)
#' ```
#'
#' And a sample is generated by
#' ```
#' x = fn(...fn(fn(x0).sample()).sample()).sample()
#' ```
#'
#' where the ellipses (`...`) represent `n-2` composed calls to `fn`, `fn`
#' constructs a `tfd$Distribution`-like instance, and `x0` is a fixed initializing `Tensor`.
#'
#' @section References:
#'  - [George Papamakarios, Theo Pavlakou, and Iain Murray. Masked Autoregressive Flow for Density Estimation. In _Neural Information Processing Systems_, 2017.](https://arxiv.org/abs/1705.07057)
#'  - [Aaron van den Oord, Nal Kalchbrenner, Oriol Vinyals, Lasse Espeholt, Alex Graves, and Koray Kavukcuoglu. Conditional Image Generation with PixelCNN Decoders. In _Neural Information Processing Systems_, 2016.](https://arxiv.org/abs/1606.05328)
#'
#' @param distribution_fn Function which constructs a `tfd$Distribution`-like instance from a `Tensor`
#' (e.g., `sample0`). The function must respect the "autoregressive property",
#' i.e., there exists a permutation of event such that each coordinate is a
#' diffeomorphic function of on preceding coordinates.
#' @param sample0 Initial input to `distribution_fn`; used to
#' build the distribution in `__init__` which in turn specifies this
#' distribution's properties, e.g., `event_shape`, `batch_shape`, `dtype`.
#' If unspecified, then `distribution_fn` should be default constructable.
#' @param num_steps Number of times `distribution_fn` is composed from samples,
#' e.g., `num_steps=2` implies `distribution_fn(distribution_fn(sample0)$sample(n))$sample()`.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_autoregressive <- function(distribution_fn,
                               sample0 = NULL,
                               num_steps = NULL,
                               validate_args = FALSE,
                               allow_nan_stats = TRUE,
                               name = "Autoregressive") {
  args <- list(
    distribution_fn = distribution_fn,
    sample0 = sample0,
    num_steps = as_nullable_integer(num_steps),
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$Autoregressive,
          args)
}

#' Marginal distribution of a Gaussian process at finitely many points.
#'
#' A Gaussian process (GP) is an indexed collection of random variables, any
#' finite collection of which are jointly Gaussian. While this definition applies
#' to finite index sets, it is typically implicit that the index set is infinite;
#' in applications, it is often some finite dimensional real or complex vector
#' space. In such cases, the GP may be thought of as a distribution over
#' (real- or complex-valued) functions defined over the index set.
#'
#' Just as Gaussian distributions are fully specified by their first and second
#' moments, a Gaussian process can be completely specified by a mean and
#' covariance function.
#' Let `S` denote the index set and `K` the space in which
#' each indexed random variable takes its values (again, often R or C). The mean
#' function is then a map `m: S -> K`, and the covariance function, or kernel, is
#' a positive-definite function `k: (S x S) -> K`. The properties of functions
#' drawn from a GP are entirely dictated (up to translation) by the form of the
#' kernel function.
#'
#' This `Distribution` represents the marginal joint distribution over function
#' values at a given finite collection of points `[x[1], ..., x[N]]` from the
#' index set `S`. By definition, this marginal distribution is just a
#' multivariate normal distribution, whose mean is given by the vector
#' `[ m(x[1]), ..., m(x[N]) ]` and whose covariance matrix is constructed from
#' pairwise applications of the kernel function to the given inputs:
#'
#' ```
#' | k(x[1], x[1])    k(x[1], x[2])  ...  k(x[1], x[N]) |
#' | k(x[2], x[1])    k(x[2], x[2])  ...  k(x[2], x[N]) |
#' |      ...              ...                 ...      |
#' | k(x[N], x[1])    k(x[N], x[2])  ...  k(x[N], x[N]) |
#' ```
#'
#' For this to be a valid covariance matrix, it must be symmetric and positive
#' definite; hence the requirement that `k` be a positive definite function
#' (which, by definition, says that the above procedure will yield PD matrices).
#'
#' We also support the inclusion of zero-mean Gaussian noise in the model, via
#' the `observation_noise_variance` parameter. This augments the generative model
#' to
#'
#' ```
#' f ~ GP(m, k)
#' (y[i] | f, x[i]) ~ Normal(f(x[i]), s)
#' ```
#' where
#' * `m` is the mean function
#' * `k` is the covariance kernel function
#' * `f` is the function drawn from the GP
#' * `x[i]` are the index points at which the function is observed
#' * `y[i]` are the observed values at the index points
#' * `s` is the scale of the observation noise.
#'
#' Note that this class represents an *unconditional* Gaussian process; it does
#' not implement posterior inference conditional on observed function
#' evaluations. This class is useful, for example, if one wishes to combine a GP
#' prior with a non-conjugate likelihood using MCMC to sample from the posterior.
#'
#' Mathematical Details
#'
#' The probability density function (pdf) is a multivariate normal whose
#' parameters are derived from the GP's properties:
#'
#' ```
#' pdf(x; index_points, mean_fn, kernel) = exp(-0.5 * y) / Z
#' K = (kernel.matrix(index_points, index_points) +
#'     (observation_noise_variance + jitter) * eye(N))
#' y = (x - mean_fn(index_points))^T @ K @ (x - mean_fn(index_points))
#' Z = (2 * pi)**(.5 * N) |det(K)|**(.5)
#' ```
#'
#' where:
#' * `index_points` are points in the index set over which the GP is defined,
#' * `mean_fn` is a callable mapping the index set to the GP's mean values,
#' * `kernel` is `PositiveSemidefiniteKernel`-like and represents the covariance
#' function of the GP,
#' * `observation_noise_variance` represents (optional) observation noise.
#' * `jitter` is added to the diagonal to ensure positive definiteness up to
#' machine precision (otherwise Cholesky-decomposition is prone to failure),
#' * `eye(N)` is an N-by-N identity matrix.
#'
#' @param  kernel `PositiveSemidefiniteKernel`-like instance representing the
#' GP's covariance function.
#' @param index_points `float` `Tensor` representing finite (batch of) vector(s) of
#' points in the index set over which the GP is defined. Shape has the
#' form `[b1, ..., bB, e1, f1, ..., fF]` where `F` is the number of feature
#' dimensions and must equal `kernel$feature_ndims` and `e1` is the number
#' (size) of index points in each batch (we denote it `e1` to distinguish
#' it from the numer of inducing index points, denoted `e2` below).
#' Ultimately the GaussianProcess distribution corresponds to an
#' `e1`-dimensional multivariate normal. The batch shape must be
#' broadcastable with `kernel$batch_shape`, the batch shape of
#' `inducing_index_points`, and any batch dims yielded by `mean_fn`.
#' @param mean_fn function that acts on index points to produce a (batch
#' of) vector(s) of mean values at those index points. Takes a `Tensor` of
#' shape `[b1, ..., bB, f1, ..., fF]` and returns a `Tensor` whose shape is
#' (broadcastable with) `[b1, ..., bB]`. Default value: `NULL` implies constant zero function.
#' @param observation_noise_variance `float` `Tensor` representing the variance
#' of the noise in the Normal likelihood distribution of the model. May be
#' batched, in which case the batch shape must be broadcastable with the
#' shapes of all other batched parameters (`kernel$batch_shape`, `index_points`, etc.).
#' Default value: `0.`
#' @param jitter `float` scalar `Tensor` added to the diagonal of the covariance
#' matrix to ensure positive definiteness of the covariance matrix. Default value: `1e-6`.
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_gaussian_process <- function(kernel,
                                 index_points,
                                 mean_fn = NULL,
                                 observation_noise_variance = 0,
                                 jitter = 1e-6,
                                 validate_args = FALSE,
                                 allow_nan_stats = FALSE,
                                 name = "GaussianProcess") {
  args <- list(
    kernel = kernel,
    index_points = index_points,
    mean_fn = mean_fn,
    observation_noise_variance = observation_noise_variance,
    jitter = jitter,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$GaussianProcess, args)
}

#' Posterior predictive distribution in a conjugate GP regression model.
#'
#' @inherit tfd_normal return params
#' @inheritParams tfd_gaussian_process
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#'
#' @param observation_index_points Tensor representing finite collection, or batch
#'  of collections, of points in the index set for which some data has been observed.
#'  Shape has the form \[b1, ..., bB, e, f1, ..., fF\] where F is the number of
#'  feature dimensions and must equal `kernel$feature_ndims`, and e is the number
#'  (size) of index points in each batch. \[b1, ..., bB, e\] must be broadcastable
#'  with the shape of observations, and \[b1, ..., bB\] must be broadcastable with
#'  the shapes of all other batched parameters (kernel.batch_shape, index_points, etc).
#'  The default value is None, which corresponds to the empty set of observations,
#'  and simply results in the prior predictive model (a GP with noise of variance
#'  `predictive_noise_variance`).
#' @param observations Tensor representing collection, or batch of collections,
#'  of observations corresponding to observation_index_points. Shape has the
#'  form \[b1, ..., bB, e\], which must be brodcastable with the batch and example
#'  shapes of observation_index_points. The batch shape \[b1, ..., bB\ ] must be
#'  broadcastable with the shapes of all other batched parameters (kernel.batch_shape,
#'  index_points, etc.). The default value is None, which corresponds to the empty
#'  set of observations, and simply results in the prior predictive model (a GP
#'  with noise of variance `predictive_noise_variance`).
#' @param predictive_noise_variance Tensor representing the variance in the posterior
#'  predictive model. If None, we simply re-use observation_noise_variance for the
#'  posterior predictive noise. If set explicitly, however, we use this value. This
#'  allows us, for example, to omit predictive noise variance (by setting this to zero)
#'  to obtain noiseless posterior predictions of function values, conditioned on noisy
#'  observations.
#' @param mean_fn callable that acts on `index_points` to produce a collection, or
#'  batch of collections, of mean values at index_points. Takes a Tensor of shape
#'  \[b1, ..., bB, f1, ..., fF\] and returns a Tensor whose shape is broadcastable
#'  with \[b1, ..., bB\]. Default value: None implies the constant zero function.
#'
#' @export
tfd_gaussian_process_regression_model <- function(kernel,
                                                  index_points = NULL,
                                                  observation_index_points = NULL,
                                                  observations = NULL,
                                                  observation_noise_variance = 0.0,
                                                  predictive_noise_variance = NULL,
                                                  mean_fn = NULL,
                                                  jitter = 1e-06,
                                                  validate_args = FALSE,
                                                  allow_nan_stats = FALSE,
                                                  name="GaussianProcessRegressionModel") {
  tfp$distributions$GaussianProcessRegressionModel(
    kernel = kernel,
    index_points = index_points,
    observation_index_points = observation_index_points,
    observations = observations,
    observation_noise_variance = observation_noise_variance,
    predictive_noise_variance = predictive_noise_variance,
    mean_fn = mean_fn,
    jitter = jitter,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )
}


#' Sample distribution via independent draws.
#'
#' This distribution is useful for reducing over a collection of independent,
#' identical draws. It is otherwise identical to the input distribution.
#'
#' Mathematical Details
#' The probability function is,
#' ```
#' p(x) = prod{ p(x[i]) : i = 0, ..., (n - 1) }
#' ```
#' @param distribution The base distribution instance to transform. Typically an
#' instance of `Distribution`.
#' @param sample_shape `integer` scalar or vector `Tensor` representing the shape of a
#' single sample.
#' @param name The name for ops managed by the distribution.
#' Default value: `NULL` (i.e., `'Sample' + distribution$name`).
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_sample_distribution <- function(distribution,
                                    sample_shape = list(),
                                    validate_args = FALSE,
                                    name = NULL) {
  args <- list(
    distribution = distribution,
    sample_shape = normalize_shape(sample_shape),
    validate_args = validate_args,
    name = name
  )

  do.call(tfp$distributions$Sample, args)
}

#' Blockwise distribution
#'
#' @inherit tfd_normal return params
#' @param distributions list of Distribution instances. All distribution instances
#'  must have the same batch_shape and all must have `event_ndims==1``, i.e., be
#'  vector-variate distributions.
#' @param dtype_override samples of distributions will be cast to this dtype. If
#'  unspecified, all distributions must have the same dtype. Default value:
#'  `NULL` (i.e., do not cast).
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_blockwise <- function(distributions,
                          dtype_override=NULL,
                          validate_args=FALSE,
                          allow_nan_stats=FALSE,
                          name='Blockwise') {

  tfp$distributions$Blockwise(
    distributions = distributions,
    dtype_override = dtype_override,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )
}

#' Vector Deterministic Distribution
#'
#' The VectorDeterministic distribution is parameterized by a batch point loc in R^k.
#' The distribution is supported at this point only, and corresponds to a random
#' variable that is constant, equal to loc.
#'
#' See [Degenerate rv](https://en.wikipedia.org/wiki/Degenerate_distribution).
#'
#' @inherit tfd_normal return params
#'
#' @param loc Numeric Tensor of shape \[B1, ..., Bb, k\], with b >= 0, k >= 0 The
#'  point (or batch of points) on which this distribution is supported.
#' @param atol Non-negative Tensor of same dtype as loc and broadcastable shape.
#'  The absolute tolerance for comparing closeness to loc. Default is 0.
#' @param rtol Non-negative Tensor of same dtype as loc and broadcastable shape.
#'  The relative tolerance for comparing closeness to loc. Default is 0.
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_vector_deterministic <- function(loc,
                                     atol = NULL,
                                     rtol = NULL,
                                     validate_args = FALSE,
                                     allow_nan_stats = TRUE,
                                     name = 'VectorDeterministic') {

  tfp$distributions$VectorDeterministic(
    loc = loc,
    atol = atol,
    rtol = rtol,
    validate_args = FALSE,
    allow_nan_stats = TRUE,
    name = 'VectorDeterministic'
  )
}

#' ExpRelaxedOneHotCategorical distribution with temperature and logits.
#'
#' @inherit tfd_normal return params
#' @param temperature An 0-D Tensor, representing the temperature of a set of
#'  ExpRelaxedCategorical distributions. The temperature should be positive.
#' @param logits An N-D Tensor, N >= 1, representing the log probabilities of a
#'  set of ExpRelaxedCategorical distributions. The first N - 1 dimensions index
#'  into a batch of independent distributions and the last dimension represents a
#'  vector of logits for each class. Only one of logits or probs should be passed
#'  in.
#' @param probs An N-D Tensor, N >= 1, representing the probabilities of a set of
#'  ExpRelaxedCategorical distributions. The first N - 1 dimensions index into a
#'  batch of independent distributions and the last dimension represents a vector
#'  of probabilities for each class. Only one of logits or probs should be passed
#'  in.
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_exp_relaxed_one_hot_categorical <- function(temperature,
                                                logits = NULL,
                                                probs = NULL,
                                                validate_args = FALSE,
                                                allow_nan_stats = TRUE,
                                                name="ExpRelaxedOneHotCategorical") {
  tfp$distributions$ExpRelaxedOneHotCategorical(
    temperature = temperature,
    logits = logits,
    probs = probs,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )
}

#' Double-sided Maxwell distribution.
#'
#' This distribution is useful to compute measure valued derivatives for Gaussian
#' distributions. See Mohamed et al. (2019) for more details.
#'
#' Mathematical details
#'
#' The double-sided Maxwell distribution generalizes the Maxwell distribution to
#' the entire real line.
#'
#' ```
#' pdf(x; mu, sigma) = 1/(sigma*sqrt(2*pi)) * ((x-mu)/sigma)^2 * exp(-0.5 ((x-mu)/sigma)^2)
#' ```
#'
#' where `loc = mu` and `scale = sigma`.
#' The DoublesidedMaxwell distribution is a member of the
#' [location-scale family](https://en.wikipedia.org/wiki/Location-scale_family),
#' i.e., it can be constructed as,
#'
#' ```
#' X ~ DoublesidedMaxwell(loc=0, scale=1)
#' Y = loc + scale * X
#' ```
#' The double-sided Maxwell is a symmetric distribution that extends the
#' one-sided maxwell from R+ to the entire real line. Their densities are
#' therefore the same up to a factor of 0.5.
#'
#' It has several methods for generating random variates from it. The version
#' here uses 3 Gaussian variates and a uniform variate to generate the samples
#' The sampling path is:
#'
#' ```mu + sigma* sgn(U-0.5)* sqrt(X^2 + Y^2 + Z^2) U~Unif; X,Y,Z ~N(0,1)```
#'
#' In the sampling process above, the random variates generated by
#' sqrt(X^2 + Y^2 + Z^2) are samples from the one-sided Maxwell
#' (or Maxwell-Boltzmann) distribution.
#'
#' @section References:
#' - [Mohamed, et all, "Monte Carlo Gradient Estimation in Machine Learning.",2019](https://arxiv.org/abs/1906.10652)
#' - B. Heidergott, et al "Sensitivity estimation for Gaussian systems", 2008.  European Journal of Operational Research, vol. 187, pp193-207.
#' - G. Pflug. "Optimization of Stochastic Models: The Interface Between Simulation and Optimization", 2002. Chp. 4.2, pg 247.
#'
#' @inherit tfd_normal return params
#' @param loc Floating point tensor; location of the distribution
#' @param scale Floating point tensor; the scales of the distribution.
#' Must contain only positive values.
#' @param name string prefixed to Ops created by this class. Default value: 'doublesided_maxwell'.
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_doublesided_maxwell <- function(loc,
                                    scale,
                                    validate_args = FALSE,
                                    allow_nan_stats = TRUE,
                                    name = "doublesided_maxwell") {
  tfp$distributions$DoublesidedMaxwell(
    loc = loc,
    scale = scale,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )
}

#' The finite discrete distribution.
#'
#' The FiniteDiscrete distribution is parameterized by either probabilities or
#' log-probabilities of a set of `K` possible outcomes, which is defined by
#' a strictly ascending list of `K` values.
#'
#' Note: log_prob, prob, cdf, mode, and entropy are differentiable with respect
#' to `logits` or `probs` but not with respect to `outcomes`.
#'
#' Mathematical Details
#'
#' The probability mass function (pmf) is,
#'
#' ```pmf(x; pi, qi) = prod_j pi_j**[x == qi_j]```
#'
#' @inherit tfd_normal return params
#' @param outcomes A 1-D floating or integer `Tensor`, representing a list of
#' possible outcomes in strictly ascending order.
#' @param logits A floating N-D `Tensor`, `N >= 1`, representing the log
#' probabilities of a set of FiniteDiscrete distributions. The first `N - 1`
#' dimensions index into a batch of independent distributions and the
#' last dimension represents a vector of logits for each discrete value.
#' Only one of `logits` or `probs` should be passed in.
#' @param probs A floating  N-D `Tensor`, `N >= 1`, representing the probabilities
#' of a set of FiniteDiscrete distributions. The first `N - 1` dimensions
#' index into a batch of independent distributions and the last dimension
#' represents a vector of probabilities for each discrete value. Only one
#' of `logits` or `probs` should be passed in.
#' @param rtol `Tensor` with same `dtype` as `outcomes`. The relative tolerance for
#' floating number comparison. Only effective when `outcomes` is a floating
#' `Tensor`. Default is `10 * eps`.
#' @param atol `Tensor` with same `dtype` as `outcomes`. The absolute tolerance for
#' floating number comparison. Only effective when `outcomes` is a floating
#' `Tensor`. Default is `10 * eps`.
#' @param name string prefixed to Ops created by this class.
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_finite_discrete <- function(outcomes,
                                logits = NULL,
                                probs = NULL,
                                rtol = NULL,
                                atol = NULL,
                                validate_args = FALSE,
                                allow_nan_stats = TRUE,
                                name = "FiniteDiscrete") {
  tfp$distributions$FiniteDiscrete(
    outcomes = outcomes,
    logits = logits,
    probs = probs,
    rtol = rtol,
    atol = atol,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )
}

#' The Generalized Pareto distribution.
#'
#' The Generalized Pareto distributions are a family of continuous distributions
#' on the reals. Special cases include `Exponential` (when `loc = 0`,
#' `concentration = 0`), `Pareto` (when `concentration > 0`,
#' `loc = scale / concentration`), and `Uniform` (when `concentration = -1`).
#'
#' This distribution is often used to model the tails of other distributions.
#' As a member of the location-scale family,
#' `X ~ GeneralizedPareto(loc=loc, scale=scale, concentration=conc)` maps to
#' `Y ~ GeneralizedPareto(loc=0, scale=1, concentration=conc)` via
#' `Y = (X - loc) / scale`.
#'
#' For positive concentrations, the distribution is equivalent to a hierarchical
#' Exponential-Gamma model with `X|rate ~ Exponential(rate)` and
#' `rate ~ Gamma(concentration=1 / concentration, scale=scale / concentration)`.
#' In the following, `samps1` and `samps2` are identically distributed:
#'
#' ```
#' genp <- tfd_generalized_pareto(loc = 0, scale = scale, concentration = conc)
#' samps1 <- genp %>% tfd_sample(1000)
#' jd <- tfd_joint_distribution_named(
#'   list(
#'     rate =  tfd_gamma(1 / genp$concentration, genp$scale / genp$concentration),
#'     x = function(rate) tfd_exponential(rate)))
#' samps2 <- jd %>% tfd_sample(1000) %>% .$x
#' ```
#'
#' The support of the distribution is always lower bounded by `loc`. When
#' `concentration < 0`, the support is also upper bounded by
#' `loc + scale / abs(concentration)`.
#'
#' Mathematical Details
#'
#' The probability density function (pdf) is,
#'
#' ```
#' pdf(x; mu, sigma, shp, x > mu) =   (1 + shp * (x - mu) / sigma)**(-1 / shp - 1) / sigma
#' ```
#'
#' where:
#'  * `concentration = shp`, any real value,
#'  * `scale = sigma`, `sigma > 0`,
#'  * `loc = mu`.
#'
#'  The cumulative density function (cdf) is,
#'
#'  ```
#'  cdf(x; mu, sigma, shp, x > mu) = 1 - (1 + shp * (x - mu) / sigma)**(-1 / shp)
#'  ```
#'
#' Distribution parameters are automatically broadcast in all functions; see
#' examples for details.
#' Samples of this distribution are reparameterized (pathwise differentiable).
#'
#' @inherit tfd_normal return params
#' @param loc The location / shift of the distribution. GeneralizedPareto is a
#' location-scale distribution. This parameter lower bounds the
#' distribution's support. Must broadcast with `scale`, `concentration`.
#' Floating point `Tensor`.
#' @param scale The scale of the distribution. GeneralizedPareto is a
#' location-scale distribution, so doubling the `scale` doubles a sample
#' and halves the density. Strictly positive floating point `Tensor`. Must
#' broadcast with `loc`, `concentration`.
#' @param concentration The shape parameter of the distribution. The larger the
#' magnitude, the more the distribution concentrates near `loc` (for
#' `concentration >= 0`) or near `loc - (scale/concentration)` (for
#' `concentration < 0`). Floating point `Tensor`.
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_generalized_pareto <- function(loc,
                                   scale,
                                   concentration,
                                   validate_args = FALSE,
                                   allow_nan_stats = TRUE,
                                   name = NULL) {
  tfp$distributions$GeneralizedPareto(
    loc = loc,
    scale = scale,
    concentration = concentration,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )
}

#' The Logit-Normal distribution
#'
#' The Logit-Normal distribution models positive-valued random variables whose
#' logit (i.e., sigmoid_inverse, i.e., `log(p) - log1p(-p)`) is normally
#' distributed with mean `loc` and standard deviation `scale`. It is
#' constructed as the sigmoid transformation, (i.e., `1 / (1 + exp(-x))`) of a
#' Normal distribution.
#'
#' @inherit tfd_normal return params
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_logit_normal <- function(loc,
                             scale,
                             validate_args = FALSE,
                             allow_nan_stats = TRUE,
                             name = "LogitNormal") {
  tfp$distributions$LogitNormal(
    loc = loc,
    scale = scale,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )
}

#' The LogNormal distribution
#'
#' The LogNormal distribution models positive-valued random variables
#' whose logarithm is normally distributed with mean `loc` and
#' standard deviation `scale`. It is constructed as the exponential
#' transformation of a Normal distribution.
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @inherit tfd_normal return params
#' @export
tfd_log_normal <- function(loc,
                           scale,
                           validate_args = FALSE,
                           allow_nan_stats = TRUE,
                           name = "LogNormal") {
  tfp$distributions$LogNormal(
    loc = loc,
    scale = scale,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )
}

#' Modified PERT distribution for modeling expert predictions.
#'
#' The PERT distribution is a loc-scale family of Beta distributions
#' fit onto a real interval between `low` and `high` values set by the user,
#' along with a `peak` to indicate the expert's most frequent prediction,
#' and `temperature` to control how sharp the peak is.
#'
#' The distribution is similar to a [Triangular distribution](https://en.wikipedia.org/wiki/Triangular_distribution)
#' (i.e. `tfd.Triangular`) but with a smooth peak.
#'
#' Mathematical Details
#'
#' In terms of a Beta distribution, PERT can be expressed as
#' ```
#' PERT ~ loc + scale * Beta(concentration1, concentration0)
#' ```
#' where
#' ```
#' loc = low
#' scale = high - low
#' concentration1 = 1 + temperature * (peak - low)/(high - low)
#' concentration0 = 1 + temperature * (high - peak)/(high - low)
#' temperature > 0
#' ```
#'
#' The support is `[low, high]`.  The `peak` must fit in that interval:
#' `low < peak < high`.  The `temperature` is a positive parameter that
#' controls the shape of the distribution. Higher values yield a sharper peak.
#' The standard PERT distribution is obtained when `temperature = 4`.
#'
#' @param low lower bound
#' @param peak most frequent value
#' @param high upper bound
#' @param temperature controls the shape of the distribution
#' @inherit tfd_normal return params
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_pert <- function(low,
                     peak,
                     high,
                     temperature = 4,
                     validate_args = FALSE,
                     allow_nan_stats = FALSE,
                     name = "Pert") {
  tfp$distributions$PERT(
    low = low,
    peak = peak,
    high = high,
    temperature = temperature,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )
}

#' Plackett-Luce distribution over permutations.
#'
#' The Plackett-Luce distribution is defined over permutations of
#' fixed length. It is parameterized by a positive score vector of same length.
#' This class provides methods to create indexed batches of PlackettLuce
#' distributions. If the provided `scores` is rank 2 or higher, for
#' every fixed set of leading dimensions, the last dimension represents one
#' single PlackettLuce distribution. When calling distribution
#' functions (e.g. `dist.log_prob(x)`), `scores` and `x` are broadcast to the
#' same shape (if possible). In all cases, the last dimension of `scores, x`
#' represents single PlackettLuce distributions.
#'
#' Mathematical Details
#'
#' The Plackett-Luce is a distribution over permutation vectors `p` of length `k`
#' where the permutation `p` is an arbitrary ordering of `k` indices
#' `{0, 1, ..., k-1}`.
#'
#' The probability mass function (pmf) is,
#' ```
#' pmf(p; s) = prod_i s_{p_i} / (Z - Z_i)
#' Z = sum_{j=0}^{k-1} s_j
#' Z_i = sum_{j=0}^{i-1} s_{p_j} for i>0 and 0 for i=0
#' ```
#'
#' where `scores = s = [s_0, ..., s_{k-1}]`, `s_i >= 0`.
#'
#' Samples from Plackett-Luce distribution are generated sequentially as follows.
#'
#' ```
#' Initialize normalization `N_0 = Z`
#' For `i` in `{0, 1, ..., k-1}`
#'   1. Sample i-th element of permutation
#'      `p_i ~ Categorical(probs=[s_0/N_i, ..., s_{k-1}/N_i])`
#'   2. Update normalization
#'      `N_{i+1} = N_i-s_{p_i}`
#'   3. Mask out sampled index for subsequent rounds
#'      `s_{p_i} = 0`
#' Return p
#' ```
#'
#' Alternately, an equivalent way to sample from this distribution is to sort
#' Gumbel perturbed log-scores (Aditya et al. 2019)
#'
#' ```
#' p = argsort(log s + g) ~ PlackettLuce(s)
#' g = [g_0, ..., g_{k-1}], g_i~ Gumbel(0, 1)
#' ```
#'
#' @section References:
#' -  Aditya Grover, Eric Wang, Aaron Zweig, Stefano Ermon. Stochastic Optimization of Sorting Networks via Continuous Relaxations. ICLR 2019.
#'
#' @param scores An N-D `Tensor`, `N >= 1`, representing the scores of a set of
#' elements to be permuted. The first `N - 1` dimensions index into a
#' batch of independent distributions and the last dimension represents a
#' vector of scores for the elements.
#' @param dtype The type of the event samples (default: int32).
#' @inherit tfd_normal return params
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_plackett_luce <- function(scores,
                              dtype = tf$int32,
                              validate_args = FALSE,
                              allow_nan_stats = FALSE,
                              name = "PlackettLuce") {
  tfp$distributions$PlackettLuce(
    scores = scores,
    dtype = dtype,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )
}

#' ProbitBernoulli distribution.
#'
#' The ProbitBernoulli distribution with `probs` parameter, i.e., the probability
#' of a `1` outcome (vs a `0` outcome). Unlike a regular Bernoulli distribution,
#' which uses the logistic (aka 'sigmoid') function to go from the un-constrained
#' parameters to probabilities, this distribution uses the CDF of the
#' [standard normal distribution](https://en.wikipedia.org/wiki/Normal_distribution):
#' ```
#' p(x=1; probits) = 0.5 * (1 + erf(probits / sqrt(2)))
#' p(x=0; probits) = 1 - p(x=1; probits)
#' ```
#' Where `erf` is the [error function](https://en.wikipedia.org/wiki/Error_function).
#' A typical application of this distribution is in
#' [probit  regression](https://en.wikipedia.org/wiki/Probit_model).
#' @inherit tfd_normal return params
#'
#' @param probits An N-D `Tensor` representing the probit-odds of a `1` event. Each
#' entry in the `Tensor` parameterizes an independent ProbitBernoulli
#' distribution where the probability of an event is normal_cdf(probits).
#' Only one of `probits` or `probs` should be passed in.
#' @param probs An N-D `Tensor` representing the probability of a `1`
#' event. Each entry in the `Tensor` parameterizes an independent
#' ProbitBernoulli distribution. Only one of `probits` or `probs` should be
#' passed in.
#' @param dtype The type of the event samples. Default: `int32`.
#'
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_probit_bernoulli <- function(probits = NULL,
                                 probs = NULL,
                                 dtype = tf$int32,
                                 validate_args = FALSE,
                                 allow_nan_stats = TRUE,
                                 name = "ProbitBernoulli") {
  args <- list(
    probits = probits,
    probs = probs,
    dtype = dtype,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$ProbitBernoulli, args)
}

#' The matrix Wishart distribution on positive definite matrices
#'
#' This distribution is defined by a scalar number of degrees of freedom df and
#' an instance of LinearOperator, which provides matrix-free access to a
#' symmetric positive definite operator, which defines the scale matrix.
#'
#' Mathematical Details
#'
#' The probability density function (pdf) is,
#' ```
#' pdf(X; df, scale) = det(X)**(0.5 (df-k-1)) exp(-0.5 tr[inv(scale) X]) / Z
#' Z = 2**(0.5 df k) |det(scale)|**(0.5 df) Gamma_k(0.5 df)
#' ```
#' where:
#' * `df >= k` denotes the degrees of freedom,
#' * `scale` is a symmetric, positive definite, `k x k` matrix,
#' * `Z` is the normalizing constant, and,
#' * `Gamma_k` is the [multivariate Gamma function](
#'  https://en.wikipedia.org/wiki/Multivariate_gamma_function).
#'
#' @param df float or double tensor, the degrees of freedom of the
#' distribution(s). df must be greater than or equal to k.
#' @param scale `float` or `double` instance of `LinearOperator`.
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
#' @inherit tfd_normal return params
#'
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_wishart_linear_operator <- function(df,
                                        scale,
                                        input_output_cholesky = FALSE,
                                        validate_args = FALSE,
                                        allow_nan_stats = TRUE,
                                        name = "WishartLinearOperator") {
  args <- list(
    df = df,
    scale = scale,
    input_output_cholesky = input_output_cholesky,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$WishartLinearOperator, args)
}

#' The matrix Wishart distribution parameterized with Cholesky factors.
#'
#' This distribution is defined by a scalar degrees of freedom `df` and a scale
#' matrix, expressed as a lower triangular Cholesky factor.
#'
#' Mathematical Details
#'
#' The probability density function (pdf) is,
#' ```
#' pdf(X; df, scale) = det(X)**(0.5 (df-k-1)) exp(-0.5 tr[inv(scale) X]) / Z
#' Z = 2**(0.5 df k) |det(scale)|**(0.5 df) Gamma_k(0.5 df)
#' ```
#' where:
#' * `df >= k` denotes the degrees of freedom,
#' * `scale` is a symmetric, positive definite, `k x k` matrix,
#' * `Z` is the normalizing constant, and,
#' * `Gamma_k` is the [multivariate Gamma function](
#'  https://en.wikipedia.org/wiki/Multivariate_gamma_function).
#'
#' @param df float or double tensor, the degrees of freedom of the
#' distribution(s). df must be greater than or equal to k.
#' @param scale_tril `float` or `double` `Tensor`. The Cholesky factorization
#' of the symmetric positive definite scale matrix of the distribution.
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
#' @inherit tfd_normal return params
#'
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_wishart_tri_l <- function(df,
                              scale_tril,
                              input_output_cholesky = FALSE,
                              validate_args = FALSE,
                              allow_nan_stats = TRUE,
                              name = "WishartTriL") {
  args <- list(
    df = df,
    scale_tril = scale_tril,
    input_output_cholesky = input_output_cholesky,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$WishartTriL, args)
}

#' The Pixel CNN++ distribution
#'
#' Pixel CNN++ (Salimans et al., 2017) models a distribution over image
#' data, parameterized by a neural network. It builds on Pixel CNN and
#' Conditional Pixel CNN, as originally proposed by
#' (van den Oord et al., 2016).
#' The model expresses the joint distribution over pixels as
#' the product of conditional distributions:
#' `p(x|h) = prod{ p(x[i] | x[0:i], h) : i=0, ..., d }`, in which
#' `p(x[i] | x[0:i], h) : i=0, ..., d` is the
#' probability of the `i`-th pixel conditional on the pixels that preceded it in
#' raster order (color channels in RGB order, then left to right, then top to
#' bottom). `h` is optional additional data on which to condition the image
#' distribution, such as class labels or VAE embeddings. The Pixel CNN++
#' network enforces the dependency structure among pixels by applying a mask to
#' the kernels of the convolutional layers that ensures that the values for each
#' pixel depend only on other pixels up and to the left.
#' Pixel values are modeled with a mixture of quantized logistic distributions,
#' which can take on a set of distinct integer values (e.g. between 0 and 255
#' for an 8-bit image).
#' Color intensity `v` of each pixel is modeled as:
#' `v ~ sum{q[i] * quantized_logistic(loc[i], scale[i]) : i = 0, ..., k }`,
#' in which `k` is the number of mixture components and the `q[i]` are the
#' Categorical probabilities over the components.
#'
#' @section References:
#' - [Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P. Kingma. PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications. In _International Conference on Learning Representations_, 2017.](https://pdfs.semanticscholar.org/9e90/6792f67cbdda7b7777b69284a81044857656.pdf)
#' - [Aaron van den Oord, Nal Kalchbrenner, Oriol Vinyals, Lasse Espeholt, Alex Graves, and Koray Kavukcuoglu. Conditional Image Generation with PixelCNN Decoders. In _Neural Information Processing Systems_, 2016.](https://arxiv.org/abs/1606.05328)
#' - [Aaron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel Recurrent Neural Networks. In _International Conference on Machine Learning_, 2016.](https://arxiv.org/pdf/1601.06759.pdf)
#'
#' @param image_shape 3D `TensorShape` or tuple for the `[height, width, channels]`
#' dimensions of the image.
#' @param conditional_shape `TensorShape` or tuple for the shape of the
#' conditional input, or `NULL` if there is no conditional input.
#' @param num_resnet `integer`, the number of layers (shown in Figure 2 of https://arxiv.org/abs/1606.05328)
#' within each highest-level block of Figure 2 of https://pdfs.semanticscholar.org/9e90/6792f67cbdda7b7777b69284a81044857656.pdf.
#' @param num_hierarchies `integer`, the number of hightest-level blocks (separated by
#' expansions/contractions of dimensions in Figure 2 of https://pdfs.semanticscholar.org/9e90/6792f67cbdda7b7777b69284a81044857656.pdf.)
#' @param num_filters `integer`, the number of convolutional filters.
#' @param num_logistic_mix `integer`, number of components in the logistic mixture
#' distribution.
#' @param receptive_field_dims `tuple`, height and width in pixels of the receptive
#' field of the convolutional layers above and to the left of a given
#' pixel. The width (second element of the tuple) should be odd. Figure 1
#' (middle) of https://arxiv.org/abs/1606.05328 shows a receptive field of (3, 5)
#' (the row containing the current pixel is included in the height).
#' The default of (3, 3) was used to produce the results in https://pdfs.semanticscholar.org/9e90/6792f67cbdda7b7777b69284a81044857656.pdf.
#' @param dropout_p `float`, the dropout probability. Should be between 0 and 1.
#' @param resnet_activation `string`, the type of activation to use in the resnet blocks.
#' May be 'concat_elu', 'elu', or 'relu'.
#' @param use_weight_norm `logical`, if `TRUE` then use weight normalization (works
#' only in Eager mode).
#' @param use_data_init `logical`, if `TRUE` then use data-dependent initialization
#' (has no effect if `use_weight_norm` is `FALSE`).
#' @param high `integer`, the maximum value of the input data (255 for an 8-bit image).
#' @param low `integer`, the minimum value of the input data.
#' @param dtype Data type of the `Distribution`.
#' @param name `string`, the name of the `Distribution`.
#' @inherit tfd_normal return
#'
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_pixel_cnn <- function(image_shape,
                          conditional_shape = NULL,
                          num_resnet = 5,
                          num_hierarchies = 3,
                          num_filters = 160,
                          num_logistic_mix = 10,
                          receptive_field_dims = c(3, 3),
                          dropout_p = 0.5,
                          resnet_activation = 'concat_elu',
                          use_weight_norm = TRUE,
                          use_data_init = TRUE,
                          high = 255,
                          low = 0,
                          dtype = tf$float32,
                          name = 'PixelCNN') {
  args <- list(
    image_shape = normalize_shape(image_shape),
    conditional_shape = normalize_shape(conditional_shape),
    num_resnet = as.integer(num_resnet),
    num_hierarchies = as.integer(num_hierarchies),
    num_filters = as.integer(num_filters),
    num_logistic_mix = as.integer(num_logistic_mix),
    receptive_field_dims = normalize_shape(receptive_field_dims),
    dropout_p = dropout_p,
    resnet_activation = resnet_activation,
    use_weight_norm = use_weight_norm,
    use_data_init = use_data_init,
    high = as.integer(high),
    low = as.integer(low),
    dtype = dtype,
    name = name
  )

  do.call(tfp$distributions$PixelCNN, args)
}

#' Beta-Binomial compound distribution
#'
#' The Beta-Binomial distribution is parameterized by (a batch of) `total_count`
#' parameters, the number of trials per draw from Binomial distributions where
#' the probabilities of success per trial are drawn from underlying Beta
#' distributions; the Beta distributions are parameterized by `concentration1`
#' (aka 'alpha') and `concentration0` (aka 'beta').
#' Mathematically, it is (equivalent to) a special case of the
#' Dirichlet-Multinomial over two classes, although the computational
#' representation is slightly different: while the Beta-Binomial is a
#' distribution over the number of successes in `total_count` trials, the
#' two-class Dirichlet-Multinomial is a distribution over the number of successes
#' and failures.
#'
#' Mathematical Details
#'
#' The Beta-Binomial is a distribution over the number of successes in
#' `total_count` independent Binomial trials, with each trial having the same
#' probability of success, the underlying probability being unknown but drawn
#' from a Beta distribution with known parameters.
#' The probability mass function (pmf) is,
#'
#' ```
#' pmf(k; n, a, b) = Beta(k + a, n - k + b) / Z
#' Z = (k! (n - k)! / n!) * Beta(a, b)
#' ```
#'
#' where:
#' * `concentration1 = a > 0`,
#' * `concentration0 = b > 0`,
#' * `total_count = n`, `n` a positive integer,
#' * `n!` is `n` factorial,
#' * `Beta(x, y) = Gamma(x) Gamma(y) / Gamma(x + y)` is the
#' [beta function](https://en.wikipedia.org/wiki/Beta_function), and
#' * `Gamma` is the [gamma function](https://en.wikipedia.org/wiki/Gamma_function).
#'
#' Dirichlet-Multinomial is a [compound distribution](https://en.wikipedia.org/wiki/Compound_probability_distribution),
#' i.e., its samples are generated as follows.
#'
#' 1. Choose success probabilities:
#'   `probs ~ Beta(concentration1, concentration0)`
#' 2. Draw integers representing the number of successes:
#'   `counts ~ Binomial(total_count, probs)`
#' Distribution parameters are automatically broadcast in all functions; see
#' examples for details.
#'
#' @param  total_count Non-negative integer-valued tensor, whose dtype is the same
#' as `concentration1` and `concentration0`. The shape is broadcastable to
#' `[N1,..., Nm]` with `m >= 0`. When `total_count` is broadcast with
#' `concentration1` and `concentration0`, it defines the distribution as a
#' batch of `N1 x ... x Nm` different Beta-Binomial distributions. Its
#' components should be equal to integer values.
#' @param concentration1 Positive floating-point `Tensor` indicating mean number of
#' successes. Specifically, the expected number of successes is
#' `total_count * concentration1 / (concentration1 + concentration0)`.
#' @param concentration0 Positive floating-point `Tensor` indicating mean number of
#' failures; see description of `concentration1` for details.
#'
#' @inherit tfd_normal return params
#' @family distributions
#' @seealso For usage examples see e.g. [tfd_sample()], [tfd_log_prob()], [tfd_mean()].
#' @export
tfd_beta_binomial <- function(total_count,
                              concentration1,
                              concentration0,
                              validate_args = FALSE,
                              allow_nan_stats = TRUE,
                              name = "BetaBinomial") {
  args <- list(
    total_count = total_count,
    concentration1 = concentration1,
    concentration0 = concentration0,
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )

  do.call(tfp$distributions$BetaBinomial,
          args)
}

