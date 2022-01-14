#' A d-variate Multivariate Normal TriL Keras layer from `d+d*(d+1)/ 2` params
#'
#' @inheritParams keras::layer_dense
#' @inheritParams layer_distribution_lambda
#'
#' @param event_size Integer vector tensor representing the shape of single draw from this distribution.
#' @param validate_args  Logical, default FALSE. When TRUE distribution parameters are checked
#'  for validity despite possibly degrading runtime performance. When FALSE invalid inputs may
#'  silently render incorrect outputs. Default value: FALSE.
#'
#' @return a Keras layer
#' @family distribution_layers
#' @seealso For an example how to use in a Keras model, see [layer_independent_normal()].
#' @export
layer_multivariate_normal_tri_l <- function(object,
                                            event_size,
                                            convert_to_tensor_fn = tfp$distributions$Distribution$sample,
                                            validate_args = FALSE,
                                            ...) {
  args <- list(
    event_size = as.integer(event_size),
    convert_to_tensor_fn = convert_to_tensor_fn,
    validate_args = validate_args,
    ...
  )

  create_layer(
    tfp$python$layers$distribution_layer$MultivariateNormalTriL,
    object,
    args
  )
}

#' An Independent-Bernoulli Keras layer from prod(event_shape) params
#'
#' @inheritParams keras::layer_dense
#' @inheritParams layer_distribution_lambda
#'
#' @param event_shape Scalar integer representing the size of single draw from this distribution.
#' @param sample_dtype dtype of samples produced by this distribution.
#'  Default value: NULL (i.e., previous layer's dtype).
#' @param validate_args  Logical, default FALSE. When TRUE distribution parameters are checked
#'  for validity despite possibly degrading runtime performance. When FALSE invalid inputs may
#'  silently render incorrect outputs. Default value: FALSE.
#'  @param ... Additional arguments passed to `args` of `keras::create_layer`.
#'
#' @return a Keras layer
#' @family distribution_layers
#' @seealso For an example how to use in a Keras model, see [layer_independent_normal()].
#'
#' @export
layer_independent_bernoulli <- function(object,
                                        event_shape,
                                        convert_to_tensor_fn = tfp$distributions$Distribution$sample,
                                        sample_dtype = NULL,
                                        validate_args = FALSE,
                                        ...) {
  args <- list(
    event_shape = as.integer(event_shape),
    convert_to_tensor_fn = convert_to_tensor_fn,
    sample_dtype = sample_dtype,
    validate_args = validate_args,
    ...
  )

  create_layer(
    tfp$python$layers$distribution_layer$IndependentBernoulli,
    object,
    args
  )
}

#' Keras layer enabling plumbing TFP distributions through Keras models
#'
#' @inheritParams keras::layer_dense
#'
#' @param make_distribution_fn A callable that takes previous layer outputs and returns a `tfd$distributions$Distribution` instance.
#' @param convert_to_tensor_fn A callable that takes a tfd$Distribution instance and returns a
#'  tf$Tensor-like object. Default value: `tfd$distributions$Distribution$sample`.
#' @param ... Additional arguments passed to `args` of `keras::create_layer`.
#' @return a Keras layer
#' @family distribution_layers
#' @seealso For an example how to use in a Keras model, see [layer_independent_normal()].
#' @export
layer_distribution_lambda <- function(object,
                                      make_distribution_fn,
                                      convert_to_tensor_fn = tfp$distributions$Distribution$sample,
                                      ...) {
  args <- list(
    make_distribution_fn = make_distribution_fn,
    convert_to_tensor_fn = convert_to_tensor_fn,
    ...
  )

  create_layer(
    tfp$python$layers$distribution_layer$DistributionLambda,
    object,
    args
  )
}

#' Pass-through layer that adds a KL divergence penalty to the model loss
#'
#' @inheritParams keras::layer_dense
#'
#' @param distribution_b Distribution instance corresponding to b as in  `KL[a, b]`.
#'  The previous layer's output is presumed to be a Distribution instance and is a.
#' @param use_exact_kl Logical indicating if KL divergence should be
#'  calculated exactly via `tfp$distributions$kl_divergence` or via Monte Carlo approximation.
#'  Default value: FALSE.
#' @param test_points_reduce_axis Integer vector or scalar representing dimensions
#'  over which to reduce_mean while calculating the Monte Carlo approximation of the KL divergence.
#'  As is with all tf$reduce_* ops, NULL means reduce over all dimensions;
#'  () means reduce over none of them. Default value: () (i.e., no reduction).
#' @param test_points_fn A callable taking a `tfp$distributions$Distribution` instance and returning a tensor
#'  used for random test points to approximate the KL divergence.
#'  Default value: tf$convert_to_tensor.
#' @param weight Multiplier applied to the calculated KL divergence for each Keras batch member.
#' Default value: NULL (i.e., do not weight each batch member).
#' @param ... Additional arguments passed to `args` of `keras::create_layer`.

#' @return a Keras layer
#' @family distribution_layers
#' @seealso For an example how to use in a Keras model, see [layer_independent_normal()].
#' @export
layer_kl_divergence_add_loss <- function(object,
                                         distribution_b,
                                         use_exact_kl = FALSE,
                                         test_points_reduce_axis = NULL,
                                         test_points_fn = tf$convert_to_tensor,
                                         weight = NULL,
                                         ...) {
  args <- list(
    distribution_b = distribution_b,
    use_exact_kl = use_exact_kl,
    test_points_reduce_axis = test_points_reduce_axis,
    test_points_fn = test_points_fn,
    weight = weight,
    ...
  )

  create_layer(
    tfp$python$layers$distribution_layer$KLDivergenceAddLoss,
    object,
    args
  )
}

#' Regularizer that adds a KL divergence penalty to the model loss
#'
#' When using Monte Carlo approximation (e.g., `use_exact = FALSE`), it is presumed that the input
#'   distribution's concretization (i.e., `tf$convert_to_tensor(distribution)`) corresponds to a random
#'   sample. To override this behavior, set test_points_fn.
#'
#' @inheritParams keras::layer_dense
#' @inheritParams layer_kl_divergence_add_loss
#' @return a Keras layer
#'
#' @family distribution_layers
#' @seealso For an example how to use in a Keras model, see [layer_independent_normal()].
#' @export
layer_kl_divergence_regularizer <- function(object,
                                            distribution_b,
                                            use_exact_kl = FALSE,
                                            test_points_reduce_axis = NULL,
                                            test_points_fn = tf$convert_to_tensor,
                                            weight = NULL,
                                            ...) {
  args <- list(
    distribution_b,
    use_exact_kl = use_exact_kl,
    test_points_reduce_axis = test_points_reduce_axis,
    test_points_fn = test_points_fn,
    weight = weight,
    ...
  )

  create_layer(
    tfp$python$layers$distribution_layer$KLDivergenceRegularizer,
    object,
    args
  )
}

#' A `d`-variate OneHotCategorical Keras layer from `d` params.
#'
#' Typical choices for `convert_to_tensor_fn` include:
#' - `tfp$distributions$Distribution$sample`
#' - `tfp$distributions$Distribution$mean`
#' - `tfp$distributions$Distribution$mode`
#' - `tfp$distributions$OneHotCategorical$logits`
#'
#' @param event_size Scalar `integer` representing the size of single draw from this distribution.
#' @param sample_dtype `dtype` of samples produced by this distribution.
#'  Default value: `NULL` (i.e., previous layer's `dtype`).
#' @inheritParams keras::layer_dense
#' @inheritParams layer_distribution_lambda
#' @inheritParams layer_multivariate_normal_tri_l
#' @return a Keras layer
#' @family distribution_layers
#' @seealso For an example how to use in a Keras model, see [layer_independent_normal()].
#' @export
layer_one_hot_categorical <- function(object,
                                      event_size,
                                      convert_to_tensor_fn = tfp$distributions$Distribution$sample,
                                      sample_dtype = NULL,
                                      validate_args = FALSE,
                                      ...) {
  args <- list(
    event_size = as.integer(event_size),
    convert_to_tensor_fn = convert_to_tensor_fn,
    sample_dtype = sample_dtype,
    validate_args = validate_args,
    ...
  )

  create_layer(
    tfp$python$layers$distribution_layer$OneHotCategorical,
    object,
    args
  )
}

#' A OneHotCategorical mixture Keras layer from `k * (1 + d)` params.
#'
#' `k` (i.e., `num_components`) represents the number of component
#' `OneHotCategorical` distributions and `d` (i.e., `event_size`) represents the
#' number of categories within each `OneHotCategorical` distribution.
#'
#' Typical choices for `convert_to_tensor_fn` include:
#' - `tfp$distributions$Distribution$sample`
#' - `tfp$distributions$Distribution$mean`
#' - `tfp$distributions$Distribution$mode`
#'
#' @param num_components Scalar `integer` representing the number of mixture
#' components. Must be at least 1. (If `num_components=1`, it's more
#' efficient to use the `OneHotCategorical` layer.)
#' @inheritParams keras::layer_dense
#' @inheritParams layer_one_hot_categorical
#' @return a Keras layer
#' @family distribution_layers
#' @seealso For an example how to use in a Keras model, see [layer_independent_normal()].
#' @export
layer_categorical_mixture_of_one_hot_categorical <- function(object,
                                                             event_size,
                                                             num_components,
                                                             convert_to_tensor_fn = tfp$distributions$Distribution$sample,
                                                             sample_dtype = NULL,
                                                             validate_args = FALSE,
                                                             ...) {
  args <- list(
    event_size = as.integer(event_size),
    num_components = as.integer(num_components),
    convert_to_tensor_fn = convert_to_tensor_fn,
    sample_dtype = sample_dtype,
    validate_args = validate_args,
    ...
  )

  create_layer(
    tfp$python$layers$distribution_layer$CategoricalMixtureOfOneHotCategorical,
    object,
    args
  )
}

#' An independent Poisson Keras layer.
#'
#' @inheritParams keras::layer_dense
#' @inheritParams layer_independent_bernoulli
#' @return a Keras layer
#' @family distribution_layers
#' @seealso For an example how to use in a Keras model, see [layer_independent_normal()].
#' @export
layer_independent_poisson <- function(object,
                                      event_shape,
                                      convert_to_tensor_fn = tfp$distributions$Distribution$sample,
                                      validate_args = FALSE,
                                      ...) {
  args <- list(
    event_shape = as.integer(event_shape),
    convert_to_tensor_fn = convert_to_tensor_fn,
    validate_args = validate_args,
    ...
  )

  create_layer(tfp$python$layers$distribution_layer$IndependentPoisson,
               object,
               args)
}

#' An independent Logistic Keras layer.
#'
#' @inheritParams keras::layer_dense
#' @inheritParams layer_independent_bernoulli
#' @return a Keras layer
#' @family distribution_layers
#' @seealso For an example how to use in a Keras model, see [layer_independent_normal()].
#' @export
layer_independent_logistic <- function(object,
                                       event_shape,
                                       convert_to_tensor_fn = tfp$distributions$Distribution$sample,
                                       validate_args = FALSE,
                                       ...) {
  args <- list(
    event_shape = as.integer(event_shape),
    convert_to_tensor_fn = convert_to_tensor_fn,
    validate_args = validate_args,
    ...
  )

  create_layer(tfp$python$layers$distribution_layer$IndependentLogistic,
               object,
               args)
}


#' An independent Normal Keras layer.
#'
#' @inheritParams keras::layer_dense
#' @inheritParams layer_independent_bernoulli
#' @return a Keras layer
#' @family distribution_layers
#' @examples
#' \donttest{
#' library(keras)
#' input_shape <- c(28, 28, 1)
#' encoded_shape <- 2
#' n <- 2
#' model <- keras_model_sequential(
#'   list(
#'     layer_input(shape = input_shape),
#'     layer_flatten(),
#'     layer_dense(units = n),
#'     layer_dense(units = params_size_independent_normal(encoded_shape)),
#'     layer_independent_normal(event_shape = encoded_shape)
#'     )
#'   )
#' }
#' @export
layer_independent_normal <- function(object,
                                     event_shape,
                                     convert_to_tensor_fn = tfp$distributions$Distribution$sample,
                                     validate_args = FALSE,
                                     ...) {
  args <- list(
    event_shape = as.integer(event_shape),
    convert_to_tensor_fn = convert_to_tensor_fn,
    validate_args = validate_args,
    ...
  )

  create_layer(tfp$python$layers$distribution_layer$IndependentNormal,
               object,
               args)
}

#' A mixture (same-family) Keras layer.
#'
#' @param num_components Number of component distributions in the mixture distribution.
#' @param component_layer Function that, given a tensor of shape
#' `batch_shape + [num_components, component_params_size]`, returns a
#' `tfd.Distribution`-like instance that implements the component
#' distribution (with batch shape `batch_shape + [num_components]`) --
#' e.g., a TFP distribution layer.
#' @inheritParams keras::layer_dense
#' @inheritParams layer_independent_bernoulli
#' @return a Keras layer
#' @family distribution_layers
#' @seealso For an example how to use in a Keras model, see [layer_independent_normal()].
#' @export
layer_mixture_same_family <- function(object,
                                     num_components,
                                     component_layer,
                                     convert_to_tensor_fn = tfp$distributions$Distribution$sample,
                                     validate_args = FALSE,
                                     ...) {
  args <- list(
    num_components = as.integer(num_components),
    component_layer = component_layer,
    convert_to_tensor_fn = convert_to_tensor_fn,
    validate_args = validate_args,
    ...
  )

  create_layer(tfp$python$layers$distribution_layer$MixtureSameFamily,
               object,
               args)
}

#' A mixture distribution Keras layer, with independent normal components.
#'
#' @param num_components Number of component distributions in the mixture distribution.
#' @param event_shape integer vector `Tensor` representing the shape of single
#' draw from this distribution.
#' @inheritParams keras::layer_dense
#' @inheritParams layer_independent_bernoulli
#' @return a Keras layer
#' @family distribution_layers
#' @seealso For an example how to use in a Keras model, see [layer_independent_normal()].
#' @export
layer_mixture_normal <- function(object,
                                 num_components,
                                 event_shape = list(),
                                 convert_to_tensor_fn = tfp$distributions$Distribution$sample,
                                 validate_args = FALSE,
                                 ...) {
  args <- list(
    num_components = as.integer(num_components),
    event_shape = normalize_shape(event_shape),
    convert_to_tensor_fn = convert_to_tensor_fn,
    validate_args = validate_args,
    ...
  )

  create_layer(tfp$python$layers$distribution_layer$MixtureNormal,
               object,
               args)
}

#' A mixture distribution Keras layer, with independent logistic components.
#'
#' @inheritParams keras::layer_dense
#' @inheritParams layer_mixture_normal
#' @return a Keras layer
#' @family distribution_layers
#' @seealso For an example how to use in a Keras model, see [layer_independent_normal()].
#' @export
layer_mixture_logistic <- function(object,
                                   num_components,
                                   event_shape = list(),
                                   convert_to_tensor_fn = tfp$distributions$Distribution$sample,
                                   validate_args = FALSE,
                                   ...) {
  args <- list(
    num_components = as.integer(num_components),
    event_shape = normalize_shape(event_shape),
    convert_to_tensor_fn = convert_to_tensor_fn,
    validate_args = validate_args,
    ...
  )

  create_layer(tfp$python$layers$distribution_layer$MixtureLogistic,
               object,
               args)
}


#' A Variational Gaussian Process Layer.
#'
#' Create a Variational Gaussian Process distribution whose `index_points` are
#' the inputs to the layer. Parameterized by number of inducing points and a
#' `kernel_provider`, which should be a `tf.keras.Layer` with an @property that
#' late-binds variable parameters to a `tfp.positive_semidefinite_kernel.PositiveSemidefiniteKernel`
#' instance (this requirement has to do with the way that variables must be created
#' in a keras model). The mean_fn is an optional argument which, if omitted, will
#' be automatically configured to be a constant function with trainable variable
#' output.
#'
#' @inheritParams keras::layer_dense
#' @return a Keras layer
#'
#' @param num_inducing_points number of inducing points in the Variational Gaussian
#'  Process distribution.
#' @param kernel_provider a `Layer` instance equipped with an `@property`, which
#'  yields a `PositiveSemidefiniteKernel` instance. The latter is used to parametrize
#'  the constructed Variational Gaussian Process distribution returned by calling
#'  the layer.
#' @param event_shape the shape of the output of the layer. This translates to a
#'  batch of underlying Variational Gaussian Process distributions. For example,
#'  `event_shape = 3` means we are modelling a batch of 3 distributions over functions.
#'  We can think oof this as a distribution over 3-dimensional veector-valued
#'  functions.
#' @param inducing_index_points_initializer a `tf.keras.initializer.Initializer`
#'  used to initialize the trainable `inducing_index_points variables`. Training
#'  VGP's is pretty sensitive to choice of initial inducing index point locations.
#'  A reasonable heuristic is to scatter them near the data, not too close to each
#'  other.
#' @param unconstrained_observation_noise_variance_initializer a `tf.keras.initializer.Initializer`
#'  used to initialize the unconstrained observation noise variable. The observation
#'  noise variance is computed from this variable via the `tf.nn.softplus` function.
#' @param mean_fn a callable that maps layer inputs to mean function values.
#'  Passed to the mean_fn parameter of Variational Gaussian Process distribution.
#'  If omitted, defaults to a constant function with trainable variable value.
#' @param jitter a small term added to the diagonal of various kernel matrices for
#'  numerical stability.
#' @param name name to give to this layer and the scope of ops and variables it
#'  contains.
#'
#' @export
layer_variational_gaussian_process <- function(object,
                                               num_inducing_points,
                                               kernel_provider,
                                               event_shape = 1,
                                               inducing_index_points_initializer = NULL,
                                               unconstrained_observation_noise_variance_initializer = NULL,
                                               mean_fn = NULL,
                                               jitter = 1e-06,
                                               name = NULL) {
  unconstrained_observation_noise_variance_initializer <-
    if (is.null(unconstrained_observation_noise_variance_initializer))
      keras::initializer_constant(-10)
  else
    unconstrained_observation_noise_variance_initializer

  args <- list(
    num_inducing_points = as.integer(num_inducing_points),
    kernel_provider = kernel_provider,
    event_shape = normalize_shape(event_shape),
    inducing_index_points_initializer = inducing_index_points_initializer,
    unconstrained_observation_noise_variance_initializer = unconstrained_observation_noise_variance_initializer,
    mean_fn = mean_fn,
    jitter = jitter,
    name = name
  )

  create_layer(tfp$layers$VariationalGaussianProcess, object, args)
}
