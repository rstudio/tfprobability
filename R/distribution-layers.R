#' A d-variate MVNTriL Keras layer from `d+d*(d+1)/ 2` params.
#'
#' @inheritParams keras::layer_dense
#' @inheritParams layer_distribution_lambda
#'
#' @param event_size Integer vector tensor representing the shape of single draw from this distribution.
#' @param validate_args  Logical, default FALSE. When TRUE distribution parameters are checked
#'  for validity despite possibly degrading runtime performance. When FALSE invalid inputs may
#'  silently render incorrect outputs. Default value: FALSE.
#'
#' @family Probabilistic layers (require TensorFlow probability)
#'
#' @return a Keras layer that wraps a MultivariateNormalTriL distribution
#' @family distribution_layers
#' @export
layer_multivariate_normal_tri_l <- function(object,
                                           event_size,
                                           convert_to_tensor_fn = tfp$distributions$Distribution$sample,
                                           validate_args = FALSE,
                                           batch_input_shape = NULL,
                                           input_shape = NULL,
                                           batch_size = NULL,
                                           dtype = NULL,
                                           name = NULL,
                                           trainable = NULL,
                                           weights = NULL) {
  args <- list(
    event_size = as.integer(event_size),
    convert_to_tensor_fn = convert_to_tensor_fn,
    validate_args = validate_args,
    input_shape = normalize_shape(input_shape),
    batch_input_shape = normalize_shape(batch_input_shape),
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
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
#'
#' @return a Keras layer that wraps a Bernoulli distribution
#' @family distribution_layers
#'
#' @export
layer_independent_bernoulli <- function(object,
                                        event_shape,
                                        convert_to_tensor_fn = tfp$distributions$Distribution$sample,
                                        sample_dtype = NULL,
                                        validate_args = FALSE,
                                        batch_input_shape = NULL,
                                        input_shape = NULL,
                                        batch_size = NULL,
                                        dtype = NULL,
                                        name = NULL,
                                        trainable = NULL,
                                        weights = NULL) {
  args <- list(
    event_shape = as.integer(event_shape),
    convert_to_tensor_fn = convert_to_tensor_fn,
    sample_dtype = sample_dtype,
    validate_args = validate_args,
    input_shape = normalize_shape(input_shape),
    batch_input_shape = normalize_shape(batch_input_shape),
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  )

  create_layer(
    tfp$python$layers$distribution_layer$IndependentBernoulli,
    object,
    args
  )
}

#' Keras layer enabling plumbing TFP distributions through Keras models.
#'
#' @inheritParams keras::layer_dense
#'
#' @param make_distribution_fn A callable that takes previous layer outputs and returns a `tfd$distributions$Distribution` instance.
#' @param convert_to_tensor_fn A callable that takes a tfd$Distribution instance and returns a
#'  tf$Tensor-like object. Default value: `tfd$distributions$Distribution$sample`.
#' @family distribution_layers
#' @export
layer_distribution_lambda <- function(object,
                                      make_distribution_fn,
                                      convert_to_tensor_fn = tfp$distributions$Distribution$sample,
                                      batch_input_shape = NULL,
                                      input_shape = NULL,
                                      batch_size = NULL,
                                      dtype = NULL,
                                      name = NULL,
                                      trainable = NULL,
                                      weights = NULL) {
  args <- list(
    make_distribution_fn = make_distribution_fn,
    convert_to_tensor_fn = convert_to_tensor_fn,
    input_shape = normalize_shape(input_shape),
    batch_input_shape = normalize_shape(batch_input_shape),
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
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
#'
#' @family Probabilistic layers (require TensorFlow probability)
#'
#' @return a Keras layer that adds a KL divergence penalty to the model loss
#' @family distribution_layers
#' @export
layer_kl_divergence_add_loss <- function(object,
                                         distribution_b,
                                         use_exact_kl = FALSE,
                                         test_points_reduce_axis = NULL,
                                         test_points_fn = tf$convert_to_tensor,
                                         weight = NULL,
                                         input_shape = NULL,
                                         batch_input_shape = NULL,
                                         batch_size = NULL,
                                         dtype = NULL,
                                         name = NULL,
                                         trainable = NULL,
                                         weights = NULL) {
  args <- list(
    distribution_b = distribution_b,
    use_exact_kl = use_exact_kl,
    test_points_reduce_axis = test_points_reduce_axis,
    test_points_fn = test_points_fn,
    weight = weight,
    input_shape = normalize_shape(input_shape),
    batch_input_shape = normalize_shape(batch_input_shape),
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  )

  create_layer(
    tfp$python$layers$distribution_layer$KLDivergenceAddLoss,
    object,
    args
  )
}

#' Regularizer that adds a KL divergence penalty to the model loss.
#'
#' When using Monte Carlo approximation (e.g., `use_exact = FALSE`), it is presumed that the input
#'   distribution's concretization (i.e., `tf$convert_to_tensor(distribution)`) corresponds to a random
#'   sample. To override this behavior, set test_points_fn.
#'
#' @inheritParams keras::layer_dense
#'
#' @inheritParams keras::layer_dense
#'
#' @param distribution Distribution instance corresponding to b as in  `KL[a, b]`.
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
#'
#' @family Probabilistic layers (require TensorFlow probability)
#'
#' @family distribution_layers
#' @export
layer_kl_divergence_regularizer <- function(object,
                                      distribution_b,
                                      use_exact_kl = FALSE,
                                      test_points_reduce_axis = NULL,
                                      test_points_fn = tf$convert_to_tensor,
                                      weight = NULL,
                                      input_shape = NULL,
                                      batch_input_shape = NULL,
                                      batch_size = NULL,
                                      dtype = NULL,
                                      name = NULL,
                                      trainable = NULL,
                                      weights = NULL) {
  args <- list(
    distribution_b,
    use_exact_kl = use_exact_kl,
    test_points_reduce_axis = test_points_reduce_axis,
    test_points_fn = test_points_fn,
    weight = weight,
    input_shape = input_shape,
    batch_input_shape = batch_input_shape,
    batch_size = batch_size,
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  )

  create_layer(
    tfp$python$layers$distribution_layer$KLDivergenceRegularizer,
    object,
    args
  )
}
