#' Masked Autoencoder for Distribution Estimation
#'
#' `layer_autoregressive` takes as input a Tensor of shape `[..., event_size]`
#' and returns a Tensor of shape `[..., event_size, params]`.
#' The output satisfies the autoregressive property.  That is, the layer is
#' configured with some permutation `ord` of `{0, ..., event_size-1}` (i.e., an
#' ordering of the input dimensions), and the output `output[batch_idx, i, ...]`
#' for input dimension `i` depends only on inputs `x[batch_idx, j]` where
#' `ord(j) < ord(i)`.
#'
#' The autoregressive property allows us to use
#' `output[batch_idx, i]` to parameterize conditional distributions:
#' `p(x[batch_idx, i] | x[batch_idx, ] for ord(j) < ord(i))`
#' which give us a tractable distribution over input `x[batch_idx]`:
#'
#' `p(x[batch_idx]) = prod_i p(x[batch_idx, ord(i)] | x[batch_idx, ord(0:i)])`
#'
#' For example, when `params` is 2, the output of the layer can parameterize
#' the location and log-scale of an autoregressive Gaussian distribution.
#'
#' @inheritParams keras::layer_dense
#'
#' @param params integer specifying the number of parameters to output per input.
#' @param event_shape `list`-like of positive integers (or a single int),
#' specifying the shape of the input to this layer, which is also the
#' event_shape of the distribution parameterized by this layer.  Currently
#' only rank-1 shapes are supported.  That is, event_shape must be a single
#' integer.  If not specified, the event shape is inferred when this layer
#' is first called or built.
#' @param hidden_units `list`-like of non-negative integers, specifying
#' the number of units in each hidden layer.
#' @param input_order Order of degrees to the input units: 'random',
#' 'left-to-right', 'right-to-left', or an array of an explicit order. For
#' example, 'left-to-right' builds an autoregressive model:
#' `p(x) = p(x1) p(x2 | x1) ... p(xD | x<D)`.  Default: 'left-to-right'.
#' @param hidden_degrees Method for assigning degrees to the hidden units:
#' 'equal', 'random'.  If 'equal', hidden units in each layer are allocated
#' equally (up to a remainder term) to each degree.  Default: 'equal'.
#' @param activation An activation function.  See `keras::layer_dense`. Default: `NULL`.
#' @param use_bias Whether or not the dense layers constructed in this layer
#' should have a bias term.  See `keras::layer_dense`.  Default: `TRUE`.
#' @param kernel_initializer Initializer for the kernel weights matrix.  Default: 'glorot_uniform'.
#' @param validate_args `logical`, default `FALSE`. When `TRUE`, layer
#' parameters are checked for validity despite possibly degrading runtime
#' performance. When `FALSE` invalid inputs may silently render incorrect outputs.
#' @param ... Additional keyword arguments passed to the `keras::layer_dense` constructed by this layer.
#' @family layers
#' @export
layer_autoregressive <- function(object,
                                 params,
                                 event_shape = NULL,
                                 hidden_units = NULL,
                                 input_order = "left-to-right",
                                 hidden_degrees = "equal",
                                 activation = NULL,
                                 use_bias = TRUE,
                                 kernel_initializer = "glorot_uniform",
                                 validate_args = FALSE,
                                 ...) {
  args <- list(
    params = as.integer(params),
    event_shape = normalize_shape(event_shape),
    hidden_units = as.integer(hidden_units),
    input_order = input_order,
    hidden_degrees = hidden_degrees,
    activation = activation,
    use_bias = use_bias,
    kernel_initializer = kernel_initializer,
    validate_args = validate_args,
    ...
  )

  create_layer(
    tfp$python$bijectors$masked_autoregressive$AutoregressiveLayer,
    object,
    args
  )
}

#' Dense Variational Layer
#'
#' This layer uses variational inference to fit a "surrogate" posterior to the
#'  distribution over both the `kernel` matrix and the `bias` terms which are
#'  otherwise used in a manner similar to `layer_dense()`.
#'  This layer fits the "weights posterior" according to the following generative
#'  process:
#'    ```none
#'    [K, b] ~ Prior()
#'    M = matmul(X, K) + b
#'    Y ~ Likelihood(M)
#'    ```
#'
#' @inheritParams keras::layer_dense
#'
#' @param make_posterior_fn function taking `tf$size(kernel)`,
#'   `tf$size(bias)`, `dtype` and returns another callable which takes an
#'   input and produces a `tfd$Distribution` instance.
#' @param make_prior_fn function taking `tf$size(kernel)`, `tf$size(bias)`,
#'   `dtype` and returns another callable which takes an input and produces a
#'   `tfd$Distribution` instance.
#' @param kl_weight Amount by which to scale the KL divergence loss between prior
#'   and posterior.
#' @param kl_use_exact Logical indicating that the analytical KL divergence
#'   should be used rather than a Monte Carlo approximation.
#' @param activation An activation function.  See `keras::layer_dense`. Default: `NULL`.
#' @param use_bias Whether or not the dense layers constructed in this layer
#' should have a bias term.  See `keras::layer_dense`.  Default: `TRUE`.
#' @param ... Additional keyword arguments passed to the `keras::layer_dense` constructed by this layer.
#' @family layers
#' @export
layer_dense_variational <- function(object,
                                    units,
                                    make_posterior_fn,
                                    make_prior_fn,
                                    kl_weight = NULL,
                                    kl_use_exact = FALSE,
                                    activation = NULL,
                                    use_bias = TRUE,
                                    ...) {
  args <- list(
    units = as.integer(units),
    make_posterior_fn = make_posterior_fn,
    make_prior_fn = make_prior_fn,
    kl_weight = kl_weight,
    kl_use_exact = kl_use_exact,
    activation = activation,
    use_bias = use_bias,
    ...
  )

  create_layer(
    tfp$layers$DenseVariational,
    object,
    args
  )
}

#' Variable Layer
#'
#' Simply returns a (trainable) variable, regardless of input.
#'  This layer implements the mathematical function `f(x) = c` where `c` is a
#'  constant, i.e., unchanged for all `x`. Like other Keras layers, the constant
#'  is `trainable`.  This layer can also be interpretted as the special case of
#'  `layer_dense()` when the `kernel` is forced to be the zero matrix
#'  (`tf$zeros`).
#'
#' @inheritParams keras::layer_dense
#'
#' @param shape integer or integer vector specifying the shape of the output of this layer.
#' @param dtype TensorFlow `dtype` of the variable created by this layer.
# "   Default value: `NULL` (i.e., `tf$as_dtype(k_floatx())`).
#' @param activation An activation function.  See `keras::layer_dense`. Default: `NULL`.
#' @param initializer Initializer for the `constant` vector.
#' @param regularizer Regularizer function applied to the `constant` vector.
#' @param constraint Constraint function applied to the `constant` vector.
#' @param ... Additional keyword arguments passed to the `keras::layer_dense` constructed by this layer.
#' @family layers
#' @export
layer_variable <- function(object,
                           shape,
                           dtype = NULL,
                           activation = NULL,
                           initializer = "zeros",
                           regularizer = NULL,
                           constraint = NULL,
                           ...) {
  args <- list(
    shape = normalize_shape(shape),
    dtype = dtype,
    activation = activation,
    initializer = initializer,
    regularizer = regularizer,
    constraint = constraint,
    ...
  )

  create_layer(
    tfp$layers$VariableLayer,
    object,
    args
  )
}

#' Densely-connected layer class with reparameterization estimator.
#'
#' This layer implements the Bayesian variational inference analogue to
#' a dense layer by assuming the `kernel` and/or the `bias` are drawn
#' from distributions.
#'
#' By default, the layer implements a stochastic
#' forward pass via sampling from the kernel and bias posteriors,
#'
#' ```
#' kernel, bias ~ posterior
#' outputs = activation(matmul(inputs, kernel) + bias)
#' ```
#'
#' It uses the reparameterization estimator (Kingma and Welling, 2014)
#' which performs a Monte Carlo approximation of the distribution integrating
#' over the `kernel` and `bias`.
#'
#' The arguments permit separate specification of the surrogate posterior
#' (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
#' distributions.
#'
#' Upon being built, this layer adds losses (accessible via the `losses`
#' property) representing the divergences of `kernel` and/or `bias` surrogate
#' posteriors and their respective priors. When doing minibatch stochastic
#' optimization, make sure to scale this loss such that it is applied just once
#' per epoch (e.g. if `kl` is the sum of `losses` for each element of the batch,
#' you should pass `kl / num_examples_per_epoch` to your optimizer).
#' You can access the `kernel` and/or `bias` posterior and prior distributions
#' after the layer is built via the `kernel_posterior`, `kernel_prior`,
#' `bias_posterior` and `bias_prior` properties.
#'
#' @section References:
#' - [Diederik Kingma and Max Welling. Auto-Encoding Variational Bayes. In _International Conference on Learning Representations_, 2014.](https://arxiv.org/abs/1312.6114)
#'
#' @inheritParams keras::layer_dense
#'
#' @param units integer dimensionality of the output space
#' @param activation Activation function. Set it to None to maintain a linear activation.
#' @param activity_regularizer Regularizer function for the output.
#' @param kernel_posterior_fn Function which creates `tfd$Distribution` instance representing the surrogate
#' posterior of the `kernel` parameter. Default value: `default_mean_field_normal_fn()`.
#' @param kernel_posterior_tensor_fn Function which takes a `tfd$Distribution` instance and returns a representative
#' value. Default value: `function(d) d %>% tfd_sample()`.
#' @param kernel_prior_fn Function which creates `tfd$Distribution` instance. See `default_mean_field_normal_fn` docstring for required
#' parameter signature. Default value: `tfd_normal(loc = 0, scale = 1)`.
#' @param kernel_divergence_fn Function which takes the surrogate posterior distribution, prior distribution and random variate
#' sample(s) from the surrogate posterior and computes or approximates the KL divergence. The
#' distributions are `tfd$Distribution`-like instances and the sample is a `Tensor`.
#' @param bias_posterior_fn Function which creates a `tfd$Distribution` instance representing the surrogate
#' posterior of the `bias` parameter. Default value:  `default_mean_field_normal_fn(is_singular = TRUE)` (which creates an
#' instance of `tfd_deterministic`).
#' @param bias_posterior_tensor_fn Function which takes a `tfd$Distribution` instance and returns a representative
#' value. Default value: `function(d) d %>% tfd_sample()`.
#' @param bias_prior_fn Function which creates `tfd` instance. See `default_mean_field_normal_fn` docstring for required parameter
#' signature. Default value: `NULL` (no prior, no variational inference)
#' @param bias_divergence_fn Function which takes the surrogate posterior distribution, prior distribution and random variate sample(s)
#' from the surrogate posterior and computes or approximates the KL divergence. The
#' distributions are `tfd$Distribution`-like instances and the sample is a `Tensor`.
#' @param ... Additional keyword arguments passed to the `keras::layer_dense` constructed by this layer.
#' @family layers
#' @export
layer_dense_reparameterization <- function(object,
                                           units,
                                           activation = NULL,
                                           activity_regularizer = NULL,
                                           trainable = TRUE,
                                           kernel_posterior_fn = tfp$layers$util$default_mean_field_normal_fn(),
                                           kernel_posterior_tensor_fn = function(d)
                                             d %>% tfd_sample(),
                                           kernel_prior_fn = tfp$layers$util$default_multivariate_normal_fn,
                                           kernel_divergence_fn = function(q, p, ignore)
                                             tfd_kl_divergence(q, p),
                                           bias_posterior_fn = tfp$layers$util$default_mean_field_normal_fn(is_singular = TRUE),
                                           bias_posterior_tensor_fn = function(d)
                                             d %>% tfd_sample(),
                                           bias_prior_fn = NULL,
                                           bias_divergence_fn = function(q, p, ignore)
                                             tfd_kl_divergence(q, p),
                                           ...) {
  args <- list(
    units = as.integer(units),
    activation = activation,
    activity_regularizer = activity_regularizer,
    trainable = trainable,
    kernel_posterior_fn = kernel_posterior_fn,
    kernel_posterior_tensor_fn = kernel_posterior_tensor_fn,
    kernel_prior_fn = kernel_prior_fn,
    kernel_divergence_fn = kernel_divergence_fn,
    bias_posterior_fn = bias_posterior_fn,
    bias_posterior_tensor_fn = bias_posterior_tensor_fn,
    bias_prior_fn = bias_prior_fn,
    bias_divergence_fn = bias_divergence_fn,
    ...
  )

  create_layer(tfp$layers$DenseReparameterization,
               object,
               args)
}
