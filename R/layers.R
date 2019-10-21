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
#' @inherit layer_autoregressive_transform return params
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
    hidden_units = as_integer_list(hidden_units),
    input_order = input_order,
    hidden_degrees = hidden_degrees,
    activation = activation,
    use_bias = use_bias,
    kernel_initializer = kernel_initializer,
    validate_args = validate_args,
    ...
  )

  create_layer(
    tfp$python$bijectors$masked_autoregressive$AutoregressiveNetwork,
    object,
    args
  )
}

#' An autoregressive normalizing flow layer, given a `layer_autoregressive`.
#'
#' Following [Papamakarios et al. (2017)](https://arxiv.org/abs/1705.07057), given
#' an autoregressive model \eqn{p(x)} with conditional distributions in the location-scale
#' family, we can construct a normalizing flow for \eqn{p(x)}.
#'
#' Specifically, suppose made is a `[layer_autoregressive()]` -- a layer implementing
#' a Masked Autoencoder for Distribution Estimation (MADE) -- that computes location
#' and log-scale parameters \eqn{made(x)[i]} for each input \eqn{x[i]}. Then we can represent
#' the autoregressive model \eqn{p(x)} as \eqn{x = f(u)} where \eqn{u} is drawn
#' from from some base distribution and where \eqn{f} is an invertible and
#' differentiable function (i.e., a Bijector) and \eqn{f^{-1}(x)} is defined by:
#'
#' ```
#' library(tensorflow)
#' library(zeallot)
#' f_inverse <- function(x) {
#'   c(shift, log_scale) %<-% tf$unstack(made(x), 2, axis = -1L)
#'   (x - shift) * tf$math$exp(-log_scale)
#' }
#' ```
#'
#' Given a [layer_autoregressive()] made, a [layer_autoregressive_transform()]
#' transforms an input `tfd_*` \eqn{p(u)} to an output `tfd_*` \eqn{p(x)} where
#' \eqn{x = f(u)}.
#'
#' @seealso [tfb_masked_autoregressive_flow()] and [layer_autoregressive()]
#'
#'
#' @inheritParams keras::layer_dense
#' @return a Keras layer
#' @param made A `Made` layer, which must output two parameters for each input.
#' @param ... Additional parameters passed to Keras Layer.
#'
#' @references
#' [Papamakarios et al. (2017)](https://arxiv.org/abs/1705.07057)
#'
#'
#' @export
layer_autoregressive_transform <- function(object, made, ...) {

  args <- list(
    made = made,
    ...
  )

  create_layer(
    tfp$layers$AutoregressiveTransform,
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
#' @inherit layer_autoregressive_transform return params
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
#' @inherit layer_autoregressive_transform return params
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
#' @inherit layer_autoregressive_transform return params
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

#' Densely-connected layer class with Flipout estimator.
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
#' It uses the Flipout estimator (Wen et al., 2018), which performs a Monte
#' Carlo approximation of the distribution integrating over the `kernel` and
#' `bias`. Flipout uses roughly twice as many floating point operations as the
#' reparameterization estimator but has the advantage of significantly lower
#' variance.
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
#'
#' @section References:
#' - [Yeming Wen, Paul Vicol, Jimmy Ba, Dustin Tran, and Roger Grosse. Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches. In _International Conference on Learning Representations_, 2018.](https://arxiv.org/abs/1803.04386)
#'
#' @inherit layer_dense_reparameterization return params
#' @inheritParams keras::layer_dense
#' @param seed scalar `integer` which initializes the random number generator.
#' Default value: `NULL` (i.e., use global seed).
#'
#' @family layers
#' @export
layer_dense_flipout <- function(object,
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
                                seed = NULL,
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
    seed = as.integer(seed),
    ...
  )

  create_layer(tfp$layers$DenseFlipout,
               object,
               args)
}

#' Densely-connected layer class with local reparameterization estimator.
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
#' It uses the local reparameterization estimator (Kingma et al., 2015),
#' which performs a Monte Carlo approximation of the distribution on the hidden
#' units induced by the `kernel` and `bias`. The default `kernel_posterior_fn`
#' is a normal distribution which factorizes across all elements of the weight
#' matrix and bias vector. Unlike that paper's multiplicative parameterization, this
#' distribution has trainable location and scale parameters which is known as
#' an additive noise parameterization (Molchanov et al., 2017).
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
#' - [Diederik Kingma, Tim Salimans, and Max Welling. Variational Dropout and the Local Reparameterization Trick. In _Neural Information Processing Systems_, 2015.](https://arxiv.org/abs/1506.02557)
#' - [Dmitry Molchanov, Arsenii Ashukha, Dmitry Vetrov. Variational Dropout Sparsifies Deep Neural Networks. In _International Conference on Machine Learning_, 2017.](https://arxiv.org/abs/1701.05369)
#'
#' @inherit layer_dense_reparameterization return params
#' @inheritParams keras::layer_dense
#' @family layers
#' @export
layer_dense_local_reparameterization <- function(object,
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

  create_layer(tfp$layers$DenseLocalReparameterization,
               object,
               args)
}

#' 1D convolution layer (e.g. temporal convolution).
#'
#' This layer creates a convolution kernel that is convolved
#' (actually cross-correlated) with the layer input to produce a tensor of
#' outputs. It may also include a bias addition and activation function
#' on the outputs. It assumes the `kernel` and/or `bias` are drawn from distributions.
#'
#' This layer implements the Bayesian variational inference analogue to
#' a dense layer by assuming the `kernel` and/or the `bias` are drawn
#' from distributions.
#'
#' By default, the layer implements a stochastic forward pass via sampling from the kernel and bias posteriors,
#'
#' ```
#' outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
#' ```
#' where f denotes the layer's calculation. It uses the reparameterization
#' estimator (Kingma and Welling, 2014), which performs a Monte Carlo
#' approximation of the distribution integrating over the `kernel` and `bias`.
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
#' @param filters Integer, the dimensionality of the output space (i.e. the number
#' of filters in the convolution).
#' @param kernel_size An integer or list of a single integer, specifying the
#' length of the 1D convolution window.
#' @param strides An integer or list of a single integer,
#' specifying the stride length of the convolution.
#' Specifying any stride value != 1 is incompatible with specifying
#' any `dilation_rate` value != 1.
#' @param padding One of `"valid"` or `"same"` (case-insensitive).
#' @param data_format A string, one of `channels_last` (default) or
#' `channels_first`. The ordering of the dimensions in the inputs.
#' `channels_last` corresponds to inputs with shape `(batch, length,
#' channels)` while `channels_first` corresponds to inputs with shape
#' `(batch, channels, length)`.
#' @param dilation_rate An integer or tuple/list of a single integer, specifying
#' the dilation rate to use for dilated convolution.
#' Currently, specifying any `dilation_rate` value != 1 is
#' incompatible with specifying any `strides` value != 1.
#' @inherit layer_dense_reparameterization return params
#' @inheritParams keras::layer_conv_1d
#'
#' @family layers
#' @export
layer_conv_1d_reparameterization <- function(object,
                                             filters,
                                             kernel_size,
                                             strides = 1,
                                             padding = 'valid',
                                             data_format = 'channels_last',
                                             dilation_rate = 1,
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
    filters = as.integer(filters),
    kernel_size = as.integer(kernel_size),
    strides = as.integer(strides),
    padding = padding,
    data_format = data_format,
    dilation_rate = as.integer(dilation_rate),
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

  create_layer(tfp$layers$Convolution1DReparameterization,
               object,
               args)
}

#' 1D convolution layer (e.g. temporal convolution) with Flipout
#'
#' This layer creates a convolution kernel that is convolved
#' (actually cross-correlated) with the layer input to produce a tensor of
#' outputs. It may also include a bias addition and activation function
#' on the outputs. It assumes the `kernel` and/or `bias` are drawn from distributions.
#'
#' This layer implements the Bayesian variational inference analogue to
#' a dense layer by assuming the `kernel` and/or the `bias` are drawn
#' from distributions.
#'
#' By default, the layer implements a stochastic forward pass via sampling from the kernel and bias posteriors,
#'
#' ```
#' outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
#' ```
#' where f denotes the layer's calculation. It uses the Flipout
#' estimator (Wen et al., 2018), which performs a Monte Carlo approximation
#' of the distribution integrating over the `kernel` and `bias`. Flipout uses
#' roughly twice as many floating point operations as the reparameterization
#' estimator but has the advantage of significantly lower variance.
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
#' - [Yeming Wen, Paul Vicol, Jimmy Ba, Dustin Tran, and Roger Grosse. Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches. In _International Conference on Learning Representations_, 2018.](https://arxiv.org/abs/1803.04386)
#'
#' @param filters Integer, the dimensionality of the output space (i.e. the number
#' of filters in the convolution).
#' @param kernel_size An integer or list of a single integer, specifying the
#' length of the 1D convolution window.
#' @param strides An integer or list of a single integer,
#' specifying the stride length of the convolution.
#' Specifying any stride value != 1 is incompatible with specifying
#' any `dilation_rate` value != 1.
#' @param padding One of `"valid"` or `"same"` (case-insensitive).
#' @param data_format A string, one of `channels_last` (default) or
#' `channels_first`. The ordering of the dimensions in the inputs.
#' `channels_last` corresponds to inputs with shape `(batch, length,
#' channels)` while `channels_first` corresponds to inputs with shape
#' `(batch, channels, length)`.
#' @param dilation_rate An integer or tuple/list of a single integer, specifying
#' the dilation rate to use for dilated convolution.
#' Currently, specifying any `dilation_rate` value != 1 is
#' incompatible with specifying any `strides` value != 1.
#' @inherit layer_dense_reparameterization return params
#' @inheritParams keras::layer_conv_1d
#'
#' @family layers
#' @export
layer_conv_1d_flipout <- function(object,
                                  filters,
                                  kernel_size,
                                  strides = 1,
                                  padding = 'valid',
                                  data_format = 'channels_last',
                                  dilation_rate = 1,
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
    filters = as.integer(filters),
    kernel_size = as.integer(kernel_size),
    strides = as.integer(strides),
    padding = padding,
    data_format = data_format,
    dilation_rate = as.integer(dilation_rate),
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

  create_layer(tfp$layers$Convolution1DFlipout,
               object,
               args)
}

#' 2D convolution layer (e.g. spatial convolution over images)
#'
#' This layer creates a convolution kernel that is convolved
#' (actually cross-correlated) with the layer input to produce a tensor of
#' outputs. It may also include a bias addition and activation function
#' on the outputs. It assumes the `kernel` and/or `bias` are drawn from distributions.
#'
#' This layer implements the Bayesian variational inference analogue to
#' a dense layer by assuming the `kernel` and/or the `bias` are drawn
#' from distributions.
#'
#' By default, the layer implements a stochastic forward pass via sampling from the kernel and bias posteriors,
#'
#' ```
#' outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
#' ```
#' where f denotes the layer's calculation. It uses the reparameterization
#' estimator (Kingma and Welling, 2014), which performs a Monte Carlo
#' approximation of the distribution integrating over the `kernel` and `bias`.
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
#' @param filters Integer, the dimensionality of the output space (i.e. the number
#' of filters in the convolution).
#' @param kernel_size An integer or list of a single integer, specifying the
#' length of the 1D convolution window.
#' @param strides An integer or list of a single integer,
#' specifying the stride length of the convolution.
#' Specifying any stride value != 1 is incompatible with specifying
#' any `dilation_rate` value != 1.
#' @param padding One of `"valid"` or `"same"` (case-insensitive).
#' @param data_format A string, one of `channels_last` (default) or
#' `channels_first`. The ordering of the dimensions in the inputs.
#' `channels_last` corresponds to inputs with shape `(batch, length,
#' channels)` while `channels_first` corresponds to inputs with shape
#' `(batch, channels, length)`.
#' @param dilation_rate An integer or tuple/list of a single integer, specifying
#' the dilation rate to use for dilated convolution.
#' Currently, specifying any `dilation_rate` value != 1 is
#' incompatible with specifying any `strides` value != 1.
#' @inherit layer_dense_reparameterization return params
#' @inheritParams keras::layer_conv_2d
#'
#' @family layers
#' @export
layer_conv_2d_reparameterization <- function(object,
                                             filters,
                                             kernel_size,
                                             strides = 1,
                                             padding = 'valid',
                                             data_format = 'channels_last',
                                             dilation_rate = 1,
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
    filters = as.integer(filters),
    kernel_size = as.integer(kernel_size),
    strides = as.integer(strides),
    padding = padding,
    data_format = data_format,
    dilation_rate = as.integer(dilation_rate),
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

  create_layer(tfp$layers$Convolution2DReparameterization,
               object,
               args)
}

#' 2D convolution layer (e.g. spatial convolution over images) with Flipout
#'
#' This layer creates a convolution kernel that is convolved
#' (actually cross-correlated) with the layer input to produce a tensor of
#' outputs. It may also include a bias addition and activation function
#' on the outputs. It assumes the `kernel` and/or `bias` are drawn from distributions.
#'
#' This layer implements the Bayesian variational inference analogue to
#' a dense layer by assuming the `kernel` and/or the `bias` are drawn
#' from distributions.
#'
#' By default, the layer implements a stochastic forward pass via sampling from the kernel and bias posteriors,
#'
#' ```
#' outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
#' ```
#' where f denotes the layer's calculation. It uses the Flipout
#' estimator (Wen et al., 2018), which performs a Monte Carlo approximation
#' of the distribution integrating over the `kernel` and `bias`. Flipout uses
#' roughly twice as many floating point operations as the reparameterization
#' estimator but has the advantage of significantly lower variance.
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
#' - [Yeming Wen, Paul Vicol, Jimmy Ba, Dustin Tran, and Roger Grosse. Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches. In _International Conference on Learning Representations_, 2018.](https://arxiv.org/abs/1803.04386)
#'
#' @param filters Integer, the dimensionality of the output space (i.e. the number
#' of filters in the convolution).
#' @param kernel_size An integer or list of a single integer, specifying the
#' length of the 1D convolution window.
#' @param strides An integer or list of a single integer,
#' specifying the stride length of the convolution.
#' Specifying any stride value != 1 is incompatible with specifying
#' any `dilation_rate` value != 1.
#' @param padding One of `"valid"` or `"same"` (case-insensitive).
#' @param data_format A string, one of `channels_last` (default) or
#' `channels_first`. The ordering of the dimensions in the inputs.
#' `channels_last` corresponds to inputs with shape `(batch, length,
#' channels)` while `channels_first` corresponds to inputs with shape
#' `(batch, channels, length)`.
#' @param dilation_rate An integer or tuple/list of a single integer, specifying
#' the dilation rate to use for dilated convolution.
#' Currently, specifying any `dilation_rate` value != 1 is
#' incompatible with specifying any `strides` value != 1.
#' @inherit layer_dense_reparameterization return params
#' @inheritParams keras::layer_conv_2d
#'
#' @family layers
#' @export
layer_conv_2d_flipout <- function(object,
                                  filters,
                                  kernel_size,
                                  strides = 1,
                                  padding = 'valid',
                                  data_format = 'channels_last',
                                  dilation_rate = 1,
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
    filters = as.integer(filters),
    kernel_size = as.integer(kernel_size),
    strides = as.integer(strides),
    padding = padding,
    data_format = data_format,
    dilation_rate = as.integer(dilation_rate),
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

  create_layer(tfp$layers$Convolution2DFlipout,
               object,
               args)
}

#' 3D convolution layer (e.g. spatial convolution over volumes)
#'
#' This layer creates a convolution kernel that is convolved
#' (actually cross-correlated) with the layer input to produce a tensor of
#' outputs. It may also include a bias addition and activation function
#' on the outputs. It assumes the `kernel` and/or `bias` are drawn from distributions.
#'
#' This layer implements the Bayesian variational inference analogue to
#' a dense layer by assuming the `kernel` and/or the `bias` are drawn
#' from distributions.
#'
#' By default, the layer implements a stochastic forward pass via sampling from the kernel and bias posteriors,
#'
#' ```
#' outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
#' ```
#' where f denotes the layer's calculation. It uses the reparameterization
#' estimator (Kingma and Welling, 2014), which performs a Monte Carlo
#' approximation of the distribution integrating over the `kernel` and `bias`.
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
#' @param filters Integer, the dimensionality of the output space (i.e. the number
#' of filters in the convolution).
#' @param kernel_size An integer or list of a single integer, specifying the
#' length of the 1D convolution window.
#' @param strides An integer or list of a single integer,
#' specifying the stride length of the convolution.
#' Specifying any stride value != 1 is incompatible with specifying
#' any `dilation_rate` value != 1.
#' @param padding One of `"valid"` or `"same"` (case-insensitive).
#' @param data_format A string, one of `channels_last` (default) or
#' `channels_first`. The ordering of the dimensions in the inputs.
#' `channels_last` corresponds to inputs with shape `(batch, length,
#' channels)` while `channels_first` corresponds to inputs with shape
#' `(batch, channels, length)`.
#' @param dilation_rate An integer or tuple/list of a single integer, specifying
#' the dilation rate to use for dilated convolution.
#' Currently, specifying any `dilation_rate` value != 1 is
#' incompatible with specifying any `strides` value != 1.
#' @inherit layer_dense_reparameterization return params
#' @inheritParams keras::layer_conv_3d
#'
#' @family layers
#' @export
layer_conv_3d_reparameterization <- function(object,
                                             filters,
                                             kernel_size,
                                             strides = 1,
                                             padding = 'valid',
                                             data_format = 'channels_last',
                                             dilation_rate = 1,
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
    filters = as.integer(filters),
    kernel_size = as.integer(kernel_size),
    strides = as.integer(strides),
    padding = padding,
    data_format = data_format,
    dilation_rate = as.integer(dilation_rate),
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

  create_layer(tfp$layers$Convolution3DReparameterization,
               object,
               args)
}

#' 3D convolution layer (e.g. spatial convolution over volumes) with Flipout
#'
#' This layer creates a convolution kernel that is convolved
#' (actually cross-correlated) with the layer input to produce a tensor of
#' outputs. It may also include a bias addition and activation function
#' on the outputs. It assumes the `kernel` and/or `bias` are drawn from distributions.
#'
#' This layer implements the Bayesian variational inference analogue to
#' a dense layer by assuming the `kernel` and/or the `bias` are drawn
#' from distributions.
#'
#' By default, the layer implements a stochastic forward pass via sampling from the kernel and bias posteriors,
#'
#' ```
#' outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
#' ```
#' where f denotes the layer's calculation. It uses the Flipout
#' estimator (Wen et al., 2018), which performs a Monte Carlo approximation
#' of the distribution integrating over the `kernel` and `bias`. Flipout uses
#' roughly twice as many floating point operations as the reparameterization
#' estimator but has the advantage of significantly lower variance.
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
#' - [Yeming Wen, Paul Vicol, Jimmy Ba, Dustin Tran, and Roger Grosse. Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches. In _International Conference on Learning Representations_, 2018.](https://arxiv.org/abs/1803.04386)
#'
#' @param filters Integer, the dimensionality of the output space (i.e. the number
#' of filters in the convolution).
#' @param kernel_size An integer or list of a single integer, specifying the
#' length of the 1D convolution window.
#' @param strides An integer or list of a single integer,
#' specifying the stride length of the convolution.
#' Specifying any stride value != 1 is incompatible with specifying
#' any `dilation_rate` value != 1.
#' @param padding One of `"valid"` or `"same"` (case-insensitive).
#' @param data_format A string, one of `channels_last` (default) or
#' `channels_first`. The ordering of the dimensions in the inputs.
#' `channels_last` corresponds to inputs with shape `(batch, length,
#' channels)` while `channels_first` corresponds to inputs with shape
#' `(batch, channels, length)`.
#' @param dilation_rate An integer or tuple/list of a single integer, specifying
#' the dilation rate to use for dilated convolution.
#' Currently, specifying any `dilation_rate` value != 1 is
#' incompatible with specifying any `strides` value != 1.
#' @inherit layer_dense_reparameterization return params
#' @inheritParams keras::layer_conv_3d
#'
#' @family layers
#' @export
layer_conv_3d_flipout <- function(object,
                                  filters,
                                  kernel_size,
                                  strides = 1,
                                  padding = 'valid',
                                  data_format = 'channels_last',
                                  dilation_rate = 1,
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
    filters = as.integer(filters),
    kernel_size = as.integer(kernel_size),
    strides = as.integer(strides),
    padding = padding,
    data_format = data_format,
    dilation_rate = as.integer(dilation_rate),
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

  create_layer(tfp$layers$Convolution3DFlipout,
               object,
               args)
}



