#' Computes`Y = g(X) = X`
#'
#' @param validate_args Logical, default FALSE. Whether to validate input with asserts. If validate_args is
#'  FALSE, and the inputs are invalid, correct behavior is not guaranteed.
#' @param name name prefixed to Ops created by this class.
#' @return a bijector instance.
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @family bijectors
#' @export
tfb_identity <- function(validate_args = FALSE,
                         name = "identity") {
  args <- list(validate_args = validate_args,
               name = name)

  do.call(tfp$bijectors$Identity, args)
}

#' Computes`Y = g(X) = 1 / (1 + exp(-X))`
#'
#' @inherit tfb_identity return params
#'
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_sigmoid <- function(validate_args = FALSE,
                        name = "sigmoid") {
  args <- list(validate_args = validate_args,
               name = name)

  do.call(tfp$bijectors$Sigmoid, args)
}

#' Computes`Y=g(X)=exp(X)`
#'
#' @inherit tfb_identity return params
#'
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_exp <- function(validate_args = FALSE,
                    name = "exp") {
  args <- list(validate_args = validate_args,
               name = name)

  do.call(tfp$bijectors$Exp, args)
}

#' Computes`Y = g(X) = Abs(X)`, element-wise
#'
#' This non-injective bijector allows for transformations of scalar distributions
#' with the absolute value function, which maps `(-inf, inf)` to `[0, inf)`.
#' * For `y` in `(0, inf)`, `tfb_absolute_value$inverse(y)` returns the set inverse
#' `{x in (-inf, inf) : |x| = y}` as a tuple, `-y, y`.
#' `tfb_absolute_value$inverse(0)` returns `0, 0`, which is not the set inverse
#' (the set inverse is the singleton `{0}`), but "works" in conjunction with
#' `TransformedDistribution` to produce a left semi-continuous pdf.
#' For `y < 0`, `tfb_absolute_value$inverse(y)` happily returns the wrong thing, `-y, y`
#'  This is done for efficiency.  If `validate_args == TRUE`, `y < 0` will raise an exception.
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_absolute_value <- function(validate_args = FALSE,
                               name = "absolute_value") {
  args <- list(validate_args = validate_args,
               name = name)

  do.call(tfp$bijectors$AbsoluteValue, args)
}

#' Affine bijector
#'
#' This Bijector is initialized with shift Tensor and scale arguments,
#' giving the forward operation: `Y = g(X) = scale @ X + shift`
#' where the scale term is logically equivalent to:
#' `
#' scale =
#'     scale_identity_multiplier * tf.diag(tf.ones(d)) +
#'     tf.diag(scale_diag) +
#'     scale_tril +
#'     scale_perturb_factor @ diag(scale_perturb_diag) @ tf.transpose([scale_perturb_factor]))
#' `
#'
#'  If NULL of `scale_identity_multiplier`, `scale_diag`, or `scale_tril` are specified then
#'   `scale += IdentityMatrix` Otherwise specifying a scale argument has the semantics of
#'    `scale += Expand(arg)`, i.e., `scale_diag != NULL` means `scale += tf$diag(scale_diag)`.
#'
#' @param shift Floating-point Tensor. If this is set to NULL, no shift is applied.
#' @param scale_identity_multiplier floating point rank 0 Tensor representing a scaling done
#'  to the identity matrix. When `scale_identity_multiplier = scale_diag = scale_tril = NULL` then
#'  `scale += IdentityMatrix`. Otherwise no scaled-identity-matrix is added to `scale`.
#' @param scale_diag Floating-point Tensor representing the diagonal matrix.
#' `scale_diag` has shape `[N1, N2, ...  k]`, which represents a k x k diagonal matrix.
#' When NULL no diagonal term is added to `scale`.
#' @param scale_tril Floating-point Tensor representing the lower triangular matrix.
#' `scale_tril` has shape `[N1, N2, ...  k, k]`, which represents a k x k lower triangular matrix.
#' When NULL no `scale_tril` term is added to `scale`. The upper triangular elements above the diagonal are ignored.
#' @param scale_perturb_factor Floating-point Tensor representing factor matrix with last
#'  two dimensions of shape `(k, r)` When NULL, no rank-r update is added to scale.
#' @param scale_perturb_diag Floating-point Tensor representing the diagonal matrix.
#'  `scale_perturb_diag` has shape `[N1, N2, ...  r]`, which represents an r x r diagonal matrix.
#'  When NULL low rank updates will take the form `scale_perturb_factor * scale_perturb_factor.T`.
#' @param adjoint Logical indicating whether to use the scale matrix as specified or its adjoint.
#' Default value: FALSE.
#' @inherit tfb_identity return params
#' @param dtype `tf$DType` to prefer when converting args to Tensors. Else, we fall back to a
#'  common dtype inferred from the args, finally falling back to float32.
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_affine <- function(shift = NULL,
                       scale_identity_multiplier = NULL,
                       scale_diag = NULL,
                       scale_tril = NULL,
                       scale_perturb_factor = NULL,
                       scale_perturb_diag = NULL,
                       adjoint = FALSE,
                       validate_args = FALSE,
                       name = "affine",
                       dtype = NULL) {
  args <- list(
    shift = shift,
    scale_identity_multiplier = scale_identity_multiplier,
    scale_diag = scale_diag,
    scale_tril = scale_tril,
    scale_perturb_factor = scale_perturb_factor,
    scale_perturb_diag = scale_perturb_diag,
    adjoint = adjoint,
    validate_args = validate_args,
    name = name,
    dtype = dtype
  )
  do.call(tfp$bijectors$Affine, args)
}

#' Computes`Y = g(X; shift, scale) = scale @ X + shift`
#'
#' `shift` is a numeric Tensor and scale is a LinearOperator.
#' If `X` is a scalar then the forward transformation is: `scale * X + shift`
#' where `*` denotes broadcasted elementwise product.
#'
#' @param shift Floating-point Tensor.
#' @param scale Subclass of LinearOperator. Represents the (batch) positive definite matrix `M` in `R^{k x k}`.
#' @param adjoint Logical indicating whether to use the scale matrix as specified or its adjoint.
#' Default value: FALSE.
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_affine_linear_operator <- function(shift = NULL,
                                       scale = NULL,
                                       adjoint = FALSE,
                                       validate_args = FALSE,
                                       name = "affine_linear_operator") {
  args <- list(
    shift = shift,
    scale = scale,
    adjoint = adjoint,
    validate_args = validate_args,
    name = name
  )
  do.call(tfp$bijectors$AffineLinearOperator, args)
}


#' AffineScalar bijector
#'
#' This Bijector is initialized with shift Tensor and scale arguments, giving the forward operation:
#' `Y = g(X) = scale * X + shift`
#' If `scale` is not specified, then the bijector has the semantics of scale = 1..
#' Similarly, if `shift` is not specified, then the bijector has the semantics of shift = 0..
#'
#' @param shift Floating-point Tensor. If this is set to NULL, no shift is applied.
#' @param scale Floating-point Tensor. If this is set to NULL, no scale is applied.
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export

tfb_affine_scalar <- function(shift = NULL,
                              scale = NULL,
                              validate_args = FALSE,
                              name = "affine_scalar") {
  args <- list(
    shift = shift,
    scale = scale,
    validate_args = validate_args,
    name = name
  )
  do.call(tfp$bijectors$AffineScalar, args)
}

#' Computes`Y = g(X)` s.t. `X = g^-1(Y) = (Y - mean(Y)) / std(Y)`
#'
#' Applies Batch Normalization (Ioffe and Szegedy, 2015) to samples from a
#' data distribution. This can be used to stabilize training of normalizing
#' flows (Papamakarios et al., 2016; Dinh et al., 2017)
#'
#' When training Deep Neural Networks (DNNs), it is common practice to
#' normalize or whiten features by shifting them to have zero mean and
#' scaling them to have unit variance.
#'
#' The `inverse()` method of the BatchNormalization bijector, which is used in
#' the log-likelihood computation of data samples, implements the normalization
#' procedure (shift-and-scale) using the mean and standard deviation of the
#' current minibatch.
#'
#' Conversely, the `forward()` method of the bijector de-normalizes samples (e.g.
#' `X*std(Y) + mean(Y)` with the running-average mean and standard deviation
#' computed at training-time. De-normalization is useful for sampling.
#'
#' During training time, BatchNormalization.inverse and BatchNormalization.forward are not
#'  guaranteed to be inverses of each other because `inverse(y)` uses statistics of the current minibatch,
#'  while `forward(x)` uses running-average statistics accumulated from training.
#'  In other words, `tfb_batch_normalization()$inverse(tfb_batch_normalization()$forward(...))` and
#'  `tfb_batch_normalization()$forward(tfb_batch_normalization()$inverse(...))` will be identical when
#'   training=FALSE but may be different when training=TRUE.
#'
#' @section References:
#' - [Sergey Ioffe and Christian Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. In _International Conference on Machine Learning_, 2015.](https://arxiv.org/abs/1502.03167)
#' - [Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density Estimation using Real NVP. In _International Conference on Learning Representations_, 2017.](https://arxiv.org/abs/1605.08803)
#' - [George Papamakarios, Theo Pavlakou, and Iain Murray. Masked Autoregressive Flow for Density Estimation. In _Neural Information Processing Systems_, 2017.](https://arxiv.org/abs/1705.07057)
#'
#' @param batchnorm_layer `tf$layers$BatchNormalization` layer object. If NULL, defaults to
#' `tf$layers$BatchNormalization(gamma_constraint=tf$nn$relu(x) + 1e-6)`.
#' This ensures positivity of the scale variable.
#' @param training If TRUE, updates running-average statistics during call to inverse().
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_batch_normalization <- function(batchnorm_layer = NULL,
                                    training = TRUE,
                                    validate_args = FALSE,
                                    name = "batch_normalization") {
  args <- list(
    batchnorm_layer = batchnorm_layer,
    training = training,
    validate_args = validate_args,
    name = name
  )
  do.call(tfp$bijectors$BatchNormalization, args)
}

#' Bijector which applies a list of bijectors to blocks of a Tensor
#'
#' More specifically, given `[F_0, F_1, ... F_n]` which are scalar or vector
#' bijectors this bijector creates a transformation which operates on the vector
#' `[x_0, ... x_n]` with the transformation `[F_0(x_0), F_1(x_1) ..., F_n(x_n)]`
#' where `x_0, ..., x_n` are blocks (partitions) of the vector.
#'
#' @param bijectors A non-empty list of bijectors.
#' @param block_sizes A 1-D integer Tensor with each element signifying the
#' length of the block of the input vector to pass to the corresponding
#' bijector. The length of block_sizes must be be equal to the length of
#' bijectors. If left as NULL, a vector of 1's is used.
#' @param validate_args Logical indicating whether arguments should be checked for correctness.
#' @param name String, name given to ops managed by this object. Default:
#' E.g., `tfb_blockwise(list(tfb_exp(), tfb_softplus()))$name == 'blockwise_of_exp_and_softplus'`.
#' @inherit tfb_identity return
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_blockwise <- function(bijectors,
                          block_sizes = NULL,
                          validate_args = FALSE,
                          name = NULL) {
  args <- list(
    bijectors = bijectors,
    block_sizes = as_nullable_integer(block_sizes),
    validate_args = validate_args,
    name = name
  )
  do.call(tfp$bijectors$Blockwise, args)
}


#' Bijector which applies a sequence of bijectors
#'
#' @param bijectors list of bijector instances. An empty list makes this
#' bijector equivalent to the Identity bijector.
#' @param validate_args Logical indicating whether arguments should be checked for correctness.
#' @param name String, name given to ops managed by this object. Default:
#' E.g., `tfb_chain(list(tfb_exp(), tfb_softplus()))$name == "chain_of_exp_of_softplus"`.
#' @inherit tfb_identity return
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_chain <- function(bijectors = NULL,
                      validate_args = FALSE,
                      name = NULL) {
  args <- list(bijectors = bijectors,
               validate_args = validate_args,
               name = name)
  do.call(tfp$bijectors$Chain, args)
}



#' Computes`g(X) = X @ X.T` where `X` is lower-triangular, positive-diagonal matrix
#'
#' Note: the upper-triangular part of X is ignored (whether or not its zero).
#'
#' The surjectivity of g as a map from  the set of n x n positive-diagonal
#' lower-triangular matrices to the set of SPD matrices follows immediately from
#' executing the Cholesky factorization algorithm on an SPD matrix `A` to produce a
#' positive-diagonal lower-triangular matrix `L` such that `A = L @ L.T`.
#'
#' To prove the injectivity of g, suppose that `L_1` and `L_2` are lower-triangular
#' with positive diagonals and satisfy `A = L_1 @ L_1.T = L_2 @ L_2.T`. Then
#' `inv(L_1) @ A @ inv(L_1).T = [inv(L_1) @ L_2] @ [inv(L_1) @ L_2].T = I`.
#' Setting `L_3 := inv(L_1) @ L_2`, that `L_3` is a positive-diagonal
#' lower-triangular matrix follows from `inv(L_1)` being positive-diagonal
#' lower-triangular (which follows from the diagonal of a triangular matrix being
#' its spectrum), and that the product of two positive-diagonal lower-triangular
#' matrices is another positive-diagonal lower-triangular matrix.
#' A simple inductive argument (proceeding one column of `L_3` at a time) shows
#' that, if `I = L_3 @ L_3.T`, with `L_3` being lower-triangular with positive-
#' diagonal, then `L_3 = I`. Thus, `L_1 = L_2`, proving injectivity of g.
#'
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_cholesky_outer_product <- function(validate_args = FALSE,
                                       name = "cholesky_outer_product") {
  args <- list(validate_args = validate_args,
               name = name)

  do.call(tfp$bijectors$CholeskyOuterProduct, args)
}

#' Maps the Cholesky factor of M to the Cholesky factor of `M^{-1}`
#'
#' The forward and inverse calculations are conceptually identical to:
#' `forward <- function(x) tf$cholesky(tf$linalg$inv(tf$matmul(x, x, adjoint_b=TRUE)))`
#' `inverse = forward`
#' However, the actual calculations exploit the triangular structure of the matrices.
#'
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_cholesky_to_inv_cholesky <- function(validate_args = FALSE,
                                         name = "cholesky_to_inv_cholesky") {
  args <- list(validate_args = validate_args,
               name = name)

  do.call(tfp$bijectors$CholeskyToInvCholesky, args)
}

#' Computes`Y = g(X) = DCT(X)`, where DCT type is indicated by the type arg
#'
#' The [discrete cosine transform](https://en.wikipedia.org/wiki/Discrete_cosine_transform)
#' efficiently applies a unitary DCT operator. This can be useful for mixing and decorrelating across
#' the innermost event dimension.
#' The inverse `X = g^{-1}(Y) = IDCT(Y)`, where IDCT is DCT-III for type==2.
#' This bijector can be interleaved with Affine bijectors to build a cascade of
#' structured efficient linear layers as in Moczulski et al., 2016.
#' Note that the operator applied is orthonormal (i.e. norm='ortho').
#'
#' @section References:
#' - [Moczulski M, Denil M, Appleyard J, de Freitas N. ACDC: A structured efficient linear layer. In _International Conference on Learning Representations_, 2016.](https://arxiv.org/abs/1511.05946)
#'
#' @inherit tfb_identity return params
#' @param dct_type integer, the DCT type performed by the forward transformation.
#' Currently, only 2 and 3 are supported.
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_discrete_cosine_transform <-
  function(validate_args = FALSE,
           dct_type = 2,
           name = "dct") {
    args <- list(validate_args = validate_args,
                 dct_type = dct_type,
                 name = name)

    do.call(tfp$bijectors$DiscreteCosineTransform, args)
  }

#' Computes`Y = g(X) = exp(X) - 1`
#'
#' This Bijector is no different from `tfb_chain(list(tfb_affine_scalar(shift=-1), tfb_exp()))`.
#' However, this makes use of the more numerically stable routines
#' `tf$math$expm1` and `tf$log1p`.
#'
#' Note: the expm1(.) is applied element-wise but the Jacobian is a reduction
#' over the event space.
#'
#' @inherit tfb_identity return params
#'
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_expm1 <- function(validate_args = FALSE,
                      name = "expm1") {
  args <- list(validate_args = validate_args,
               name = name)

  do.call(tfp$bijectors$Expm1, args)
}


#' Transforms vectors to triangular
#'
#' Triangular matrix elements are filled in a clockwise spiral.
#' Given input with shape `batch_shape + [d]`, produces output with
#' shape `batch_shape + [n, n]`, where `n = (-1 + sqrt(1 + 8 * d))/2`.
#' This follows by solving the quadratic equation `d = 1 + 2 + ... + n = n * (n + 1)/2`.
#'
#' @param upper Logical representing whether output matrix should be upper triangular (TRUE)
#'  or lower triangular (FALSE, default).
#' @inherit tfb_identity return params
#'
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_fill_triangular <- function(upper = FALSE,
                                validate_args = FALSE,
                                name = "fill_triangular") {
  args <- list(upper = upper,
               validate_args = validate_args,
               name = name)

  do.call(tfp$bijectors$FillTriangular, args)
}

#' Computes`Y = g(X) = exp(-exp(-(X - loc) / scale))`
#'
#' This bijector maps inputs from `[-inf, inf]` to `[0, 1]`. The inverse of the
#' bijector applied to a uniform random variable `X ~ U(0, 1)` gives back a
#' random variable with the [Gumbel distribution](https://en.wikipedia.org/wiki/Gumbel_distribution):
#'
#' `Y ~ Gumbel(loc, scale)`
#' `pdf(y; loc, scale) = exp(-( (y - loc) / scale + exp(- (y - loc) / scale) ) ) / scale`
#'
#' @param loc Float-like Tensor that is the same dtype and is broadcastable with scale.
#' This is loc in `Y = g(X) = exp(-exp(-(X - loc) / scale))`.
#' @param scale Positive Float-like Tensor that is the same dtype and is broadcastable with loc.
#' This is scale in `Y = g(X) = exp(-exp(-(X - loc) / scale))`.
#' @inherit tfb_identity return params
#'
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_gumbel <- function(loc = 0,
                       scale = 1,
                       validate_args = FALSE,
                       name = "gumbel") {
  args <- list(
    loc = loc,
    scale = scale,
    validate_args = validate_args,
    name = name
  )

  do.call(tfp$bijectors$Gumbel, args)
}

#' Bijector constructed from custom functions
#'
#' @param forward_fn Function implementing the forward transformation.
#' @param inverse_fn Function implementing the inverse transformation.
#' @param inverse_log_det_jacobian_fn Function implementing the log_det_jacobian of the forward transformation.
#' @param forward_log_det_jacobian_fn Function implementing the log_det_jacobian of the inverse transformation.
#' @param forward_event_shape_fn Function implementing non-identical static event shape changes. Default: shape is assumed unchanged.
#' @param forward_event_shape_tensor_fn Function implementing non-identical event shape changes. Default: shape is assumed unchanged.
#' @param inverse_event_shape_fn Function implementing non-identical static event shape changes. Default: shape is assumed unchanged.
#' @param inverse_event_shape_tensor_fn Function implementing non-identical event shape changes. Default: shape is assumed unchanged.
#' @param is_constant_jacobian Logical indicating that the Jacobian is constant for all input arguments.
#' @param forward_min_event_ndims Integer indicating the minimal dimensionality this bijector acts on.
#' @param inverse_min_event_ndims Integer indicating the minimal dimensionality this bijector acts on.
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_inline <- function(forward_fn = NULL,
                       inverse_fn = NULL,
                       inverse_log_det_jacobian_fn = NULL,
                       forward_log_det_jacobian_fn = NULL,
                       forward_event_shape_fn = NULL,
                       forward_event_shape_tensor_fn = NULL,
                       inverse_event_shape_fn = NULL,
                       inverse_event_shape_tensor_fn = NULL,
                       is_constant_jacobian = NULL,
                       validate_args = FALSE,
                       forward_min_event_ndims = NULL,
                       inverse_min_event_ndims = NULL,
                       name = "inline") {
  args <- list(
    forward_fn = forward_fn,
    inverse_fn = inverse_fn,
    inverse_log_det_jacobian_fn = inverse_log_det_jacobian_fn,
    forward_log_det_jacobian_fn = forward_log_det_jacobian_fn,
    forward_event_shape_fn = forward_event_shape_fn,
    forward_event_shape_tensor_fn = forward_event_shape_tensor_fn,
    inverse_event_shape_fn = inverse_event_shape_fn,
    inverse_event_shape_tensor_fn = inverse_event_shape_tensor_fn,
    is_constant_jacobian = is_constant_jacobian,
    forward_min_event_ndims = as_nullable_integer(forward_min_event_ndims),
    inverse_min_event_ndims = as_nullable_integer(inverse_min_event_ndims),
    validate_args = validate_args,
    name = name
  )

  do.call(tfp$bijectors$Inline, args)
}

#' Bijector which inverts another Bijector
#'
#' Creates a Bijector which swaps the meaning of inverse and forward.
#' Note: An inverted bijector's inverse_log_det_jacobian is often more
#' efficient if the base bijector implements _forward_log_det_jacobian. If
#' _forward_log_det_jacobian is not implemented then the following code is
#' used:
#' `y = b$inverse(x)`
#' ` -b$inverse_log_det_jacobian(y)`
#'
#' @param bijector Bijector instance.
#' @inherit tfb_identity return params
#'
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_invert <- function(bijector,
                       validate_args = FALSE,
                       name = NULL) {
  args <- list(bijector = bijector,
               validate_args = validate_args,
               name = name)

  do.call(tfp$bijectors$Invert, args)
}

#' Computes`Y = g(X) = (1 - (1 - X)**(1 / b))**(1 / a)`, with X in `[0, 1]`
#'
#' This bijector maps inputs from `[0, 1]` to `[0, 1]`. The inverse of the
#' bijector applied to a uniform random variable X ~ U(0, 1) gives back a
#' random variable with the [Kumaraswamy distribution](https://en.wikipedia.org/wiki/Kumaraswamy_distribution):
#' `Y ~ Kumaraswamy(a, b)`
#' `pdf(y; a, b, 0 <= y <= 1) = a * b * y ** (a - 1) * (1 - y**a) ** (b - 1)`
#'
#' @param concentration1 float scalar indicating the transform power, i.e.,
#' `Y = g(X) = (1 - (1 - X)**(1 / b))**(1 / a) where a is concentration1.`
#' @param concentration0 float scalar indicating the transform power,
#' i.e., `Y = g(X) = (1 - (1 - X)**(1 / b))**(1 / a)` where b is concentration0.
#' @inherit tfb_identity return params
#'
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_kumaraswamy <- function(concentration1 = NULL,
                            concentration0 = NULL,
                            validate_args = FALSE,
                            name = "kumaraswamy") {
  args <- list(
    concentration1 = concentration1,
    concentration0 = concentration0,
    validate_args = validate_args,
    name = name
  )

  do.call(tfp$bijectors$Kumaraswamy, args)
}


#' Affine MaskedAutoregressiveFlow bijector
#'
#' The affine autoregressive flow (Papamakarios et al., 2016) provides a
#' relatively simple framework for user-specified (deep) architectures to learn a
#' distribution over continuous events. Regarding terminology,
#'
#' "Autoregressive models decompose the joint density as a product of
#' conditionals, and model each conditional in turn. Normalizing flows
#' transform a base density (e.g. a standard Gaussian) into the target density
#' by an invertible transformation with tractable Jacobian." (Papamakarios et al., 2016)
#'
#' In other words, the "autoregressive property" is equivalent to the
#' decomposition, `p(x) = prod{ p(x[perm[i]] | x[perm[0:i]]) : i=0, ..., d }`
#' where perm is some permutation of `{0, ..., d}`. In the simple case where
#' the permutation is identity this reduces to:
#'
#' `p(x) = prod{ p(x[i] | x[0:i]) : i=0, ..., d }`. The provided
#' shift_and_log_scale_fn, tfb_masked_autoregressive_default_template, achieves
#' this property by zeroing out weights in its masked_dense layers.
#' In TensorFlow Probability, "normalizing flows" are implemented as
#' tfp.bijectors.Bijectors. The forward "autoregression" is implemented
#' using a tf.while_loop and a deep neural network (DNN) with masked weights
#' such that the autoregressive property is automatically met in the inverse.
#' A TransformedDistribution using MaskedAutoregressiveFlow(...) uses the
#' (expensive) forward-mode calculation to draw samples and the (cheap)
#' reverse-mode calculation to compute log-probabilities. Conversely, a
#' TransformedDistribution using Invert(MaskedAutoregressiveFlow(...)) uses
#' the (expensive) forward-mode calculation to compute log-probabilities and the
#' (cheap) reverse-mode calculation to compute samples.
#'
#' Given a shift_and_log_scale_fn, the forward and inverse transformations are
#' (a sequence of) affine transformations. A "valid" shift_and_log_scale_fn
#' must compute each shift (aka loc or "mu" in Germain et al. (2015)])
#' and log(scale) (aka "alpha" in Germain et al. (2015)) such that ech
#' are broadcastable with the arguments to forward and inverse, i.e., such
#' that the calculations in forward, inverse below are possible.
#'
#' For convenience, tfb_masked_autoregressive_default_template is offered as a
#' possible shift_and_log_scale_fn function. It implements the MADE
#' architecture (Germain et al., 2015). MADE is a feed-forward network that
#' computes a shift and log(scale) using masked_dense layers in a deep
#' neural network. Weights are masked to ensure the autoregressive property. It
#' is possible that this architecture is suboptimal for your task. To build
#' alternative networks, either change the arguments to
#' tfb_masked_autoregressive_default_template, use the masked_dense function to
#' roll-out your own, or use some other architecture, e.g., using tf.layers.
#' Warning: no attempt is made to validate that the shift_and_log_scale_fn
#' enforces the "autoregressive property".
#'
#' Assuming shift_and_log_scale_fn has valid shape and autoregressive semantics,
#' the forward transformation is
#'
#' ```
#' def forward(x):
#'    y = zeros_like(x)
#'    event_size = x.shape[-event_dims:].num_elements()
#'    for _ in range(event_size):
#'      shift, log_scale = shift_and_log_scale_fn(y)
#'      y = x * tf.exp(log_scale) + shift
#'    return y
#' ```
#'
#' and the inverse transformation is
#'
#' ```
#' def inverse(y):
#'   shift, log_scale = shift_and_log_scale_fn(y)
#'   return (y - shift) / tf.exp(log_scale)
#' ```
#'
#' Notice that the inverse does not need a for-loop. This is because in the
#' forward pass each calculation of shift and log_scale is based on the y
#' calculated so far (not x). In the inverse, the y is fully known, thus is
#' equivalent to the scaling used in forward after event_size passes, i.e.,
#' the "last" y used to compute shift, log_scale.
#' (Roughly speaking, this also proves the transform is bijective.)
#'
#' @section References:
#' - [Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE: Masked Autoencoder for Distribution Estimation. In _International Conference on Machine Learning_, 2015.](https://arxiv.org/abs/1502.03509)
#' - [Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, and Max Welling. Improving Variational Inference with Inverse Autoregressive Flow. In _Neural Information Processing Systems_, 2016.](https://arxiv.org/abs/1606.04934)
#' - [George Papamakarios, Theo Pavlakou, and Iain Murray. Masked Autoregressive Flow for Density Estimation. In _Neural Information Processing Systems_, 2017.](https://arxiv.org/abs/1705.07057)
#'
#' @param shift_and_log_scale_fn Function which computes shift and log_scale from both the
#' forward domain (x) and the inverse domain (y).
#' Calculation must respect the "autoregressive property". Suggested default:
#' tfb_masked_autoregressive_default_template(hidden_layers=...).
#' Typically the function contains `tf$Variables` and is wrapped using `tf$make_template`.
#'  Returning NULL for either (both) shift, log_scale is equivalent to (but more efficient than) returning zero.
#' @param is_constant_jacobian Logical, default: FALSE. When TRUE the implementation assumes log_scale
#' does not depend on the forward domain (x) or inverse domain (y) values.
#' (No validation is made; is_constant_jacobian=FALSE is always safe but possibly computationally inefficient.)
#' @param unroll_loop Logical indicating whether the `tf$while_loop` in _forward should be replaced with a
#' static for loop. Requires that the final dimension of x be known at graph construction time. Defaults to FALSE.
#' @param event_ndims integer, the intrinsic dimensionality of this bijector.
#' 1 corresponds to a simple vector autoregressive bijector as implemented by the
#' `tfb_masked_autoregressive_default_template`, 2 might be useful for a 2D convolutional shift_and_log_scale_fn and so on.
#'
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_masked_autoregressive_flow <-
  function(shift_and_log_scale_fn,
           is_constant_jacobian = FALSE,
           unroll_loop = FALSE,
           event_ndims = 1L,
           validate_args = FALSE,
           name = NULL) {
    args <- list(
      shift_and_log_scale_fn = shift_and_log_scale_fn,
      is_constant_jacobian = is_constant_jacobian,
      unroll_loop = unroll_loop,
      event_ndims = as.integer(event_ndims),
      validate_args = validate_args,
      name = name
    )

    do.call(tfp$bijectors$MaskedAutoregressiveFlow, args)
  }

#' Masked Autoregressive Density Estimator
#'
#' This will be wrapped in a make_template to ensure the variables are only
#' created once. It takes the input and returns the loc ("mu" in
#' Germain et al. (2015)) and log_scale ("alpha" in Germain et al. (2015)) from
#' the MADE network.
#'
#' Warning: This function uses masked_dense to create randomly initialized
#' `tf$Variables`. It is presumed that these will be fit, just as you would any
#' other neural architecture which uses `tf$layers$dense`.
#'
#' About Hidden Layers
#' Each element of hidden_layers should be greater than the input_depth
#' (i.e., `input_depth = tf$shape(input)[-1]` where input is the input to the
#' neural network). This is necessary to ensure the autoregressivity property.
#'
#' About Clipping
#' This function also optionally clips the log_scale (but possibly not its
#' gradient). This is useful because if log_scale is too small/large it might
#' underflow/overflow making it impossible for the MaskedAutoregressiveFlow
#' bijector to implement a bijection. Additionally, the log_scale_clip_gradient
#' bool indicates whether the gradient should also be clipped. The default does
#' not clip the gradient; this is useful because it still provides gradient
#' information (for fitting) yet solves the numerical stability problem. I.e.,
#' log_scale_clip_gradient = FALSE means `grad[exp(clip(x))] = grad[x] exp(clip(x))`
#' rather than the usual `grad[clip(x)] exp(clip(x))`.
#'
#' @section References:
#' - [Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE: Masked Autoencoder for Distribution Estimation. In _International Conference on Machine Learning_, 2015.](https://arxiv.org/abs/1502.03509)
#'
#' @param hidden_layers list-like of non-negative integer, scalars indicating the number
#'  of units in each hidden layer. Default: `list(512, 512)`.
#' @param shift_only logical indicating if only the shift term shall be
#' computed. Default: FALSE.
#' @param activation Activation function (callable). Explicitly setting to NULL implies a linear activation.
#' @param log_scale_min_clip float-like scalar Tensor, or a Tensor with the same shape as log_scale. The minimum value to clip by. Default: -5.
#' @param log_scale_max_clip float-like scalar Tensor, or a Tensor with the same shape as log_scale. The maximum value to clip by. Default: 3.
#' @param log_scale_clip_gradient logical indicating that the gradient of tf$clip_by_value should be preserved. Default: FALSE.
#' @param name A name for ops managed by this function. Default: "tfb_masked_autoregressive_default_template".
#' @param ... `tf$layers$dense` arguments
#'
#' @return list of:
#' - shift: `Float`-like `Tensor` of shift terms
#' - log_scale: `Float`-like `Tensor` of log(scale) terms
#'
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_masked_autoregressive_default_template <- function(hidden_layers,
                                                   shift_only = FALSE,
                                                   activation = tf$nn$relu,
                                                   log_scale_min_clip = -5,
                                                   log_scale_max_clip = 3,
                                                   log_scale_clip_gradient = FALSE,
                                                   name = NULL,
                                                   ...) {
  tfp$bijectors$masked_autoregressive_default_template(
    as.integer(hidden_layers),
    shift_only,
    activation,
    log_scale_min_clip,
    log_scale_max_clip,
    log_scale_clip_gradient,
    name,
    ...
  )
}

#' Autoregressively masked dense layer
#'
#' Analogous to `tf$layers$dense`.
#'
#' See Germain et al. (2015)for detailed explanation.
#'
#' @section References:
#' - [Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE: Masked Autoencoder for Distribution Estimation. In _International Conference on Machine Learning_, 2015.](https://arxiv.org/abs/1502.03509)
#'
#' @param inputs Tensor input.
#' @param units integer scalar representing the dimensionality of the output space.
#' @param num_blocks integer scalar representing the number of blocks for the MADE masks.
#' @param exclusive logical scalar representing whether to zero the diagonal of
#' the mask, used for the first layer of a MADE.
#' @param kernel_initializer Initializer function for the weight matrix.
#' If NULL (default), weights are initialized using the `tf$glorot_random_initializer`
#' @param reuse logical scalar representing whether to reuse the weights of a previous layer by the same name.
#' @param name string used to describe ops managed by this function.
#' @param ... `tf$layers$dense` arguments
#'
#' @return tensor
#'
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_masked_dense <- function(inputs,
                         units,
                         num_blocks = NULL,
                         exclusive = FALSE,
                         kernel_initializer = NULL,
                         reuse = NULL,
                         name = NULL,
                         ...) {
  tfp$bijectors$masked_dense(
    inputs,
    as.integer(units),
    as.integer(num_blocks),
    exclusive,
    kernel_initializer,
    reuse,
    name,
    ...
  )
}

#' Build a scale-and-shift function using a multi-layer neural network
#'
#' This will be wrapped in a make_template to ensure the variables are only
#' created once. It takes the d-dimensional input `x[0:d]` and returns the `D-d`
#' dimensional outputs loc ("mu") and log_scale ("alpha").
#'
#' The default template does not support conditioning and will raise an
#' exception if condition_kwargs are passed to it. To use conditioning in
#' real nvp bijector, implement a conditioned shift/scale template that
#' handles the condition_kwargs.
#'
#' @section References:
#' - [George Papamakarios, Theo Pavlakou, and Iain Murray. Masked Autoregressive Flow for Density Estimation. In _Neural Information Processing Systems_, 2017.](https://arxiv.org/abs/1705.07057)
#' @param hidden_layers list-like of non-negative integer, scalars indicating the number
#'  of units in each hidden layer. Default: `list(512, 512)`.
#' @param shift_only logical indicating if only the shift term shall be
#' computed (i.e. NICE bijector). Default: FALSE.
#' @param activation Activation function (callable). Explicitly setting to NULL implies a linear activation.
#' @param name A name for ops managed by this function. Default: "tfb_real_nvp_default_template".
#' @param ... tf$layers$dense arguments
#'
#' @inherit tfb_masked_autoregressive_default_template return
#'
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_real_nvp_default_template <- function(hidden_layers,
                                      shift_only = FALSE,
                                      activation = tf$nn$relu,
                                      name = NULL,
                                      ...) {
  tfp$bijectors$real_nvp_default_template(as.integer(hidden_layers),
                                          shift_only,
                                          activation,
                                          name,
                                          ...)
}

#' RealNVP affine coupling layer for vector-valued events
#'
#' Real NVP models a normalizing flow on a D-dimensional distribution via a
#' single D-d-dimensional conditional distribution (Dinh et al., 2017):
#' `y[d:D] = x[d:D] * tf.exp(log_scale_fn(x[0:d])) + shift_fn(x[0:d])`
#' `y[0:d] = x[0:d]`
#' The last D-d units are scaled and shifted based on the first d units only,
#' while the first d units are 'masked' and left unchanged. Real NVP's
#' shift_and_log_scale_fn computes vector-valued quantities.
#' For scale-and-shift transforms that do not depend on any masked units, i.e.
#' d=0, use the tfb_affine bijector with learned parameters instead.
#' Masking is currently only supported for base distributions with
#' event_ndims=1. For more sophisticated masking schemes like checkerboard or
#' channel-wise masking (Papamakarios et al., 2016), use the tfb_permute
#' bijector to re-order desired masked units into the first d units. For base
#' distributions with event_ndims > 1, use the tfb_reshape bijector to
#' flatten the event shape.
#'
#' Recall that the MAF bijector (Papamakarios et al., 2016) implements a
#' normalizing flow via an autoregressive transformation. MAF and IAF have
#' opposite computational tradeoffs - MAF can train all units in parallel but
#' must sample units sequentially, while IAF must train units sequentially but
#' can sample in parallel. In contrast, Real NVP can compute both forward and
#' inverse computations in parallel. However, the lack of an autoregressive
#' transformations makes it less expressive on a per-bijector basis.
#'
#' A "valid" shift_and_log_scale_fn must compute each shift (aka loc or
#' "mu" in Papamakarios et al. (2016) and log(scale) (aka "alpha" in
#' Papamakarios et al. (2016)) such that each are broadcastable with the
#' arguments to forward and inverse, i.e., such that the calculations in
#' forward, inverse below are possible. For convenience,
#' real_nvp_default_nvp is offered as a possible shift_and_log_scale_fn function.
#'
#' NICE (Dinh et al., 2014) is a special case of the Real NVP bijector
#' which discards the scale transformation, resulting in a constant-time
#' inverse-log-determinant-Jacobian. To use a NICE bijector instead of Real
#' NVP, shift_and_log_scale_fn should return (shift, NULL), and
#' is_constant_jacobian should be set to TRUE in the RealNVP constructor.
#' Calling tfb_real_nvp_default_template with shift_only=TRUE returns one such
#' NICE-compatible shift_and_log_scale_fn.
#'
#' Caching: the scalar input depth D of the base distribution is not known at
#' construction time. The first call to any of forward(x), inverse(x),
#' inverse_log_det_jacobian(x), or forward_log_det_jacobian(x) memoizes
#' D, which is re-used in subsequent calls. This shape must be known prior to
#'  graph execution (which is the case if using `tf$layers`).
#'
#' @section References:
#' - [George Papamakarios, Theo Pavlakou, and Iain Murray. Masked Autoregressive Flow for Density Estimation. In _Neural Information Processing Systems_, 2017.](https://arxiv.org/abs/1705.07057)
#' - [Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density Estimation using Real NVP. In _International Conference on Learning Representations_, 2017.](https://arxiv.org/abs/1605.08803)
#' - [Laurent Dinh, David Krueger, and Yoshua Bengio. NICE: Non-linear Independent Components Estimation._arXiv preprint arXiv:1410.8516_,2014.](https://arxiv.org/abs/1410.8516)
#' - [Eric Jang. Normalizing Flows Tutorial, Part 2: Modern Normalizing Flows. Technical Report_, 2018.](https://blog.evjang.com/2018/01/nf2.html)
#'
#' @param num_masked integer indicating that the first d units of the event
#' should be masked. Must be in the closed interval `[1, D-1]`, where D
#' is the event size of the base distribution.
#' @param shift_and_log_scale_fn Function which computes shift and log_scale from both the
#' forward domain (x) and the inverse domain (y).
#' Calculation must respect the "autoregressive property". Suggested default:
#' `tfb_real_nvp_default_template(hidden_layers=...)`.
#' Typically the function contains `tf$Variables` and is wrapped using `tf$make_template`.
#'  Returning NULL for either (both) shift, log_scale is equivalent to (but more efficient than) returning zero.
#' @param is_constant_jacobian Logical, default: FALSE. When TRUE the implementation assumes log_scale
#' does not depend on the forward domain (x) or inverse domain (y) values.
#' (No validation is made; is_constant_jacobian=FALSE is always safe but possibly computationally inefficient.)
#'
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_real_nvp <-
  function(num_masked,
           shift_and_log_scale_fn,
           is_constant_jacobian = FALSE,
           validate_args = FALSE,
           name = NULL) {
    args <- list(
      num_masked = as.integer(num_masked),
      shift_and_log_scale_fn = shift_and_log_scale_fn,
      is_constant_jacobian = is_constant_jacobian,
      validate_args = validate_args,
      name = name
    )

    do.call(tfp$bijectors$RealNVP, args)
  }


#' Computes `g(L) = inv(L)`, where L is a lower-triangular matrix
#'
#' L must be nonsingular; equivalently, all diagonal entries of L must be nonzero.
#' The input must have rank >= 2.  The input is treated as a batch of matrices
#' with batch shape `input.shape[:-2]`, where each matrix has dimensions
#' `input.shape[-2]` by `input.shape[-1]` (hence `input.shape[-2]` must equal `input.shape[-1]`).
#'
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_matrix_inverse_tri_l <- function(validate_args = FALSE,
                                     name = "matrix_inverse_tril") {
  args <- list(validate_args = validate_args,
               name = name)

  do.call(tfp$bijectors$MatrixInverseTriL, args)
}


#' Matrix-vector multiply using LU decomposition
#'
#' This bijector is identical to the "Convolution1x1" used in Glow (Kingma and Dhariwal, 2018).
#'
#' Warning: this bijector never verifies the scale matrix (as parameterized by LU
#' ecomposition) is invertible. Ensuring this is the case is the caller's responsibility.
#'
#' @section References:
#' - [Diederik P. Kingma, Prafulla Dhariwal. Glow: Generative Flow with Invertible 1x1 Convolutions. _arXiv preprint arXiv:1807.03039_, 2018.](https://arxiv.org/abs/1807.03039)
#'
#' @param lower_upper The LU factorization as returned by `tf$linalg$lu`.
#' @param permutation The LU factorization permutation as returned by `tf$linalg$lu`.
#'
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_matvec_lu <- function(lower_upper,
                          permutation,
                          validate_args = FALSE,
                          name = NULL) {
  args <- list(
    lower_upper = lower_upper,
    permutation = permutation,
    validate_args = validate_args,
    name = name
  )

  do.call(tfp$bijectors$MatvecLU, args)
}

#' Computes`Y = g(X) = NormalCDF(x)`
#'
#' This bijector maps inputs from `[-inf, inf]` to `[0, 1]`. The inverse of the
#' bijector applied to a uniform random variable `X ~ U(0, 1)` gives back a
#' random variable with the [Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution):
#'
#'  `Y ~ Normal(0, 1)`
#' `pdf(y; 0., 1.) = 1 / sqrt(2 * pi) * exp(-y ** 2 / 2)`
#'
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_normal_cdf <- function(validate_args = FALSE,
                           name = "normal") {
  args <- list(validate_args = validate_args,
               name = name)

  do.call(tfp$bijectors$NormalCDF, args)
}

#' Bijector which maps a tensor x_k that has increasing elements in the last dimension to an unconstrained tensor y_k
#'
#' Both the domain and the codomain of the mapping is `[-inf, inf]`, however,
#' the input of the forward mapping must be strictly increasing.
#' The inverse of the bijector applied to a normal random vector `y ~ N(0, 1)`
#' gives back a sorted random vector with the same distribution `x ~ N(0, 1)`
#' where x = sort(y)
#'
#' On the last dimension of the tensor, Ordered bijector performs:
#' `y[0] = x[0]`
#' `y[1:] = tf$log(x[1:] - x[:-1])`
#'
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_ordered <- function(validate_args = FALSE,
                        name = "ordered") {
  args <- list(validate_args = validate_args,
               name = name)

  do.call(tfp$bijectors$Ordered, args)
}


#' Permutes the rightmost dimension of a Tensor
#'
#' @param permutation An integer-like vector-shaped Tensor representing the
#' permutation to apply to the axis dimension of the transformed Tensor.
#' @param axis Scalar integer Tensor representing the dimension over which to tf$gather.
#' axis must be relative to the end (reading left to right) thus must be negative.
#' Default value: -1 (i.e., right-most).
#'
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_permute <- function(permutation,
                        axis = -1L,
                        validate_args = FALSE,
                        name = NULL) {
  args <- list(
    permutation = as.integer(permutation),
    axis = as_nullable_integer(axis),
    validate_args = validate_args,
    name = name
  )

  do.call(tfp$bijectors$Permute, args)
}

#' Computes`Y = g(X) = (1 + X * c)**(1 / c)`, where `X >= -1 / c`
#'
#' The [power transform](https://en.wikipedia.org/wiki/Power_transform) maps
#' inputs from `[0, inf]` to `[-1/c, inf]`; this is equivalent to the inverse of this bijector.
#' This bijector is equivalent to the Exp bijector when c=0.
#'
#' @param power float scalar indicating the transform power, i.e.,
#' `Y = g(X) = (1 + X * c)**(1 / c)` where c is the power.
#'
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_power_transform <- function(power,
                                validate_args = FALSE,
                                name = "power_transform") {
  args <- list(power = power,
               validate_args = validate_args,
               name = name)

  do.call(tfp$bijectors$PowerTransform, args)
}


#' A Bijector that computes `b(x) = 1. / x`
#'
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_reciprocal <- function(validate_args = FALSE,
                           name = "reciprocal") {
  args <- list(validate_args = validate_args,
               name = name)

  do.call(tfp$bijectors$Reciprocal, args)
}

#' Reshapes the event_shape of a Tensor
#'
#' The semantics generally follow that of `tf$reshape()`, with a few differences:
#'   * The user must provide both the input and output shape, so that
#'     the transformation can be inverted. If an input shape is not
#'     specified, the default assumes a vector-shaped input, i.e.,
#'     `event_shape_in = list(-1)`.
#'   * The Reshape bijector automatically broadcasts over the leftmost
#'   dimensions of its input (sample_shape and batch_shape); only
#'   the rightmost event_ndims_in dimensions are reshaped. The
#'   number of dimensions to reshape is inferred from the provided
#'   event_shape_in (`event_ndims_in = length(event_shape_in))`.
#'
#' @param event_shape_out An integer-like vector-shaped Tensor
#' representing the event shape of the transformed output.
#' @param event_shape_in An optional integer-like vector-shape Tensor
#' representing the event shape of the input. This is required in
#' order to define inverse operations; the default of list(-1) assumes a vector-shaped input.
#'
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_reshape <- function(event_shape_out,
                        event_shape_in = c(-1),
                        validate_args = FALSE,
                        name = NULL) {
  args <- list(
    event_shape_out = normalize_shape(event_shape_out),
    event_shape_in = normalize_shape(event_shape_in),
    validate_args = validate_args,
    name = name
  )

  do.call(tfp$bijectors$Reshape, args)
}


#' Transforms unconstrained vectors to TriL matrices with positive diagonal
#'
#' This is implemented as a simple tfb_chain of tfb_fill_triangular followed by
#' tfb_transform_diagonal, and provided mostly as a convenience.
#' The default setup is somewhat opinionated, using a Softplus transformation followed by a
#'  small shift (1e-5) which attempts to avoid numerical issues from zeros on the diagonal.
#'
#' @param diag_bijector Bijector instance, used to transform the output diagonal to be positive.
#' Default value: NULL (i.e., `tfb_softplus()`).
#' @param diag_shift Float value broadcastable and added to all diagonal entries after applying the
#' diag_bijector. Setting a positive value forces the output diagonal entries to be positive, but
#' prevents inverting the transformation for matrices with diagonal entries less than this value.
#' Default value: 1e-5.
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_scale_tri_l <- function(diag_bijector = NULL,
                            diag_shift = 1e-5,
                            validate_args = FALSE,
                            name = "scale_tril") {
  args <- list(
    diag_bijector = diag_bijector,
    diag_shift = diag_shift,
    validate_args = validate_args,
    name = name
  )
  do.call(tfp$bijectors$ScaleTriL, args)
}

#' Computes`Y = g(X) = Sinh( (Arcsinh(X) + skewness) * tailweight )`
#'
#' For skewness in `(-inf, inf)` and tailweight in `(0, inf)`, this
#' transformation is a diffeomorphism of the real line `(-inf, inf)`.
#' The inverse transform is `X = g^{-1}(Y) = Sinh( ArcSinh(Y) / tailweight - skewness )`.
#' The SinhArcsinh transformation of the Normal is described in
#' [Sinh-arcsinh distributions](https://oro.open.ac.uk/22510/)
#'
#' This Bijector allows a similar transformation of any distribution supported on `(-inf, inf)`.
#'
#' # Meaning of the parameters
#' * If skewness = 0 and tailweight = 1, this transform is the identity.
#' * Positive (negative) skewness leads to positive (negative) skew.
#' * positive skew means, for unimodal X centered at zero, the mode of Y is "tilted" to the right.
#' * positive skew means positive values of Y become more likely, and negative values become less likely.
#' * Larger (smaller) tailweight leads to fatter (thinner) tails.
#' * Fatter tails mean larger values of |Y| become more likely.
#' * If X is a unit Normal, tailweight < 1 leads to a distribution that is "flat" around Y = 0, and a very steep drop-off in the tails.
#' * If X is a unit Normal, tailweight > 1 leads to a distribution more peaked at the mode with heavier tails.
#' To see the argument about the tails, note that for |X| >> 1 and |X| >> (|skewness| * tailweight)**tailweight, we have
#' Y approx 0.5 X**tailweight e**(sign(X) skewness * tailweight).
#'
#' @param skewness Skewness parameter.  Float-type Tensor.  Default is 0 of type float32.
#' @param tailweight  Tailweight parameter.  Positive Tensor of same dtype as skewness and broadcastable shape.  Default is 1 of type float32.
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_sinh_arcsinh <- function(skewness = NULL,
                             tailweight = NULL,
                             validate_args = FALSE,
                             name = "SinhArcsinh") {
  args <- list(
    skewness = skewness,
    tailweight = tailweight,
    validate_args = validate_args,
    name = name
  )
  do.call(tfp$bijectors$SinhArcsinh, args)
}

#' Computes `Y = g(X) = exp([X 0]) / sum(exp([X 0]))`
#'
#' To implement [softmax](https://en.wikipedia.org/wiki/Softmax_function) as a
#' bijection, the forward transformation appends a value to the input and the
#' inverse removes this coordinate. The appended coordinate represents a pivot,
#' e.g., softmax(x) = exp(x-c) / sum(exp(x-c)) where c is the implicit last
#' coordinate.
#'
#' At first blush it may seem like the [Invariance of domain](https://en.wikipedia.org/wiki/Invariance_of_domain)
#' theorem implies this implementation is not a bijection. However, the appended dimension
#' makes the (forward) image non-open and the theorem does not directly apply.
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_softmax_centered <- function(validate_args = FALSE,
                                 name = "softmax_centered") {
  args <- list(validate_args = validate_args,
               name = name)
  do.call(tfp$bijectors$SoftmaxCentered, args)
}

#' Computes `Y = g(X) = Log[1 + exp(X)]`
#'
#' The softplus Bijector has the following two useful properties:
#' * The domain is the positive real numbers
#' * softplus(x) approx x, for large x, so it does not overflow as easily as the Exp Bijector.
#'
#' The optional nonzero hinge_softness parameter changes the transition at zero.
#' With hinge_softness = c, the bijector is:
#'
#' ````
#' f_c(x) := c * g(x / c) = c * Log[1 + exp(x / c)].
#' ```
#'
#' For large x >> 1,
#'
#' ```
#' c * Log[1 + exp(x / c)] approx c * Log[exp(x / c)] = x
#' ```
#'
#' so the behavior for large x is the same as the standard softplus.
#' As c > 0 approaches 0 from the right, f_c(x) becomes less and less soft,
#' approaching max(0, x).
#' * c = 1 is the default.
#' * c > 0 but small means f(x) approx ReLu(x) = max(0, x).
#' * c < 0 flips sign and reflects around the y-axis: f_{-c}(x) = -f_c(-x).
#' * c = 0 results in a non-bijective transformation and triggers an exception.
#' Note: log(.) and exp(.) are applied element-wise but the Jacobian is a reduction over the event space.
#'
#' @param hinge_softness Nonzero floating point Tensor.  Controls the softness of what
#' would otherwise be a kink at the origin.  Default is 1.0.
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_softplus <- function(hinge_softness = NULL,
                         validate_args = FALSE,
                         name = "softplus") {
  args <- list(hinge_softness = hinge_softness,
               validate_args = validate_args,
               name = name)
  do.call(tfp$bijectors$Softplus, args)
}

#' Computes `Y = g(X) = X / (1 + |X|)`
#'
#' The softsign Bijector has the following two useful properties:
#' * The domain is all real numbers
#' * softsign(x) approx sgn(x), for large |x|.
#'
#' @inheritParams tfb_identity
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_softsign <- function(validate_args = FALSE,
                         name = "softsign") {
  args <- list(validate_args = validate_args,
               name = name)
  do.call(tfp$bijectors$Softsign, args)
}

#' Computes`g(X) = X^2`; X is a positive real number.
#'
#' g is a bijection between the non-negative real numbers (R_+) and the non-negative real numbers.
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_square <- function(validate_args = FALSE,
                       name = "square") {
  args <- list(validate_args = validate_args,
               name = name)
  do.call(tfp$bijectors$Square, args)
}

#' Computes `Y = tanh(X)`
#'
#' `Y = tanh(X)`, therefore Y in `(-1, 1)`.
#'
#' This can be achieved by an affine transform of the Sigmoid bijector, i.e., it is equivalent to
#'
#' \code{tfb_chain(list(tfb_affine(shift = -1, scale = 2),
#'                tfb_sigmoid(),
#'                tfb_affine(scale = 2)))}
#'
#'
#' However, using the Tanh bijector directly is slightly faster and more numerically stable.
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_tanh <- function(validate_args = FALSE,
                     name = "tanh") {
  args <- list(validate_args = validate_args,
               name = name)
  do.call(tfp$bijectors$Tanh, args)
}

#' Applies a Bijector to the diagonal of a matrix
#'
#' @param diag_bijector Bijector instance used to transform the diagonal.
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_transform_diagonal <- function(diag_bijector,
                                   validate_args = FALSE,
                                   name = "transform_diagonal") {
  args <- list(diag_bijector = diag_bijector,
               validate_args = validate_args,
               name = name)
  do.call(tfp$bijectors$TransformDiagonal, args)
}

#' Computes`Y = g(X) = transpose_rightmost_dims(X, rightmost_perm)`
#'
#' This bijector is semantically similar to tf.transpose except that it
#' transposes only the rightmost "event" dimensions. That is, unlike
#' `tf$transpose` the perm argument is itself a permutation of
#' `tf$range(rightmost_transposed_ndims)` rather than `tf$range(tf$rank(x))`,
#' i.e., users specify the (rightmost) dimensions to permute, not all dimensions.
#'
#' The actual (forward) transformation is:
#'
#' \code{sample_batch_ndims <- tf$rank(x) - tf$size(perm)
#' perm = tf$concat(list(tf$range(sample_batch_ndims), sample_batch_ndims + perm),axis=0)
#' tf$transpose(x, perm)}
#'
#' @param perm Positive integer vector-shaped Tensor representing permutation of
#' rightmost dims (for forward transformation).  Note that the 0th index
#' represents the first of the rightmost dims and the largest value must be
#' rightmost_transposed_ndims - 1 and corresponds to `tf$rank(x) - 1`.
#' Only one of perm and rightmost_transposed_ndims can (and must) be specified.
#' Default value: `tf$range(start=rightmost_transposed_ndims, limit=-1, delta=-1)`.
#' @param rightmost_transposed_ndims Positive integer scalar-shaped Tensor
#' representing the number of rightmost dimensions to permute.
#' Only one of perm and rightmost_transposed_ndims can (and must) be
#' specified. Default value: `tf$size(perm)`.
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_transpose <- function(perm = NULL,
                          rightmost_transposed_ndims = NULL,
                          validate_args = FALSE,
                          name = "transpose") {
  args <- list(
    perm = as_nullable_integer(perm),
    rightmost_transposed_ndims = as_nullable_integer(rightmost_transposed_ndims),
    validate_args = validate_args,
    name = name
  )
  do.call(tfp$bijectors$Transpose, args)
}

#' Computes`Y = g(X) = 1 - exp((-X / scale) ** concentration)` where X >= 0
#'
#' This bijector maps inputs from `[0, inf]` to `[0, 1]`. The inverse of the
#' bijector applied to a uniform random variable X ~ U(0, 1) gives back a
#' random variable with the [Weibull distribution](https://en.wikipedia.org/wiki/Weibull_distribution):
#'
#' `Y ~ Weibull(scale, concentration)`
#' `pdf(y; scale, concentration, y >= 0) = (concentration / scale) * (y / scale)**(concentration - 1) * exp(-(y / scale)**concentration)`
#'
#'
#' @param scale Positive Float-type Tensor that is the same dtype and is
#' broadcastable with concentration.
#' This is l in `Y = g(X) = 1 - exp((-x / l) ** k)`.
#' @param concentration Positive Float-type Tensor that is the same dtype and is
#' broadcastable with scale.
#' This is k in `Y = g(X) = 1 - exp((-x / l) ** k)`.
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_weibull <- function(scale = 1,
                        concentration = 1,
                        validate_args = FALSE,
                        name = "weibull") {
  args <- list(
    scale = scale,
    concentration = concentration,
    validate_args = validate_args,
    name = name
  )
  do.call(tfp$bijectors$Weibull, args)
}


#' Maps unconstrained reals to Cholesky-space correlation matrices.
#'
#' This bijector is a mapping between `R^{n}` and the `n`-dimensional manifold of
#' Cholesky-space correlation matrices embedded in `R^{m^2}`, where `n` is the
#' `(m - 1)`th triangular number; i.e. `n = 1 + 2 + ... + (m - 1)`.
#'
#' Mathematical Details
#'
#' The image of unconstrained reals under the `CorrelationCholesky` bijector is
#' the set of correlation matrices which are positive definite.
#' A [correlation matrix](https://en.wikipedia.org/wiki/Correlation_and_dependence#Correlation_matrices)
#' can be characterized as a symmetric positive semidefinite matrix with 1s on
#' the main diagonal. However, the correlation matrix is positive definite if no
#' component can be expressed as a linear combination of the other components.
#' For a lower triangular matrix `L` to be a valid Cholesky-factor of a positive
#' definite correlation matrix, it is necessary and sufficient that each row of
#' `L` have unit Euclidean norm. To see this, observe that if `L_i` is the
#' `i`th row of the Cholesky factor corresponding to the correlation matrix `R`,
#' then the `i`th diagonal entry of `R` satisfies:
#' ```
#' 1 = R_i,i = L_i . L_i = ||L_i||^2
#' ```
#' where '.' is the dot product of vectors and `||...||` denotes the Euclidean
#' norm. Furthermore, observe that `R_i,j` lies in the interval `[-1, 1]`. By the
#' Cauchy-Schwarz inequality:
#' ````
#' |R_i,j| = |L_i . L_j| <= ||L_i|| ||L_j|| = 1
#' ````
#' This is a consequence of the fact that `R` is symmetric positive definite with
#' 1s on the main diagonal.
#' The LKJ distribution with `input_output_cholesky=TRUE` generates samples from
#' (and computes log-densities on) the set of Cholesky factors of positive
#' definite correlation matrices. The `CorrelationCholesky` bijector provides
#' a bijective mapping from unconstrained reals to the support of the LKJ
#' distribution.
#'
#' @section References:
#' - [Stan Manual. Section 24.2. Cholesky LKJ Correlation Distribution.](https://mc-stan.org/docs/2_18/functions-reference/cholesky-lkj-correlation-distribution.html)
#' - Daniel Lewandowski, Dorota Kurowicka, and Harry Joe, "Generating random correlation matrices based on vines and extended onion method," Journal of Multivariate Analysis 100 (2009), pp 1989-2001.
#'
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_correlation_cholesky <- function(validate_args = FALSE,
                                     name = "correlation_cholesky") {
  args <- list(validate_args = validate_args,
               name = name)
  do.call(tfp$bijectors$CorrelationCholesky, args)
}

#' Computes the cumulative sum of a tensor along a specified axis.
#'
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @param axis `int` indicating the axis along which to compute the cumulative sum.
#'  Note that positive (and zero) values are not supported
#'
#' @export
tfb_cumsum <- function(axis = -1,
                       validate_args = FALSE,
                       name='cumsum') {
  tfp$bijectors$Cumsum(
    axis = as.integer(axis),
    validate_args = validate_args,
    name = name
  )
}

#' Bijector which applies a Stick Breaking procedure.
#'
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_iterated_sigmoid_centered <- function(validate_args = FALSE,
                                          name = 'iterated_sigmoid') {
  tfp$bijectors$IteratedSigmoidCentered(
    validate_args = validate_args,
    name = name
  )
}

#' Compute `Y = g(X; shift) = X + shift`.
#'
#' where `shift` is a numeric `Tensor`.
#' @inherit tfb_identity return params
#' @family bijectors
#' @param shift floating-point tensor
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_shift <- function(shift,
                      validate_args = FALSE,
                      name = 'shift') {
  tfp$bijectors$Shift(shift = shift,
                      validate_args = validate_args,
                      name = name)
}

#' Pads a value to the `event_shape` of a `Tensor`.
#'
#' The semantics of `bijector_pad` generally follow that of `tf$pad()`
#' except that `bijector_pad`'s `paddings` argument applies to the rightmost
#' dimensions. Additionally, the new argument `axis` enables overriding the
#' dimensions to which `paddings` is applied. Like `paddings`, the `axis`
#' argument is also relative to the rightmost dimension and must therefore be
#' negative.
#' The argument `paddings` is a vector of `integer` pairs each representing the
#' number of left and/or right `constant_values` to pad to the corresponding
#' righmost dimensions. That is, unless `axis` is specified`, specifiying `k`
#' different `paddings` means the rightmost `k` dimensions will be "grown" by the
#' sum of the respective `paddings` row. When `axis` is specified, it indicates
#' the dimension to which the corresponding `paddings` element is applied. By
#' default `axis` is `NULL` which means it is logically equivalent to
#' `range(start=-len(paddings), limit=0)`, i.e., the rightmost dimensions.
#'
#' @inherit tfb_identity return params
#' @param paddings A vector-shaped `Tensor` of `integer` pairs representing the number
#' of elements to pad on the left and right, respectively.
#' Default value: `list(reticulate::tuple(0L, 1L))`.
#' @param mode One of `'CONSTANT'`, `'REFLECT'`, or `'SYMMETRIC'`
#' (case-insensitive). For more details, see `tf$pad`.
#' @param constant_values In "CONSTANT" mode, the scalar pad value to use. Must be
#' same type as `tensor`. For more details, see `tf$pad`.
#' @param axis The dimensions for which `paddings` are applied. Must be 1:1 with
#' `paddings` or `NULL`.
#' Default value: `NULL` (i.e., `tf$range(start = -length(paddings), limit = 0)`).
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_pad <- function(paddings = list(c(0, 1)),
                    mode = 'CONSTANT',
                    constant_values = 0,
                    axis = NULL,
                    validate_args = FALSE,
                    name = NULL) {
  tfp$bijectors$Pad(
    paddings = as_integer_list(paddings),
    mode = mode,
    constant_values = constant_values,
    axis = as_nullable_integer(axis),
    validate_args = validate_args,
    name = name
  )
}

#' Compute `Y = g(X; scale) = scale @ X`
#'
#' In TF parlance, the `scale` term is logically equivalent to:
#' ```
#' scale = tf$diag(scale_diag)
#' ```
#' The `scale` term is applied without materializing a full dense matrix.
#'
#' @inherit tfb_identity return params
#' @param scale_diag Floating-point `Tensor` representing the diagonal matrix.
#' `scale_diag` has shape `[N1, N2, ...  k]`, which represents a k x k
#' diagonal matrix.
#' @param adjoint `logical` indicating whether to use the `scale` matrix as
#' specified or its adjoint. Default value: `FALSE`.
#' @param dtype `tf$DType` to prefer when converting args to `Tensor`s. Else, we
#' fall back to a common dtype inferred from the args, finally falling back
#' to `float32`.
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_scale_matvec_diag <- function(scale_diag,
                                  adjoint = FALSE,
                                  validate_args = FALSE,
                                  name = 'scale_matvec_diag',
                                  dtype = NULL) {
  tfp$bijectors$ScaleMatvecDiag(
    scale_diag = scale_diag,
    adjoint = adjoint,
    validate_args = validate_args,
    name = name,
    dtype = dtype
  )
}

#' Compute `Y = g(X; scale) = scale @ X`.
#'
#' `scale` is a `LinearOperator`.
#' If `X` is a scalar then the forward transformation is: `scale * X`
#' where `*` denotes broadcasted elementwise product.
#' @inherit tfb_identity return params
#' @param scale  Subclass of `LinearOperator`. Represents the (batch, non-singular)
#' linear transformation by which the `Bijector` transforms inputs.
#' @param adjoint `logical` indicating whether to use the `scale` matrix as
#' specified or its adjoint. Default value: `FALSE`.
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_scale_matvec_linear_operator <- function(scale,
                                             adjoint = FALSE,
                                             validate_args = FALSE,
                                             name = 'scale_matvec_linear_operator') {
  tfp$bijectors$ScaleMatvecLinearOperator(
    scale = scale,
    adjoint = adjoint,
    validate_args = validate_args,
    name = name
  )
}

#' Matrix-vector multiply using LU decomposition.
#'
#' This bijector is identical to the "Convolution1x1" used in Glow (Kingma and Dhariwal, 2018).
#'
#' @section References:
#' - [Diederik P. Kingma, Prafulla Dhariwal. Glow: Generative Flow with Invertible 1x1 Convolutions. _arXiv preprint arXiv:1807.03039_, 2018.](https://arxiv.org/abs/1807.03039)
#'
#' @inherit tfb_identity return params
#' @param  lower_upper The LU factorization as returned by `tf$linalg$lu`.
#' @param permutation The LU factorization permutation as returned by `tf$linalg$lu`.
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_scale_matvec_lu <- function(lower_upper,
                                permutation,
                                validate_args = FALSE,
                                name = NULL) {
  tfp$bijectors$ScaleMatvecLU(
    lower_upper = lower_upper,
    permutation = permutation,
    validate_args = validate_args,
    name = name
  )
}

#' Compute `Y = g(X; scale) = scale @ X`.
#'
#' The `scale` term is presumed lower-triangular and non-singular (ie, no zeros
#' on the diagonal), which permits efficient determinant calculation (linear in
#' matrix dimension, instead of cubic).
#'
#' @inherit tfb_identity return params
#' @param scale_tril Floating-point `Tensor` representing the lower triangular
#' matrix. `scale_tril` has shape `[N1, N2, ...  k, k]`, which represents a
#' k x k lower triangular matrix.
#' When `NULL` no `scale_tril` term is added to `scale`.
#' The upper triangular elements above the diagonal are ignored.
#' @param adjoint `logical` indicating whether to use the `scale` matrix as
#' specified or its adjoint. Note that lower-triangularity is taken into
#' account first: the region above the diagonal of `scale_tril` is treated
#' as zero (irrespective of the `adjoint` setting). A lower-triangular
#' input with `adjoint=TRUE` will behave like an upper triangular
#' transform. Default value: `FALSE`.
#' @param dtype `tf$DType` to prefer when converting args to `Tensor`s. Else, we
#' fall back to a common dtype inferred from the args, finally falling back
#' to float32.
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_scale_matvec_tri_l <- function(scale_tril,
                                   adjoint = FALSE,
                                   validate_args = FALSE,
                                   name = 'scale_matvec_tril',
                                   dtype = NULL) {
  tfp$bijectors$ScaleMatvecTriL(
    scale_tril = scale_tril,
    adjoint = adjoint,
    validate_args = validate_args,
    name = name,
    dtype = dtype
  )
}

#' A piecewise rational quadratic spline, as developed in Conor et al.(2019).
#'
#' This transformation represents a monotonically increasing piecewise rational
#' quadratic function. Outside of the bounds of `knot_x`/`knot_y`, the transform
#' behaves as an identity function.
#'
#' Typically this bijector will be used as part of a chain, with splines for
#' trailing `x` dimensions conditioned on some of the earlier `x` dimensions, and
#' with the inverse then solved first for unconditioned dimensions, then using
#' conditioning derived from those inverses, and so forth.
#'
#' For each argument, the innermost axis indexes bins/knots and batch axes
#' index axes of `x`/`y` spaces. A `RationalQuadraticSpline` with a separate
#' transform for each of three dimensions might have `bin_widths` shaped
#' `[3, 32]`. To use the same spline for each of `x`'s three dimensions we may
#' broadcast against `x` and use a `bin_widths` parameter shaped `[32]`.
#'
#' Parameters will be broadcast against each other and against the input
#' `x`/`y`s, so if we want fixed slopes, we can use kwarg `knot_slopes=1`.
#' A typical recipe for acquiring compatible bin widths and heights would be:
#'
#' ```
#' nbins <- unconstrained_vector$shape[-1]
#' range_min <- 1
#' range_max <- 1
#' min_bin_size = 1e-2
#' scale <- range_max - range_min - nbins * min_bin_size
#' bin_widths = tf$math$softmax(unconstrained_vector) * scale + min_bin_size
#' ```
#'
#' @section References:
#' - [Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios. Neural Spline Flows. _arXiv preprint arXiv:1906.04032_, 2019.](https://arxiv.org/abs/1906.04032)
#' @inherit tfb_identity return params
#' @param bin_widths The widths of the spans between subsequent knot `x` positions,
#' a floating point `Tensor`. Must be positive, and at least 1-D. Innermost
#' axis must sum to the same value as `bin_heights`. The knot `x` positions
#' will be a first at `range_min`, followed by knots at `range_min +
#' cumsum(bin_widths, axis=-1)`.
#' @param bin_heights The heights of the spans between subsequent knot `y`
#' positions, a floating point `Tensor`. Must be positive, and at least
#' 1-D. Innermost axis must sum to the same value as `bin_widths`. The knot
#' `y` positions will be a first at `range_min`, followed by knots at
#' `range_min + cumsum(bin_heights, axis=-1)`.
#' @param knot_slopes The slope of the spline at each knot, a floating point
#' `Tensor`. Must be positive. `1`s are implicitly padded for the first and
#' last implicit knots corresponding to `range_min` and `range_min +
#' sum(bin_widths, axis=-1)`. Innermost axis size should be 1 less than
#' that of `bin_widths`/`bin_heights`, or 1 for broadcasting.
#' @param range_min The `x`/`y` position of the first knot, which has implicit
#' slope `1`. `range_max` is implicit, and can be computed as `range_min +
#'  sum(bin_widths, axis=-1)`. Scalar floating point `Tensor`.
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_rational_quadratic_spline <- function(bin_widths,
                                          bin_heights,
                                          knot_slopes,
                                          range_min = -1,
                                          validate_args = FALSE,
                                          name = NULL) {
  tfp$bijectors$RationalQuadraticSpline(
    bin_widths = bin_widths,
    bin_heights = bin_heights,
    knot_slopes = knot_slopes,
    range_min = range_min,
    validate_args = validate_args,
    name = name
  )
}

#' Compute `Y = g(X) = exp(-exp(-(X - loc) / scale))`, the Gumbel CDF.
#'
#' This bijector maps inputs from `[-inf, inf]` to `[0, 1]`. The inverse of the
#' bijector applied to a uniform random variable `X ~ U(0, 1)` gives back a
#' random variable with the [Gumbel distribution](https://en.wikipedia.org/wiki/Gumbel_distribution):
#'
#' ```
#' Y ~ GumbelCDF(loc, scale)
#' pdf(y; loc, scale) = exp(-( (y - loc) / scale + exp(- (y - loc) / scale) ) ) / scale
#' ```
#' @param loc Float-like `Tensor` that is the same dtype and is
#' broadcastable with `scale`.
#' This is `loc` in `Y = g(X) = exp(-exp(-(X - loc) / scale))`.
#' @param scale Positive Float-like `Tensor` that is the same dtype and is
#' broadcastable with `loc`.
#' This is `scale` in `Y = g(X) = exp(-exp(-(X - loc) / scale))`.
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_gumbel_cdf <- function(loc = 0,
                           scale = 1,
                           validate_args = FALSE,
                           name = "gumbel_cdf") {
  args <- list(
    loc = loc,
    scale = scale,
    validate_args = validate_args,
    name = name
  )

  do.call(tfp$bijectors$GumbelCDF, args)
}

#' Compute `Y = g(X) = 1 - exp((-X / scale) ** concentration), X >= 0`.
#'
#' This bijector maps inputs from `[0, inf]` to `[0, 1]`. The inverse of the
#' bijector applied to a uniform random variable `X ~ U(0, 1)` gives back a
#' random variable with the
#' [Weibull distribution](https://en.wikipedia.org/wiki/Weibull_distribution):
#' ```
#' Y ~ Weibull(scale, concentration)
#' pdf(y; scale, concentration, y >= 0) =
#'   (concentration / scale) * (y / scale)**(concentration - 1) *
#'     exp(-(y / scale)**concentration)
#' ```
#'
#' Likwewise, the forward of this bijector is the Weibull distribution CDF.
#'
#' @param scale Positive Float-type `Tensor` that is the same dtype and is
#' broadcastable with `concentration`.
#' This is `l` in `Y = g(X) = 1 - exp((-x / l) ** k)`.
#' @param concentration Positive Float-type `Tensor` that is the same dtype and is
#' broadcastable with `scale`.
#' This is `k` in `Y = g(X) = 1 - exp((-x / l) ** k)`.
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_weibull_cdf <- function(scale = 1,
                            concentration = 1,
                            validate_args = FALSE,
                            name = "weibull_cdf") {
  args <- list(
    scale = scale,
    concentration = concentration,
    validate_args = validate_args,
    name = name
  )
  do.call(tfp$bijectors$WeibullCDF, args)
}

#' Computes`Y = g(X) = (1 - (1 - X)**(1 / b))**(1 / a)`, with X in `[0, 1]`
#'
#' This bijector maps inputs from `[0, 1]` to `[0, 1]`. The inverse of the
#' bijector applied to a uniform random variable X ~ U(0, 1) gives back a
#' random variable with the [Kumaraswamy distribution](https://en.wikipedia.org/wiki/Kumaraswamy_distribution):
#' `Y ~ Kumaraswamy(a, b)`
#' `pdf(y; a, b, 0 <= y <= 1) = a * b * y ** (a - 1) * (1 - y**a) ** (b - 1)`
#'
#' @param concentration1 float scalar indicating the transform power, i.e.,
#' `Y = g(X) = (1 - (1 - X)**(1 / b))**(1 / a) where a is concentration1.`
#' @param concentration0 float scalar indicating the transform power,
#' i.e., `Y = g(X) = (1 - (1 - X)**(1 / b))**(1 / a)` where b is concentration0.
#' @inherit tfb_identity return params
#'
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_kumaraswamy_cdf <- function(concentration1 = 1,
                                concentration0 = 1,
                                validate_args = FALSE,
                                name = "kumaraswamy_cdf") {
  args <- list(
    concentration1 = concentration1,
    concentration0 = concentration0,
    validate_args = validate_args,
    name = name
  )

  do.call(tfp$bijectors$KumaraswamyCDF, args)
}

#' Compute `Y = g(X; scale) = scale * X`.
#'
#' Examples:
#' ```
#' Y <- 2 * X
#' b <- tfb_scale(scale = 2)
#' ```
#'
#' @inherit tfb_identity return params
#' @param scale Floating-point `Tensor`.
#' @param log_scale Floating-point `Tensor`. Logarithm of the scale. If this is set
#' to `NULL`, no scale is applied. This should not be set if `scale` is set.
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_scale <- function(scale = NULL,
                      log_scale = NULL,
                      validate_args = FALSE,
                      name = 'scale') {

  if (tfp_version() > "0.10") {
    tfp$bijectors$Scale(scale = scale,
                        log_scale = log_scale,
                        validate_args = validate_args,
                        name = name)
  } else {
    tfp$bijectors$Scale(scale = scale,
                        validate_args = validate_args,
                        name = name)
  }

}

#' Transforms unconstrained vectors to TriL matrices with positive diagonal
#'
#' This is implemented as a simple tfb_chain of tfb_fill_triangular followed by
#' tfb_transform_diagonal, and provided mostly as a convenience.
#' The default setup is somewhat opinionated, using a Softplus transformation followed by a
#'  small shift (1e-5) which attempts to avoid numerical issues from zeros on the diagonal.
#'
#' @param diag_bijector Bijector instance, used to transform the output diagonal to be positive.
#' Default value: NULL (i.e., `tfb_softplus()`).
#' @param diag_shift Float value broadcastable and added to all diagonal entries after applying the
#' diag_bijector. Setting a positive value forces the output diagonal entries to be positive, but
#' prevents inverting the transformation for matrices with diagonal entries less than this value.
#' Default value: 1e-5.
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_fill_scale_tri_l <- function(diag_bijector = NULL,
                                 diag_shift = 1e-5,
                                 validate_args = FALSE,
                                 name = "fill_scale_tril") {
  args <- list(
    diag_bijector = diag_bijector,
    diag_shift = diag_shift,
    validate_args = validate_args,
    name = name
  )
  do.call(tfp$bijectors$FillScaleTriL, args)
}

#' Implements a continuous normalizing flow X->Y defined via an ODE.
#'
#' This bijector implements a continuous dynamics transformation
#' parameterized by a differential equation, where initial and terminal
#' conditions correspond to domain (X) and image (Y) i.e.
#'
#' ```
#' d/dt[state(t)] = state_time_derivative_fn(t, state(t))
#' state(initial_time) = X
#' state(final_time) = Y
#' ```
#'
#' For this transformation the value of `log_det_jacobian` follows another
#' differential equation, reducing it to computation of the trace of the jacobian
#' along the trajectory
#'
#' ```
#' state_time_derivative = state_time_derivative_fn(t, state(t))
#' d/dt[log_det_jac(t)] = Tr(jacobian(state_time_derivative, state(t)))
#' ```
#'
#' FFJORD constructor takes two functions `ode_solve_fn` and
#' `trace_augmentation_fn` arguments that customize integration of the
#' differential equation and trace estimation.
#'
#' Differential equation integration is performed by a call to `ode_solve_fn`.
#'
#' Custom `ode_solve_fn` must accept the following arguments:
#' * ode_fn(time, state): Differential equation to be solved.
#' * initial_time: Scalar float or floating Tensor representing the initial time.
#' * initial_state: Floating Tensor representing the initial state.
#' * solution_times: 1D floating Tensor of solution times.
#'
#' And return a Tensor of shape `[solution_times$shape, initial_state$shape]`
#' representing state values evaluated at `solution_times`. In addition
#' `ode_solve_fn` must support nested structures. For more details see the
#' interface of `tfp$math$ode$Solver$solve()`.
#'
#' Trace estimation is computed simultaneously with `state_time_derivative`
#' using `augmented_state_time_derivative_fn` that is generated by
#' `trace_augmentation_fn`. `trace_augmentation_fn` takes
#' `state_time_derivative_fn`, `state.shape` and `state.dtype` arguments and
#' returns a `augmented_state_time_derivative_fn` callable that computes both
#' `state_time_derivative` and unreduced `trace_estimation`.
#'
#' Custom `ode_solve_fn` and `trace_augmentation_fn` examples:
#'
#' ```
#' # custom_solver_fn: `function(f, t_initial, t_solutions, y_initial, ...)`
#' # ... : Additional arguments to pass to custom_solver_fn.
#' ode_solve_fn <- function(ode_fn, initial_time, initial_state, solution_times) {
#'   custom_solver_fn(ode_fn, initial_time, solution_times, initial_state, ...)
#' }
#' ffjord <- tfb_ffjord(state_time_derivative_fn, ode_solve_fn = ode_solve_fn)
#' ```
#'
#' ```
#' # state_time_derivative_fn: `function(time, state)`
#' # trace_jac_fn: `function(time, state)` unreduced jacobian trace function
#' trace_augmentation_fn <- function(ode_fn, state_shape, state_dtype) {
#'   augmented_ode_fn <- function(time, state) {
#'     list(ode_fn(time, state), trace_jac_fn(time, state))
#'   }
#' augmented_ode_fn
#' }
#' ffjord <- tfb_ffjord(state_time_derivative_fn, trace_augmentation_fn = trace_augmentation_fn)
#' ```
#'
#' For more details on FFJORD and continous normalizing flows see Chen et al. (2018), Grathwol et al. (2018).
#' @section References:
#'   -  Chen, T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). Neural ordinary differential equations. In Advances in neural information processing systems (pp. 6571-6583)
#'   -  [Grathwohl, W., Chen, R. T., Betterncourt, J., Sutskever, I., & Duvenaud, D. (2018). Ffjord: Free-form continuous dynamics for scalable reversible generative models. arXiv preprint arXiv:1810.01367.](https://arxiv.org/abs/1810.01367)
#'
#' @param state_time_derivative_fn  `function` taking arguments `time`
#' (a scalar representing time) and `state` (a Tensor representing the
#' state at given `time`) returning the time derivative of the `state` at
#' given `time`.
#' @param ode_solve_fn `function` taking arguments `ode_fn` (same as
#' `state_time_derivative_fn` above), `initial_time` (a scalar representing
#' the initial time of integration), `initial_state` (a Tensor of floating
#' dtype represents the initial state) and `solution_times` (1D Tensor of
#' floating dtype representing time at which to obtain the solution)
#' returning a Tensor of shape `[time_axis, initial_state$shape]`. Will take
#' `[final_time]` as the `solution_times` argument and
#' `state_time_derivative_fn` as `ode_fn` argument.
#' If `NULL` a DormandPrince solver from `tfp$math$ode` is used.
#' Default value: NULL
#' @param trace_augmentation_fn `function` taking arguments `ode_fn` (
#' `function` same as `state_time_derivative_fn` above),
#' `state_shape` (TensorShape of a the state), `dtype` (same as dtype of
#' the state) and returning a `function` taking arguments `time`
#' (a scalar representing the time at which the function is evaluted),
#' `state` (a Tensor representing the state at given `time`) that computes
#' a tuple (`ode_fn(time, state)`, `jacobian_trace_estimation`).
#' `jacobian_trace_estimation` should represent trace of the jacobian of
#' `ode_fn` with respect to `state`. `state_time_derivative_fn` will be
#' passed as `ode_fn` argument.
#' Default value: tfp$bijectors$ffjord$trace_jacobian_hutchinson
#' @param initial_time Scalar float representing time to which the `x` value of the
#' bijector corresponds to. Passed as `initial_time` to `ode_solve_fn`.
#' For default solver can be `float` or floating scalar `Tensor`.
#' Default value: 0.
#' @param final_time Scalar float representing time to which the `y` value of the
#' bijector corresponds to. Passed as `solution_times` to `ode_solve_fn`.
#' For default solver can be `float` or floating scalar `Tensor`.
#' Default value: 1.
#' @param dtype `tf$DType` to prefer when converting args to `Tensor`s. Else, we
#' fall back to a common dtype inferred from the args, finally falling
#' back to float32.
#'
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_ffjord <- function(state_time_derivative_fn,
                       ode_solve_fn = NULL,
                       trace_augmentation_fn = tfp$bijectors$ffjord$trace_jacobian_hutchinson,
                       initial_time = 0,
                       final_time = 1,
                       validate_args = FALSE,
                       dtype = tf$float32,
                       name = 'ffjord') {
  args <- list(
    state_time_derivative_fn = state_time_derivative_fn,
    ode_solve_fn = ode_solve_fn,
    trace_augmentation_fn = trace_augmentation_fn,
    initial_time = initial_time,
    final_time = final_time,
    validate_args = validate_args,
    dtype = dtype,
    name = name
  )
  do.call(tfp$bijectors$FFJORD, args)
}

#' LambertWTail transformation for heavy-tail Lambert W x F random variables.
#'
#' A random variable Y has a Lambert W x F distribution if W_tau(Y) = X has
#' distribution F, where tau = (shift, scale, tail) parameterizes the inverse
#' transformation.
#'
#' This bijector defines the transformation underlying Lambert W x F
#' distributions that transform an input random variable to an output
#' random variable with heavier tails. It is defined as
#' Y = (U * exp(0.5 * tail * U^2)) * scale + shift,  tail >= 0
#' where U = (X - shift) / scale is a shifted/scaled input random variable, and
#' tail >= 0 is the tail parameter.
#'
#' Attributes:
#' shift: shift to center (uncenter) the input data.
#' scale: scale to normalize (de-normalize) the input data.
#' tailweight: Tail parameter `delta` of heavy-tail transformation; must be >= 0.
#'
#' @param shift Floating point tensor; the shift for centering (uncentering) the
#' input (output) random variable(s).
#' @param scale Floating point tensor; the scaling (unscaling) of the input
#' (output) random variable(s). Must contain only positive values.
#' @param tailweight Floating point tensor; the tail behaviors of the output random
#' variable(s).  Must contain only non-negative values.
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_lambert_w_tail <- function(shift = NULL,
                               scale = NULL,
                               tailweight = NULL,
                               validate_args = FALSE,
                               name = "lambertw_tail") {
  args <- list(
    shift = shift,
    scale = scale,
    tailweight = tailweight,
    validate_args = validate_args,
    name = name
  )
  do.call(tfp$bijectors$LambertWTail, args)
}

#' Split a `Tensor` event along an axis into a list of `Tensor`s.
#'
#' The inverse of `split` concatenates a list of `Tensor`s along `axis`.
#'
#' @param num_or_size_splits Either an integer indicating the number of
#' splits along `axis` or a 1-D integer `Tensor` or Python list containing
#' the sizes of each output tensor along `axis`. If a list/`Tensor`, it may
#' contain at most one value of `-1`, which indicates a split size that is
#' unknown and determined from input.
#' @param axis A negative integer or scalar `int32` `Tensor`. The dimension along
#' which to split. Must be negative to enable the bijector to support
#' arbitrary batch dimensions. Defaults to -1 (note that this is different from the `tf$Split` default of `0`).
#' Must be statically known.
#'
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_split <- function(num_or_size_splits,
                      axis = -1,
                      validate_args = FALSE,
                      name = "split") {
  args <- list(
    num_or_size_splits = as_integer_list(num_or_size_splits),
    axis = as.integer(axis),
    validate_args = validate_args,
    name = name
  )
  do.call(tfp$bijectors$Split, args)
}

#' Compute `Y = g(X) = 1 - exp(-c * (exp(rate * X) - 1)`, the Gompertz CDF.
#'
#' This bijector maps inputs from `[-inf, inf]` to `[0, inf]`. The inverse of the
#' bijector applied to a uniform random variable `X ~ U(0, 1)` gives back a
#' random variable with the
#' [Gompertz distribution](https://en.wikipedia.org/wiki/Gompertz_distribution):
#' ```
#' Y ~ GompertzCDF(concentration, rate)
#' pdf(y; c, r) = r * c * exp(r * y + c - c * exp(-c * exp(r * y)))
#' ```
#' Note: Because the Gompertz distribution concentrates its mass close to zero,
#' for larger rates or larger concentrations, `bijector.forward` will quickly
#' saturate to 1.
#'
#' @param concentration Positive Float-like `Tensor` that is the same dtype and is
#' broadcastable with `concentration`.
#' This is `c` in `Y = g(X) = 1 - exp(-c * (exp(rate * X) - 1)`.
#' @param rate Positive Float-like `Tensor` that is the same dtype and is
#' broadcastable with `concentration`.
#' This is `rate` in `Y = g(X) = 1 - exp(-c * (exp(rate * X) - 1)`.
#'
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_gompertz_cdf <- function(concentration,
                             rate,
                             validate_args = FALSE,
                             name = "gompertz_cdf") {
  args <- list(
    concentration = concentration,
    rate = rate,
    validate_args = validate_args,
    name = name
  )
  do.call(tfp$bijectors$GompertzCDF, args)
}

#' Compute `Y = g(X) = (1 - exp(-rate * X)) * exp(-c * exp(-rate * X))`
#'
#' This bijector maps inputs from `[-inf, inf]` to `[0, inf]`. The inverse of the
#' bijector applied to a uniform random variable `X ~ U(0, 1)` gives back a
#' random variable with the
#' [Shifted Gompertz distribution](https://en.wikipedia.org/wiki/Shifted_Gompertz_distribution):
#' ```
#' Y ~ ShiftedGompertzCDF(concentration, rate)
#' pdf(y; c, r) = r * exp(-r * y - exp(-r * y) / c) * (1 + (1 - exp(-r * y)) / c)
#' ```
#'
#' Note: Even though this is called `ShiftedGompertzCDF`, when applied to the
#' `Uniform` distribution, this is not the same as applying a `GompertzCDF` with
#' a `Shift` bijector (i.e. the Shifted Gompertz distribution is not the same as
#' a Gompertz distribution with a location parameter).
#'
#' Note: Because the Shifted Gompertz distribution concentrates its mass close
#' to zero, for larger rates or larger concentrations, `bijector$forward` will
#' quickly saturate to 1.
#'
#' @param concentration Positive Float-like `Tensor` that is the same dtype and is
#' broadcastable with `concentration`.
#' This is `c` in `Y = g(X) = (1 - exp(-rate * X)) * exp(-c * exp(-rate * X))`.
#' @param rate Positive Float-like `Tensor` that is the same dtype and is
#' broadcastable with `concentration`.
#' This is `rate` in `Y = g(X) = (1 - exp(-rate * X)) * exp(-c * exp(-rate * X))`.
#'
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_shifted_gompertz_cdf <- function(concentration,
                                     rate,
                                     validate_args = FALSE,
                                     name = "shifted_gompertz_cdf") {
  args <- list(
    concentration = concentration,
    rate = rate,
    validate_args = validate_args,
    name = name
  )
  do.call(tfp$bijectors$ShiftedGompertzCDF, args)
}

#' Bijector that computes `Y = sinh(X)`.
#'
#' @inherit tfb_identity return params
#' @family bijectors
#' @seealso For usage examples see [tfb_forward()], [tfb_inverse()], [tfb_inverse_log_det_jacobian()].
#' @export
tfb_sinh <- function(validate_args = FALSE,
                     name = "sinh") {
  args <- list(validate_args = validate_args,
               name = name)
  do.call(tfp$bijectors$Sinh, args)
}






