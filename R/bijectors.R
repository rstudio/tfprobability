



#' Compute Y = g(X) = X.
#'
#' @param validate_args Logical, default `FALSE`. Whether to validate input with asserts. If `validate_args` is
#'  `FALSE`, and the inputs are invalid, correct behavior is not guaranteed.
#' @param name name prefixed to Ops created by this class.
#'
#' @return
#' @family bijectors
#' @export

bijector_identity <- function(validate_args = FALSE,
                              name = "identity") {
  args <- list(validate_args = validate_args,
               name = name)

  do.call(tfp$bijectors$Identity, args)
}

#' Bijector which computes `Y = g(X) = 1 / (1 + exp(-X))`.
#'
#' @inheritParams bijector_identity
#'
#' @return
#' @export

bijector_sigmoid <- function(validate_args = FALSE,
                             name = "sigmoid") {
  args <- list(validate_args = validate_args,
               name = name)

  do.call(tfp$bijectors$Sigmoid, args)
}

#' Bijector which computes `Y=g(X)=exp(X)``
#'
#' @inheritParams bijector_identity
#'
#' @return
#' @export

bijector_exp <- function(validate_args = FALSE,
                         name = "exp") {
  args <- list(validate_args = validate_args,
               name = name)

  do.call(tfp$bijectors$Exp, args)
}

#' Bijector which computes `Y = g(X) = Abs(X)`, element-wise.
#'
#' This non-injective bijector allows for transformations of scalar distributions
#' with the absolute value function, which maps `(-inf, inf)` to `[0, inf)`.
#' * For `y in (0, inf)`, `AbsoluteValue.inverse(y)` returns the set inverse
#' `{x in (-inf, inf) : |x| = y}` as a tuple, `-y, y`.
#' `AbsoluteValue.inverse(0)` returns `0, 0`, which is not the set inverse
#' (the set inverse is the singleton `{0}`), but "works" in conjunction with
#' `TransformedDistribution` to produce a left semi-continuous pdf.
#' For `y < 0`, `AbsoluteValue.inverse(y)` happily returns the wrong thing, `-y, y`.
#'  This is done for efficiency.  If `validate_args == True`, `y < 0` will raise an exception.

#' @inheritParams bijector_identity
#'
#' @return
#' @export

bijector_absolute_value <- function(validate_args = FALSE,
                                    name = "absolute_value") {
  args <- list(validate_args = validate_args,
               name = name)

  do.call(tfp$bijectors$AbsoluteValue, args)
}

#' Instantiates the `Affine` bijector.
#'
#' This `Bijector` is initialized with `shift` `Tensor` and `scale` arguments,
#' giving the forward operation: `Y = g(X) = scale @ X + shift``
#' where the `scale` term is logically equivalent to:
#' scale =
#'     scale_identity_multiplier * tf.diag(tf.ones(d)) +
#'     tf.diag(scale_diag) +
#'     scale_tril +
#'     scale_perturb_factor @ diag(scale_perturb_diag) @ tf.transpose([scale_perturb_factor]))
#'
#'  If none of `scale_identity_multiplier`, `scale_diag`, or `scale_tril` are specified then
#'   `scale += IdentityMatrix`. Otherwise specifying a `scale` argument has the semantics of
#'    `scale += Expand(arg)`, i.e., `scale_diag != NULL` means `scale += tf$diag(scale_diag)`.
#'
#' @param shift Floating-point `Tensor`. If this is set to `NULL`, no shift is applied.
#' @param scale_identity_multiplier floating point rank 0 `Tensor` representing a scaling done
#'  to the identity matrix. When `scale_identity_multiplier = scale_diag = scale_tril = None` then
#'  `scale += IdentityMatrix`. Otherwise no scaled-identity-matrix is added to `scale`.
#' @param scale_diag Floating-point `Tensor` representing the diagonal matrix.
#' `scale_diag` has shape `[N1, N2, ...  k]`, which represents a k x k diagonal matrix.
#' When `NULL` no diagonal term is added to `scale`.
#' @param scale_tril Floating-point `Tensor` representing the lower triangular matrix.
#' `scale_tril` has shape `[N1, N2, ...  k, k]`, which represents a k x k lower triangular matrix.
#' When `None` no `scale_tril` term is added to `scale`. The upper triangular elements above the diagonal are ignored.
#' @param scale_perturb_factor Floating-point `Tensor` representing factor matrix with last
#'  two dimensions of shape `(k, r)`. When `NULL`, no rank-r update is added to `scale`.
#' @param scale_perturb_diag Floating-point `Tensor` representing the diagonal matrix.
#'  `scale_perturb_diag` has shape `[N1, N2, ...  r]`, which represents an `r x r` diagonal matrix.
#'  When `None` low rank updates will take the form `scale_perturb_factor * scale_perturb_factor.T`.
#' @param adjoint Logical indicating whether to use the `scale` matrix as specified or its adjoint.
#' Default value: `FALSE`.
#' @inheritParams bijector_identity
#' @param dtype `tf$DType` to prefer when converting args to `Tensor`s. Else, we fall back to a
#'  common dtype inferred from the args, finally falling back to float32.

#' @export

bijector_affine <- function(shift = NULL,
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

#' Compute `Y = g(X; shift, scale) = scale @ X + shift`.
#'
#' `shift` is a numeric `Tensor` and `scale` is a `LinearOperator`.
#' If `X` is a scalar then the forward transformation is: `scale * X + shift`
#' where `*` denotes broadcasted elementwise product.
#'
#' @param shift Floating-point `Tensor`.
#' @param scale Subclass of `LinearOperator`. Represents the (batch) positive definite matrix `M` in `R^{k x k}`.
#' @param adjoint Logical indicating whether to use the `scale` matrix as specified or its adjoint.
#' Default value: `False`.
#' @inheritParams bijector_identity
#'
#' @export

bijector_affine_linear_operator <- function(shift = NULL,
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


#' Instantiates the `AffineScalar` bijector.
#'
#' This `Bijector` is initialized with `shift` `Tensor` and `scale` arguments, giving the forward operation:
#' `Y = g(X) = scale * X + shift`
#' If `scale` is not specified, then the bijector has the semantics of `scale = 1.`.
#' Similarly, if `shift` is not specified, then the bijector has the semantics of `shift = 0.`.
#'
#' @param shift Floating-point `Tensor`. If this is set to `None`, no shift is applied.
#' @param scale Floating-point `Tensor`. If this is set to `None`, no scale is applied.
#' @inheritParams bijector_identity
#' @export

bijector_affine_scalar <- function(shift = NULL,
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

#' Compute `Y = g(X) s.t. X = g^-1(Y) = (Y - mean(Y)) / std(Y)`.
#'
#' Applies Batch Normalization [(Ioffe and Szegedy, 2015)][1] to samples from a
#' data distribution. This can be used to stabilize training of normalizing
#' flows ([Papamakarios et al., 2016][3]; [Dinh et al., 2017][2])
#'
#' When training Deep Neural Networks (DNNs), it is common practice to
#' normalize or whiten features by shifting them to have zero mean and
#' scaling them to have unit variance.
#'
#' The `inverse()` method of the `BatchNormalization` bijector, which is used in
#' the log-likelihood computation of data samples, implements the normalization
#' procedure (shift-and-scale) using the mean and standard deviation of the
#' current minibatch.
#'
#' Conversely, the `forward()` method of the bijector de-normalizes samples (e.g.
#' `X*std(Y) + mean(Y)` with the running-average mean and standard deviation
#' computed at training-time. De-normalization is useful for sampling.
#'
#' During training time, `BatchNormalization.inverse` and `BatchNormalization.forward` are not
#'  guaranteed to be inverses of each other because `inverse(y)` uses statistics of the current minibatch,
#'  while `forward(x)` uses running-average statistics accumulated from training.
#'  In other words, `BatchNormalization.inverse(BatchNormalization.forward(...))` and
#'  `BatchNormalization.forward(BatchNormalization.inverse(...))` will be identical when
#'   `training=FALSE` but may be different when `training=TRUE`.
#'
#' References
#' [1]: Sergey Ioffe and Christian Szegedy. Batch Normalization: Accelerating Deep Network Training
#'  by Reducing Internal Covariate Shift.
#'  In _International Conference on Machine Learning_, 2015. https://arxiv.org/abs/1502.03167
#' [2]: Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density Estimation using Real NVP.
#' In _International Conference on Learning Representations_, 2017. https://arxiv.org/abs/1605.08803
#' [3]: George Papamakarios, Theo Pavlakou, and Iain Murray. Masked Autoregressive Flow for Density Estimation.
#' In _Neural Information Processing Systems_, 2017. https://arxiv.org/abs/1705.07057

#'
#' @param batchnorm_layer `tf$layers$BatchNormalization` layer object. If `NULL`, defaults to
#' `tf$layers$BatchNormalization(gamma_constraint=tf.nn.relu(x) + 1e-6)`.
#' This ensures positivity of the scale variable.

#' @param training If TRUE, updates running-average statistics during call to `inverse()`.
#' @inheritParams bijector_identity
#' @export

bijector_batch_normalization <- function(batchnorm_layer = NULL,
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

#' Bijector which applies a list of bijectors to blocks of a `Tensor`.
#'
#' More specifically, given [F_0, F_1, ... F_n] which are scalar or vector
#' bijectors this bijector creates a transformation which operates on the vector
#' [x_0, ... x_n] with the transformation [F_0(x_0), F_1(x_1) ..., F_n(x_n)]
#' where x_0, ..., x_n are blocks (partitions) of the vector.
#'
#' @param bijectors A non-empty list of bijectors.
#' @param block_sizes A 1-D integer `Tensor` with each element signifying the
#' length of the block of the input vector to pass to the corresponding
#' bijector. The length of `block_sizes` must be be equal to the length of
#' `bijectors`. If left as NULL, a vector of 1's is used.
#' @param validate_args Logical indicating whether arguments should be checked for correctness.
#' @param name String, name given to ops managed by this object. Default:
#' E.g., `Blockwise([Exp(), Softplus()]).name ==   'blockwise_of_exp_and_softplus'`.

#' @export
bijector_blockwise <- function(bijectors,
                               block_sizes = NULL,
                               validate_args = FALSE,
                               name = NULL) {
  args <- list(
    bijectors = bijectors,
    block_sizes = as.integer(block_sizes),
    validate_args = validate_args,
    name = name
  )
  do.call(tfp$bijectors$Blockwise, args)
}


