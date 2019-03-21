#' @export
forward <-
  function(bijector, x, name="forward") {
    UseMethod("forward")
  }


#' Returns the forward Bijector evaluation, i.e., X = g(Y).
#'
#' @param bijector The bijector to apply
#' @param x Tensor. The input to the "forward" evaluation.
#' @param name  The name to give this op.
#'
#' @return Tensor
#' @export

forward.tensorflow_probability.python.bijectors.bijector.Bijector <-
  function(bijector, x, name="forward") {
    bijector$forward(as_tf_float(x), name)
  }

#' @export
inverse <-
  function(bijector, y, name="inverse") {
    UseMethod("inverse")
  }


#' Returns the inverse Bijector evaluation, i.e., X = g^{-1}(Y).
#'
#' @param bijector  The bijector to apply
#' @param y Tensor. The input to the "inverse" evaluation.
#' @name The name to give this op.
#'
#' @return Tensor, if this bijector is injective. If not injective, returns the k-tuple containing the unique k points (x1, ..., xk) such that g(xi) = y.
#' @export

inverse.tensorflow_probability.python.bijectors.bijector.Bijector <-
  function(bijector, y, name="inverse") {
    bijector$inverse(as_tf_float(y), name)
  }

#' @export
forward_log_det_jacobian <-
  function(bijector, x, event_ndims, name="forward_log_det_jacobian") {
    UseMethod("forward_log_det_jacobian")
  }


#' Returns both the forward_log_det_jacobian.
#'
#' @param bijector The bijector to apply
#' @param x Tensor. The input to the "forward" Jacobian determinant evaluation.
#' @param event_ndims Number of dimensions in the probabilistic events being transformed.
#'  Must be greater than or equal to bijector$forward_min_event_ndims. The result is summed over the final
#'  dimensions to produce a scalar Jacobian determinant for each event, i.e. it has shape
#'   x$shape$ndims - event_ndims dimensions.
#' @param name The name to give this op.
#'
#' @return Tensor, if this bijector is injective. If not injective this is not implemented.
#' @export

forward_log_det_jacobian.tensorflow_probability.python.bijectors.bijector.Bijector <-
  function(bijector, x, event_ndims, name="forward_log_det_jacobian") {
    bijector$forward_log_det_jacobian(as_tf_float(x), as.integer(event_ndims), name)
  }

#' @export
inverse_log_det_jacobian <-
  function(bijector, y, event_ndims, name="inverse_log_det_jacobian") {
    UseMethod("inverse_log_det_jacobian")
  }

#' Returns the (log o det o Jacobian o inverse)(y).
#'
#' @param bijector The bijector to apply
#' @param y Tensor. The input to the "inverse" Jacobian determinant evaluation.
#' @param event_ndims Number of dimensions in the probabilistic events being transformed.
#'  Must be greater than or equal to bijector$inverse_min_event_ndims. The result is summed over the final
#'  dimensions to produce a scalar Jacobian determinant for each event, i.e. it has shape
#'   x$shape$ndims - event_ndims dimensions.
#' @param name The name to give this op.
#'
#' @return ildj: Tensor, if this bijector is injective. If not injective, returns the tuple of local log det
#' Jacobians, log(det(Dg_i^{-1}(y))), where g_i is the restriction of g to the ith partition Di.
#' @export
#'
inverse_log_det_jacobian.tensorflow_probability.python.bijectors.bijector.Bijector <-
  function(bijector, y, event_ndims, name="inverse_log_det_jacobian") {
    bijector$inverse_log_det_jacobian(as_tf_float(y), as.integer(event_ndims), name)
  }

