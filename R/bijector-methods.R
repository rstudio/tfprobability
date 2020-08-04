#' Returns the forward Bijector evaluation, i.e., `X = g(Y)`.
#'
#' @param bijector The bijector to apply
#' @param x Tensor. The input to the "forward" evaluation.
#' @param name name of the operation
#' @return a tensor
#' @family bijector_methods
#' @examples
#' \donttest{
#'   b <- tfb_affine_scalar(shift = 1, scale = 2)
#'   x <- 10
#'   b %>% tfb_forward(x)
#' }
#' @export
tfb_forward <-
  function(bijector, x, name ="forward") {
    bijector$forward(as_float_tensor(x), name)
  }

#' Returns the inverse Bijector evaluation, i.e., `X = g^{-1}(Y)`.
#'
#' @param bijector  The bijector to apply
#' @param y Tensor. The input to the "inverse" evaluation.
#' @param name name of the operation
#' @return a tensor
#' @family bijector_methods
#' @examples
#' \donttest{
#'   b <- tfb_affine_scalar(shift = 1, scale = 2)
#'   x <- 10
#'   y <- b %>% tfb_forward(x)
#'   b %>% tfb_inverse(y)
#' }
#' @export
tfb_inverse <-
  function(bijector, y, name="inverse") {
    bijector$inverse(as_float_tensor(y), name)
  }

#' Returns the result of the forward evaluation of the log determinant of the Jacobian
#'
#' @param bijector The bijector to apply
#' @param x Tensor. The input to the "forward" Jacobian determinant evaluation.
#' @param event_ndims Number of dimensions in the probabilistic events being transformed.
#'  Must be greater than or equal to bijector$forward_min_event_ndims. The result is summed over the final
#'  dimensions to produce a scalar Jacobian determinant for each event, i.e. it has shape
#'   x$shape$ndims - event_ndims dimensions.
#' @param name name of the operation
#' @return a tensor
#' @family bijector_methods
#' @examples
#' \donttest{
#'   b <- tfb_affine_scalar(shift = 1, scale = 2)
#'   x <- 10
#'   b %>% tfb_forward_log_det_jacobian(x, event_ndims = 0)
#' }
#' @export
tfb_forward_log_det_jacobian <-
  function(bijector, x, event_ndims, name="forward_log_det_jacobian") {
    bijector$forward_log_det_jacobian(as_float_tensor(x), as.integer(event_ndims), name)
  }

#' Returns the result of the inverse evaluation of the log determinant of the Jacobian
#'
#' @param bijector The bijector to apply
#' @param y Tensor. The input to the "inverse" Jacobian determinant evaluation.
#' @param event_ndims Number of dimensions in the probabilistic events being transformed.
#'  Must be greater than or equal to bijector$inverse_min_event_ndims. The result is summed over the final
#'  dimensions to produce a scalar Jacobian determinant for each event, i.e. it has shape
#'   x$shape$ndims - event_ndims dimensions.
#' @param name name of the operation
#' @return a tensor
#' @family bijector_methods
#' @examples
#' \donttest{
#'   b <- tfb_affine_scalar(shift = 1, scale = 2)
#'   x <- 10
#'   y <- b %>% tfb_forward(x)
#'   b %>% tfb_inverse_log_det_jacobian(y, event_ndims = 0)
#' }
#' @export
tfb_inverse_log_det_jacobian <-
  function(bijector, y, event_ndims, name="inverse_log_det_jacobian") {
    bijector$inverse_log_det_jacobian(as_float_tensor(y), as.integer(event_ndims), name)
  }

