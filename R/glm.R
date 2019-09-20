#' Runs multiple Fisher scoring steps.
#'
#' @param x float-like, matrix-shaped Tensor where each row represents a sample's
#'  features.
#' @param ... other arguments passed to specific methods.
#'
#' @export
glm_fit <- function(x, ...) {
  UseMethod("glm_fit")
}

#' @inheritParams glm_fit
#' @export
glm_fit.default <- function(x, ...) {
  glm_fit(tensorflow::tf$convert_to_tensor(x), ...)
}

#' Runs multiple Fisher scoring steps.
#'
#' @inheritParams glm_fit
#' @param response vector-shaped Tensor where each element represents a sample's
#'  observed response (to the corresponding row of features). Must have same `dtype`
#'  as `x`.
#' @param model `tfp$glm$ExponentialFamily-like` instance which implicitly
#'  characterizes a negative log-likelihood loss by specifying the distribuion's
#'  mean, gradient_mean, and variance.
#' @param model_coefficients_start Optional (batch of) vector-shaped Tensor representing
#'  the initial model coefficients, one for each column in `x`. Must have same `dtype`
#'  as model_matrix. Default value: Zeros.
#' @param predicted_linear_response_start Optional Tensor with shape, `dtype` matching
#'  `response`; represents offset shifted initial linear predictions based on
#'  `model_coefficients_start`. Default value: offset if model_coefficients is `NULL`,
#'  and `tf$linalg$matvec(x, model_coefficients_start) + offset` otherwise.
#' @param l2_regularizer Optional scalar Tensor representing L2 regularization penalty.
#'  Default: `NULL` ie. no regularization.
#' @param dispersion Optional (batch of) Tensor representing response dispersion.
#' @param offset Optional Tensor representing constant shift applied to `predicted_linear_response`.
#' @param convergence_criteria_fn callable taking: `is_converged_previous`, `iter_`,
#'  `model_coefficients_previous`, `predicted_linear_response_previous`, `model_coefficients_next`,
#'  `predicted_linear_response_next`, `response`, `model`, `dispersion` and returning
#'  a logical Tensor indicating that Fisher scoring has converged.
#' @param learning_rate Optional (batch of) scalar Tensor used to dampen iterative progress.
#'  Typically only needed if optimization diverges, should be no larger than 1 and typically
#'  very close to 1. Default value: `NULL` (i.e., 1).
#' @param fast_unsafe_numerics Optional Python bool indicating if faster, less numerically
#'  accurate methods can be employed for computing the weighted least-squares solution. Default
#'  value: TRUE (i.e., "fast but possibly diminished accuracy").
#' @param maximum_iterations Optional maximum number of iterations of Fisher scoring to run;
#'  "and-ed" with result of `convergence_criteria_fn`. Default value: `NULL` (i.e., infinity).
#' @param name usesed as name prefix to ops created by this function. Default value: "fit".
#'
#' @export
glm_fit.tensorflow.tensor <- function(x,
                                      response,
                                      model,
                                      model_coefficients_start = NULL,
                                      predicted_linear_response_start = NULL,
                                      l2_regularizer = NULL,
                                      dispersion = NULL,
                                      offset = NULL,
                                      convergence_criteria_fn = NULL,
                                      learning_rate = NULL,
                                      fast_unsafe_numerics=TRUE,
                                      maximum_iterations = NULL,
                                      name = NULL) {
  out <- tfp$glm$fit(
    model_matrix = x,
    response = response,
    model = model,
    model_coefficients_start = model_coefficients_start,
    predicted_linear_response_start = predicted_linear_response_start,
    l2_regularizer = l2_regularizer,
    dispersion = dispersion,
    offset = offset,
    convergence_criteria_fn = convergence_criteria_fn,
    learning_rate = learning_rate,
    fast_unsafe_numerics = fast_unsafe_numerics,
    maximum_iterations = maximum_iterations,
    name = name
  )
  class(out) <- c("glm_fit")
  out
}
