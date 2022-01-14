#' Runs multiple Fisher scoring steps
#'
#' @param x float-like, matrix-shaped Tensor where each row represents a sample's
#'  features.
#' @param ... other arguments passed to specific methods.
#'
#' @seealso [glm_fit.tensorflow.tensor()]
#'
#' @return A `glm_fit` object with parameter estimates, number of iterations,
#'  etc.
#'
#' @export
glm_fit <- function(x, ...) {
  UseMethod("glm_fit")
}

#' Runs one Fisher scoring step
#'
#' @inheritParams glm_fit
#' @seealso [glm_fit_one_step.tensorflow.tensor()]
#'
#' @return A `glm_fit` object with parameter estimates, number of iterations,
#'  etc.
#'
#' @export
glm_fit_one_step <- function(x, ...) {
  UseMethod("glm_fit_one_step")
}

#' @inheritParams glm_fit
#' @export
glm_fit.default <- function(x, ...) {
  glm_fit(tensorflow::tf$convert_to_tensor(x), ...)
}

#' @inheritParams glm_fit
#' @export
glm_fit_one_step.default <- function(x, ...) {
  glm_fit_one_step(tensorflow::tf$convert_to_tensor(x), ...)
}

#' Runs multiple Fisher scoring steps
#'
#' @inheritParams glm_fit
#' @param response vector-shaped Tensor where each element represents a sample's
#'  observed response (to the corresponding row of features). Must have same `dtype`
#'  as `x`.
#' @param model a string naming the model (see [glm_families]) or a `tfp$glm$ExponentialFamily-like`
#'  instance which implicitly characterizes a negative log-likelihood loss by specifying
#'  the distribuion's mean, gradient_mean, and variance.
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
#' @family glm_fit
#'
#' @return A `glm_fit` object with parameter estimates, and
#'  number of required steps.
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
                                      name = NULL,
                                      ...) {

  if (is.character(model)) model <- family_from_string(model)

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

#' Runs one Fisher Scoring step
#' @inheritParams glm_fit.tensorflow.tensor
#' @family glm_fit
#'
#' @return A `glm_fit` object with parameter estimates, and
#'  number of required steps.
#'
#' @export
glm_fit_one_step.tensorflow.tensor <- function(x,
                             response,
                             model,
                             model_coefficients_start=NULL,
                             predicted_linear_response_start=NULL,
                             l2_regularizer=NULL,
                             dispersion=NULL,
                             offset=NULL,
                             learning_rate=NULL,
                             fast_unsafe_numerics=TRUE,
                             name=NULL,
                             ...) {

  if (is.character(model)) model <- family_from_string(model)

  out <- tfp$glm$fit_one_step(
    model_matrix = x,
    response = response,
    model = model,
    model_coefficients_start = model_coefficients_start,
    predicted_linear_response_start = predicted_linear_response_start,
    l2_regularizer = l2_regularizer,
    dispersion = dispersion,
    offset = offset,
    learning_rate = learning_rate,
    fast_unsafe_numerics = fast_unsafe_numerics,
    name = name
  )
  class(out) <- "glm_fit"
  out
}


#' GLM families
#'
#' A list of models that can be used as the `model` argument in [glm_fit()]:
#'
#' * `Bernoulli`: `Bernoulli(probs=mean)` where `mean = sigmoid(matmul(X, weights))`
#' * `BernoulliNormalCDF`: `Bernoulli(probs=mean)` where `mean = Normal(0, 1).cdf(matmul(X, weights))`
#' * `GammaExp`: `Gamma(concentration=1, rate=1 / mean)` where `mean = exp(matmul(X, weights))`
#' * `GammaSoftplus`: `Gamma(concentration=1, rate=1 / mean)` where `mean = softplus(matmul(X, weights))`
#' * `LogNormal`: `LogNormal(loc=log(mean) - log(2) / 2, scale=sqrt(log(2)))` where
#'  `mean = exp(matmul(X, weights))`.
#' * `LogNormalSoftplus`: `LogNormal(loc=log(mean) - log(2) / 2, scale=sqrt(log(2)))` where
#'  `mean = softplus(matmul(X, weights))`
#' * `Normal`: `Normal(loc=mean, scale=1)` where `mean = matmul(X, weights)`.
#' * `NormalReciprocal`: `Normal(loc=mean, scale=1)` where `mean = 1 / matmul(X, weights)`
#' * `Poisson`: `Poisson(rate=mean)` where `mean = exp(matmul(X, weights))`.
#' * `PoissonSoftplus`: `Poisson(rate=mean)` where `mean = softplus(matmul(X, weights))`.
#'
#' @return list of models that can be used as the `model` argument in [glm_fit()]
#'
#' @family glm_fit
#' @name glm_families
#' @rdname glm_families
NULL

family_from_string <- function(model) {

  if (model == "Bernoulli")
    tfp$glm$Bernoulli()
  else if (model == "BernoulliNormalCDF")
    tfp$glm$BernoulliNormalCDF()
  else if (model == "GammaExp")
    tfp$glm$GammaExp()
  else if (model == "GammaSoftplus")
    tfp$glm$GammaSoftplus()
  else if (model == "LogNormal")
    tfp$glm$LogNormal()
  else if (model == "LogNormalSoftplus")
    tfp$glm$LogNormalSoftplus()
  else if (model == "Normal")
    tfp$glm$Normal()
  else if (model == "NormalReciprocal")
    tfp$glm$NormalReciprocal()
  else if (model == "Poisson")
    tfp$glm$Poisson()
  else if (model == "PoissonSoftplus")
    tfp$glm$PoissonSoftplus()
  else
    stop("Model ", model, "not implemented", call. = FALSE)
}
