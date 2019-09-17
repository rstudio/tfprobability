
glm_fit <- function(x, ...) {
  UseMethod("glm_fit")
}

glm_fit.default <- function(x, ...) {
  glm_fit(tensorflow::tf$convert_to_tensor(x), ...)
}

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

glm_fit.tf_dataset <- function(x,
                                feature_columns,
                                model,
                                model_coefficients_start = NULL,
                                predicted_linear_response_start = NULL,
                                l2_regularizer = NULL,
                                dispersion = NULL,
                                offset = NULL,
                                convergence_criteria_fn = NULL,
                                learning_rate = NULL,
                                fast_unsafe_numerics=True,
                                maximum_iterations = NULL,
                                name = NULL) {
  NULL
}

