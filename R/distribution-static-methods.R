#' number of `params` needed to create a MultivariateNormalTriL distribution
#'
#' @param event_size event size of this distribution
#' @return a scalar
#' @export
params_size_multivariate_normal_tri_l <- function(event_size) {
  tfp$layers$MultivariateNormalTriL$params_size(as.integer(event_size))
}

#' number of `params` needed to create an IndependentBernoulli distribution
#'
#' @inheritParams params_size_multivariate_normal_tri_l
#' @return a scalar
#' @export
params_size_independent_bernoulli <- function(event_size) {
  tfp$layers$IndependentBernoulli$params_size(as.integer(event_size))
}

#' number of `params` needed to create a OneHotCategorical distribution
#'
#' @inheritParams params_size_multivariate_normal_tri_l
#' @return a scalar
#' @export
params_size_one_hot_categorical <- function(event_size) {
  tfp$layers$OneHotCategorical$params_size(as.integer(event_size))
}

#' number of `params` needed to create a CategoricalMixtureOfOneHotCategorical distribution
#'
#' @inheritParams params_size_multivariate_normal_tri_l
#' @param num_components number of components in the mixture
#' @return a scalar
#' @export
params_size_categorical_mixture_of_one_hot_categorical <- function(event_size, num_components) {
  tfp$layers$CategoricalMixtureOfOneHotCategorical$params_size(as.integer(event_size), as.integer(num_components))
}

#' number of `params` needed to create an IndependentPoisson distribution
#'
#' @inheritParams params_size_multivariate_normal_tri_l
#' @return a scalar
#' @export
params_size_independent_poisson <- function(event_size) {
  tfp$layers$IndependentPoisson$params_size(as.integer(event_size))
}

#' number of `params` needed to create an IndependentLogistic distribution
#'
#' @inheritParams params_size_multivariate_normal_tri_l
#' @return a scalar
#' @export
params_size_independent_logistic <- function(event_size) {
  tfp$layers$IndependentLogistic$params_size(as.integer(event_size))
}

#' number of `params` needed to create an IndependentNormal distribution
#'
#' @inheritParams params_size_multivariate_normal_tri_l
#' @return a scalar
#' @export
params_size_independent_normal <- function(event_size) {
  tfp$layers$IndependentNormal$params_size(as.integer(event_size))
}

#' number of `params` needed to create a MixtureSameFamily distribution
#'
#' @param num_components Number of component distributions in the mixture distribution.
#' @param component_params_size Number of parameters needed to create a single component distribution.
#' @return a scalar
#' @export
params_size_mixture_same_family <- function(num_components, component_params_size) {
  tfp$layers$MixtureSameFamily$params_size(as.integer(num_components), as.integer(component_params_size))
}

#' number of `params` needed to create a MixtureNormal distribution
#'
#' @param num_components Number of component distributions in the mixture distribution.
#' @param event_shape Number of parameters needed to create a single component distribution.
#' @return a scalar
#' @export
params_size_mixture_normal <- function(num_components, event_shape) {
  tfp$layers$MixtureNormal$params_size(as.integer(num_components), as.integer(event_shape))
}

#' number of `params` needed to create a MixtureLogistic distribution
#'
#' @param num_components Number of component distributions in the mixture distribution.
#' @param event_shape Number of parameters needed to create a single component distribution.
#' @return a scalar
#' @export
params_size_mixture_logistic <- function(num_components, event_shape) {
  tfp$layers$MixtureLogistic$params_size(as.integer(num_components), as.integer(event_shape))
}
