#' number of `params` needed to create a MultivariateNormalTriL distribution
#'
#' @param event_size event size of this distribution
#' @export
params_size_multivariate_normal_tri_l <- function(event_size) {
  tfp$layers$MultivariateNormalTriL$params_size(event_size)
}

#' number of `params` needed to create an IndependentBernoulli distribution
#'
#' @inheritParams params_size_multivariate_normal_tri_l
#' @export
params_size_independent_bernoulli <- function(event_size) {
  tfp$layers$IndependentBernoulli$params_size(event_size)
}

#' number of `params` needed to create a OneHotCategorical distribution
#'
#' @inheritParams params_size_multivariate_normal_tri_l
#' @export
params_size_one_hot_categorical <- function(event_size) {
  tfp$layers$OneHotCategorical$params_size(event_size)
}

#' number of `params` needed to create a CategoricalMixtureOfOneHotCategorical distribution
#'
#' @inheritParams params_size_multivariate_normal_tri_l
#' @param num_components number of components in the mixture
#' @export
params_size_categorical_mixture_of_one_hot_categorical <- function(event_size, num_components) {
  tfp$layers$CategoricalMixtureOfOneHotCategorical$params_size(event_size, num_components)
}
