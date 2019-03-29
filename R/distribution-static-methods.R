#' number of `params` needed to create a single distribution
#' @param event_size event size of this distribution
#' @export
params_size_multivariate_normal_tri_l <- function(event_size) {
  tfp$layers$MultivariateNormalTriL$params_size(event_size)
}

#' number of `params` needed to create a single distribution
#' @inheritParams params_size_multivariate_normal_tri_l
#' @export
params_size_independent_bernoulli <- function(event_size) {
  tfp$layers$IndependentBernoulli$params_size(event_size)
}
