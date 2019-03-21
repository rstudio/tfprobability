
params_size_multivariate_normal_tri_l <- function(event_size) {
  tfp$layers$MultivariateNormalTriL$params_size(event_size)
}

params_size_independent_bernoulli <- function(event_size) {
  tfp$layers$IndependentBernoulli$params_size(event_size)
}
