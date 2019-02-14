
params_size_multivariate_normal_tril <- function(event_size) {
  tfp$layers$MultivariateNormalTriL$params_size(event_size)
}

params_size_independent_bernoulli <- function(event_size) {
  tfp$layers$IndependentBernoulli$params_size(event_size)
}
