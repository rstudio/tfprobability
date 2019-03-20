context("distribution methods")

source("utils.R")

test_succeeds("can sample from distributions", {

  d <- tfd_normal(0,1)
  x <- d %>% tfd_sample(c(2, 2))
  expect_length(dim(x), 2)

  x <- d %>% tfd_sample()
  expect_length(dim(x), 0)

})

test_succeeds("can compute log probability", {

  d <- tfd_normal(0,1)
  x <- d %>% tfd_log_prob(c(0.22, 0.44, 3))
  expect_length(dim(x), 1)

})

test_succeeds("can compute probability", {

  d <- tfd_normal(0,1)
  x <- d %>% tfd_prob(matrix(1:4, ncol = 2))
  expect_length(dim(x), 2)

})

test_succeeds("can compute log cdf", {

  d <- tfd_normal(0,1)
  x <- d %>% tfd_log_cdf(matrix(1:4, ncol = 2))
  expect_length(dim(x), 2)

})

test_succeeds("can compute cdf", {

  d <- tfd_normal(0,1)
  x <- d %>% tfd_cdf(matrix(1:4, ncol = 2))
  expect_length(dim(x), 2)

})

test_succeeds("can compute log survival function", {

  d <- tfd_normal(0,1)
  x <- d %>% tfd_log_survival_function(matrix(1:4, ncol = 2))
  expect_length(dim(x), 2)

})

test_succeeds("can compute survival function", {

  d <- tfd_normal(0,1)
  x <- d %>% tfd_survival_function(matrix(1:4, ncol = 2))
  expect_length(dim(x), 2)

})


test_succeeds("can compute mean", {

  d <- tfd_normal(0,1)
  x <- d %>% tfd_mean()
  expect_length(dim(x), 0)

})

test_succeeds("can compute entropy", {

  d <- tfd_normal(0,1)
  x <- d %>% tfd_entropy()
  expect_length(dim(x), 0)

})

test_succeeds("can compute quantiles", {

  d <- tfd_normal(0,1)
  x <- d %>% tfd_quantile(0.25)
  expect_length(dim(x), 0)

})

test_succeeds("can compute variance", {

  d <- tfd_normal(0,1)
  x <- d %>% tfd_variance()
  expect_length(dim(x), 0)

})

test_succeeds("can compute stddev", {

  d <- tfd_normal(0,1)
  x <- d %>% tfd_stddev()
  expect_length(dim(x), 0)

})

test_succeeds("can compute covariance", {

  d <- tfd_multivariate_normal_diag(loc = c(1, -1),
                                             scale_diag = c(1, 2))
  x <- d %>% tfd_covariance()
  expect_length(dim(x), 2)

})

test_succeeds("can compute mode", {

  d <- tfd_bernoulli(probs = c(0.7, 0.3))
  x <- d %>% tfd_mode()
  expect_length(dim(x), 1)

})

test_succeeds("can compute cross entropy", {

  d <- tfd_multivariate_normal_diag(loc = c(1, -1),
                                             scale_diag = c(1, 2))
  other <- tfd_multivariate_normal_diag(loc = c(0, 0),
                                                 scale_diag = c(1, 1))
  x <- d %>% tfd_cross_entropy(other)
  expect_length(dim(x), 0)

})

# KL[p, q] =  H[p, q] - H[p]
test_succeeds("can compute cross entropy", {

  d <- tfd_multivariate_normal_diag(loc = c(1, -1),
                                             scale_diag = c(1, 2))
  other <- tfd_multivariate_normal_diag(loc = c(0, 0),
                                                 scale_diag = c(1, 1))
  cx <- d %>% tfd_cross_entropy(other) %>% tensor_value()
  ent <- d %>% tfd_entropy() %>% tensor_value()
  kl <- d %>% tfd_kl_divergence(other) %>% tensor_value()
  expect_equal(kl, cx - ent, tolerance = 1e-6)
})
