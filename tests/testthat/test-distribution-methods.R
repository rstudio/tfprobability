context("distribution methods")

source("utils.R")

test_succeeds("can sample from distributions", {

  d <- tfd_normal(0,1)
  x <- d %>% sample(c(2, 2))
  expect_length(dim(x), 2)

  x <- d %>% sample()
  expect_length(dim(x), 0)

})

test_succeeds("can compute log probability", {

  d <- tfd_normal(0,1)
  x <- d %>% log_prob(c(0.22, 0.44, 3))
  expect_length(dim(x), 1)

})

test_succeeds("can compute probability", {

  d <- tfd_normal(0,1)
  x <- d %>% prob(matrix(1:4, ncol = 2))
  expect_length(dim(x), 2)

})

test_succeeds("can compute log cdf", {

  d <- tfd_normal(0,1)
  x <- d %>% log_cdf(matrix(1:4, ncol = 2))
  expect_length(dim(x), 2)

})

test_succeeds("can compute cdf", {

  d <- tfd_normal(0,1)
  x <- d %>% cdf(matrix(1:4, ncol = 2))
  expect_length(dim(x), 2)

})

test_succeeds("can compute log survival function", {

  d <- tfd_normal(0,1)
  x <- d %>% log_survival_function(matrix(1:4, ncol = 2))
  expect_length(dim(x), 2)

})

test_succeeds("can compute survival function", {

  d <- tfd_normal(0,1)
  x <- d %>% survival_function(matrix(1:4, ncol = 2))
  expect_length(dim(x), 2)

})


test_succeeds("can compute mean", {

  d <- tfd_normal(0,1)
  x <- d %>% mean()
  expect_length(dim(x), 0)

})

test_succeeds("can compute entropy", {

  d <- tfd_normal(0,1)
  x <- d %>% entropy()
  expect_length(dim(x), 0)

})

test_succeeds("can compute quantiles", {

  d <- tfd_normal(0,1)
  x <- d %>% quantile(0.25)
  expect_length(dim(x), 0)

})

test_succeeds("can compute variance", {

  d <- tfd_normal(0,1)
  x <- d %>% variance()
  expect_length(dim(x), 0)

})

test_succeeds("can compute stddev", {

  d <- tfd_normal(0,1)
  x <- d %>% stddev()
  expect_length(dim(x), 0)

})

test_succeeds("can compute covariance", {

  d <- tfd_multivariate_normal_diag(loc = c(1, -1),
                                             scale_diag = c(1, 2))
  x <- d %>% covariance()
  expect_length(dim(x), 2)

})

test_succeeds("can compute mode", {

  d <- tfd_bernoulli(probs = c(0.7, 0.3))
  x <- d %>% mode()
  expect_length(dim(x), 1)

})

test_succeeds("can compute cross entropy", {

  d <- tfd_multivariate_normal_diag(loc = c(1, -1),
                                             scale_diag = c(1, 2))
  other <- tfd_multivariate_normal_diag(loc = c(0, 0),
                                                 scale_diag = c(1, 1))
  x <- d %>% cross_entropy(other)
  expect_length(dim(x), 0)

})

# KL[p, q] =  H[p, q] - H[p]
test_succeeds("can compute cross entropy", {

  d <- tfd_multivariate_normal_diag(loc = c(1, -1),
                                             scale_diag = c(1, 2))
  other <- tfd_multivariate_normal_diag(loc = c(0, 0),
                                                 scale_diag = c(1, 1))
  cx <- d %>% cross_entropy(other) %>% tensor_value()
  ent <- d %>% entropy() %>% tensor_value()
  kl <- d %>% kl_divergence(other) %>% tensor_value()
  expect_equal(kl, cx - ent, tolerance = 1e-6)
})
