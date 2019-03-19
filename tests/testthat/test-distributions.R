context("distributions")

source("utils.R")

test_succeeds("Define a batch of two scalar valued Normals", {
  d <- tfd_normal(loc = c(1, 2), scale = c(11, 22))
  x <- d %>% sample(c(2, 2))
  expect_equal(d$batch_shape$as_list(), 2)

})

test_succeeds("Initialize a 3-batch, 2-variate scaled-identity Gaussian.", {
  d <- tfd_multivariate_normal_diag(loc = c(1,-1),
                                             scale_identity_multiplier = c(1, 2, 3))
  x <- d %>% sample()
  expect_equal(d$batch_shape$as_list(), 3)

})

test_succeeds("Make independent distribution from a 2-batch Normal.", {

  d <- tfd_normal(
    loc = c(-1., 1, 5, 2),
    scale = c(0.1, 0.5, 1.4, 6)
  )
  i <- tfd_independent(
    distribution = d,
    reinterpreted_batch_ndims = 1
  )
  i
  expect_equal(i$event_shape$as_list(), 4)

})

test_succeeds("Make independent distribution from a 28*28-batch Bernoulli.", {

  d <- tfd_bernoulli(
   probs = matrix(rep(0.5, 28 * 28), ncol = 28)
  )
  i <- tfd_independent(
    distribution = d,
    reinterpreted_batch_ndims = 2
  )
  i
  expect_equal(i$event_shape$ndims, 2)

})

test_succeeds("Create a log normal distribution from a normal one.", {

  d <- tfd_transformed(
    distribution = tfd_normal(loc = 0, scale = 1),
    bijector = tfb_exp()
  )
  expect_equal(d$event_shape$ndims, 0)

})


