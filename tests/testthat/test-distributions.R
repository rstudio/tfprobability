context("distributions")

source("utils.R")

test_succeeds("Define a batch of two scalar valued Normals", {
  d <- tfd_normal(loc = c(1, 2), scale = c(11, 22))
  x <- d %>% tfd_sample(c(2, 2))
  expect_equal(d$batch_shape$as_list(), 2)

})

test_succeeds("Initialize a 3-batch, 2-variate scaled-identity Gaussian.", {
  d <- tfd_multivariate_normal_diag(loc = c(1,-1),
                                    scale_identity_multiplier = c(1, 2, 3))
  x <- d %>% tfd_sample()
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

test_succeeds("Relaxed one hot categorical distribution works", {
  s <- tfd_relaxed_one_hot_categorical(temperature = 0, logits = c(1, 1)) %>%
    tfd_sample(10) %>%
    tensor_value()

  expect_equal(s, array(NaN, c(10, 2)))

  s <- tfd_relaxed_one_hot_categorical(temperature = 1e-10, logits = c(1e5, -1e5)) %>%
    tfd_sample(10) %>%
    tensor_value()

  expect_equal(s, array(c(rep(1, 10), rep(0, 10)), dim = c(10, 2)))

  s <- tfd_relaxed_one_hot_categorical(temperature = 1e-10, probs = c(0, 1)) %>%
    tfd_sample(10) %>%
    tensor_value()

  expect_equal(s, array(c(rep(0, 10), rep(1, 10)), dim = c(10, 2)))
})

test_succeeds("One hot categorical distribution works", {
  s <- tfd_one_hot_categorical(logits = c(-1e5, 1e5)) %>%
    tfd_sample(10) %>%
    tensor_value()

  expect_equal(s, array(c(rep(0, 10), rep(1, 10)), dim = c(10, 2)))

  s <-  tfd_one_hot_categorical(probs = c(1, 0)) %>%
    tfd_sample(10) %>%
    tensor_value()

  expect_equal(s, array(c(rep(1, 10), rep(0, 10)), dim = c(10, 2)))

  s <- tfd_one_hot_categorical(probs = c(0.5, 0.5)) %>%
    tfd_sample(10) %>%
    tensor_value()

  expect_identical(s %in% c(0, 1), rep(TRUE, 20))
})

test_succeeds("Relaxed Bernoulli distribution works", {

  s <- tfd_relaxed_bernoulli(temperature = 1e-10, logits = c(1e5, -1e5)) %>%
    tfd_sample(10) %>%
    tensor_value()

  expect_equal(s, array(c(rep(1, 10), rep(0, 10)), dim = c(10, 2)))

  s <- tfd_relaxed_bernoulli(temperature = 1e-10, probs = c(0, 1)) %>%
    tfd_sample(10) %>%
    tensor_value()

  expect_equal(s, array(c(rep(0, 10), rep(1, 10)), dim = c(10, 2)))
})

test_succeeds("Zipf distribution works", {

  batch_size <- 12
  power <- rep(3, batch_size)
  x <- c(-3, -0.5, 0, 2, 2.2, 3, 3.1, 4, 5, 5.5, 6, 7.2)

  zipf <- tfd_zipf(power = power, interpolate_nondiscrete = FALSE)
  log_pmf <- zipf %>% tfd_log_prob(x)
  expect_equal(log_pmf$get_shape()$as_list(), batch_size)
})

test_succeeds("Wishart distribution works", {

 s <- matrix(c(1, 2, 2, 5), ncol = 2, byrow = TRUE)
 df <- 4
 d <- tfd_wishart(df = df, scale_tril = tf$linalg$cholesky(s))
 expect_equal(tfd_mean(d) %>% tensor_value(), df * s)
})

test_succeeds("VonMisesFisher distribution works", {

  mean_dirs <- tf$nn$l2_normalize(
    matrix(c(1, 1, -2, 1, 0, -1), ncol =2, byrow = TRUE),
    axis = -1L)
  concentration <- matrix(c(0, 0.1, 2, 40, 1000))
  d = tfd_von_mises_fisher(
    mean_direction = mean_dirs,
    concentration = concentration,
    #validate_args = TRUE, ### this does not work
    allow_nan_stats = FALSE)
  expect_equal(d$batch_shape$as_list(), c(5, 3))
  expect_equal(d$event_shape$as_list(), c(2))
})

test_succeeds("VonMises distribution works", {

  x <- c(2, 3, 4, 5, 6, 7)
  d <- tfd_von_mises(0.1, 0)
  log_prob <- d %>% tfd_log_prob(x)
  expect_equivalent(log_prob %>% tensor_value(), rep(-log(2 * pi), 6), tol = 1e6)
})

test_succeeds("VectorSinhArcsinhDiag distribution works", {

  n <- 10
  scale_diag <- runif(n)
  scale_identity_multiplier <- 1
  loc = rnorm(n)
  norm = tfd_multivariate_normal_diag(
    loc = loc,
    scale_diag = scale_diag,
    scale_identity_multiplier = scale_identity_multiplier,
    validate_args = TRUE)
  vsad = tfd_vector_sinh_arcsinh_diag(
    loc = loc,
    scale_diag = scale_diag,
    scale_identity_multiplier = scale_identity_multiplier,
    validate_args = TRUE)

  x <- matrix(rnorm(5 * n), ncol = n)
  normal_prob <- norm %>% tfd_prob(x)
  vsad_prob <- vsad %>% tfd_prob(x)
  expect_equal(normal_prob %>% tensor_value(), vsad_prob %>% tensor_value())
})

test_succeeds("VectorLaplaceLinearOperator distribution works", {

  mu <- c(1, 2, 3)
  cov <-
    matrix(
      c(0.36,  0.12,  0.06, 0.12,  0.29,-0.13,  0.06,-0.13,  0.26),
      nrow = 3,
      byrow = TRUE
    )
  scal <- tf$cholesky(cov) %>% tf$cast(tf$float32)
  vla <- tfd_vector_laplace_linear_operator(
    loc = mu,
    scal = tf$linalg$LinearOperatorLowerTriangular(scal / tf$sqrt(2))
  )
  vla %>% tfd_covariance() %>% tensor_value()
  expect_equal(vla %>% tfd_covariance() %>% tensor_value(), cov, tol = 1e6)
})

test_succeeds("VectorLaplaceDiag distribution works", {

  d <- tfd_vector_laplace_diag(loc = matrix(rep(0, 6), ncol =3))
  expect_equivalent(d %>% tfd_stddev() %>% tensor_value(), rep(sqrt(2), 3), tol = 1e8)
})


