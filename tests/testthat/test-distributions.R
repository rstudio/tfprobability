context("distributions")

source("utils.R")

test_succeeds("Normal distribution works", {
  d <- tfd_normal(loc = c(1, 2), scale = c(11, 22))
  x <- d %>% tfd_sample(c(2, 2))
  expect_equal(d$batch_shape$as_list(), 2)

})

test_succeeds("MultivariateNormalDiag distribution works", {
  d <- tfd_multivariate_normal_diag(loc = c(1,-1),
                                    scale_identity_multiplier = c(1, 2, 3))
  x <- d %>% tfd_sample()
  expect_equal(d$batch_shape$as_list(), 3)

})

test_succeeds("Independent distribution works", {

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

  # Make independent distribution from a 28*28-batch Bernoulli.
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

test_succeeds("Transformed distribution works", {

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
  expect_equivalent(log_prob %>% tensor_value(), rep(-log(2 * pi), 6), tol = 1e-6)
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
  expect_equal(vla %>% tfd_covariance() %>% tensor_value(), cov, tol = 1e-6)
})

test_succeeds("VectorLaplaceDiag distribution works", {

  d <- tfd_vector_laplace_diag(loc = matrix(rep(0, 6), ncol =3))
  expect_equivalent(d %>% tfd_stddev() %>% tensor_value(), rep(sqrt(2), 3), tol = 1e-8)
})

test_succeeds("VectorExponentialDiag distribution works", {

  d <- tfd_vector_exponential_diag(loc = c(-1, 1),scale_diag = c(1, -5))
  expect_equivalent(d %>% tfd_mean() %>% tensor_value(), c(-1 + 1, 1 - 5), tol = 1e-8)
})

test_succeeds("VectorExponentialDiag distribution works", {

  s <- matrix(c(1, 0.1, 0.1, 1), ncol = 2)
  d <- tfd_vector_exponential_linear_operator(scale = tf$linalg$LinearOperatorFullMatrix(s))
  expect_equivalent(d %>% tfd_mean() %>% tensor_value(), c(1.1, 1.1), tol = 1e-8)
})

test_succeeds("VectorDiffeoMixture distribution works", {

  dims <- 5L
  d <- tfd_vector_diffeomixture(
    mix_loc = list(c(0, 1)),
    temperature = list(1),
    distribution = tfd_normal(loc = 0, scale = 1),
    loc = list(NULL, rep(2, 5)),
    scale = list(
      tf$linalg$LinearOperatorScaledIdentity(
        num_rows = dims,
        multiplier = 1.1,
        is_positive_definite = TRUE),
      tf$linalg$LinearOperatorDiag(
        diag = seq(2.5, 3.5,  length.out = 5),
        is_positive_definite = TRUE)))
  expect_equal((d %>% tfd_mean())$get_shape()$as_list(), c(1, 5))
})


test_succeeds("VariationalGaussianProcess distribution works", {

  # Important:
  # This test only creates the distribution and does not train it.
  # Consider enhancing as per examples in
  # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/distributions/variational_gaussian_process.py

  # Create kernel with trainable parameters, and trainable observation noise
  # variance variable. Each of these is constrained to be positive.
  amplitude <- tf$nn$softplus(tf$Variable(-1, name = 'amplitude'))
  length_scale <-
    1e-5 + tf$nn$softplus(tf$Variable(-3, name = 'length_scale'))
  kernel = tfp$positive_semidefinite_kernels$ExponentiatedQuadratic(amplitude = amplitude,
                                                                    length_scale = length_scale)
  observation_noise_variance <- tf$nn$softplus(tf$Variable(0, name = 'observation_noise_variance'))
  # Create trainable inducing point locations and variational parameters.
  num_inducing_points <- 20L
  initial_inducing_points <-
    matrix(seq(-13, 13, length.out = num_inducing_points), nrow = num_inducing_points) %>%
    tf$cast(tf$float32)
  inducing_index_points <- tf$Variable(initial_inducing_points, name = 'inducing_index_points')
  variational_inducing_observations_loc <-
    tf$Variable(rep(0, num_inducing_points) %>% tf$cast(tf$float32),
                name = 'variational_inducing_observations_loc')
  variational_inducing_observations_scale <-
    tf$Variable(diag(num_inducing_points) %>% tf$cast(tf$float32),
                name = 'variational_inducing_observations_scale')
  # These are the index point locations over which we'll construct the
  # (approximate) posterior predictive distribution.
  num_predictive_index_points <- 500
  index_points <-
    matrix(seq(-13, 13, length.out = num_predictive_index_points), nrow = num_predictive_index_points) %>% tf$cast(tf$float32)
  # Construct our variational GP Distribution instance.
  vgp = tfd_variational_gaussian_process(
    kernel,
    index_points = index_points,
    inducing_index_points = inducing_index_points,
    variational_inducing_observations_loc = variational_inducing_observations_loc,
    variational_inducing_observations_scale = variational_inducing_observations_scale,
    observation_noise_variance = observation_noise_variance
  )
})

test_succeeds("Uniform distribution works", {

  d <- tfd_uniform(low = 3, high = c(5, 6, 7))
  expect_equivalent(d %>% tfd_mean() %>% tensor_value(), c(4, 4.5, 5), tol = 1e-6)
})

test_succeeds("Truncated normal distribution works", {

  d <- tfd_truncated_normal(loc = c(0, 1),
                            scale = 1,
                            low = c(-1, 0),
                            high = c(1, 1))

  m <- d %>% tfd_mean()
  expect_equal(m$get_shape()$as_list(), 2)
})

test_succeeds("Triangular distribution works", {

  d <- tfd_triangular(low = 3, high = 7, peak = 5)
  expect_equivalent(d %>% tfd_mean() %>% tensor_value(), 5)
})

test_succeeds("Student T distribution works", {

  d <- tfd_student_t(df = c(5,6), loc = 0, scale = 1)
  sds <- d %>% tfd_stddev() %>% tensor_value()
  expect_gt(sds[1], sds[2])
})

test_succeeds("Student T process works", {

  num_points <- 100
  index_points <- seq(-1., 1., length.out = num_points) %>% matrix(nrow = num_points)
  kernel <- tfp$positive_semidefinite_kernels$ExponentiatedQuadratic()
  d <- tfd_student_t_process(df = 3, kernel = kernel, index_points = index_points)
  noisy_samples <- d %>% tfd_sample(10)
  expect_equal(noisy_samples$get_shape()$as_list(), c(10, 100))
})

test_succeeds("SinhArcsinh distribution works", {

  n <- 10
  scale <- runif(n)
  loc <- rnorm(n)
  norm = tfd_normal(
    loc = loc,
    scale = scale)
  vsad = tfd_sinh_arcsinh(
    loc = loc,
    scale = scale)

  x <- matrix(rnorm(5 * n), ncol = n)
  normal_prob <- norm %>% tfd_prob(x)
  vsad_prob <- vsad %>% tfd_prob(x)
  expect_equal(normal_prob %>% tensor_value(), vsad_prob %>% tensor_value(), tol = 1e-6)
})

test_succeeds("Quantized distribution works", {

  scale <- 1
  loc <- 0
  q = tfd_quantized(tfd_normal(
    loc = loc,
    scale = scale))

  x <- c(0.1, 0.4, 1.2)
  q_prob <- q %>% tfd_cdf(x)
  expect_equal(q_prob %>% tensor_value() %>% which.max(), 3)
})

test_succeeds("Poisson distribution works", {

  lambda <- c(1, 3, 2.5)
  d <- tfd_poisson(rate = lambda)

  expect_equivalent(d %>% tfd_stddev() %>% tensor_value(), sqrt(lambda), tol = 1e-7)
})

test_succeeds("PoissonLogNormalQuadratureCompound distribution works", {

  d <-
    tfd_poisson_log_normal_quadrature_compound(
      loc = c(0.,-0.5),
      scale = 1,
      quadrature_size = 10
    )

  expect_equal((d %>% tfd_stddev())$get_shape()$as_list(), 2)
})

test_succeeds("Pareto distribution works", {

  d <- tfd_pareto(2)
  expect_equal(d %>% tfd_mode() %>% tensor_value(), 1)
})

test_succeeds("NegativeBinomial distribution works", {

  d <- tfd_negative_binomial(total_count = 23, probs = 0.1)
  nb_mean <- function(r, p) r * p /(1 - p)
  expect_equal(d %>% tfd_mean() %>% tensor_value(), nb_mean(23, 0.1), tol = 1e7)
})

test_succeeds("MultivariateNormalTriL distribution works", {

  mu <- c(1, 2, 3)
  cov <- matrix(c(0.36,  0.12,  0.06, 0.12,  0.29, -0.13,  0.06, -0.13,  0.26), nrow = 3, byrow =TRUE)
  scale <- tf$cholesky(cov)
  d <- tfd_multivariate_normal_tri_l(loc = mu, scale_tril = scale)
  expect_equivalent(d %>% tfd_mean() %>% tensor_value(), mu)
})

test_succeeds("MultivariateNormalLinearOperator distribution works", {

  mu <- c(1, 2, 3)
  cov <- matrix(c(0.36,  0.12,  0.06, 0.12,  0.29, -0.13,  0.06, -0.13,  0.26), nrow = 3, byrow =TRUE)
  scale <- tf$cholesky(cov)
  d <- tfd_multivariate_normal_linear_operator(loc = mu, scale = tf$linalg$LinearOperatorLowerTriangular(scale))
  expect_equivalent(d %>% tfd_covariance() %>% tensor_value(), cov)
})

test_succeeds("MultivariateNormalFullCovariance distribution works", {

  mu <- c(1, 2, 3)
  cov <- matrix(c(0.36,  0.12,  0.06, 0.12,  0.29, -0.13,  0.06, -0.13,  0.26), nrow = 3, byrow =TRUE)
  d <- tfd_multivariate_normal_full_covariance(loc = mu, covariance_matrix = cov)
  expect_equivalent(d %>% tfd_mean() %>% tensor_value(), mu)
})

test_succeeds("MultivariateNormalDiagPlusLowRank distribution works", {

  # Initialize a single 3-variate Gaussian with covariance `cov = S @ S.T`,
  # `S = diag(d) + U @ diag(m) @ U.T`. The perturbation, `U @ diag(m) @ U.T`, is
  # a rank-2 update.
  mu <- c(-0.5, 0, 0.5)
  d <- c(1.5, 0.5, 2)
  U <- matrix(c(1, 2, -1, 1, 2, -0.5), nrow = 3, byrow = TRUE)
  m <- c(4, 5)
  d <- tfd_multivariate_normal_diag_plus_low_rank(loc = mu,
                                                  scale_diag = d,
                                                  scale_perturb_factor = U,
                                                  scale_perturb_diag = m)
  expect_equal((d %>% tfd_prob(c(-1, 0, 1)))$get_shape()$as_list(), list())
})

test_succeeds("MultivariateStudentTLinearOperator distribution works", {

  df <- 3
  loc <- c(1, 2, 3)
  scale <- matrix(c(0.6, 0, 0, 0.2, 0.5, 0, 0.1, -0.3, 0.4), nrow = 3, byrow = TRUE)
  sigma = tf$matmul(scale, scale, adjoint_b = TRUE)
  d <- tfd_multivariate_student_t_linear_operator(
    df = df,
    loc = loc,
    scale = tf$linalg$LinearOperatorLowerTriangular(scale))
  expect_equivalent(d %>% tfd_covariance() %>% tensor_value(), cov * 3)
})

test_succeeds("Multinomial distribution works", {

  p <- list(c(.1, .2, .7), c(.3, .3, .4))
  total_count <- c(4, 5)
  d <- tfd_multinomial(total_count = total_count, probs = p)
  counts <- list(c(2, 1, 1), c(3, 1, 1))
  expect_equal((d %>% tfd_prob(counts))$get_shape()$as_list(), 2)
  expect_equal((d %>% tfd_sample(5))$get_shape()$as_list(), c(5, 2, 3))
})

test_succeeds("Mixture distribution works", {

  mix <- 0.3
  d <- tfd_mixture(
        cat = tfd_categorical(probs = c(mix, 1 -mix)),
    components = list(
      tfd_normal(loc = -1, scale = 0.1),
      tfd_normal(loc = 1, scale = 0.5)))
  expect_equal((d %>% tfd_sample(5))$get_shape()$as_list(), 5)
})




