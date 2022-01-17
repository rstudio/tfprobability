context("distributions")

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

  d <- tfd_transformed_distribution(
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

 skip_if_tfp_above("0.8")
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

  skip_if_tfp_above("0.9")
  n <- 10
  scale_diag <- runif(n)
  scale_identity_multiplier <- 1
  loc = rnorm(n)
  norm = tfd_multivariate_normal_diag(
    loc = loc,
    scale_diag = scale_diag,
    validate_args = TRUE)
  vsad = tfd_vector_sinh_arcsinh_diag(
    loc = loc,
    scale_diag = scale_diag,
    validate_args = TRUE)

  x <- matrix(rnorm(5 * n), ncol = n)
  normal_prob <- norm %>% tfd_prob(x)
  vsad_prob <- vsad %>% tfd_prob(x)
  expect_equal(normal_prob %>% tensor_value(), vsad_prob %>% tensor_value())
})

test_succeeds("VectorLaplaceLinearOperator distribution works", {

  skip_if_tfp_above("0.9")
  mu <- c(1, 2, 3)
  cov <-
    matrix(
      c(0.36,  0.12,  0.06, 0.12,  0.29,-0.13,  0.06,-0.13,  0.26),
      nrow = 3,
      byrow = TRUE
    )
  scal <- tf$linalg$cholesky(cov) %>% tf$cast(tf$float32)
  vla <- tfd_vector_laplace_linear_operator(
    loc = mu,
    scal = tf$linalg$LinearOperatorLowerTriangular(scal / tf$sqrt(2))
  )
  vla %>% tfd_covariance() %>% tensor_value()
  expect_equal(vla %>% tfd_covariance() %>% tensor_value(), cov, tol = 1e-6)
})

test_succeeds("VectorLaplaceDiag distribution works", {

  skip_if_tfp_above("0.9")
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
  skip("Batch dim behavior changed")
  expect_equivalent(d %>% tfd_mean() %>% tensor_value(), c(1.1, 1.1), tol = 1e-8)
})

test_succeeds("VectorDiffeoMixture distribution works", {

  skip_if_tfp_above("0.10")

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
  kernel = (get_psd_kernels())$ExponentiatedQuadratic(amplitude = amplitude,
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
  kernel <- (get_psd_kernels())$ExponentiatedQuadratic()
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
  expect_equal(d %>% tfd_mean() %>% tensor_value(), nb_mean(23, 0.1), tol = 1e-7)
})

test_succeeds("MultivariateNormalTriL distribution works", {

  mu <- c(1, 2, 3)
  cov <- matrix(c(0.36,  0.12,  0.06, 0.12,  0.29, -0.13,  0.06, -0.13,  0.26), nrow = 3, byrow =TRUE)
  scale <- tf$linalg$cholesky(cov)
  d <- tfd_multivariate_normal_tri_l(loc = mu, scale_tril = scale)
  expect_equivalent(d %>% tfd_mean() %>% tensor_value(), mu)
})

test_succeeds("MultivariateNormalLinearOperator distribution works", {

  mu <- c(1, 2, 3)
  cov <- matrix(c(0.36,  0.12,  0.06, 0.12,  0.29, -0.13,  0.06, -0.13,  0.26), nrow = 3, byrow =TRUE)
  scale <- tf$linalg$cholesky(cov)
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


test_succeeds("Categorical distribution works", {

  d <- tfd_categorical(logits = log(c(0.1, 0.5, 0.4)))
  empirical_prob <- tf$cast(tf$histogram_fixed_width(d %>% tfd_sample(1e4),
                                                     c(0L, 2L),
                                                     nbins = 3L),
                            dtype = tf$float32) / 1e4
  expect_equal(which.max(empirical_prob %>% tensor_value()), 2)
})

test_succeeds("Mixture same family distribution works", {

  d <- tfd_mixture_same_family(
    mixture_distribution = tfd_categorical(probs = c(0.3, 0.7)),
    components_distribution = tfd_normal(loc = c(-1, 1),
                                         scale = c(0.1, 0.5))
  )

  expect_equal(d %>% tfd_mean() %>% tensor_value(), 0.4, tol = 1e-7)
})

test_succeeds("Log normal distribution works", {

  mu <- 1.3
  v <- 0.5
  d <- tfd_log_normal(loc = mu, scale = sqrt(v))
  expected_mean <- exp(mu + v/2)
  expect_equal(d %>% tfd_mean() %>% tensor_value(), expected_mean)
})

test_succeeds("Logistic distribution works", {

  d <- tfd_logistic(loc = c(1, 2), scale = c(11, 22))
  d %>% tfd_prob(c(0, 1.5))
  expect_equal((d %>% tfd_prob(c(0, 1.5)))$get_shape()$as_list(), 2)
})

test_succeeds("LKJ distribution works", {

  d <- tfd_lkj(dimension = 3, concentration = 1.5)
  prob <- d %>% tfd_prob(array(rnorm(2 * 3 * 3), dim = c(2, 3, 3)))
  expect_equal(prob$get_shape()$as_list(), 2)
})

test_succeeds("CholeskyLKJ distribution works", {

  skip_if_tfp_below("0.9")

  d <- tfd_cholesky_lkj(dimension = 3, concentration = 1.5)
  prob <- d %>% tfd_prob(array(rnorm(2 * 3 * 3), dim = c(2, 3, 3)))
  expect_equal(prob$get_shape()$as_list(), 2)
})

test_succeeds("LinearGaussianStateSpaceModel distribution works", {

  ndims <- 2L
  step_std <- 1
  noise_std <- 5
  d <- tfd_linear_gaussian_state_space_model(
    num_timesteps = 100,
    transition_matrix = tf$linalg$LinearOperatorIdentity(ndims),
    transition_noise = tfd_multivariate_normal_diag(
      scale_diag = step_std^2 * tf$ones(list(ndims))),
    observation_matrix = tf$linalg$LinearOperatorIdentity(ndims),
    observation_noise = tfd_multivariate_normal_diag(
      scale_diag = noise_std^2 * tf$ones(list(ndims))),
    initial_state_prior = tfd_multivariate_normal_diag(
      scale_diag = noise_std^2 * tf$ones(list(ndims))))
  expect_equal((d %>% tfd_sample())$get_shape()$as_list(), c(100, 2))
})

test_succeeds("Kumaraswamy distribution works", {

  alpha <- c(1, 2, 3)
  beta <- c(1, 2, 3)
  d <- tfd_kumaraswamy(alpha, beta)
  x <- matrix(c(0.1, 0.4, 0.5, 0.2, 0.3, 0.5), ncol = 3)
  prob <- d %>% tfd_prob(x)
  expect_equal(prob$get_shape()$as_list(), c(2, 3))
})

test_succeeds("Exponential distribution works", {

  d <- tfd_exponential(rate = 0.1)
  expect_equal(d %>% tfd_mean() %>% tensor_value(), 10)
})

test_succeeds("Gamma distribution works", {

  d <- tfd_gamma(concentration = 1, rate = 2)
  expect_equal(d %>% tfd_mean() %>% tensor_value(), 0.5)
})

test_succeeds("JointDistributionSequential distribution works", {

  skip_if_tfp_below("0.9")

  d <- tfd_joint_distribution_sequential(
    list(
      # e
      tfd_independent(tfd_exponential(rate = c(100, 120)), 1, name = "e"),
      # g
      function(e) tfd_gamma(concentration = e[1], rate = e[2], name = "g"),
      # n
      tfd_normal(loc = 0, scale = 2, name = "n1"),
      # m
      function(n1, g) tfd_normal(loc = n1, scale = g, name = "n2"),
      # x
      function(n2) tfd_sample_distribution(tfd_bernoulli(logits = n2), 12, name = "s")
    ))

  x <- d %>% tfd_sample()
  expect_equal(length(x), 5)
  expect_equal((d %>% tfd_log_prob(x))$get_shape()$as_list(), list())
  expect_equal(d$resolve_graph() %>% length(), 5)

})

test_succeeds("Track JointDistributionSequential broadcasting/reshaping changes", {

  df <- cbind(iris[1:100, 1:4], as.integer(iris[1:100, 5]) - 1)

  model <- tfd_joint_distribution_sequential(
    list(# b1
      tfd_normal(loc = 0, scale = 0.5),
      # b2
      tfd_normal(loc = 0, scale = 0.5),
      # b3
      tfd_normal(loc = 0, scale = 0.5),
      # b4
      tfd_normal(loc = 0, scale = 0.5),
      function(b4, b3, b2, b1)
        tfd_independent(
          tfd_binomial(
            total_count = 1,
            logits =
              tf$expand_dims(b1, axis = -1L) * df$Sepal.Length +
              tf$expand_dims(b2, axis = -1L) * df$Sepal.Width +
              tf$expand_dims(b3, axis = -1L) * df$Petal.Length +
              tf$expand_dims(b4, axis = -1L) * df$Petal.Width
          ),
          reinterpreted_batch_ndims = 1
        )
    )
  )

  # this SHOULD work as well but does not
  # model <- tfd_joint_distribution_sequential(
  #   list(# b1
  #     tfd_normal(loc = 0, scale = 0.5),
  #     # b2
  #     tfd_normal(loc = 0, scale = 0.5),
  #     # b3
  #     tfd_normal(loc = 0, scale = 0.5),
  #     # b4
  #     tfd_normal(loc = 0, scale = 0.5),
  #     function(b4, b3, b2, b1)
  #       tfd_independent(
  #         tfd_binomial(
  #           total_count = 1,
  #           logits =
  #             b1[reticulate::py_ellipsis(), tf$newaxis] * df$Sepal.Length +
  #             b2[reticulate::py_ellipsis(), tf$newaxis] * df$Sepal.Width +
  #             b3[reticulate::py_ellipsis(), tf$newaxis] * df$Petal.Length +
  #             b4[reticulate::py_ellipsis(), tf$newaxis] * df$Petal.Width
  #         ),
  #         reinterpreted_batch_ndims = 1
  #       )
  #   )
  # )

  samples <- model %>% tfd_sample(3)
  model %>% tfd_log_prob(samples)
})

test_succeeds("JointDistributionNamed distribution works", {

  skip_if_tfp_below("0.9")
  d <- tfd_joint_distribution_named(
    list(
      e = tfd_independent(tfd_exponential(rate = c(100, 120)), 1),
      g = function(e) tfd_gamma(concentration = e[1], rate = e[2]),
      n = tfd_normal(loc = 0, scale = 2),
      m = function(n, g) tfd_normal(loc = n, scale = g),
      x = function(m) tfd_sample_distribution(tfd_bernoulli(logits = m), 12)
    ))

  x <- d %>% tfd_sample()
  expect_equal(length(x), 5)
  expect_equal((d %>% tfd_log_prob(x))$get_shape()$as_list(), list())
  expect_equal(d$resolve_graph() %>% length(), 5)

})

test_succeeds("Inverse Gaussian distribution works", {

  d <- tfd_inverse_gaussian(loc = 1, concentration = 1)
  expect_equal(d %>% tfd_mean() %>% tensor_value(), 1)
})

test_succeeds("Inverse Gamma distribution works", {

  d <- tfd_inverse_gamma(concentration = 1.5, scale = 2)
  expect_equal(d %>% tfd_mean() %>% tensor_value(), 2/(1.5-1))
})

test_succeeds("Horseshoe distribution works", {

  d <- tfd_horseshoe(scale = 2)
  expect_equal(d %>% tfd_mean() %>% tensor_value(), 0)
})

test_succeeds("Hidden Markov Model distribution works", {

  # Represent a cold day with 0 and a hot day with 1.
  # Suppose the first day of a sequence has a 0.8 chance of being cold.
  # We can model this using the categorical distribution:
  initial_distribution <- tfd_categorical(probs = c(0.8, 0.2))
  # Suppose a cold day has a 30% chance of being followed by a hot day
  # and a hot day has a 20% chance of being followed by a cold day.
  # We can model this as:
  transition_distribution <- tfd_categorical(probs = matrix(c(0.7, 0.3, 0.2, 0.8), nrow = 2, byrow = TRUE) %>% tf$cast(tf$float32))
  # Suppose additionally that on each day the temperature is
  # normally distributed with mean and standard deviation 0 and 5 on
  # a cold day and mean and standard deviation 15 and 10 on a hot day.
  # We can model this with:
  observation_distribution <- tfd_normal(loc = c(0, 15), scale = c(5, 10))
  # We can combine these distributions into a single week long
  # hidden Markov model with:
  d <- tfd_hidden_markov_model(
    initial_distribution = initial_distribution,
    transition_distribution = transition_distribution,
    observation_distribution = observation_distribution,
    num_steps = 7)
  # The expected temperatures for each day are given by:
  d %>% tfd_mean() %>% tensor_value() # shape [7], elements approach 9.0
  # The log pdf of a week of temperature 0 is:
  d %>% tfd_log_prob(rep(0, 7)) %>% tensor_value()
})

test_succeeds("Half normal distribution works", {

  d <- tfd_half_normal(scale = 1)
  expect_equal(d %>% tfd_mean() %>% tensor_value(), 1 * sqrt(2) / sqrt(pi), tol = 1e-7)
})

test_succeeds("Half cauchy distribution works", {

  d <- tfd_half_cauchy(loc = 1, scale = 1)
  expect_equal(d %>% tfd_mode() %>% tensor_value(), 1)
})

test_succeeds("Beta distribution works", {

  d <- tfd_beta(concentration1 = 3, concentration0 = 1)
  expect_equal(d %>% tfd_mean() %>% tensor_value(), 3/4)
})

test_succeeds("Binomial distribution works", {

  d <- tfd_binomial(total_count = 7, probs = 0.3)
  expect_equal(d %>% tfd_mean() %>% tensor_value(), 7 * 0.3, tol = 1e-7)
})

test_succeeds("Cauchy distribution works", {

  d <- tfd_cauchy(loc = 1, scale = 1)
  expect_equal(d %>% tfd_entropy() %>% tensor_value(), log(4 * pi * 1))
})

test_succeeds("Gamma-Gamma distribution works", {

  d <- tfd_gamma_gamma(concentration = 1, mixing_concentration = 2, mixing_rate = 0.5)
  expect_equal(d %>% tfd_mean() %>% tensor_value(), 0.5)
})

test_succeeds("Chi distribution works", {

  d <- tfd_chi(df = 1)
  expect_equal((d %>% tfd_prob(c(0, 1)))$get_shape()$as_list() , 2)
})

test_succeeds("Chi2 distribution works", {

  d <- tfd_chi2(df = 1)
  expect_equal(d %>% tfd_mean() %>% tensor_value() , 1)
})

test_succeeds("Gumbel distribution works", {

  d <- tfd_gumbel(loc = 3, scale = 1.5)
  expect_equal(d %>% tfd_variance() %>% tensor_value() , pi^2/6 * 1.5^2, tol = 1e-7)
})

test_succeeds("Geometric distribution works", {

  d <- tfd_geometric(probs = 0.6)
  expect_equal(d %>% tfd_variance() %>% tensor_value() , (1 - 0.6)/0.6^2, tol = 1e-6)
})

test_succeeds("Dirichlet distribution works", {

  d <- tfd_dirichlet(concentration = c(1, 2, 3))
  expect_equivalent(d %>% tfd_mean() %>% tensor_value() , c(1/6, 1/3, 1/2))
})

test_succeeds("DirichletMultinomial distribution works", {

  d <- tfd_dirichlet_multinomial(total_count = 2, concentration = c(1, 2, 3))
  expect_equivalent(d %>% tfd_mean() %>% tensor_value() , c(1/3, 2/3, 1), tol = 1e-7)
})

test_succeeds("Deterministic distribution works", {

  d <- tfd_deterministic(loc = 1.22)
  expect_equal(d %>% tfd_prob(1.23) %>% tensor_value() ,0)
})

test_succeeds("Empirical distribution works", {

  d <- tfd_empirical(samples = c(0, 1, 1, 2))
  expect_equal(d %>% tfd_cdf(1) %>% tensor_value(), 0.75)
})

test_succeeds("BatchReshape distribution works", {

  dims <- 2
  new_batch_shape <- c(1, 2, -1)
  old_batch_shape <- 6
  scale <- matrix(rep(1, old_batch_shape * dims), nrow = old_batch_shape)
  mvn <- tfd_multivariate_normal_diag(scale_diag = scale)
  d <- tfd_batch_reshape(
    distribution = mvn,
    batch_shape = new_batch_shape,
    validate_args = TRUE)
  expect_equal(d$batch_shape$as_list() %>% length(), 3)
})

test_succeeds("Autoregressive distribution works", {

  if (tfp_version() >= 0.8) {
    fill_triangular <- tfp$math$fill_triangular
  } else {
    fill_triangular <- tfp$distributions$fill_triangular
  }

  normal_fn <- function(event_size) {
    n <- event_size * (event_size + 1) / 2 %>% trunc()
    n <- tf$cast(n, tf$int32)
    p <- 2 * tfd_uniform()$sample(n) - 1

    affine <- tfb_affine(scale_tril = fill_triangular(0.25 * p))

    fn <- function(samples) {
      scale <- tf$exp(affine %>% tfb_forward(samples))
      tfd_independent(tfd_normal(loc = 0, scale = scale, validate_args = TRUE),
        reinterpreted_batch_ndims = 1)
    }
    fn
  }

  batch_and_event_shape <- c(3L, 2L, 4L)
  sample0 <- tf$zeros(batch_and_event_shape, dtype = tf$int32)
  skip("bijectors.Affine removed, test needs updating")
  ar <- tfd_autoregressive(normal_fn(batch_and_event_shape[3]), sample0)
  x <- ar %>% tfd_sample(c(6, 5))
  expect_equal(x$get_shape()$as_list(), c(6, 5, 3, 2, 4))

})


test_succeeds("BatchReshape distribution works", {

  num_points <- 100
  kernel <- (get_psd_kernels())$ExponentiatedQuadratic()
  index_points <-
    matrix(seq(-1, 1, length.out = num_points), nrow = num_points) %>%
    tf$cast(tf$float32)

  gp <- tfd_gaussian_process(kernel, index_points, observation_noise_variance = .05)
  samples <- gp %>% tfd_sample(10)
  expect_equal(samples$get_shape()$as_list(), c(10, 100))
})

test_succeeds("Sample distribution works", {

  d <- tfd_sample_distribution(
    tfd_independent(tfd_normal(loc = tf$zeros(list(3L, 2L)), scale = 1),
                    reinterpreted_batch_ndims = 1),
    sample_shape = list(5, 4))

  samples <- d %>% tfd_sample(list(6, 1))
  expect_equal(samples$get_shape()$as_list(), c(6, 1, 3, 5, 4, 2))
  logprob <- d %>% tfd_log_prob(samples)
  expect_equal(logprob$get_shape()$as_list(), c(6, 1, 3))
})

test_succeeds("Blockwise distribution works", {

  d <- tfd_blockwise(
    list(
      tfd_multinomial(total_count = 5, probs = c(0.1, 0.2, 0.3)),
      tfd_multivariate_normal_diag(loc = c(0,0,0,0))
      )
  )

  expect_equal(tfd_sample(d)$get_shape()$as_list(), 7)
})

test_succeeds("Vector Deterministic distribution works", {

  d <- tfd_vector_deterministic(
    loc = c(1,1,1)
  )

  expect_equal(tfd_sample(d)$get_shape()$as_list(), 3)
})

test_succeeds("Gaussian Process Regression Model works", {

  skip_if_tfp_below("0.8")

  d <- tfd_gaussian_process_regression_model(
    kernel = (get_psd_kernels())$MaternFiveHalves(),
    index_points = matrix(seq(-1, 1, length.out = 100), ncol = 1),
    observation_index_points = matrix(runif(50), ncol = 1),
    observations = rnorm(50),
    observation_noise_variance = 0.5
  )


  expect_equal(tfd_sample(d)$get_shape()$as_list(), 100)
})

test_succeeds("Exponential Relaxed One Hot Categorical distribution works", {
  d <- tfd_exp_relaxed_one_hot_categorical(
    temperature = 0.5,
    probs = c(0.1, 0.5, 0.4)
  )

  expect_equal(tfd_sample(d)$get_shape()$as_list(), 3)
})

test_succeeds("Doublesided Maxwell works", {

  skip_if_tfp_below("0.9")

  d <- tfd_doublesided_maxwell(
    loc = 0,
    scale = c(0.5, 1, 2)
  )

  expect_equal(tfd_sample(d)$get_shape()$as_list(), 3)
})

test_succeeds("Finite discrete works", {

  skip_if_tfp_below("0.9")

  d <- tfd_finite_discrete(
    c(1, 2, 3),
    probs = c(0.5, 0.2, 0.3)
  )

  expect_equal(d %>% tfd_prob(2) %>% tensor_value(), 0.2)
})

test_succeeds("Generalized Pareto discrete works", {

  skip_if_tfp_below("0.9")

  scale <- 2
  conc <- 0.5
  genp <- tfd_generalized_pareto(loc = 0, scale = scale, concentration = conc)
  mean1 <- genp %>% tfd_sample(1000) %>% tf$reduce_mean()

  jd <- tfd_joint_distribution_named(
     list(
       rate =  tfd_gamma(1 / genp$concentration, genp$scale / genp$concentration),
       x = function(rate) tfd_exponential(rate)))

  mean2 <- jd %>% tfd_sample(1000) %>% .$x %>% tf$reduce_mean()

  expect_equal(mean1 %>% tensor_value(), mean2 %>% tensor_value(), tolerance = 1)
})

test_succeeds("Logit-normal works", {

  skip_if_tfp_below("0.9")

  t <- tfd_transformed_distribution(
    tfd_normal(0, 1),
    tfb_sigmoid()
  )

  sample_mean_t <- t %>% tfd_sample(10000) %>% tf$reduce_mean()

  d <- tfd_logit_normal(0, 1)
  sample_mean_d <- d %>% tfd_sample(10000) %>% tf$reduce_mean()

  expect_equal(sample_mean_t %>% tensor_value(), sample_mean_d %>% tensor_value(), tolerance = 0.1)
})

test_succeeds("Log-normal works", {

  skip_if_tfp_below("0.9")

  t <- tfd_transformed_distribution(
    tfd_normal(0, 1),
    tfb_exp()
  )

  sample_mean_t <- t %>% tfd_sample(10000) %>% tf$reduce_mean()

  d <- tfd_log_normal(0, 1)
  sample_mean_d <- d %>% tfd_sample(10000) %>% tf$reduce_mean()

  expect_equal(sample_mean_t %>% tensor_value(), sample_mean_d %>% tensor_value(), tolerance = 0.1)
})

test_succeeds("Pert works", {

  skip_if_tfp_below("0.9")

  d1 <- tfd_pert(1, 7, 11, 4)
  d2 <- tfd_pert(1, 7, 11, 20)

  p1 <- d1 %>% tfd_prob(7)
  p2 <- d2 %>% tfd_prob(7)

  expect_lt(p1 %>% tensor_value(), p2 %>% tensor_value())
})

test_succeeds("PlackettLuce works", {

  skip_if_tfp_below("0.9")

  scores <- c(0.1, 2, 5)
  d <-  tfd_plackett_luce(scores)

  permutations <- list(list(2, 1, 0), list(1, 0, 2))

  expect_equal(d %>% tfd_prob(permutations) %>% tensor_value() %>% length(), 2)
})

test_succeeds("ProbitBernoulli works", {

  skip_if_tfp_below("0.9")

  probits <- c(1, 0)
  d <-  tfd_probit_bernoulli(probits = probits)

  samples <- d %>% tfd_sample(10)
  expect_equal(samples$get_shape() %>% length(), 2)
})

test_succeeds("WishartTriL distribution works", {

  skip_if_tfp_below("0.9")
  s <- matrix(c(1, 2, 2, 5), ncol = 2, byrow = TRUE)
  df <- 4
  d <- tfd_wishart_tri_l(df = df, scale_tril = tf$linalg$cholesky(s))
  expect_equal(tfd_mean(d) %>% tensor_value(), df * s)
})

test_succeeds("PixelCNN distribution works", {

  skip_if_tfp_below("0.10")

  library(tfdatasets)
  library(keras)

  mnist <- dataset_mnist()
  train_data <- mnist$train

  preprocess <- function(record) {
    record$x <- tf$cast(record$x, tf$float32) %>%
      tf$expand_dims(axis = -1L)
    record$y <- tf$cast(record$y, tf$float32)
    list(tuple(record$x, record$y))
  }

  batch_size <- 32

  train_ds <- tensor_slices_dataset(train_data)
  train_ds <- train_ds %>%
    dataset_take(32) %>%
    dataset_map(preprocess) %>%
    dataset_batch(batch_size)

  dist <- tfd_pixel_cnn(
    image_shape = c(28, 28, 1),
    conditional_shape = list(),
    num_resnet = 1,
    num_hierarchies = 1,
    num_filters = 2,
    num_logistic_mix = 1,
    dropout_p = .3
  )

  image_input <- layer_input(shape = c(28, 28, 1))
  label_input <- layer_input(shape = list())
  log_prob <- dist %>% tfd_log_prob(image_input, conditional_input = label_input)

  model <- keras_model(inputs = list(image_input, label_input), outputs = log_prob)
  model$add_loss(tf$negative(tf$reduce_mean(log_prob)))
  model$compile(optimizer = optimizer_adam(.001))

  model %>% fit(train_ds, epochs = 1)

  samples <- dist %>% tfd_sample(1, conditional_input = 1)
  expect_equal(dim(samples), c(1, 28, 28, 1))

})

test_succeeds("BetaBinomial distribution works", {

  skip_if_tfp_below("0.10")

  d <- tfd_beta_binomial(
    total_count = c(5, 10, 20),
    concentration1 = c(.5, 2, 3),
    concentration0 = c(4, 3, 2))
  expect_equal(d %>% tfd_log_prob(1) %>% tensor_value() %>% length(), 3)
})

test_succeeds("JointDistributionSequentialAutoBatched distribution works", {

  skip_if_tfp_below("0.11")

  d <- tfd_joint_distribution_sequential_auto_batched(
    # e ~ Exponential(rate=[100,120])
    # g ~ Gamma(concentration=e[0], rate=e[1])
    # n ~ Normal(loc=0, scale=2.)
    # m ~ Normal(loc=n, scale=g)
    # for i = 1, ..., 12:  x[i] ~ Bernoulli(logits=m)
    list(
      # e
      tfd_exponential(rate = c(100, 120)),
      # g
      function(e) tfd_gamma(concentration = e[1], rate = e[2], name = "g"),
      # n
      tfd_normal(loc = 0, scale = 2, name = "n1"),
      # m
      function(n1, g) tfd_normal(loc = n1, scale = g, name = "n2"),
      # x
      function(n2) tfd_sample_distribution(tfd_bernoulli(logits = n2), 12, name = "s")
    ))

  x <- d %>% tfd_sample()
  expect_equal(length(x), 5)
  expect_equal((d %>% tfd_log_prob(x))$get_shape()$as_list(), list())
  expect_equal(d$resolve_graph() %>% length(), 5)

})

test_succeeds("JointDistributionNamedAutoBatched distribution works", {

  skip_if_tfp_below("0.11")
  d <- tfd_joint_distribution_named_auto_batched(
    list(
      e = tfd_exponential(rate = c(100, 120)),
      g = function(e) tfd_gamma(concentration = e[1], rate = e[2]),
      n = tfd_normal(loc = 0, scale = 2),
      m = function(n, g) tfd_normal(loc = n, scale = g),
      x = function(m) tfd_sample_distribution(tfd_bernoulli(logits = m), 12)
    ))

  x <- d %>% tfd_sample()
  expect_equal(length(x), 5)
  expect_equal((d %>% tfd_log_prob(x))$get_shape()$as_list(), list())
  expect_equal(d$resolve_graph() %>% length(), 5)

})


test_succeeds("Weibull distribution works", {

  skip_if_tfp_below("0.11")

  d <- tfd_weibull(scale = c(1, 3, 45),
                   concentration = c(2.5, 22, 7))
  expect_equal(d %>% tfd_cdf(1) %>% tensor_value() %>% length(), 3)
})

test_succeeds("TruncatedCauchy distribution works", {

  skip_if_tfp_below("0.11")

  d <- tfd_truncated_cauchy(
    loc = c(0, 1),
    scale = 1,
    low = c(-1, 0),
    high = c(1, 1)
  )
  expect_equal(d %>% tfd_cdf(1) %>% tensor_value() %>% length(), 2)
})

test_succeeds("SphericalUniform distribution works", {

  skip_if_tfp_below("0.11")

  d <- tfd_spherical_uniform(
    dimension = 4,
    batch_shape = list(3)
  )
  expect_equal(d %>% tfd_sample(5) %>% tensor_value() %>% dim(), c(5, 3, 4))
})

test_succeeds("PowerSpherical distribution works", {

  skip_if_tfp_below("0.11")

  d <- tfd_power_spherical(
    mean_direction = list(list(0, 1, 0), list(1, 0, 0)),
    concentration = c(1, 2)
  )

  x <- rbind(c(0, 0, 1), c(0, 1, 0))
  expect_equal(d %>% tfd_prob(x) %>% tensor_value() %>% length(), 2)
})

test_succeeds("LogLogistic distribution works", {

  skip_if_tfp_below("0.11")

  d1 <- tfd_log_logistic(
    loc = 1,
    scale = 1
  )

  d2 <- tfd_transformed_distribution(
    distribution = tfd_logistic(loc = 1, scale = 1),
    bijector = tfb_exp()
  )

  x <- c(1, 10, 100)
  expect_equal(d1 %>% tfd_prob(x) %>% tensor_value(),
               d2 %>% tfd_prob(x) %>% tensor_value(),
               tol = 1e-6
  )
})

test_succeeds("Bates distribution works", {

  skip_if_tfp_below("0.11")

  counts <- c(1, 2, 5)
  low <- c(0, 5, 10)
  high <- 100

  d <- tfd_bates(total_count = counts, low = low, high = high)
  expect_equal(d %>% tfd_sample() %>% tensor_value() %>% length(), 3)
})

test_succeeds("GeneralizedNormal distribution works", {

  skip_if_tfp_below("0.11")

  d <- tfd_generalized_normal(loc = c(1, 2), scale = c(11, 22), power = c(1, 1))
  x <- d %>% tfd_sample(c(2, 2))
  expect_equal(d$batch_shape$as_list(), 2)

})

test_succeeds("JohnsonSU distribution works", {

  skip_if_tfp_below("0.11")

  d <- tfd_johnson_s_u(skewness = -2, tailweight = 2, loc = 1.1, scale = 1.5)
  expect_equal(d %>% tfd_log_prob(1) %>% tensor_value() %>% length(), 1)

})

test_succeeds("ContinuousBernoulli distribution works", {

  skip_if_tfp_below("0.11")

  d <- tfd_continuous_bernoulli(logits = c(0.1, 0.9))
  expect_equal(d %>% tfd_log_prob(1) %>% tensor_value() %>% length(), 2)

})

test_succeeds("Skellam distribution works", {

  skip_if_tfp_below("0.12")

  d <- tfd_skellam(rate1 = 0.1, rate2 = 0.7)
  expect_equal(d %>% tfd_log_prob(1) %>% tensor_value() %>% length(), 1)

})

test_succeeds("ExpGamma distribution works", {

  skip_if_tfp_below("0.12")
  d <- tfd_exp_gamma(concentration = 1, rate = 1)
  expect_equal(d %>% tfd_mean() %>% tensor_value() %>% length(), 1)
})

test_succeeds("ExpInverseGamma distribution works", {

  skip_if_tfp_below("0.12")
  d <- tfd_exp_inverse_gamma(concentration = 1, scale = 1)
  expect_equal(d %>% tfd_mean() %>% tensor_value() %>% length(), 1)
})

