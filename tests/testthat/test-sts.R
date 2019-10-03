
context("sts")

source("utils.R")

test_succeeds("local level state space model works", {

  ll <- sts_local_level_state_space_model(
    num_timesteps = 50,
    level_scale = 0.5,
    initial_state_prior = tfd_multivariate_normal_diag(scale_diag = list(1)))

  y <- ll %>% tfd_sample()
  expect_equal(y$get_shape()$as_list(), c(50, 1))
  lp <- ll %>% tfd_log_prob(y)
  expect_equal(lp$get_shape()$as_list() %>% length(), 0)

  # Passing additional parameter dimensions constructs a batch of models. The
  # overall batch shape is the broadcast batch shape of the parameters:
  ll <- sts_local_level_state_space_model(
    num_timesteps = 50,
    level_scale = rep(1, 10),
    initial_state_prior = tfd_multivariate_normal_diag(scale_diag = tf$ones(list(10, 10, 1))))

  y <- ll %>% tfd_sample(5)
  expect_equal(y$get_shape()$as_list(), c(5, 10, 10,50, 1))
  lp <- ll %>% tfd_log_prob(y)
  expect_equal(lp$get_shape()$as_list(), c(5, 10, 10))
})

test_succeeds("local linear trend state space model works", {

  llt <- sts_local_linear_trend_state_space_model(
    num_timesteps = 50,
    level_scale = 0.5,
    slope_scale = 0.5,
    initial_state_prior = tfd_multivariate_normal_diag(scale_diag = list(1, 1)))

  y <- llt %>% tfd_sample()
  expect_equal(y$get_shape()$as_list(), c(50, 1))
  lp <- llt %>% tfd_log_prob(y)
  expect_equal(lp$get_shape()$as_list() %>% length(), 0)
})

test_succeeds("semi local linear trend state space model works", {

  sll <- sts_semi_local_linear_trend_state_space_model(
    num_timesteps = 50,
    level_scale = 0.5,
    slope_mean = 0.2,
    autoregressive_coef = 0.9,
    slope_scale = 0.5,
    initial_state_prior = tfd_multivariate_normal_diag(scale_diag = list(1, 1)))

  y <- sll %>% tfd_sample()
  expect_equal(y$get_shape()$as_list(), c(50, 1))
  lp <- sll %>% tfd_log_prob(y)
  expect_equal(lp$get_shape()$as_list() %>% length(), 0)
})

test_succeeds("seasonal works", {

  month_of_year <- sts_seasonal(
    num_seasons = 12,
    num_steps_per_season = list(31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31),
    drift_scale_prior = tfd_log_normal(loc = -1, scale = 0.1),
    initial_effect_prior = tfd_normal(loc = 0, scale = 5),
    name='month_of_year')
})

test_succeeds("seasonal state space model works", {

  sss <- sts_seasonal_state_space_model(
    num_timesteps = 30,
    num_seasons = 7,
    drift_scale = 0.1,
    initial_state_prior = tfd_multivariate_normal_diag(scale_diag = rep(1, 7)),
    num_steps_per_season = 24)

  y <- sss %>% tfd_sample()
  expect_equal(y$get_shape()$as_list(), c(30, 1))
  lp <- sss %>% tfd_log_prob(y)
  expect_equal(lp$get_shape()$as_list() %>% length(), 0)
})

test_succeeds("sum works", {

  ts <- rep(1.1:7.1, 4)
  llt <- sts_local_linear_trend(observed_time_series = ts, name='local_trend')
  dof <- ts %>% sts_seasonal(num_seasons = 7, name='day_of_week_effect')
  sum <- ts %>% sts_sum(components = list(llt, dof))

  expect_equal(sum$latent_size, llt$latent_size + dof$latent_size)
})

test_succeeds("additive state space model works", {

  local_ssm <- sts_local_linear_trend_state_space_model(
    num_timesteps = 30,
    level_scale = 0.5,
    slope_scale = 0.1,
    initial_state_prior = tfd_multivariate_normal_diag(
      loc = list(0, 0), scale_diag = list(1, 1)))

  day_of_week_ssm <- sts_seasonal_state_space_model(
    num_timesteps = 30,
    num_seasons = 7,
    drift_scale = 0.1,
    initial_state_prior = tfd_multivariate_normal_diag(
      loc = rep(0,7), scale_diag = rep(1, 7)))

  additive_ssm <- sts_additive_state_space_model(
    component_ssms = list(local_ssm, day_of_week_ssm),
    observation_noise_scale = 0.1)

  y <- additive_ssm %>% tfd_sample()
  expect_equal(y$get_shape()$as_list(), c(30, 1))
})


test_succeeds("sts_linear_regression works", {

  holiday_indicators <- matrix(0, nrow = 31, ncol =3)
  holiday_indicators[23, 0] <- 1
  holiday_indicators[24, 1] <- 1
  holiday_indicators[30, 2] <- 1

  holidays <- sts_linear_regression(design_matrix = holiday_indicators)

  ts <- rep(1.1:7.1, 4)
  seasonal <- ts %>% sts_seasonal(num_seasons = 7)

  model <- ts %>% sts_sum(components = list(holidays, seasonal))
})

test_succeeds("sts_dynamic_linear_regression works", {

  model <- sts_dynamic_linear_regression(
    design_matrix = matrix(31 * 3, nrow = 31)
  )
})

test_succeeds("dynamic linear regression state space model works", {

  m <- matrix(777, nrow = 42, ncol = 2) %>% tf$cast(tf$float32)

  model <-sts_dynamic_linear_regression_state_space_model(
    num_timesteps = 42,
    design_matrix = m,
    drift_scale = 3.14,
    initial_state_prior = tfd_multivariate_normal_diag(scale_diag = c(1, 2)),
    observation_noise_scale = 1)

  y <- model %>% tfd_sample()
  lp <- model %>% tfd_log_prob(y)

  expect_equal(y$get_shape()$as_list() %>% length(), 2)
  expect_equal(lp$get_shape()$as_list() %>% length(), 0)
})

test_succeeds("dynamic linear regression state space model works with batches of models", {

  m <- matrix(777, nrow = 42, ncol = 2) %>% tf$cast(tf$float32)

  model <-sts_dynamic_linear_regression_state_space_model(
    num_timesteps = 42,
    design_matrix = m,
    drift_scale = c(3.14, 1),
    initial_state_prior = tfd_multivariate_normal_diag(scale_diag = c(1, 2)),
    observation_noise_scale = c(1, 2)
  )

  y <- model %>% tfd_sample(3)
  lp <- model %>% tfd_log_prob(y)

  expect_equal(y$get_shape()$as_list() %>% length(), 4)
  expect_equal(lp$get_shape()$as_list() %>% length(), 2)
})

test_succeeds("sts_autoregressive works", {

  model <- sts_autoregressive(order = 3)
})

test_succeeds("autoregressive state space model works", {

  model <- sts_autoregressive_state_space_model(
    num_timesteps = 50,
    coefficients = c(0.8,-0.1),
    level_scale = 0.5,
    initial_state_prior = tfd_multivariate_normal_diag(scale_diag = c(1, 1))
  )

  y <- model %>% tfd_sample()
  lp <- model %>% tfd_log_prob(y)

  expect_equal(y$get_shape()$as_list() %>% length(), 2)
  expect_equal(lp$get_shape()$as_list() %>% length(), 0)
})

test_succeeds("autoregressive state space model works with batches of models", {

  model <- sts_autoregressive_state_space_model(
    num_timesteps = 50,
    coefficients = c(0.8,-0.1),
    level_scale = rep(1, 10),
    initial_state_prior = tfd_multivariate_normal_diag(scale_diag = array(1, dim = c(10, 10, 2)))
  )

  y <- model %>% tfd_sample(5)
  lp <- model %>% tfd_log_prob(y)

  expect_equal(y$get_shape()$as_list() %>% length(), 5)
  expect_equal(lp$get_shape()$as_list() %>% length(), 3)
})

test_succeeds("sts_sparse_linear_regression works", {

  model <- sts_sparse_linear_regression(design_matrix = matrix(31 * 3, nrow = 31),
                                        weights_prior_scale = 0.1)
})

test_succeeds("constrained seasonal state space model works", {

  sss <- sts_constrained_seasonal_state_space_model(
    num_timesteps = 30,
    num_seasons = 7,
    drift_scale = 0.1,
    initial_state_prior = tfd_multivariate_normal_diag(scale_diag = rep(1, 7 - 1)),
    num_steps_per_season = 24)

  y <- sss %>% tfd_sample()
  expect_equal(y$get_shape()$as_list(), c(30, 1))
  lp <- sss %>% tfd_log_prob(y)
  expect_equal(lp$get_shape()$as_list() %>% length(), 0)
})

