
context("sts")

source("utils.R")

test_succeeds("local level state space model works", {

  skip_if_tfp_below("0.7")

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

  ll <- sts_local_linear_trend_state_space_model(
    num_timesteps = 50,
    level_scale = 0.5,
    slope_scale = 0.5,
    initial_state_prior = tfd_multivariate_normal_diag(scale_diag = list(1, 1)))

  y <- ll %>% tfd_sample()
  expect_equal(y$get_shape()$as_list(), c(50, 1))
  lp <- ll %>% tfd_log_prob(y)
  expect_equal(lp$get_shape()$as_list() %>% length(), 0)
})

test_succeeds("semi local linear trend state space model works", {

  skip_if_tfp_below("0.7")

  ll <- sts_semi_local_linear_trend_state_space_model(
    num_timesteps = 50,
    level_scale = 0.5,
    slope_mean = 0.2,
    autoregressive_coef = 0.9,
    slope_scale = 0.5,
    initial_state_prior = tfd_multivariate_normal_diag(scale_diag = list(1, 1)))

  y <- ll %>% tfd_sample()
  expect_equal(y$get_shape()$as_list(), c(50, 1))
  lp <- ll %>% tfd_log_prob(y)
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
  day_of_week <- sts_seasonal_state_space_model(
    num_timesteps = 30,
    num_seasons = 7,
    drift_scale = 0.1,
    initial_state_prior = tfd_multivariate_normal_diag(scale_diag = rep(1, 7)),
    num_steps_per_season = 24)
})
