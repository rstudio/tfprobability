context("sts-functions")

test_succeeds("sts_build_factored_variational_loss works", {
  skip_if_eager()

  observed_time_series <-
    rep(c(3.5, 4.1, 4.5, 3.9, 2.4, 2.1, 1.2), 5) + rep(c(1.1, 1.5, 2.4, 3.1, 4.0), each = 7)

  day_of_week <-
    observed_time_series %>% sts_seasonal(num_seasons = 7)
  local_linear_trend <-
    observed_time_series %>% sts_local_linear_trend()
  model <-
    observed_time_series %>% sts_sum(components = list(day_of_week, local_linear_trend))

  optimizer <- tf$compat$v1$train$AdamOptimizer(0.1)

  build_variational_loss <- function() {
    res <-
      observed_time_series %>% sts_build_factored_variational_loss(model = model)
    variational_loss <- res[[1]]
    variational_loss
  }

  loss_and_dists <-
    observed_time_series %>% sts_build_factored_variational_loss(model = model)
  variational_loss <- loss_and_dists[[1]]
  train_op <- optimizer$minimize(variational_loss)
  with (tf$Session() %as% sess,  {
    sess$run(tf$compat$v1$global_variables_initializer())
    for (step in 1:5) {
      res <- sess$run(train_op)
    }
    avg_loss <-
      Map(function(x)
        sess$run(variational_loss), 1:2) %>% unlist() %>% mean()

    variational_distributions <- loss_and_dists[[2]]
    posterior_samples <-
      Map(function(d)
        d %>% tfd_sample(50),
        variational_distributions) %>%
      sess$run()

  })

  expect_length(avg_loss, 1)
  expect_length(posterior_samples, 4)
})

test_succeeds("sts_fit_with_hmc works", {

  observed_time_series <-
    rep(c(3.5, 4.1, 4.5, 3.9, 2.4, 2.1, 1.2), 5) + rep(c(1.1, 1.5, 2.4, 3.1, 4.0), each = 7)

  if (tensorflow::tf_version() >= "2.0")
    observed_time_series <- tensorflow::tf$convert_to_tensor(observed_time_series, dtype = tensorflow::tf$float64)


  day_of_week <-
    observed_time_series %>% sts_seasonal(num_seasons = 7)
  local_linear_trend <-
    observed_time_series %>% sts_local_linear_trend()
  model <-
    observed_time_series %>% sts_sum(components = list(day_of_week, local_linear_trend))

  states_and_results <-
    observed_time_series %>% sts_fit_with_hmc(
      model,
      num_results = 10,
      num_warmup_steps = 5,
      num_variational_steps = 15
    )
  posterior_samples <- states_and_results[[1]]
  expect_length(posterior_samples, 4)

})

test_succeeds("sts_one_step_predictive works", {

  observed_time_series <-
    rep(c(3.5, 4.1, 4.5, 3.9, 2.4, 2.1, 1.2), 5) + rep(c(1.1, 1.5, 2.4, 3.1, 4.0), each = 7)

  if (tensorflow::tf_version() >= "2.0")
    observed_time_series <- tensorflow::tf$convert_to_tensor(observed_time_series, dtype = tensorflow::tf$float64)

  day_of_week <-
    observed_time_series %>% sts_seasonal(num_seasons = 7)
  local_linear_trend <-
    observed_time_series %>% sts_local_linear_trend()
  model <-
    observed_time_series %>% sts_sum(components = list(day_of_week, local_linear_trend))

  states_and_results <-
    observed_time_series %>% sts_fit_with_hmc(
      model,
      num_results = 10,
      num_warmup_steps = 5,
      num_variational_steps = 15
    )
  samples <- states_and_results[[1]]

  preds <- observed_time_series %>%
    sts_one_step_predictive(model,
                            parameter_samples = samples,
                            timesteps_are_event_shape = TRUE)
  pred_means <- preds %>% tfd_mean()
  pred_sds <- preds %>% tfd_stddev()
  skip("Batch dim behavior changed")
  expect_equal(preds$event_shape %>% length(), 2)

})

test_succeeds("sts_forecast works", {

  observed_time_series <-
    rep(c(3.5, 4.1, 4.5, 3.9, 2.4, 2.1, 1.2), 5) + rep(c(1.1, 1.5, 2.4, 3.1, 4.0), each = 7)

  if (tensorflow::tf_version() >= "2.0")
    observed_time_series <- tensorflow::tf$convert_to_tensor(observed_time_series, dtype = tensorflow::tf$float64)

  day_of_week <-
    observed_time_series %>% sts_seasonal(num_seasons = 7)
  local_linear_trend <-
    observed_time_series %>% sts_local_linear_trend()
  model <-
    observed_time_series %>% sts_sum(components = list(day_of_week, local_linear_trend))

  states_and_results <-
    observed_time_series %>% sts_fit_with_hmc(
      model,
      num_results = 10,
      num_warmup_steps = 5,
      num_variational_steps = 15
    )
  samples <- states_and_results[[1]]

  preds <-
    observed_time_series %>% sts_forecast(model,
                                          parameter_samples = samples,
                                          num_steps_forecast = 50)
  predictions <- preds %>% tfd_sample(10)
  expect_equal(predictions$get_shape()$as_list() %>% length(), 3)

})

test_succeeds("sts_decompose_by_component works", {

  observed_time_series <-
    array(rnorm(2 * 1 * 12), dim = c(2, 1, 12))

  day_of_week <-
    observed_time_series %>% sts_seasonal(num_seasons = 7, name = "seasonal")
  local_linear_trend <-
    observed_time_series %>% sts_local_linear_trend(name = "local_linear")
  model <-
    observed_time_series %>% sts_sum(components = list(day_of_week, local_linear_trend))
  states_and_results <- observed_time_series %>% sts_fit_with_hmc(
    model,
    num_results = 10,
    num_warmup_steps = 5,
    num_variational_steps = 15
  )

  samples <- states_and_results[[1]]

  component_dists <-
    observed_time_series %>% sts_decompose_by_component(model = model,
                                                        parameter_samples = samples)

  day_of_week_effect_mean <- component_dists[[1]] %>% tfd_mean()
  expect_equal(day_of_week_effect_mean$get_shape()$as_list() %>% length(),
               3)

})

test_succeeds("sts_build_factored_surrogate_posterior works", {
  skip_if_tfp_below("0.8")

  observed_time_series <-
    rep(c(3.5, 4.1, 4.5, 3.9, 2.4, 2.1, 1.2), 5) + rep(c(1.1, 1.5, 2.4, 3.1, 4.0), each = 7)

  day_of_week <-
    observed_time_series %>% sts_seasonal(num_seasons = 7)
  local_linear_trend <-
    observed_time_series %>% sts_local_linear_trend()
  model <-
    observed_time_series %>% sts_sum(components = list(day_of_week, local_linear_trend))

  optimizer <- tf$compat$v1$train$AdamOptimizer(0.1)

  # build the surrogate posterior variables outside of a training loop,
  # then fit them by optimizing a loss of your choice
  # or use vi_fit_surrogate_posterior to automate the loss construction and fitting
  surrogate_posterior <-
    model %>% sts_build_factored_surrogate_posterior()

  loss_curve <- vi_fit_surrogate_posterior(
    target_log_prob_fn = model$joint_log_prob(observed_time_series),
    surrogate_posterior = surrogate_posterior,
    optimizer = optimizer,
    num_steps = 20
  )

  if (tf$executing_eagerly()) {
    posterior_samples <- surrogate_posterior %>% tfd_sample(50)
  } else {
    with (tf$control_dependencies(list(loss_curve)), {
      posterior_samples <- surrogate_posterior %>% tfd_sample(50)
    })
  }

  expect_length(posterior_samples, 4)

})

test_succeeds("sts_sample_uniform_initial_state works", {

  model <- sts_sparse_linear_regression(design_matrix = matrix(31 * 3, nrow = 31),
                                        weights_prior_scale = 0.1)
  p <- model$parameters[[1]]
  init <- sts_sample_uniform_initial_state(parameter = p, init_sample_shape = list(2, 2))
  expect_equal(init$get_shape()$as_list() %>% length(), 2)

})

test_succeeds("sts_decompose_forecast_by_component works", {

  observed_time_series <-
    array(rnorm(2 * 1 * 12), dim = c(2, 1, 12))

  day_of_week <-
    observed_time_series %>% sts_seasonal(num_seasons = 7, name = "seasonal")
  local_linear_trend <-
    observed_time_series %>% sts_local_linear_trend(name = "local_linear")
  model <-
    observed_time_series %>% sts_sum(components = list(day_of_week, local_linear_trend))
  states_and_results <- observed_time_series %>% sts_fit_with_hmc(
    model,
    num_results = 10,
    num_warmup_steps = 5,
    num_variational_steps = 15
  )

  samples <- states_and_results[[1]]

  forecast_dist <-
    observed_time_series %>% sts_forecast(model,
                                          parameter_samples = samples,
                                          num_steps_forecast = 50)

  component_forecasts <-
    sts_decompose_forecast_by_component(model, forecast_dist, samples)

  day_of_week_effect_mean <- component_forecasts[[1]] %>% tfd_mean()
  expect_equal(day_of_week_effect_mean$get_shape()$as_list() %>% length(),
               3)
})
