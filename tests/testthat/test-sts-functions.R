# context("sts-functions")
#
# source("utils.R")
#
# test_succeeds("sts_build_factored_variational_loss works", {
#   observed_time_series <-
#     rep(c(3.5, 4.1, 4.5, 3.9, 2.4, 2.1, 1.2), 5) + rep(c(1.1, 1.5, 2.4, 3.1, 4.0), each = 7)
#
#   day_of_week <-
#     observed_time_series %>% sts_seasonal(num_seasons = 7)
#   local_linear_trend <-
#     observed_time_series %>% sts_local_linear_trend()
#   model <-
#     observed_time_series %>% sts_sum(components = list(day_of_week, local_linear_trend))
#
#   optimizer <- tf$compat$v1$train$AdamOptimizer(0.1)
#
#   build_variational_loss <- function() {
#     res <-
#       observed_time_series %>% sts_build_factored_variational_loss(model = model)
#     variational_loss <- res[[1]]
#     variational_loss
#   }
#
#   if (tf$executing_eagerly()) {
#     for (step in 1:5) {
#       optimizer$minimize(build_variational_loss)
#       # Draw multiple samples to reduce Monte Carlo error in the optimized variational bounds.
#       avg_loss <-
#         Map(function(x)
#           build_variational_loss() %>% as.numeric(), 1:3) %>% unlist() %>% mean()
#
#       variational_distributions <-
#         (observed_time_series %>% sts_build_factored_variational_loss(model = model))[[2]]
#       posterior_samples <-
#         Map(
#           function(d)
#             d %>% tfd_sample(50),
#           variational_distributions
#         )
#     }
#
#   } else {
#     loss_and_dists <-
#       observed_time_series %>% sts_build_factored_variational_loss(model = model)
#     variational_loss <- loss_and_dists[[1]]
#     train_op <- optimizer$minimize(variational_loss)
#     with (tf$Session() %as% sess,  {
#       sess$run(tf$compat$v1$global_variables_initializer())
#       for (step in 1:5) {
#         res <- sess$run(train_op)
#       }
#       avg_loss <-
#         Map(function(x)
#           sess$run(variational_loss), 1:2) %>% unlist() %>% mean()
#
#       variational_distributions <- loss_and_dists[[2]]
#       posterior_samples <-
#         Map(
#           function(d)
#             d %>% tfd_sample(50),
#           variational_distributions
#         ) %>%
#         sess$run()
#
#     })
#
#   }
#
#   expect_length(avg_loss, 1)
#   expect_length(posterior_samples, 4)
# })
#
# test_succeeds("sts_fit_with_hmc works", {
#   observed_time_series <-
#     rep(c(3.5, 4.1, 4.5, 3.9, 2.4, 2.1, 1.2), 5) + rep(c(1.1, 1.5, 2.4, 3.1, 4.0), each = 7)
#
#   day_of_week <-
#     observed_time_series %>% sts_seasonal(num_seasons = 7)
#   local_linear_trend <-
#     observed_time_series %>% sts_local_linear_trend()
#   model <-
#     observed_time_series %>% sts_sum(components = list(day_of_week, local_linear_trend))
#
#   states_and_results <-
#     observed_time_series %>% sts_fit_with_hmc(
#       model,
#       num_results = 10,
#       num_warmup_steps = 5,
#       num_variational_steps = 15
#     )
#   posterior_samples <- states_and_results[[1]]
#   expect_length(posterior_samples, 4)
#
# })
#
# test_succeeds("sts_one_step_predictive works", {
#   skip_if_tfp_below("0.7")
#
#   observed_time_series <-
#     rep(c(3.5, 4.1, 4.5, 3.9, 2.4, 2.1, 1.2), 5) + rep(c(1.1, 1.5, 2.4, 3.1, 4.0), each = 7)
#
#   day_of_week <-
#     observed_time_series %>% sts_seasonal(num_seasons = 7)
#   local_linear_trend <-
#     observed_time_series %>% sts_local_linear_trend()
#   model <-
#     observed_time_series %>% sts_sum(components = list(day_of_week, local_linear_trend))
#
#   states_and_results <-
#     observed_time_series %>% sts_fit_with_hmc(
#       model,
#       num_results = 10,
#       num_warmup_steps = 5,
#       num_variational_steps = 15
#     )
#   samples <- states_and_results[[1]]
#
#   preds <-
#     observed_time_series %>% sts_one_step_predictive(model, parameter_samples = samples)
#   pred_means <- preds %>% tfd_mean()
#   pred_sds <- preds %>% tfd_stddev()
#   expect_equal(preds$event_shape %>% length(), 2)
#
# })
#
# test_succeeds("sts_forecast works", {
#   skip_if_tfp_below("0.7")
#
#   observed_time_series <-
#     rep(c(3.5, 4.1, 4.5, 3.9, 2.4, 2.1, 1.2), 5) + rep(c(1.1, 1.5, 2.4, 3.1, 4.0), each = 7)
#
#   day_of_week <-
#     observed_time_series %>% sts_seasonal(num_seasons = 7)
#   local_linear_trend <-
#     observed_time_series %>% sts_local_linear_trend()
#   model <-
#     observed_time_series %>% sts_sum(components = list(day_of_week, local_linear_trend))
#
#   states_and_results <-
#     observed_time_series %>% sts_fit_with_hmc(
#       model,
#       num_results = 10,
#       num_warmup_steps = 5,
#       num_variational_steps = 15
#     )
#   samples <- states_and_results[[1]]
#
#   preds <-
#     observed_time_series %>% sts_forecast(model,
#                                           parameter_samples = samples,
#                                           num_steps_forecast = 50)
#   predictions <- preds %>% tfd_sample(10)
#   expect_equal(predictions$get_shape()$as_list() %>% length(), 3)
#
# })
#
# test_succeeds("sts_decompose_by_component works", {
#
#   skip_if_tfp_below("0.7")
#
#   observed_time_series <-
#     array(rnorm(2 * 1 * 12), dim = c(2, 1, 12))
#
#   day_of_week <-
#     observed_time_series %>% sts_seasonal(num_seasons = 7, name = "seasonal")
#   local_linear_trend <-
#     observed_time_series %>% sts_local_linear_trend(name = "local_linear")
#   model <-
#     observed_time_series %>% sts_sum(components = list(day_of_week, local_linear_trend))
#   states_and_results <- observed_time_series %>% sts_fit_with_hmc(
#     model,
#     num_results = 10,
#     num_warmup_steps = 5,
#     num_variational_steps = 15
#   )
#
#   samples <- states_and_results[[1]]
#
#   component_dists <-
#     observed_time_series %>% sts_decompose_by_component(model = model,
#                                                         parameter_samples = samples)
#
#   day_of_week_effect_mean <- component_dists[[1]] %>% tfd_mean()
#   expect_equal(day_of_week_effect_mean$get_shape()$as_list() %>% length(), 3)
#
# })
