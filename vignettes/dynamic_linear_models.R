## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  eval = FALSE
)

## -----------------------------------------------------------------------------
#  library(tensorflow)
#  library(tfprobability)
#
#  library(tidyverse)
#  library(zeallot)
#
#  # As the data does not seem to be available at the address given in Petris et al. any more,
#  # we put it on the TensorFlow for R blog for download
#  # download from:
#  # https://github.com/rstudio/ai-blog/tree/master/_posts/2019-06-25-dynamic_linear_models_tfprobability/data/capm.txt
#  df <- read_table(
#    "capm.txt",
#    col_types = list(X1 = col_date(format = "%Y.%m"))) %>%
#    rename(month = X1)
#  df %>% glimpse()

## -----------------------------------------------------------------------------
#  # excess returns of the asset under study
#  ibm <- df$IBM - df$RKFREE
#  # market excess returns
#  x <- df$MARKET - df$RKFREE
#
#  fit <- lm(ibm ~ x)
#  summary(fit)

## -----------------------------------------------------------------------------
#  # zoom in on ibm
#  ts <- ibm %>% matrix()
#  # forecast 12 months
#  n_forecast_steps <- 12
#  ts_train <- ts[1:(length(ts) - n_forecast_steps), 1, drop = FALSE]
#
#  # make sure we work with float32 here
#  ts_train <- tf$cast(ts_train, tf$float32)
#  ts <- tf$cast(ts, tf$float32)

## -----------------------------------------------------------------------------
#  # define the model on the complete series
#  linreg <- ts %>%
#    sts_dynamic_linear_regression(
#      design_matrix = cbind(rep(1, length(x)), x) %>% tf$cast(tf$float32)
#    )

## -----------------------------------------------------------------------------
#  fit_with_vi <-
#    function(ts,
#             ts_train,
#             model,
#             n_iterations,
#             n_param_samples,
#             n_forecast_steps,
#             n_forecast_samples) {
#
#      optimizer <- tf$compat$v1$train$AdamOptimizer(0.1)
#
#      loss_and_dists <-
#        ts_train %>% sts_build_factored_variational_loss(model = model)
#      variational_loss <- loss_and_dists[[1]]
#      train_op <- optimizer$minimize(variational_loss)
#
#      with (tf$Session() %as% sess,  {
#
#        # step 1: train the model using variational inference
#        sess$run(tf$compat$v1$global_variables_initializer())
#        for (step in 1:n_iterations) {
#          sess$run(train_op)
#          loss <- sess$run(variational_loss)
#          if (step %% 1 == 0)
#            cat("Loss: ", as.numeric(loss), "\n")
#        }
#        # step 2: obtain forecasts
#        variational_distributions <- loss_and_dists[[2]]
#        posterior_samples <-
#          Map(
#            function(d)
#              d %>% tfd_sample(n_param_samples),
#            variational_distributions %>% reticulate::py_to_r() %>% unname()
#          )
#        forecast_dists <-
#          ts_train %>% sts_forecast(model, posterior_samples, n_forecast_steps)
#        fc_means <- forecast_dists %>% tfd_mean()
#        fc_sds <- forecast_dists %>% tfd_stddev()
#
#        # step 3: obtain smoothed and filtered estimates from the Kálmán filter
#        ssm <- model$make_state_space_model(length(ts_train), param_vals = posterior_samples)
#        c(smoothed_means, smoothed_covs) %<-% ssm$posterior_marginals(ts_train)
#        c(., filtered_means, filtered_covs, ., ., ., .) %<-% ssm$forward_filter(ts_train)
#
#        c(posterior_samples, fc_means, fc_sds, smoothed_means, smoothed_covs, filtered_means, filtered_covs) %<-%
#          sess$run(list(posterior_samples, fc_means, fc_sds, smoothed_means, smoothed_covs, filtered_means, filtered_covs))
#
#      })
#
#      list(
#        variational_distributions,
#        posterior_samples,
#        fc_means[, 1],
#        fc_sds[, 1],
#        smoothed_means,
#        smoothed_covs,
#        filtered_means,
#        filtered_covs
#      )
#    }

## -----------------------------------------------------------------------------
#  # number of VI steps
#  n_iterations <- 300
#  # sample size for posterior samples
#  n_param_samples <- 50
#  # sample size to draw from the forecast distribution
#  n_forecast_samples <- 50
#
#  # call fit_vi defined above
#  c(
#    param_distributions,
#    param_samples,
#    fc_means,
#    fc_sds,
#    smoothed_means,
#    smoothed_covs,
#    filtered_means,
#    filtered_covs
#  ) %<-% fit_vi(
#    ts,
#    ts_train,
#    model,
#    n_iterations,
#    n_param_samples,
#    n_forecast_steps,
#    n_forecast_samples
#  )
#

## -----------------------------------------------------------------------------
#  smoothed_means_intercept <- smoothed_means[, , 1] %>% colMeans()
#  smoothed_means_slope <- smoothed_means[, , 2] %>% colMeans()
#
#  smoothed_sds_intercept <- smoothed_covs[, , 1, 1] %>% colMeans() %>% sqrt()
#  smoothed_sds_slope <- smoothed_covs[, , 2, 2] %>% colMeans() %>% sqrt()
#
#  filtered_means_intercept <- filtered_means[, , 1] %>% colMeans()
#  filtered_means_slope <- filtered_means[, , 2] %>% colMeans()
#
#  filtered_sds_intercept <- filtered_covs[, , 1, 1] %>% colMeans() %>% sqrt()
#  filtered_sds_slope <- filtered_covs[, , 2, 2] %>% colMeans() %>% sqrt()
#
#  forecast_df <- df %>%
#    select(month, IBM) %>%
#    add_column(pred_mean = c(rep(NA, length(ts_train)), fc_means)) %>%
#    add_column(pred_sd = c(rep(NA, length(ts_train)), fc_sds)) %>%
#    add_column(smoothed_means_intercept = c(smoothed_means_intercept, rep(NA, n_forecast_steps))) %>%
#    add_column(smoothed_means_slope = c(smoothed_means_slope, rep(NA, n_forecast_steps))) %>%
#    add_column(smoothed_sds_intercept = c(smoothed_sds_intercept, rep(NA, n_forecast_steps))) %>%
#    add_column(smoothed_sds_slope = c(smoothed_sds_slope, rep(NA, n_forecast_steps))) %>%
#    add_column(filtered_means_intercept = c(filtered_means_intercept, rep(NA, n_forecast_steps))) %>%
#    add_column(filtered_means_slope = c(filtered_means_slope, rep(NA, n_forecast_steps))) %>%
#    add_column(filtered_sds_intercept = c(filtered_sds_intercept, rep(NA, n_forecast_steps))) %>%
#    add_column(filtered_sds_slope = c(filtered_sds_slope, rep(NA, n_forecast_steps)))
#

## -----------------------------------------------------------------------------
#  ggplot(forecast_df, aes(x = month, y = IBM)) +
#    geom_line(color = "grey") +
#    geom_line(aes(y = pred_mean), color = "cyan") +
#    geom_ribbon(
#      aes(ymin = pred_mean - 2 * pred_sd, ymax = pred_mean + 2 * pred_sd),
#      alpha = 0.2,
#      fill = "cyan"
#    ) +
#    theme(axis.title = element_blank())

## ---- eval=TRUE, echo=FALSE, layout="l-body-outset", fig.cap = "12-point-ahead forecasts for IBM; posterior means +/- 2 standard deviations."----
knitr::include_graphics("images/capm_forecast.png")

## -----------------------------------------------------------------------------
#  ggplot(forecast_df, aes(x = month, y = smoothed_means_intercept)) +
#    geom_line(color = "orange") +
#    geom_line(aes(y = smoothed_means_slope),
#              color = "green") +
#    geom_ribbon(
#      aes(
#        ymin = smoothed_means_intercept - 2 * smoothed_sds_intercept,
#        ymax = smoothed_means_intercept + 2 * smoothed_sds_intercept
#      ),
#      alpha = 0.3,
#      fill = "orange"
#    ) +
#    geom_ribbon(
#      aes(
#        ymin = smoothed_means_slope - 2 * smoothed_sds_slope,
#        ymax = smoothed_means_slope + 2 * smoothed_sds_slope
#      ),
#      alpha = 0.1,
#      fill = "green"
#    ) +
#    coord_cartesian(xlim = c(forecast_df$month[1], forecast_df$month[length(ts) - n_forecast_steps]))  +
#    theme(axis.title = element_blank())
#

## ---- eval=TRUE, echo=FALSE, layout="l-body-outset", fig.cap = "Smoothing estimates from the Kálmán filter. Green: coefficient for dependence on excess market returns (slope), orange: vector of ones (intercept)."----
knitr::include_graphics("images/capm_smoothed.png")

## -----------------------------------------------------------------------------
#  ggplot(forecast_df, aes(x = month, y = filtered_means_intercept)) +
#    geom_line(color = "orange") +
#    geom_line(aes(y = filtered_means_slope),
#              color = "green") +
#    geom_ribbon(
#      aes(
#        ymin = filtered_means_intercept - 2 * filtered_sds_intercept,
#        ymax = filtered_means_intercept + 2 * filtered_sds_intercept
#      ),
#      alpha = 0.3,
#      fill = "orange"
#    ) +
#    geom_ribbon(
#      aes(
#        ymin = filtered_means_slope - 2 * filtered_sds_slope,
#        ymax = filtered_means_slope + 2 * filtered_sds_slope
#      ),
#      alpha = 0.1,
#      fill = "green"
#    ) +
#    coord_cartesian(ylim = c(-2, 2),
#                    xlim = c(forecast_df$month[1], forecast_df$month[length(ts) - n_forecast_steps])) +
#    theme(axis.title = element_blank())

## ---- eval=TRUE, echo=FALSE, layout="l-body-outset", fig.cap = "Filtering estimates from the Kálmán filter. Green: coefficient for dependence on excess market returns (slope), orange: vector of ones (intercept)."----
knitr::include_graphics("images/capm_filtered.png")
