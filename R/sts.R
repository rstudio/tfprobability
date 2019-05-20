#' Formal representation of a local level model
#'
#' The local level model posits a `level` evolving via a Gaussian random walk:
#' ```
#' level[t] = level[t-1] + Normal(0., level_scale)
#' ```
#'
#' The latent state is `[level]`. We observe a noisy realization of the current
#' level: `f[t] = level[t] + Normal(0., observation_noise_scale)` at each timestep.
#'
#' @param level_scale_prior optional `tfp$distribution` instance specifying a prior
#' on the `level_scale` parameter. If `NULL`, a heuristic default prior is
#' constructed based on the provided `observed_time_series`.
#' Default value: `NULL`.
#' @param initial_level_prior optional `tfp$distribution` instance specifying a
#' prior on the initial level. If `NULL`, a heuristic default prior is
#' constructed based on the provided `observed_time_series`.
#' Default value: `NULL`.
#' @param observed_time_series optional `float` `tensor` of shape
#' `batch_shape + [T, 1]` (omitting the trailing unit dimension is also
#' supported when `T > 1`), specifying an observed time series.
#' Any priors not explicitly set will be given default values according to
#' the scale of the observed time series (or batch of time series). May
#' optionally be an instance of `sts_masked_time_series`, which includes
#' a mask `tensor` to specify timesteps with missing observations.
#' Default value: `NULL`.
#' @param name the name of this model component. Default value: 'LocalLevel'.
#'
#' @family sts
#'
#' @export
sts_local_level <- function(observed_time_series = NULL,
                            level_scale_prior = NULL,
                            initial_level_prior = NULL,
                            name = NULL) {
  args <- list(
    level_scale_prior = level_scale_prior,
    initial_level_prior = initial_level_prior,
    observed_time_series = observed_time_series,
    name = name
  )
  do.call(tfp$sts$LocalLevel, args)

}

#' State space model for a local level
#'
#' A state space model (SSM) posits a set of latent (unobserved) variables that
#' evolve over time with dynamics specified by a probabilistic transition model
#' `p(z[t+1] | z[t])`. At each timestep, we observe a value sampled from an
#' observation model conditioned on the current state, `p(x[t] | z[t])`. The
#' special case where both the transition and observation models are Gaussians
#' with mean specified as a linear function of the inputs, is known as a linear
#' Gaussian state space model and supports tractable exact probabilistic
#' calculations; see `tfd_linear_gaussian_state_space_model` for
#' details.
#' The local level model is a special case of a linear Gaussian SSM, in which the
#' latent state posits a `level` evolving via a Gaussian random walk:
#' ```
#' level[t] = level[t-1] + Normal(0., level_scale)
#' ```
#'
#' The latent state is `[level]` and `[level]` is observed (with noise) at each timestep.
#'
#' The parameters `level_scale` and `observation_noise_scale` are each (a batch
#' of) scalars. The batch shape of this `Distribution` is the broadcast batch
#' shape of these parameters and of the `initial_state_prior`.
#'
#' Mathematical Details
#'
#' The local level model implements a `tfp$distributions$LinearGaussianStateSpaceModel` with
#' `latent_size = 1` and `observation_size = 1`, following the transition model:
#'```
#' transition_matrix = [[1]]
#' transition_noise ~ N(loc = 0, scale = diag([level_scale]))
#' ```
#'
#' which implements the evolution of `level` described above, and the observation model:
#' ```
#' observation_matrix = [[1]]
#' observation_noise ~ N(loc = 0, scale = observation_noise_scale)
#' ```
#'
#' @param num_timesteps Scalar `integer` `tensor` number of timesteps to model
#' with this distribution.
#' @param level_scale Scalar (any additional dimensions are treated as batch
#' dimensions) `float` `tensor` indicating the standard deviation of the
#' level transitions.
#' @param initial_state_prior instance of `tfd_multivariate_normal`
#' representing the prior distribution on latent states.  Must have
#' event shape `[1]` (as `tfd_linear_gaussian_state_space_model` requires a
#' rank-1 event shape).
#' @param observation_noise_scale Scalar (any additional dimensions are
#' treated as batch dimensions) `float` `tensor` indicating the standard
#' deviation of the observation noise.
#' @param initial_step Optional scalar `integer` `tensor` specifying the starting
#' timestep. Default value: 0.
#' @param validate_args `logical`. Whether to validate input
#' with asserts. If `validate_args` is `FALSE`, and the inputs are
#' invalid, correct behavior is not guaranteed. Default value: `FALSE`.
#' @param allow_nan_stats `logical`. If `FALSE`, raise an
#' exception if a statistic (e.g. mean/mode/etc...) is undefined for any
#' batch member. If `TRUE`, batch members with valid parameters leading to
#' undefined statistics will return NaN for this statistic. Default value: `TRUE`.
#' @param name string name prefixed to ops created by this class.
#' Default value: "LocalLevelStateSpaceModel".
#'
#' @family sts
#'
#' @export
sts_local_level_state_space_model <- function(num_timesteps,
                                              level_scale,
                                              initial_state_prior,
                                              observation_noise_scale = 0,
                                              initial_step = 0,
                                              validate_args = FALSE,
                                              allow_nan_stats = TRUE,
                                              name = NULL) {

  args <- list(
    num_timesteps = as.integer(num_timesteps),
    level_scale = level_scale,
    initial_state_prior = initial_state_prior,
    observation_noise_scale = observation_noise_scale,
    initial_step = as.integer(initial_step),
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )
  do.call(tfp$sts$LocalLevelStateSpaceModel, args)
}


#' Formal representation of a local linear trend model
#'
#' The local linear trend model posits a `level` and `slope`, each
#' evolving via a Gaussian random walk:
#' ```
#' level[t] = level[t-1] + slope[t-1] + Normal(0., level_scale)
#' slope[t] = slope[t-1] + Normal(0., slope_scale)
#' ```
#'
#' The latent state is the two-dimensional tuple `[level, slope]`. At each
#' timestep we observe a noisy realization of the current level:
#' `f[t] = level[t] + Normal(0., observation_noise_scale)`.
#' This model is appropriate for data where the trend direction and magnitude (latent
#' `slope`) is consistent within short periods but may evolve over time.
#'
#' Note that this model can produce very high uncertainty forecasts, as
#' uncertainty over the slope compounds quickly. If you expect your data to
#' have nonzero long-term trend, i.e. that slopes tend to revert to some mean,
#' then the `SemiLocalLinearTrend` model may produce sharper forecasts.
#'
#' @param slope_scale_prior optional `tfd$Distribution` instance specifying a prior
#' on the `slope_scale` parameter. If `NULL`, a heuristic default prior is
#' constructed based on the provided `observed_time_series`. Default value: `NULL`.
#' @param initial_slope_prior optional `tfd$Distribution` instance specifying a
#' prior on the initial slope. If `NULL`, a heuristic default prior is
#' constructed based on the provided `observed_time_series`. Default value: `NULL`.
#' @param name the name of this model component. Default value: 'LocalLinearTrend'.
#'
#' @inheritParams sts_local_level
#' @family sts
#'
#' @export
sts_local_linear_trend <- function(observed_time_series = NULL,
                                   level_scale_prior = NULL,
                                   slope_scale_prior = NULL,
                                   initial_level_prior = NULL,
                                   initial_slope_prior = NULL,
                                   name = NULL) {
  args <- list(
    level_scale_prior = level_scale_prior,
    slope_scale_prior = slope_scale_prior,
    initial_level_prior = initial_level_prior,
    initial_slope_prior = initial_slope_prior,
    observed_time_series = observed_time_series,
    name = name
  )
  do.call(tfp$sts$LocalLinearTrend, args)

}

#' State space model for a local linear trend
#'
#' A state space model (SSM) posits a set of latent (unobserved) variables that
#' evolve over time with dynamics specified by a probabilistic transition model
#' `p(z[t+1] | z[t])`. At each timestep, we observe a value sampled from an
#' observation model conditioned on the current state, `p(x[t] | z[t])`. The
#' special case where both the transition and observation models are Gaussians
#' with mean specified as a linear function of the inputs, is known as a linear
#' Gaussian state space model and supports tractable exact probabilistic
#' calculations; see `tfd_linear_gaussian_state_space_model` for details.
#'
#' The local linear trend model is a special case of a linear Gaussian SSM, in
#' which the latent state posits a `level` and `slope`, each evolving via a
#' Gaussian random walk:
#'
#' ```
#' level[t] = level[t-1] + slope[t-1] + Normal(0., level_scale)
#' slope[t] = slope[t-1] + Normal(0., slope_scale)
#' ```
#'
#' The latent state is the two-dimensional tuple `[level, slope]`. The
#' `level` is observed at each timestep.
#'
#' The parameters `level_scale`, `slope_scale`, and `observation_noise_scale`
#' are each (a batch of) scalars. The batch shape of this `Distribution` is the
#' broadcast batch shape of these parameters and of the `initial_state_prior`.
#'
#' Mathematical Details
#'
#' The linear trend model implements a `tfd_linear_gaussian_state_space_model`
#' with `latent_size = 2` and `observation_size = 1`, following the transition model:
#'
#' ```
#' transition_matrix = [[1., 1.]
#'                      [0., 1.]]
#' transition_noise ~ N(loc = 0, scale = diag([level_scale, slope_scale]))
#' ```
#'
#' which implements the evolution of `[level, slope]` described above, and the observation model:
#'
#' ```
#' observation_matrix = [[1., 0.]]
#' observation_noise ~ N(loc= 0 , scale = observation_noise_scale)
#' ```
#' which picks out the first latent component, i.e., the `level`, as the
#' observation at each timestep.
#'
#' @param slope_scale Scalar (any additional dimensions are treated as batch
#' dimensions) `float` `tensor` indicating the standard deviation of the
#' slope transitions.
#' @param name string prefixed to ops created by this class.
#' Default value: "LocalLinearTrendStateSpaceModel".
#'
#' @inheritParams sts_local_level_state_space_model
#' @family sts
#'
#' @export
sts_local_linear_trend_state_space_model <- function(num_timesteps,
                                                     level_scale,
                                                     slope_scale,
                                                     initial_state_prior,
                                                     observation_noise_scale = 0,
                                                     initial_step = 0,
                                                     validate_args = FALSE,
                                                     allow_nan_stats = TRUE,
                                                     name = NULL) {
  args <- list(
    num_timesteps = as.integer(num_timesteps),
    level_scale = level_scale,
    slope_scale = slope_scale,
    initial_state_prior = initial_state_prior,
    observation_noise_scale = observation_noise_scale,
    initial_step = as.integer(initial_step),
    validate_args = validate_args,
    allow_nan_stats = allow_nan_stats,
    name = name
  )
  do.call(tfp$sts$LocalLinearTrendStateSpaceModel, args)
}

#' Formal representation of a semi-local linear trend model.
#'
#' Like the `sts_local_linear_trend` model, a semi-local linear trend posits a
#' latent `level` and `slope`, with the level component updated according to
#' the current slope plus a random walk:
#'
#' ```
#' level[t] = level[t-1] + slope[t-1] + Normal(0., level_scale)
#' ```
#'
#' The slope component in a `sts_semi_local_linear_trend` model evolves according to
#' a first-order autoregressive (AR1) process with potentially nonzero mean:
#'
#' ```
#' slope[t] = (slope_mean + autoregressive_coef * (slope[t-1] - slope_mean) + Normal(0., slope_scale))
#' ```
#'
#' Unlike the random walk used in `LocalLinearTrend`, a stationary
#' AR1 process (coefficient in `(-1, 1)`) maintains bounded variance over time,
#' so a `SemiLocalLinearTrend` model will often produce more reasonable
#' uncertainties when forecasting over long timescales.
#'
#' @param slope_mean_prior optional `tfd$Distribution` instance specifying a prior
#' on the `slope_mean` parameter. If `NULL`, a heuristic default prior is
#' constructed based on the provided `observed_time_series`. Default value: `NULL`.
#' @param autoregressive_coef_prior optional `tfd$Distribution` instance specifying
#' a prior on the `autoregressive_coef` parameter. If `NULL`, the default
#' prior is a standard `Normal(0, 1)`. Note that the prior may be
#' implicitly truncated by `constrain_ar_coef_stationary` and/or `constrain_ar_coef_positive`.
#' Default value: `NULL`.
#' @param constrain_ar_coef_stationary if `TRUE`, perform inference using a
#' parameterization that restricts `autoregressive_coef` to the interval
#' `(-1, 1)`, or `(0, 1)` if `force_positive_ar_coef` is also `TRUE`,
#' corresponding to stationary processes. This will implicitly truncate
#' the support of `autoregressive_coef_prior`. Default value: `TRUE`.
#' @param constrain_ar_coef_positive if `TRUE`, perform inference using a
#' parameterization that restricts `autoregressive_coef` to be positive,
#' or in `(0, 1)` if `constrain_ar_coef_stationary` is also `TRUE`. This
#' will implicitly truncate the support of `autoregressive_coef_prior`.
#' Default value: `FALSE`.
#' @param name the name of this model component. Default value: 'SemiLocalLinearTrend'.
#'
#' @inheritParams sts_local_linear_trend
#' @family sts
#'
#' @export
sts_semi_local_linear_trend <- function(observed_time_series = NULL,
                                        level_scale_prior = NULL,
                                        slope_mean_prior = NULL,
                                        slope_scale_prior = NULL,
                                        autoregressive_coef_prior = NULL,
                                        initial_level_prior = NULL,
                                        initial_slope_prior = NULL,
                                        constrain_ar_coef_stationary = TRUE,
                                        constrain_ar_coef_positive = FALSE,
                                        name = NULL) {
  args <- list(
    level_scale_prior = level_scale_prior,
    slope_mean_prior = slope_mean_prior,
    slope_scale_prior = slope_scale_prior,
    autoregressive_coef_prior = autoregressive_coef_prior,
    initial_level_prior = initial_level_prior,
    initial_slope_prior = initial_slope_prior,
    observed_time_series = observed_time_series,
    constrain_ar_coef_stationary = constrain_ar_coef_stationary,
    constrain_ar_coef_positive = constrain_ar_coef_positive,
    name = name
  )
  do.call(tfp$sts$SemiLocalLinearTrend, args)

}

#' State space model for a semi-local linear trend.
#'
#' A state space model (SSM) posits a set of latent (unobserved) variables that
#' evolve over time with dynamics specified by a probabilistic transition model
#' `p(z[t+1] | z[t])`. At each timestep, we observe a value sampled from an
#' observation model conditioned on the current state, `p(x[t] | z[t])`. The
#' special case where both the transition and observation models are Gaussians
#' with mean specified as a linear function of the inputs, is known as a linear
#' Gaussian state space model and supports tractable exact probabilistic
#' calculations; see `tfd_linear_gaussian_state_space_model` for details.
#'
#' The semi-local linear trend model is a special case of a linear Gaussian
#' SSM, in which the latent state posits a `level` and `slope`. The `level`
#' evolves via a Gaussian random walk centered at the current `slope`, while
#' the `slope` follows a first-order autoregressive (AR1) process with
#' mean `slope_mean`:
#'
#' ```
#' level[t] = level[t-1] + slope[t-1] + Normal(0, level_scale)
#' slope[t] = (slope_mean + autoregressive_coef * (slope[t-1] - slope_mean) +
#'            Normal(0., slope_scale))
#' ```
#'
#' The latent state is the two-dimensional tuple `[level, slope]`. The
#' `level` is observed at each timestep.
#' The parameters `level_scale`, `slope_mean`, `slope_scale`,
#' `autoregressive_coef`, and `observation_noise_scale` are each (a batch of)
#' scalars. The batch shape of this `Distribution` is the broadcast batch shape
#' of these parameters and of the `initial_state_prior`.
#'
#' Mathematical Details
#'
#' The semi-local linear trend model implements a
#' `tfp.distributions.LinearGaussianStateSpaceModel` with `latent_size = 2`
#' and `observation_size = 1`, following the transition model:
#'
#' ```
#' transition_matrix = [[1., 1.]
#'                      [0., autoregressive_coef]]
#' transition_noise ~ N(loc=slope_mean - autoregressive_coef * slope_mean,
#'                      scale=diag([level_scale, slope_scale]))
#' ```
#' which implements the evolution of `[level, slope]` described above, and
#' the observation model:
#'
#' ```
#' observation_matrix = [[1., 0.]]
#' observation_noise ~ N(loc=0, scale=observation_noise_scale)
#' ```
#'
#' which picks out the first latent component, i.e., the `level`, as the
#' observation at each timestep.
#'
#' @param slope_mean Scalar (any additional dimensions are treated as batch
#' dimensions) `float` `tensor` indicating the expected long-term mean of
#' the latent slope.
#' @param autoregressive_coef Scalar (any additional dimensions are treated as
#' batch dimensions) `float` `tensor` defining the AR1 process on the latent slope.
#' @param name string` prefixed to ops created by this class.
#' Default value: "SemiLocalLinearTrendStateSpaceModel".
#'
#' @inheritParams sts_local_linear_trend_state_space_model
#' @family sts
#'
#' @export
sts_semi_local_linear_trend_state_space_model <-
  function(num_timesteps,
           level_scale,
           slope_mean,
           slope_scale,
           autoregressive_coef,
           initial_state_prior,
           observation_noise_scale = 0,
           initial_step = 0,
           validate_args = FALSE,
           allow_nan_stats = TRUE,
           name = NULL) {
    args <- list(
      num_timesteps = as.integer(num_timesteps),
      level_scale = level_scale,
      slope_mean = slope_mean,
      slope_scale = slope_scale,
      autoregressive_coef = autoregressive_coef,
      initial_state_prior = initial_state_prior,
      observation_noise_scale = observation_noise_scale,
      initial_step = as.integer(initial_step),
      validate_args = validate_args,
      allow_nan_stats = allow_nan_stats,
      name = name
    )
    do.call(tfp$sts$SemiLocalLinearTrendStateSpaceModel, args)
  }


#' Formal representation of a seasonal effect model.
#'
#' A seasonal effect model posits a fixed set of recurring, discrete 'seasons',
#' each of which is active for a fixed number of timesteps and, while active,
#' contributes a different effect to the time series. These are generally not
#' meteorological seasons, but represent regular recurring patterns such as
#' hour-of-day or day-of-week effects. Each season lasts for a fixed number of
#' timesteps. The effect of each season drifts from one occurrence to the next
#' following a Gaussian random walk:
#'
#' ```
#' effects[season, occurrence[i]] = (
#'   effects[season, occurrence[i-1]] + Normal(loc=0., scale=drift_scale))
#' ```
#'
#' The `drift_scale` parameter governs the standard deviation of the random walk;
#' for example, in a day-of-week model it governs the change in effect from this
#' Monday to next Monday.
#'
#' @param num_seasons Scalar `integer` number of seasons.
#' @param num_steps_per_season `integer` number of steps in each
#' season. This may be either a scalar (shape `[]`), in which case all
#' seasons have the same length, or an array of shape `[num_seasons]`,
#' in which seasons have different length, but remain constant around
#' different cycles, or an array of shape `[num_cycles, num_seasons]`,
#' in which num_steps_per_season for each season also varies in different
#' cycle (e.g., a 4 years cycle with leap day). Default value: 1.
#' @param drift_scale_prior optional `tfd$Distribution` instance specifying a prior
#' on the `drift_scale` parameter. If `NULL`, a heuristic default prior is
#' constructed based on the provided `observed_time_series`. Default value: `NULL`.
#' @param initial_effect_prior optional `tfd$Distribution` instance specifying a
#' normal prior on the initial effect of each season. This may be either
#' a scalar `tfd_normal` prior, in which case it applies independently to
#' every season, or it may be multivariate normal (e.g.,
#' `tfd_multivariate_normal_diag`) with event shape `[num_seasons]`, in
#' which case it specifies a joint prior across all seasons. If `NULL`, a
#' heuristic default prior is constructed based on the provided
#' `observed_time_series`. Default value: `NULL`.
#' @param constrain_mean_effect_to_zero if `TRUE`, use a model parameterization
#' that constrains the mean effect across all seasons to be zero. This
#' constraint is generally helpful in identifying the contributions of
#' different model components and can lead to more interpretable
#' posterior decompositions. It may be undesirable if you plan to directly
#' examine the latent space of the underlying state space model. Default value: `TRUE`.
#' @param name the name of this model component. Default value: 'Seasonal'.
#'
#' @inheritParams sts_local_linear_trend
#' @family sts
#'
#' @export
sts_seasonal <- function(observed_time_series = NULL,
                         num_seasons,
                         num_steps_per_season = 1,
                         drift_scale_prior = NULL,
                         initial_effect_prior = NULL,
                         constrain_mean_effect_to_zero = TRUE,
                         name = NULL) {
  args <- list(
    num_seasons = as.integer(num_seasons),
    num_steps_per_season = as.integer(num_steps_per_season),
    drift_scale_prior = drift_scale_prior,
    initial_effect_prior = initial_effect_prior,
    constrain_mean_effect_to_zero = constrain_mean_effect_to_zero,
    name = name
  )
  do.call(tfp$sts$Seasonal, args)

}
