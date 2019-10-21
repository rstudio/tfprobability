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
#' @return an instance of `StructuralTimeSeries`.
#'
#' @family sts
#' @seealso For usage examples see [sts_fit_with_hmc()], [sts_forecast()], [sts_decompose_by_component()].
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
#' @return an instance of `LinearGaussianStateSpaceModel`.
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
#' @inherit sts_local_level return params
#' @family sts
#' @seealso For usage examples see [sts_fit_with_hmc()], [sts_forecast()], [sts_decompose_by_component()].
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
#' @inherit sts_local_level_state_space_model return params
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
#' @inherit sts_local_linear_trend return params
#' @family sts
#' @seealso For usage examples see [sts_fit_with_hmc()], [sts_forecast()], [sts_decompose_by_component()].
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
#' @inherit sts_local_linear_trend_state_space_model return params
#' @family sts
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
#' @inherit sts_local_linear_trend return params
#' @family sts
#' @seealso For usage examples see [sts_fit_with_hmc()], [sts_forecast()], [sts_decompose_by_component()].
#'
#' @export
sts_seasonal <- function(observed_time_series = NULL,
                         num_seasons,
                         num_steps_per_season = 1,
                         drift_scale_prior = NULL,
                         initial_effect_prior = NULL,
                         constrain_mean_effect_to_zero = TRUE,
                         name = NULL) {

  storage.mode(num_steps_per_season) <- "integer"
  args <- list(
    num_seasons = as.integer(num_seasons),
    num_steps_per_season = num_steps_per_season,
    drift_scale_prior = drift_scale_prior,
    initial_effect_prior = initial_effect_prior,
    observed_time_series = observed_time_series,
    constrain_mean_effect_to_zero = constrain_mean_effect_to_zero,
    name = name
  )

  do.call(tfp$sts$Seasonal, args)

}

#' State space model for a seasonal effect.
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
#'
#' A seasonal effect model is a special case of a linear Gaussian SSM. The
#' latent states represent an unknown effect from each of several 'seasons';
#' these are generally not meteorological seasons, but represent regular
#' recurring patterns such as hour-of-day or day-of-week effects. The effect of
#' each season drifts from one occurrence to the next, following a Gaussian random walk:
#'
#' ```
#' effects[season, occurrence[i]] = (effects[season, occurrence[i-1]] + Normal(loc=0., scale=drift_scale))
#' ```
#'
#' The latent state has dimension `num_seasons`, containing one effect for each
#' seasonal component. The parameters `drift_scale` and
#' `observation_noise_scale` are each (a batch of) scalars. The batch shape of
#' this `Distribution` is the broadcast batch shape of these parameters and of
#' the `initial_state_prior`.
#' Note: there is no requirement that the effects sum to zero.
#'
#' Mathematical Details
#'
#' The seasonal effect model implements a `tfd_linear_gaussian_state_space_model` with
#' `latent_size = num_seasons` and `observation_size = 1`. The latent state
#' is organized so that the *current* seasonal effect is always in the first
#' (zeroth) dimension. The transition model rotates the latent state to shift
#' to a new effect at the end of each season:
#'
#' ```
#' transition_matrix[t] = (permutation_matrix([1, 2, ..., num_seasons-1, 0])
#'                        if season_is_changing(t)
#'                        else eye(num_seasons)
#' transition_noise[t] ~ Normal(loc=0., scale_diag=(
#'                       [drift_scale, 0, ..., 0]
#'                       if season_is_changing(t)
#'                       else [0, 0, ..., 0]))
#' ```
#' where `season_is_changing(t)` is `True` if ``t `mod` sum(num_steps_per_season)`` is in
#' the set of final days for each season, given by `cumsum(num_steps_per_season) - 1`.
#' The observation model always picks out the effect for the current season, i.e.,
#' the first element of the latent state:
#' ```
#' observation_matrix = [[1., 0., ..., 0.]]
#' observation_noise ~ Normal(loc=0, scale=observation_noise_scale)
#' ```
#' @param num_seasons Scalar `integer` number of seasons.
#' @param drift_scale Scalar (any additional dimensions are treated as batch
#' dimensions) `float` `tensor` indicating the standard deviation of the
#' change in effect between consecutive occurrences of a given season.
#' This is assumed to be the same for all seasons.
#' @param initial_state_prior instance of `tfd_multivariate_normal`
#' representing the prior distribution on latent states; must
#' have event shape `[num_seasons]`.
#' @param  num_steps_per_season `integer` number of steps in each
#' season. This may be either a scalar (shape `[]`), in which case all
#' seasons have the same length, or an array of shape `[num_seasons]`,
#' in which seasons have different length, but remain constant around
#' different cycles, or an array of shape `[num_cycles, num_seasons]`,
#' in which num_steps_per_season for each season also varies in different
#' cycle (e.g., a 4 years cycle with leap day). Default value: 1.
#' @param name string prefixed to ops created by this class.
#' Default value: "SeasonalStateSpaceModel".
#'
#' @inherit sts_local_linear_trend_state_space_model return params
#' @family sts
#'
#' @export
sts_seasonal_state_space_model <-
  function(num_timesteps,
           num_seasons,
           drift_scale,
           initial_state_prior,
           observation_noise_scale = 0,
           num_steps_per_season = 1,
           initial_step = 0,
           validate_args = FALSE,
           allow_nan_stats = TRUE,
           name = NULL) {
    args <- list(
      num_timesteps = as.integer(num_timesteps),
      num_seasons = as.integer(num_seasons),
      drift_scale = drift_scale,
      initial_state_prior = initial_state_prior,
      observation_noise_scale = observation_noise_scale,
      num_steps_per_season = as.integer(num_steps_per_season),
      initial_step = as.integer(initial_step),
      validate_args = validate_args,
      allow_nan_stats = allow_nan_stats,
      name = name
    )
    do.call(tfp$sts$SeasonalStateSpaceModel, args)
  }

#' Sum of structural time series components.
#'
#' This class enables compositional specification of a structural time series
#' model from basic components. Given a list of component models, it represents
#' an additive model, i.e., a model of time series that may be decomposed into a
#' sum of terms corresponding to the component models.
#'
#' Formally, the additive model represents a random process
#' `g[t] = f1[t] + f2[t] + ... + fN[t] + eps[t]`, where the `f`'s are the
#' random processes represented by the components, and
#' `eps[t] ~ Normal(loc=0, scale=observation_noise_scale)` is an observation
#' noise term. See the `AdditiveStateSpaceModel` documentation for mathematical details.
#'
#' This model inherits the parameters (with priors) of its components, and
#' adds an `observation_noise_scale` parameter governing the level of noise in
#' the observed time series.
#'
#' @param components `list` of one or more StructuralTimeSeries instances.
#' These must have unique names.
#' @param constant_offset optional scalar `float` `tensor`, or batch of scalars,
#' specifying a constant value added to the sum of outputs from the
#' component models. This allows the components to model the shifted series
#' `observed_time_series - constant_offset`. If `NULL`, this is set to the
#' mean of the provided `observed_time_series`. Default value: `NULL`.
#' @param observation_noise_scale_prior optional `tfd$Distribution` instance
#' specifying a prior on `observation_noise_scale`. If `NULL`, a heuristic
#' default prior is constructed based on the provided
#' `observed_time_series`. Default value: `NULL`.
#' @param name string name of this model component; used as `name_scope`
#' for ops created by this class. Default value: 'Sum'.
#'
#' @inherit sts_local_level return params
#' @family sts
#' @seealso For usage examples see [sts_fit_with_hmc()], [sts_forecast()], [sts_decompose_by_component()].
#'
#' @export
sts_sum <- function(observed_time_series = NULL,
                    components,
                    constant_offset = NULL,
                    observation_noise_scale_prior = NULL,
                    name = NULL) {

  args <- list(
    components = components,
    observation_noise_scale_prior = observation_noise_scale_prior,
    constant_offset = constant_offset,
    observed_time_series = observed_time_series,
    name = name
  )

  do.call(tfp$sts$Sum, args)

}

#' A state space model representing a sum of component state space models.
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
#' The `sts_additive_state_space_model` represents a sum of component state space
#' models. Each of the `N` components describes a random process
#' generating a distribution on observed time series `x1[t], x2[t], ..., xN[t]`.
#' The additive model represents the sum of these
#' processes, `y[t] = x1[t] + x2[t] + ... + xN[t] + eps[t]`, where
#' `eps[t] ~ N(0, observation_noise_scale)` is an observation noise term.
#'
#' Mathematical Details
#'
#' The additive model concatenates the latent states of its component models.
#' The generative process runs each component's dynamics in its own subspace of
#' latent space, and then observes the sum of the observation models from the
#' components.
#'
#' Formally, the transition model is linear Gaussian:
#'
#' ```
#' p(z[t+1] | z[t]) ~ Normal(loc = transition_matrix.matmul(z[t]), cov = transition_cov)
#' ```
#'
#' where each `z[t]` is a latent state vector concatenating the component
#' state vectors, `z[t] = [z1[t], z2[t], ..., zN[t]]`, so it has size
#' `latent_size = sum([c.latent_size for c in components])`.
#'
#' The transition matrix is the block-diagonal composition of transition
#' matrices from the component processes:
#'
#' ```
#' transition_matrix =
#'  [[ c0.transition_matrix,  0.,                   ..., 0.                   ],
#'   [ 0.,                    c1.transition_matrix, ..., 0.                   ],
#'   [ ...                    ...                   ...                       ],
#'   [ 0.,                    0.,                   ..., cN.transition_matrix ]]
#' ```
#'
#' and the noise covariance is similarly the block-diagonal composition of
#' component noise covariances:
#'
#' ```
#' transition_cov =
#'  [[ c0.transition_cov, 0.,                ..., 0.                ],
#'   [ 0.,                c1.transition_cov, ..., 0.                ],
#'   [ ...                ...                     ...               ],
#'   [ 0.,                0.,                ..., cN.transition_cov ]]
#' ```
#'
#' The observation model is also linear Gaussian,
#'
#' ```
#' p(y[t] | z[t]) ~ Normal(loc = observation_matrix.matmul(z[t]), stddev = observation_noise_scale)
#' ```
#'
#' This implementation assumes scalar observations, so `observation_matrix` has shape `[1, latent_size]`.
#' The additive observation matrix simply concatenates the observation matrices from each component:
#'
#' ```
#' observation_matrix = concat([c0.obs_matrix, c1.obs_matrix, ..., cN.obs_matrix], axis=-1)
#' ```
#'
#' The effect is that each component observation matrix acts on the dimensions
#' of latent state corresponding to that component, and the overall expected
#' observation is the sum of the expected observations from each component.
#'
#' If `observation_noise_scale` is not explicitly specified, it is also computed
#' by summing the noise variances of the component processes:
#'
#' ```
#' observation_noise_scale = sqrt(sum([c.observation_noise_scale**2 for c in components]))
#' ```
#' @param component_ssms `list` containing one or more
#' `tfd_linear_gaussian_state_space_model` instances. The components
#' will in general implement different time-series models, with possibly
#' different `latent_size`, but they must have the same `dtype`, event
#' shape (`num_timesteps` and `observation_size`), and their batch shapes
#' must broadcast to a compatible batch shape.#'
#' @param constant_offset scalar `float` `tensor`, or batch of scalars,
#' specifying a constant value added to the sum of outputs from the
#' component models. This allows the components to model the shifted series
#' `observed_time_series - constant_offset`. Default value: `0`.#'
#' @param observation_noise_scale Optional scalar `float` `tensor` indicating the
#' standard deviation of the observation noise. May contain additional
#' batch dimensions, which must broadcast with the batch shape of elements
#' in `component_ssms`. If `observation_noise_scale` is specified for the
#' `sts_additive_state_space_model`, the observation noise scales of component
#' models are ignored. If `NULL`, the observation noise scale is derived
#' by summing the noise variances of the component models, i.e.,
#' `observation_noise_scale = sqrt(sum([ssm.observation_noise_scale**2 for ssm in component_ssms]))`.
#' @param name string prefixed to ops created by this class.
#' Default value: "AdditiveStateSpaceModel".
#' @inherit sts_local_linear_trend_state_space_model return params
#' @family sts
#'
#' @export
sts_additive_state_space_model <-
  function(component_ssms,
           constant_offset = 0,
           observation_noise_scale = NULL,
           initial_state_prior = NULL,
           initial_step = 0,
           validate_args = FALSE,
           allow_nan_stats = TRUE,
           name = NULL) {
    args <- list(
      component_ssms = component_ssms,
      constant_offset = constant_offset,
      observation_noise_scale = observation_noise_scale,
      initial_state_prior = initial_state_prior,
      initial_step = as.integer(initial_step),
      validate_args = validate_args,
      allow_nan_stats = allow_nan_stats,
      name = name
    )

    do.call(tfp$sts$AdditiveStateSpaceModel, args)
  }

#' Formal representation of a linear regression from provided covariates.
#'
#' This model defines a time series given by a linear combination of
#' covariate time series provided in a design matrix:
#' ```
#' observed_time_series <- tf$matmul(design_matrix, weights)
#' ```
#'
#' The design matrix has shape `list(num_timesteps, num_features)`.
#' The weights are treated as an unknown random variable of size `list(num_features)`
#' (both components also support batch shape), and are integrated over using the same
#' approximate inference tools as other model parameters, i.e., generally HMC or
#' variational inference.
#'
#' This component does not itself include observation noise; it defines a
#' deterministic distribution with mass at the point
#' `tf$matmul(design_matrix, weights)`. In practice, it should be combined with
#' observation noise from another component such as `sts_sum`, as demonstrated below.
#'
#' @param design_matrix float `tensor` of shape `tf$concat(list(batch_shape, list(num_timesteps, num_features)))`.
#' This may also optionally be an instance of `tf$linalg$LinearOperator`.
#' @param weights_prior `Distribution` representing a prior over the regression
#' weights. Must have event shape `list(num_features)` and batch shape
#' broadcastable to the design matrix's `batch_shape`. Alternately,
#' `event_shape` may be scalar (`list()`), in which case the prior is
#' internally broadcast as
#' `tfd_transformed_distribution(weights_prior, tfb_identity(), event_shape = list(num_features), batch_shape = design_matrix$batch_shape)`.
#' If `NULL`, defaults to `tfd_student_t(df = 5, loc = 0, scale = 10)`,
#' a weakly-informative prior loosely inspired by the
#' [Stan prior choice recommendations](https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations).
#' Default value: `NULL`.
#' @param name the name of this model component. Default value: 'LinearRegression'.
#'
#' @inherit sts_local_level return params
#' @family sts
#' @seealso For usage examples see [sts_fit_with_hmc()], [sts_forecast()], [sts_decompose_by_component()].
#'
#' @export
sts_linear_regression <- function(design_matrix,
                                  weights_prior = NULL,
                                  name = NULL) {

  args <- list(
    design_matrix = design_matrix,
    weights_prior = weights_prior,
    name = name
  )

  do.call(tfp$sts$LinearRegression, args)
}

#' Formal representation of a dynamic linear regression model.
#'
#' The dynamic linear regression model is a special case of a linear Gaussian SSM
#' and a generalization of typical (static) linear regression. The model
#' represents regression `weights` with a latent state which evolves via a
#' Gaussian random walk:
#'
#' ``` weights[t] ~ Normal(weights[t-1], drift_scale)```
#'
#' The latent state has dimension `num_features`, while the parameters
#' `drift_scale` and `observation_noise_scale` are each (a batch of) scalars. The
#' batch shape of this distribution is the broadcast batch shape of these
#' parameters, the `initial_state_prior`, and the `design_matrix`.
#' `num_features` is determined from the last dimension of `design_matrix` (equivalent to the
#' number of columns in the design matrix in linear regression).
#'
#' @param drift_scale_prior instance of `Distribution` specifying a prior on
#' the `drift_scale` parameter. If `NULL`, a heuristic default prior is
#' constructed based on the provided `observed_time_series`. Default value: `NULL`.
#' @param initial_weights_prior instance of `tfd_multivariate_normal` representing
#' the prior distribution on the latent states (the regression weights).
#' Must have event shape `list(num_features)`. If `NULL`, a weakly-informative
#' Normal(0, 10) prior is used. Default value: `NULL`.
#' @param name the name of this component. Default value: 'DynamicLinearRegression'.
#'
#' @inherit sts_local_level return params
#' @inheritParams sts_linear_regression
#' @family sts
#' @seealso For usage examples see [sts_fit_with_hmc()], [sts_forecast()], [sts_decompose_by_component()].
#'
#' @export
sts_dynamic_linear_regression <- function(observed_time_series = NULL,
                                          design_matrix,
                                          drift_scale_prior = NULL,
                                          initial_weights_prior = NULL,
                                          name = NULL) {
  args <- list(
    design_matrix = design_matrix,
    drift_scale_prior = drift_scale_prior,
    initial_weights_prior = initial_weights_prior,
    observed_time_series = observed_time_series,
    name = name
  )

  do.call(tfp$sts$DynamicLinearRegression, args)
}


#' State space model for a dynamic linear regression from provided covariates.
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
#' The dynamic linear regression model is a special case of a linear Gaussian SSM
#' and a generalization of typical (static) linear regression. The model
#' represents regression `weights` with a latent state which evolves via a
#' Gaussian random walk:
#'  ```weights[t] ~ Normal(weights[t-1], drift_scale)```
#'
#' The latent state (the weights) has dimension `num_features`, while the
#' parameters `drift_scale` and `observation_noise_scale` are each (a batch of)
#' scalars. The batch shape of this `Distribution` is the broadcast batch shape
#' of these parameters, the `initial_state_prior`, and the
#' `design_matrix`. `num_features` is determined from the last dimension of
#' `design_matrix` (equivalent to the number of columns in the design matrix in
#' linear regression).
#'
#' Mathematical Details
#'
#' The dynamic linear regression model implements a
#' `tfd_linear_gaussian_state_space_model` with `latent_size = num_features` and
#' `observation_size = 1` following the transition model:
#'
#' ```
#' transition_matrix = eye(num_features)
#' transition_noise ~ Normal(0, diag([drift_scale]))
#' ```
#'
#' which implements the evolution of `weights` described above. The observation
#' model is:
#' ```
#' observation_matrix[t] = design_matrix[t]
#' observation_noise ~ Normal(0, observation_noise_scale)
#' ```
#' @param num_timesteps Scalar `integer` `tensor`, number of timesteps to model
#' with this distribution.
#' @param design_matrix float `tensor` of shape `tf$concat(list(batch_shape, list(num_timesteps, num_features)))`.
#' This may also optionally be an instance of `tf$linalg$LinearOperator`.
#' @param drift_scale Scalar (any additional dimensions are treated as batch
#' dimensions) `float` `tensor` indicating the standard deviation of the
#' latent state transitions.
#' @param initial_state_prior instance of `tfd_multivariate_normal` representing
#' the prior distribution on latent states.  Must have
#' event shape `list(num_features)`.
#' @param observation_noise_scale  Scalar (any additional dimensions are
#' treated as batch dimensions) `float` `tensor` indicating the standard
#' deviation of the observation noise. Default value: `0`.
#' @param initial_step scalar `integer` `tensor` specifying the starting timestep.
#' Default value: `0`.
#' @param name name prefixed to ops created by this class. Default value: 'DynamicLinearRegressionStateSpaceModel'.
#'
#' @inherit sts_local_linear_trend_state_space_model return params
#' @family sts
#'
#' @export
sts_dynamic_linear_regression_state_space_model <-
  function(num_timesteps,
           design_matrix,
           drift_scale,
           initial_state_prior,
           observation_noise_scale = 0,
           initial_step = 0,
           validate_args = FALSE,
           allow_nan_stats = TRUE,
           name = NULL) {
    args <- list(
      num_timesteps = as.integer(num_timesteps),
      design_matrix = design_matrix,
      drift_scale = drift_scale,
      initial_state_prior = initial_state_prior,
      observation_noise_scale = observation_noise_scale,
      initial_step = as.integer(initial_step),
      validate_args = validate_args,
      allow_nan_stats = allow_nan_stats,
      name = name
    )

    do.call(tfp$sts$DynamicLinearRegressionStateSpaceModel, args)
  }

#' Formal representation of an autoregressive model.
#'
#' An autoregressive (AR) model posits a latent `level` whose value at each step
#' is a noisy linear combination of previous steps:
#' ```
#' level[t+1] = (sum(coefficients * levels[t:t-order:-1]) + Normal(0., level_scale))
#' ```
#'
#' The latent state is `levels[t:t-order:-1]`. We observe a noisy realization of
#' the current level: `f[t] = level[t] + Normal(0., observation_noise_scale)` at
#' each timestep.
#'
#' If `coefficients=[1.]`, the AR process is a simple random walk, equivalent to
#' a `LocalLevel` model. However, a random walk's variance increases with time,
#' while many AR processes (in particular, any first-order process with
#' `abs(coefficient) < 1`) are *stationary*, i.e., they maintain a constant
#' variance over time. This makes AR processes useful models of uncertainty.
#'
#' @param order scalar positive `integer` specifying the number of past
#' timesteps to regress on.
#' @param coefficients_prior optional `Distribution` instance specifying a
#' prior on the `coefficients` parameter. If `NULL`, a default standard
#' normal (`tfd_multivariate_normal_diag(scale_diag = tf$ones(list(order)))`) prior
#' is used. Default value: `NULL`.
#' @param level_scale_prior optional `Distribution` instance specifying a prior
#' on the `level_scale` parameter. If `NULL`, a heuristic default prior is
#' constructed based on the provided `observed_time_series`. Default value: `NULL`.
#' @param initial_state_prior optional `Distribution` instance specifying a
#' prior on the initial state, corresponding to the values of the process
#' at a set of size `order` of imagined timesteps before the initial step.
#' If `NULL`, a heuristic default prior is constructed based on the
#' provided `observed_time_series`. Default value: `NULL`.
#' @param coefficient_constraining_bijector optional `Bijector` instance
#' representing a constraining mapping for the autoregressive coefficients.
#' For example, `tfb_tanh()` constrains the coefficients to lie in
#' `(-1, 1)`, while `tfb_softplus()` constrains them to be positive, and
#' `tfb_identity()` implies no constraint. If `NULL`, the default behavior
#' constrains the coefficients to lie in `(-1, 1)` using a `tanh` bijector.
#' Default value: `NULL`.
#' @param name the name of this model component. Default value: 'Autoregressive'.
#'
#' @inherit sts_local_level return params
#' @inheritParams sts_linear_regression
#' @family sts
#' @seealso For usage examples see [sts_fit_with_hmc()], [sts_forecast()], [sts_decompose_by_component()].
#'
#' @export
sts_autoregressive <- function(observed_time_series = NULL,
                               order,
                               coefficients_prior = NULL,
                               level_scale_prior = NULL,
                               initial_state_prior = NULL,
                               coefficient_constraining_bijector = NULL,
                               name = NULL) {
  args <- list(
    order = as.integer(order),
    coefficients_prior = coefficients_prior,
    level_scale_prior = level_scale_prior,
    initial_state_prior = initial_state_prior,
    coefficient_constraining_bijector = coefficient_constraining_bijector,
    observed_time_series = observed_time_series,
    name = name
  )

  do.call(tfp$sts$Autoregressive, args)

}

#' State space model for an autoregressive process.
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
#'
#' In an autoregressive process, the expected level at each timestep is a linear
#' function of previous levels, with added Gaussian noise:
#' ```
#' level[t+1] = (sum(coefficients * levels[t:t-order:-1]) + Normal(0., level_scale))
#' ```
#'
#' The process is characterized by a vector `coefficients` whose size determines
#' the order of the process (how many previous values it looks at), and by
#' `level_scale`, the standard deviation of the noise added at each step.
#' This is formulated as a state space model by letting the latent state encode
#' the most recent values; see 'Mathematical Details' below.
#'
#' The parameters `level_scale` and `observation_noise_scale` are each (a batch
#' of) scalars, and `coefficients` is a (batch) vector of size `list(order)`. The
#' batch shape of this `Distribution` is the broadcast batch
#' shape of these parameters and of the `initial_state_prior`.
#'
#' Mathematical Details
#'
#' The autoregressive model implements a
#' `tfd_linear_gaussian_state_space_model` with `latent_size = order`
#' and `observation_size = 1`. The latent state vector encodes the recent history
#' of the process, with the current value in the topmost dimension. At each
#' timestep, the transition sums the previous values to produce the new expected
#' value, shifts all other values down by a dimension, and adds noise to the
#' current value. This is formally encoded by the transition model:
#'
#' ```
#' transition_matrix = [ coefs[0], coefs[1], ..., coefs[order]
#'                       1.,       0 ,       ..., 0.
#'                       0.,       1.,       ..., 0.
#'                       ...
#'                       0.,       0.,  ...,  1., 0.         ]
#' ```
#' ```
#' transition_noise ~ N(loc=0., scale=diag([level_scale, 0., 0., ..., 0.]))
#' ```
#'
#' The observation model simply extracts the current (topmost) value, and
#' optionally adds independent noise at each step:
#'
#' ```
#' observation_matrix = [[1., 0., ..., 0.]]
#' observation_noise ~ N(loc=0, scale=observation_noise_scale)
#' ```
#'
#' Models with `observation_noise_scale = 0` are AR processes in the formal
#' sense. Setting `observation_noise_scale` to a nonzero value corresponds to a
#' latent AR process observed under an iid noise model.
#'
#' @param num_timesteps Scalar `integer` `tensor` number of timesteps to model
#' with this distribution.
#' @param coefficients `float` `tensor` of shape `tf$concat(batch_shape, list(order))`
#' defining  the autoregressive coefficients. The coefficients are defined
#' backwards in time:
#' `coefficients[0] * level[t] + coefficients[1] * level[t-1] + ... + coefficients[order-1] * level[t-order+1]`.
#' @param level_scale Scalar (any additional dimensions are treated as batch
#' dimensions) `float` `tensor` indicating the standard deviation of the
#' transition noise at each step.
#' @param initial_state_prior instance of `tfd_multivariate_normal` representing
#' the prior distribution on latent states.  Must have event shape `list(order)`.
#' @param observation_noise_scale Scalar (any additional dimensions are
#' treated as batch dimensions) `float` `tensor` indicating the standard
#' deviation of the observation noise. Default value: 0.
#' @param initial_step Optional scalar `int` `tensor` specifying the starting
#' timestep. Default value: 0.
#' @param name name prefixed to ops created by this class. Default value: "AutoregressiveStateSpaceModel".
#'
#' @inherit sts_local_linear_trend_state_space_model return params
#' @family sts
#'
#' @export
sts_autoregressive_state_space_model <-
  function(num_timesteps,
           coefficients,
           level_scale,
           initial_state_prior,
           observation_noise_scale = 0,
           initial_step = 0,
           validate_args = FALSE,
           name = NULL) {
    args <- list(
      num_timesteps = as.integer(num_timesteps),
      coefficients = coefficients,
      level_scale = level_scale,
      initial_state_prior = initial_state_prior,
      observation_noise_scale = observation_noise_scale,
      initial_step = as.integer(initial_step),
      validate_args = validate_args,
      name = name
    )

    do.call(tfp$sts$AutoregressiveStateSpaceModel, args)
  }

#' Formal representation of a sparse linear regression.
#'
#' This model defines a time series given by a sparse linear combination of
#' covariate time series provided in a design matrix:
#'
#' ```
#' observed_time_series <- tf$matmul(design_matrix, weights)
#' ```
#'
#' This is identical to `sts_linear_regression`, except that
#' `sts_sparse_linear_regression` uses a parameterization of a Horseshoe
#' prior to encode the assumption that many of the `weights` are zero,
#' i.e., many of the covariate time series are irrelevant. See the mathematical
#' details section below for further discussion. The prior parameterization used
#' by `sts_sparse_linear_regression` is more suitable for inference than that
#' obtained by simply passing the equivalent `tfd_horseshoe` prior to
#' `sts_linear_regression`; when sparsity is desired, `sts_sparse_linear_regression` will
#' likely yield better results.
#'
#' This component does not itself include observation noise; it defines a
#' deterministic distribution with mass at the point
#' `tf$matmul(design_matrix, weights)`. In practice, it should be combined with
#' observation noise from another component such as `sts_sum`.
#'
#' Mathematical Details
#'
#' The basic horseshoe prior Carvalho et al. (2009) is defined as a Cauchy-normal scale mixture:
#' ```
#' scales[i] ~ HalfCauchy(loc=0, scale=1)
#' weights[i] ~ Normal(loc=0., scale=scales[i] * global_scale)`
#' ```
#'
#' The Cauchy scale parameters puts substantial mass near zero, encouraging
#' weights to be sparse, but their heavy tails allow weights far from zero to be
#' estimated without excessive shrinkage. The horseshoe can be thought of as a
#' continuous relaxation of a traditional 'spike-and-slab' discrete sparsity
#' prior, in which the latent Cauchy scale mixes between 'spike'
#' (`scales[i] ~= 0`) and 'slab' (`scales[i] >> 0`) regimes.
#'
#' Following the recommendations in Piironen et al. (2017), `SparseLinearRegression` implements
#' a horseshoe with the following adaptations:
#' - The Cauchy prior on `scales[i]` is represented as an InverseGamma-Normal
#' compound.
#' - The `global_scale` parameter is integrated out following a `Cauchy(0.,
#' scale=weights_prior_scale)` hyperprior, which is also represented as an
#' InverseGamma-Normal compound.
#' - All compound distributions are implemented using a non-centered
#' parameterization.
#' The compound, non-centered representation defines the same marginal prior as
#' the original horseshoe (up to integrating out the global scale),
#' but allows samplers to mix more efficiently through the heavy tails; for
#' variational inference, the compound representation implicity expands the
#' representational power of the variational model.
#'
#' Note that we do not yet implement the regularized ('Finnish') horseshoe,
#' proposed in Piironen et al. (2017) for models with weak likelihoods, because the likelihood
#' in STS models is typically Gaussian, where it's not clear that additional
#' regularization is appropriate. If you need this functionality, please
#' email tfprobability@tensorflow.org.
#'
#' The full prior parameterization implemented in `SparseLinearRegression` is
#' as follows:
#'
#' ```
#' Sample global_scale from Cauchy(0, scale=weights_prior_scale).
#' global_scale_variance ~ InverseGamma(alpha=0.5, beta=0.5)
#' global_scale_noncentered ~ HalfNormal(loc=0, scale=1)
#' global_scale = (global_scale_noncentered *
#' sqrt(global_scale_variance) *
#' weights_prior_scale)
#' Sample local_scales from Cauchy(0, 1).
#' local_scale_variances[i] ~ InverseGamma(alpha=0.5, beta=0.5)
#' local_scales_noncentered[i] ~ HalfNormal(loc=0, scale=1)
#' local_scales[i] = local_scales_noncentered[i] * sqrt(local_scale_variances[i])
#' weights[i] ~ Normal(loc=0., scale=local_scales[i] * global_scale)
#' ```
#' @param weights_prior_scale float `Tensor` defining the scale of the Horseshoe
#' prior on regression weights. Small values encourage the weights to be
#' sparse. The shape must broadcast with `weights_batch_shape`.
#' Default value: `0.1`.
#' @param weights_batch_shape if `NULL`, defaults to
#' `design_matrix.batch_shape_tensor()`. Must broadcast with the batch
#' shape of `design_matrix`. Default value: `NULL`.
#'
#' @section References:
#' - [Carvalho, C., Polson, N. and Scott, J. Handling Sparsity via the Horseshoe. AISTATS (2009).](http://proceedings.mlr.press/v5/carvalho09a/carvalho09a.pdf)
#' - [Juho Piironen, Aki Vehtari. Sparsity information and regularization in the horseshoe and other shrinkage priors (2017).](https://arxiv.org/abs/1707.01694)
#'
#' @inherit sts_linear_regression return params
#' @family sts
#' @seealso For usage examples see [sts_fit_with_hmc()], [sts_forecast()], [sts_decompose_by_component()].
#'
#' @export
sts_sparse_linear_regression <- function(design_matrix,
                                         weights_prior_scale = 0.1,
                                         weights_batch_shape = NULL,
                                         name = NULL) {
  args <- list(
    design_matrix = design_matrix,
    weights_prior_scale = weights_prior_scale,
    weights_batch_shape = weights_batch_shape,
    name = name
  )

  do.call(tfp$sts$SparseLinearRegression, args)
}

#' Seasonal state space model with effects constrained to sum to zero.
#'
#' @seealso [sts_seasonal_state_space_model()].
#'
#' Mathematical details
#'
#' The constrained model implements a reparameterization of the
#' naive `SeasonalStateSpaceModel`. Instead of directly representing the
#' seasonal effects in the latent space, the latent space of the constrained
#' model represents the difference between each effect and the mean effect.
#' The following discussion assumes familiarity with the mathematical details
#' of `SeasonalStateSpaceModel`.
#'
#' *Reparameterization and constraints*: let the seasonal effects at a given
#' timestep be `E = [e_1, ..., e_N]`. The difference between each effect `e_i`
#' and the mean effect is `z_i = e_i - sum_i(e_i)/N`. By itself, this
#' transformation is not invertible because recovering the absolute effects
#' requires that we know the mean as well. To fix this, we'll define
#' `z_N = sum_i(e_i)/N` as the mean effect. It's easy to see that this is
#' invertible: given the mean effect and the differences of the first `N - 1`
#' effects from the mean, it's easy to solve for all `N` effects. Formally,
#' we've defined the invertible linear reparameterization `Z = R E`, where
#'
#' ```
#' R = [1 - 1/N, -1/N,    ..., -1/N
#'      -1/N,    1 - 1/N, ..., -1/N,
#'      ...
#'      1/N,     1/N,     ...,  1/N]
#' ```
#'
#' represents the change of basis from 'effect coordinates' E to
#' 'residual coordinates' Z. The `Z`s form the latent space of the
#' `ConstrainedSeasonalStateSpaceModel`.
#' To constrain the mean effect `z_N` to zero, we fix the prior to zero,
#' `p(z_N) ~ N(0., 0)`, and after the transition at each timestep we project
#' `z_N` back to zero. Note that this projection is linear: to set the Nth
#' dimension to zero, we simply multiply by the identity matrix with a missing
#' element in the bottom right, i.e., `Z_constrained = P Z`,
#' where `P = eye(N) - scatter((N-1, N-1), 1)`.
#'
#' *Model*: concretely, suppose a naive seasonal effect model has initial state
#' prior `N(m, S)`, transition matrix `F` and noise covariance
#' `Q`, and observation matrix `H`. Then the corresponding constrained seasonal
#' effect model has initial state prior `N(P R m, P R S R' P')`,
#' transition matrix `P R F R^-1` and noise covariance `F R Q R' F'`, and
#' observation matrix `H R^-1`, where the change-of-basis matrix `R` and
#' constraint projection matrix `P` are as defined above. This follows
#' directly from applying the reparameterization `Z = R E`, and then enforcing
#' the zero-sum constraint on the prior and transition noise covariances.
#' In practice, because the sum of effects `z_N` is constrained to be zero, it
#' will never contribute a term to any linear operation on the latent space,
#' so we can drop that dimension from the model entirely.
#' `ConstrainedSeasonalStateSpaceModel` does this, so that it implements the
#' `N - 1` dimension latent space `z_1, ..., z_[N-1]`.
#' Note that since we constrained the mean effect to be zero, the latent
#' `z_i`'s now recover their interpretation as the *actual* effects,
#' `z_i = e_i` for `i = `1, ..., N - 1`, even though they were originally
#' defined as residuals. The `N`th effect is represented only implicitly, as
#' the nonzero mean of the first `N - 1` effects. Although the computational
#' represention is not symmetric across all `N` effects, we derived the
#' `ConstrainedSeasonalStateSpaceModel` by starting with a symmetric
#' representation and imposing only a symmetric constraint (the zero-sum
#' constraint), so the probability model remains symmetric over all `N`
#' seasonal effects.
#'
#' @inherit sts_seasonal_state_space_model return params
#' @family sts
#'
#' @export
sts_constrained_seasonal_state_space_model <-
  function(num_timesteps,
           num_seasons,
           drift_scale,
           initial_state_prior,
           observation_noise_scale = 1e-4,
           num_steps_per_season = 1,
           initial_step = 0,
           validate_args = FALSE,
           allow_nan_stats = TRUE,
           name = NULL) {
    args <- list(
      num_timesteps = as.integer(num_timesteps),
      num_seasons = as.integer(num_seasons),
      drift_scale = drift_scale,
      initial_state_prior = initial_state_prior,
      observation_noise_scale = observation_noise_scale,
      num_steps_per_season = as.integer(num_steps_per_season),
      initial_step = as.integer(initial_step),
      validate_args = validate_args,
      allow_nan_stats = allow_nan_stats,
      name = name
    )
    do.call(tfp$sts$ConstrainedSeasonalStateSpaceModel, args)
  }
