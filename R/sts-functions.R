

#' Build a loss function for variational inference in STS models.
#'
#' Variational inference searches for the distribution within some family of
#' approximate posteriors that minimizes a divergence between the approximate
#' posterior `q(z)` and true posterior `p(z|observed_time_series)`. By converting
#' inference to optimization, it's generally much faster than sampling-based
#' inference algorithms such as HMC. The tradeoff is that the approximating
#' family rarely contains the true posterior, so it may miss important aspects of
#' posterior structure (in particular, dependence between variables) and should
#' not be blindly trusted. Results may vary; it's generally wise to compare to
#' HMC to evaluate whether inference quality is sufficient for your task at hand.
#'
#' This method constructs a loss function for variational inference using the
#' Kullback-Liebler divergence `KL[q(z) || p(z|observed_time_series)]`, with an
#' approximating family given by independent Normal distributions transformed to
#' the appropriate parameter space for each parameter. Minimizing this loss (the
#' negative ELBO) maximizes a lower bound on the log model evidence
#' `-log p(observed_time_series)`. This is equivalent to the 'mean-field' method
#' implemented in Kucukelbir et al. (2017) and is a standard approach.
#' The resulting posterior approximations are unimodal; they will tend to underestimate posterior
#' uncertainty when the true posterior contains multiple modes
#' (the `KL[q||p]` divergence encourages choosing a single mode) or dependence between variables.
#'
#' @param model An instance of `StructuralTimeSeries` representing a
#' time-series model. This represents a joint distribution over
#' time-series and their parameters with batch shape `[b1, ..., bN]`.
#' @param observed_time_series `float` `tensor` of shape
#' `concat([sample_shape, model.batch_shape, [num_timesteps, 1]])` where
#' `sample_shape` corresponds to i.i.d. observations, and the trailing `[1]`
#' dimension may (optionally) be omitted if `num_timesteps > 1`. May
#' optionally be an instance of `sts_masked_time_series`, which includes
#' a mask `tensor` to specify timesteps with missing observations.
#' @param init_batch_shape Batch shape (`list`) of initial
#' states to optimize in parallel. Default value: `list()`. (i.e., just run a single optimization).
#' @param seed integer to seed the random number generator.
#' @param name name prefixed to ops created by this function. Default value: `NULL`
#' (i.e., 'build_factored_variational_loss').
#'
#' @return list of:
#' - variational_loss: `float` `Tensor` of shape
#' `tf$concat([init_batch_shape, model$batch_shape])`, encoding a stochastic
#' estimate of an upper bound on the negative model evidence `-log p(y)`.
#' Minimizing this loss performs variational inference; the gap between the
#' variational bound and the true (generally unknown) model evidence
#' corresponds to the divergence `KL[q||p]` between the approximate and true
#' posterior.
#' - variational_distributions: a named list giving
#' the approximate posterior for each model parameter. The keys are
#'  `character` parameter names in order, corresponding to
#' `[param.name for param in model.parameters]`. The values are
#' `tfd$Distribution` instances with batch shape
#' `tf$concat([init_batch_shape, model$batch_shape])`; these will typically be
#' of the form `tfd$TransformedDistribution(tfd.Normal(...), bijector=param.bijector)`.
#'
#' @section References:
#'  - [Alp Kucukelbir, Dustin Tran, Rajesh Ranganath, Andrew Gelman, and David M. Blei. Automatic Differentiation Variational Inference. In _Journal of Machine Learning Research_, 2017.](https://arxiv.org/abs/1603.00788)
#'
#' @family sts-functions
#'
#' @export
sts_build_factored_variational_loss <-
  function(observed_time_series,
           model,
           init_batch_shape = list(),
           seed = NULL,
           name = NULL) {
    tfp$sts$build_factored_variational_loss(model, observed_time_series, init_batch_shape, seed, name)

  }

#' Draw posterior samples using Hamiltonian Monte Carlo (HMC)
#'
#' Markov chain Monte Carlo (MCMC) methods are considered the gold standard of
#' Bayesian inference; under suitable conditions and in the limit of infinitely
#' many draws they generate samples from the true posterior distribution. HMC (Neal, 2011)
#' uses gradients of the model's log-density function to propose samples,
#' allowing it to exploit posterior geometry. However, it is computationally more
#' expensive than variational inference and relatively sensitive to tuning.
#'
#' This method attempts to provide a sensible default approach for fitting
#' StructuralTimeSeries models using HMC. It first runs variational inference as
#' a fast posterior approximation, and initializes the HMC sampler from the
#' variational posterior, using the posterior standard deviations to set
#' per-variable step sizes (equivalently, a diagonal mass matrix). During the
#' warmup phase, it adapts the step size to target an acceptance rate of 0.75,
#' which is thought to be in the desirable range for optimal mixing (Betancourt et al., 2014).
#'
#' @section References:
#' - [Radford Neal. MCMC Using Hamiltonian Dynamics. _Handbook of Markov Chain Monte Carlo_, 2011.](https://arxiv.org/abs/1206.1901)
#' - [M.J. Betancourt, Simon Byrne, and Mark Girolami. Optimizing The Integrator Step Size for Hamiltonian Monte Carlo.](https://arxiv.org/abs/1411.6669)
#'
#' @param num_results Integer number of Markov chain draws. Default value: `100`.
#' @param num_warmup_steps Integer number of steps to take before starting to
#' collect results. The warmup steps are also used to adapt the step size
#' towards a target acceptance rate of 0.75. Default value: `50`.
#' @param num_leapfrog_steps Integer number of steps to run the leapfrog integrator
#' for. Total progress per HMC step is roughly proportional to `step_size * num_leapfrog_steps`.
#' Default value: `15`.
#' @param initial_state Optional Python `list` of `Tensor`s, one for each model
#' parameter, representing the initial state(s) of the Markov chain(s). These
#' should have shape `tf$concat(list(chain_batch_shape, param$prior$batch_shape, param$prior$event_shape))`.
#' If `NULL`, the initial state is set automatically using a sample from a variational posterior.
#' Default value: `NULL`.
#' @param initial_step_size `list` of `tensor`s, one for each model parameter,
#' representing the step size for the leapfrog integrator. Must
#' broadcast with the shape of `initial_state`. Larger step sizes lead to
#' faster progress, but too-large step sizes make rejection exponentially
#' more likely. If `NULL`, the step size is set automatically using the
#' standard deviation of a variational posterior. Default value: `NULL`.
#' @param chain_batch_shape Batch shape (`list` or `int`) of chains to run in parallel.
#' Default value: `list()` (i.e., a single chain).
#' @param num_variational_steps `int` number of steps to run the variational
#' optimization to determine the initial state and step sizes. Default value: `150`.
#' @param variational_optimizer Optional `tf$train$Optimizer` instance to use in
#' the variational optimization. If `NULL`, defaults to `tf$train$AdamOptimizer(0.1)`.
#' Default value: `NULL`.
#' @param variational_sample_size integer number of Monte Carlo samples to use
#' in estimating the variational divergence. Larger values may stabilize
#' the optimization, but at higher cost per step in time and memory.
#' Default value: `1`.
#' @param name name prefixed to ops created by this function. Default value: `NULL` (i.e., 'fit_with_hmc').
#' @return list of:
#' - samples: `list` of `Tensors` representing posterior samples of model
#' parameters, with shapes `[concat([[num_results], chain_batch_shape,
#' param.prior.batch_shape, param.prior.event_shape]) for param in
#' model.parameters]`.
#' - kernel_results: A (possibly nested) `list` of `Tensor`s representing
#' internal calculations made within the HMC sampler.
#'
#' @inheritParams sts_build_factored_variational_loss
#' @family sts-functions
#' @examples
#' \donttest{
#' observed_time_series <-
#'   rep(c(3.5, 4.1, 4.5, 3.9, 2.4, 2.1, 1.2), 5) +
#'   rep(c(1.1, 1.5, 2.4, 3.1, 4.0), each = 7) %>%
#'   tensorflow::tf$convert_to_tensor(dtype = tensorflow::tf$float64)
#' day_of_week <- observed_time_series %>% sts_seasonal(num_seasons = 7)
#' local_linear_trend <- observed_time_series %>% sts_local_linear_trend()
#' model <- observed_time_series %>%
#'   sts_sum(components = list(day_of_week, local_linear_trend))
#' states_and_results <- observed_time_series %>%
#'   sts_fit_with_hmc(
#'     model,
#'     num_results = 10,
#'     num_warmup_steps = 5,
#'     num_variational_steps = 15)
#' }
#'
#' @export

sts_fit_with_hmc <- function(observed_time_series,
                             model,
                             num_results = 100,
                             num_warmup_steps = 50,
                             num_leapfrog_steps = 15,
                             initial_state = NULL,
                             initial_step_size = NULL,
                             chain_batch_shape = list(),
                             num_variational_steps = 150,
                             variational_optimizer = NULL,
                             variational_sample_size = 5,
                             seed = NULL,
                             name = NULL) {

  args <- list(
    model = model,
    observed_time_series = observed_time_series,
    num_results = as.integer(num_results),
    num_warmup_steps = as.integer(num_warmup_steps),
    num_leapfrog_steps = as.integer(num_leapfrog_steps),
    initial_state = initial_state,
    initial_step_size = initial_step_size,
    chain_batch_shape = chain_batch_shape,
    num_variational_steps = as.integer(num_variational_steps),
    variational_optimizer = variational_optimizer,
    seed = seed,
    name = name
  )

  if (tfp_version() >= "0.8") args$variational_sample_size = as.integer(variational_sample_size)

  do.call(tfp$sts$fit_with_hmc, args)

}

#' Compute one-step-ahead predictive distributions for all timesteps
#'
#' Given samples from the posterior over parameters, return the predictive
#' distribution over observations at each time `T`, given observations up
#' through time `T-1`.
#'
#' @param parameter_samples `list` of `tensors` representing posterior samples
#' of model parameters, with shapes
#' `list(tf$concat(list(list(num_posterior_draws), param<1>$prior$batch_shape, param<1>$prior$event_shape),
#'                 list(list(num_posterior_draws), param<2>$prior$batch_shape, param<2>$prior$event_shape),
#'                 ...
#'                 )
#' )`
#' for all model parameters.
#' This may optionally also be a named list mapping parameter names to `tensor` values.
#'
#' @inheritParams sts_build_factored_variational_loss
#'
#' @param timesteps_are_event_shape Deprecated, for backwards compatibility only. If False, the predictive distribution will return per-timestep probabilities Default value: TRUE.
#' @return forecast_dist a `tfd_mixture_same_family` instance with event shape
#' `list(num_timesteps)` and batch shape `tf$concat(list(sample_shape, model$batch_shape))`, with
#' `num_posterior_draws` mixture components. The `t`th step represents the
#' forecast distribution `p(observed_time_series[t] | observed_time_series[0:t-1], parameter_samples)`.
#'
#'
#'
#' @family sts-functions
#'
#' @export
sts_one_step_predictive <- function(observed_time_series,
                                    model,
                                    parameter_samples,
                                    timesteps_are_event_shape = TRUE) {
  args <- capture_args(match.call())
  do.call(tfp$sts$one_step_predictive, args)
}

#' Construct predictive distribution over future observations
#'
#' Given samples from the posterior over parameters, return the predictive
#' distribution over future observations for num_steps_forecast timesteps.
#'
#' @param num_steps_forecast scalar `integer` `tensor` number of steps to forecast
#'
#' @inheritParams sts_one_step_predictive
#'
#' @return forecast_dist a `tfd_mixture_same_family` instance with event shape
#' `list(num_steps_forecast, 1)` and batch shape `tf$concat(list(sample_shape, model$batch_shape))`, with
#' `num_posterior_draws` mixture components.
#'
#' @family sts-functions
#' @examples
#' \donttest{
#' observed_time_series <-
#'   rep(c(3.5, 4.1, 4.5, 3.9, 2.4, 2.1, 1.2), 5) +
#'   rep(c(1.1, 1.5, 2.4, 3.1, 4.0), each = 7) %>%
#'   tensorflow::tf$convert_to_tensor(dtype = tensorflow::tf$float64)
#' day_of_week <- observed_time_series %>% sts_seasonal(num_seasons = 7)
#' local_linear_trend <- observed_time_series %>% sts_local_linear_trend()
#' model <- observed_time_series %>%
#'   sts_sum(components = list(day_of_week, local_linear_trend))
#' states_and_results <- observed_time_series %>%
#'   sts_fit_with_hmc(
#'     model,
#'     num_results = 10,
#'     num_warmup_steps = 5,
#'     num_variational_steps = 15)
#' samples <- states_and_results[[1]]
#' preds <- observed_time_series %>%
#'   sts_forecast(model,
#'                parameter_samples = samples,
#'                num_steps_forecast = 50)
#' predictions <- preds %>% tfd_sample(10)
#' }
#'
#' @export
sts_forecast <- function(observed_time_series,
                         model,
                         parameter_samples,
                         num_steps_forecast) {
  tfp$sts$forecast(model,
                   observed_time_series,
                   parameter_samples,
                   as.integer(num_steps_forecast))
}

#' Decompose an observed time series into contributions from each component.
#'
#' This method decomposes a time series according to the posterior represention
#' of a structural time series model. In particular, it:
#' - Computes the posterior marginal mean and covariances over the additive
#' model's latent space.
#' - Decomposes the latent posterior into the marginal blocks for each
#' model component.
#' - Maps the per-component latent posteriors back through each component's
#' observation model, to generate the time series modeled by that component.
#'
#' @param model An instance of `sts_sum` representing a structural time series model.
#'
#' @inheritParams sts_one_step_predictive
#'
#' @return component_dists A named list mapping
#' component StructuralTimeSeries instances (elements of `model$components`)
#' to `Distribution` instances representing the posterior marginal
#' distributions on the process modeled by each component. Each distribution
#' has batch shape matching that of `posterior_means`/`posterior_covs`, and
#' event shape of `list(num_timesteps)`.
#'
#' @family sts-functions
#' @examples
#' \donttest{
#' observed_time_series <- array(rnorm(2 * 1 * 12), dim = c(2, 1, 12))
#' day_of_week <- observed_time_series %>% sts_seasonal(num_seasons = 7, name = "seasonal")
#' local_linear_trend <- observed_time_series %>% sts_local_linear_trend(name = "local_linear")
#' model <- observed_time_series %>%
#'   sts_sum(components = list(day_of_week, local_linear_trend))
#' states_and_results <- observed_time_series %>%
#'   sts_fit_with_hmc(
#'     model,
#'     num_results = 10,
#'     num_warmup_steps = 5,
#'     num_variational_steps = 15
#'     )
#' samples <- states_and_results[[1]]
#'
#' component_dists <- observed_time_series %>%
#'  sts_decompose_by_component(model = model, parameter_samples = samples)
#' }
#' @export
sts_decompose_by_component <- function(observed_time_series,
                                       model,
                                       parameter_samples) {
  tfp$sts$decompose_by_component(model,
                                 observed_time_series,
                                 parameter_samples)
}

#' Build a variational posterior that factors over model parameters.
#'
#' The surrogate posterior consists of independent Normal distributions for
#' each parameter with trainable `loc` and `scale`, transformed using the
#' parameter's `bijector` to the appropriate support space for that parameter.
#'
#' @param model An instance of `StructuralTimeSeries` representing a
#' time-series model. This represents a joint distribution over
#' time-series and their parameters with batch shape `[b1, ..., bN]`.#'
#' @param batch_shape Batch shape (`list`, or `integer`) of initial
#' states to optimize in parallel.
#' Default value: `list()`. (i.e., just run a single optimization).
#' @param seed integer to seed the random number generator.
#' @param name string prefixed to ops created by this function.
#' Default value: `NULL` (i.e., 'build_factored_surrogate_posterior').
#'
#' @return  variational_posterior `tfd_joint_distribution_named` defining a trainable
#' surrogate posterior over model parameters. Samples from this
#' distribution are named lists with  `character` parameter names as keys.
#'
#' @family sts-functions
#'
#' @export
sts_build_factored_surrogate_posterior <-
  function(model,
           batch_shape = list(),
           seed = NULL,
           name = NULL) {
    tfp$sts$build_factored_surrogate_posterior(model, batch_shape, seed, name)

  }

#' Initialize from a uniform `[-2, 2]` distribution in unconstrained space.
#'
#' @param parameter `sts$Parameter` named tuple instance.
#' @param return_constrained if `TRUE`, re-applies the constraining bijector
#' to return initializations in the original domain. Otherwise, returns
#' initializations in the unconstrained space.
#' Default value: `TRUE`.
#' @param init_sample_shape `sample_shape` of the sampled initializations.
#' Default value: `list()`.
#' @param seed integer to seed the random number generator.
#'
#' @return uniform_initializer `Tensor` of shape
#' `concat([init_sample_shape, parameter.prior.batch_shape, transformed_event_shape])`, where
#' `transformed_event_shape` is `parameter.prior.event_shape`, if
#' `return_constrained=TRUE`, and otherwise it is
#' `parameter$bijector$inverse_event_shape(parameter$prior$event_shape)`.
#'
#' @family sts-functions
#' @export
sts_sample_uniform_initial_state <-
  function(parameter,
           return_constrained = TRUE,
           init_sample_shape = list(),
           seed = NULL) {
    tfp$sts$sample_uniform_initial_state(parameter,
                                         return_constrained,
                                         as_integer_list(init_sample_shape),
                                         seed)
  }

#' Decompose a forecast distribution into contributions from each component.
#'
#' @inheritParams sts_decompose_by_component
#'
#' @param forecast_dist A `Distribution` instance returned by `sts_forecast()`.
#' (specifically, must be a `tfd.MixtureSameFamily` over a
#' `tfd_linear_gaussian_state_space_model` parameterized by posterior samples).
#'
#' @return component_dists A named list mapping
#' component StructuralTimeSeries instances (elements of `model$components`)
#' to `Distribution` instances representing the marginal forecast for each component.
#' Each distribution has batch shape matching `forecast_dist` (specifically,
#' the event shape is `[num_steps_forecast]`).
#'
#' @family sts-functions
#'
#' @export
sts_decompose_forecast_by_component <- function(model,
                                                forecast_dist,
                                                parameter_samples) {
  tfp$sts$decompose_forecast_by_component(model,
                                          forecast_dist,
                                          parameter_samples)
}
