
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
#' @section References:
#'  - [Alp Kucukelbir, Dustin Tran, Rajesh Ranganath, Andrew Gelman, and David M. Blei. Automatic Differentiation Variational Inference. In _Journal of Machine Learning Research_, 2017.](https://arxiv.org/abs/1603.00788)
#'
#' @family sts-functions
#'
#' @export
sts_build_factored_variational_loss <- function(observed_time_series,
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
#' @param name name prefixed to ops created by this function. Default value: `NULL` (i.e., 'fit_with_hmc').
#'
#' @inheritParams sts_build_factored_variational_loss
#' @family sts-functions
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
                             seed = NULL,
                             name = NULL) {
  tfp$sts$fit_with_hmc(
    model,
    observed_time_series,
    as.integer(num_results),
    as.integer(num_warmup_steps),
    as.integer(num_leapfrog_steps),
    initial_state,
    initial_step_size,
    chain_batch_shape,
    as.integer(num_variational_steps),
    variational_optimizer,
    seed,
    name
  )

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
#' @returns forecast_dist a `tfd_mixture_same_family` instance with event shape
#' `list(num_timesteps)` and batch shape `tf$concat(list(sample_shape, model$batch_shape))`, with
#' `num_posterior_draws` mixture components. The `t`th step represents the
#' forecast distribution `p(observed_time_series[t] | observed_time_series[0:t-1], parameter_samples)`.
#' @inheritParams sts_build_factored_variational_loss
#' @family sts-functions
#'
#' @export
sts_one_step_predictive <- function(observed_time_series,
                                    model,
                                    parameter_samples) {
  tfp$sts$one_step_predictive(model,
                              observed_time_series,
                              parameter_samples)
}

