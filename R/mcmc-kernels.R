#' Runs one step of Hamiltonian Monte Carlo.
#'
#' Hamiltonian Monte Carlo (HMC) is a Markov chain Monte Carlo (MCMC) algorithm
#' that takes a series of gradient-informed steps to produce a Metropolis
#' proposal. This class implements one random HMC step from a given
#' `current_state`. Mathematical details and derivations can be found in
#' Neal (2011).
#'
#' The `one_step` function can update multiple chains in parallel. It assumes
#' that all leftmost dimensions of `current_state` index independent chain states
#' (and are therefore updated independently). The output of
#' `target_log_prob_fn(*current_state)` should sum log-probabilities across all
#' event dimensions. Slices along the rightmost dimensions may have different
#' target distributions; for example, `current_state[0, :]` could have a
#' different target distribution from `current_state[1, :]`. These semantics are
#' governed by `target_log_prob_fn(*current_state)`. (The number of independent
#' chains is `tf.size(target_log_prob_fn(*current_state))`.)
#'
#' @section References:
#' - [Radford Neal. MCMC Using Hamiltonian Dynamics. _Handbook of Markov Chain Monte Carlo_, 2011.](https://arxiv.org/abs/1206.1901)
#' - [Bernard Delyon, Marc Lavielle, Eric, Moulines. _Convergence of a stochastic approximation version of the EM algorithm_, Ann. Statist. 27 (1999), no. 1, 94--128.](https://projecteuclid.org/euclid.aos/1018031103)
#'
#' @param target_log_prob_fn Function which takes an argument like
#' `current_state` (if it's a list `current_state` will be unpacked) and returns its
#' (possibly unnormalized) log-density under the target distribution.
#' @param step_size `Tensor` or `list` of `Tensor`s representing the step
#' size for the leapfrog integrator. Must broadcast with the shape of
#' `current_state`. Larger step sizes lead to faster progress, but
#' too-large step sizes make rejection exponentially more likely. When
#' possible, it's often helpful to match per-variable step sizes to the
#' standard deviations of the target distribution in each variable.
#' @param num_leapfrog_steps Integer number of steps to run the leapfrog integrator
#' for. Total progress per HMC step is roughly proportional to
#' `step_size * num_leapfrog_steps`.
#' @param state_gradients_are_stopped `logical` indicating that the proposed
#' new state be run through `tf$stop_gradient`. This is particularly useful
#' when combining optimization over samples from the HMC chain.
#' Default value: `FALSE` (i.e., do not apply `stop_gradient`).
#' @param step_size_update_fn Function taking current `step_size`
#' (typically a `tf$Variable`) and `kernel_results` (typically
#' `collections.namedtuple`) and returns updated step_size (`Tensor`s).
#' Default value: `NULL` (i.e., do not update `step_size` automatically).
#' @param seed integer to seed the random number generator.
#' @param store_parameters_in_results If `TRUE`, then `step_size` and
#' `num_leapfrog_steps` are written to and read from eponymous fields in
#' the kernel results objects returned from `one_step` and
#' `bootstrap_results`. This allows wrapper kernels to adjust those
#' parameters on the fly. This is incompatible with `step_size_update_fn`,
#' which must be set to `NULL`.
#' @param name string prefixed to Ops created by this function.
#' Default value: `NULL` (i.e., 'hmc_kernel').
#'
#' @family mcmc-kernels
#' @export
#'
mcmc_hamiltonian_monte_carlo <- function(target_log_prob_fn,
                                         step_size,
                                         num_leapfrog_steps,
                                         state_gradients_are_stopped = FALSE,
                                         step_size_update_fn = NULL,
                                         seed = NULL,
                                         store_parameters_in_results = FALSE,
                                         name = NULL) {
  args <- list(
    target_log_prob_fn = target_log_prob_fn,
    step_size = step_size,
    num_leapfrog_steps = as.integer(num_leapfrog_steps),
    state_gradients_are_stopped = state_gradients_are_stopped,
    step_size_update_fn = step_size_update_fn,
    seed = seed,
    store_parameters_in_results = store_parameters_in_results,
    name = name
  )

  do.call(tfp$mcmc$HamiltonianMonteCarlo, args)
}

# hack due to https://github.com/r-lib/pkgdown/issues/330
tfp_mcmc_sssa <- function() tfp$mcmc$simple_step_size_adaptation

#' Adapts the inner kernel's `step_size` based on `log_accept_prob`.
#'
#' The simple policy multiplicatively increases or decreases the `step_size` of
#' the inner kernel based on the value of `log_accept_prob`. It is based on
#' equation 19 of Andrieu and Thoms (2008). Given enough steps and small
#' enough `adaptation_rate` the median of the distribution of the acceptance
#' probability will converge to the `target_accept_prob`. A good target
#' acceptance probability depends on the inner kernel. If this kernel is
#' `HamiltonianMonteCarlo`, then 0.6-0.9 is a good range to aim for. For
#' `RandomWalkMetropolis` this should be closer to 0.25. See the individual
#' kernels' docstrings for guidance.
#'
#' In general, adaptation prevents the chain from reaching a stationary
#' distribution, so obtaining consistent samples requires `num_adaptation_steps`
#' be set to a value somewhat smaller than the number of burnin steps.
#' However, it may sometimes be helpful to set `num_adaptation_steps` to a larger
#' value during development in order to inspect the behavior of the chain during
#' adaptation.
#'
#' The step size is assumed to broadcast with the chain state, potentially having
#' leading dimensions corresponding to multiple chains. When there are fewer of
#' those leading dimensions than there are chain dimensions, the corresponding
#' dimensions in the `log_accept_prob` are averaged (in the direct space, rather
#' than the log space) before being used to adjust the step size. This means that
#' this kernel can do both cross-chain adaptation, or per-chain step size
#' adaptation, depending on the shape of the step size.
#'
#' For example, if your problem has a state with shape `[S]`, your chain state
#' has shape `[C0, C1, Y]` (meaning that there are `C0 * C1` total chains) and
#' `log_accept_prob` has shape `[C0, C1]` (one acceptance probability per chain),
#' then depending on the shape of the step size, the following will happen:
#' - Step size has shape `[]`, `[S]` or `[1]`, the `log_accept_prob` will be averaged
#' across its `C0` and `C1` dimensions. This means that you will learn a shared
#' step size based on the mean acceptance probability across all chains. This
#' can be useful if you don't have a lot of steps to adapt and want to average
#' away the noise.
#' - Step size has shape `[C1, 1]` or `[C1, S]`, the `log_accept_prob` will be
#' averaged across its `C0` dimension. This means that you will learn a shared
#' step size based on the mean acceptance probability across chains that share
#' the coordinate across the `C1` dimension. This can be useful when the `C1`
#' dimension indexes different distributions, while `C0` indexes replicas of a
#' single distribution, all sampled in parallel.
#' - Step size has shape `[C0, C1, 1]` or `[C0, C1, S]`, then no averaging will
#' happen. This means that each chain will learn its own step size. This can be
#' useful when all chains are sampling from different distributions. Even when
#' all chains are for the same distribution, this can help during the initial
#' warmup period.
#' - Step size has shape `[C0, 1, 1]` or `[C0, 1, S]`, the `log_accept_prob` will be
#' averaged across its `C1` dimension. This means that you will learn a shared
#' step size based on the mean acceptance probability across chains that share
#' the coordinate across the `C0` dimension. This can be useful when the `C0`
#' dimension indexes different distributions, while `C1` indexes replicas of a
#' single distribution, all sampled in parallel.
#'
#' @section References:
#' - [Andrieu, Christophe, Thoms, Johannes. A tutorial on adaptive MCMC. _Statistics and Computing_, 2008.](https://people.eecs.berkeley.edu/~jordan/sail/readings/andrieu-thoms.pdf)
#' - http://andrewgelman.com/2017/12/15/burn-vs-warm-iterative-simulation-algorithms/#comment-627745
#' - [Betancourt, M. J., Byrne, S., & Girolami, M. (2014). _Optimizing The Integrator Step Size for Hamiltonian Monte Carlo_.)(http://arxiv.org/abs/1411.6669)
#'
#' @param inner_kernel `TransitionKernel`-like object.
#' @param num_adaptation_steps Scalar `integer` `Tensor` number of initial steps to
#' during which to adjust the step size. This may be greater, less than, or
#' equal to the number of burnin steps.
#' @param target_accept_prob A floating point `Tensor` representing desired
#' acceptance probability. Must be a positive number less than 1. This can
#' either be a scalar, or have shape `list(num_chains)`. Default value: `0.75`
#' (the center of asymptotically optimal rate for HMC).
#' @param adaptation_rate `Tensor` representing amount to scale the current
#' `step_size`.
#' @param step_size_setter_fn A function with the signature
#' `(kernel_results, new_step_size) -> new_kernel_results` where
#' `kernel_results` are the results of the `inner_kernel`, `new_step_size`
#' is a `Tensor` or a nested collection of `Tensor`s with the same
#' structure as returned by the `step_size_getter_fn`, and
#' `new_kernel_results` are a copy of `kernel_results` with the step
#' size(s) set.
#' @param step_size_getter_fn A function with the signature
#' `(kernel_results) -> step_size` where `kernel_results` are the results
#' of the `inner_kernel`, and `step_size` is a floating point `Tensor` or a
#' nested collection of such `Tensor`s.
#' @param log_accept_prob_getter_fn A function with the signature
#' `(kernel_results) -> log_accept_prob` where `kernel_results` are the
#' results of the `inner_kernel`, and `log_accept_prob` is a floating point
#' `Tensor`. `log_accept_prob` can either be a scalar, or have shape
#' `list(num_chains)`. If it's the latter, `step_size` should also have the same
#' leading dimension.
#' @param validate_args `Logical`. When `True` kernel parameters are checked
#' for validity. When `False` invalid inputs may silently render incorrect
#' outputs.
#' @param name string prefixed to Ops created by this class. Default: "simple_step_size_adaptation".
#'
#' @family mcmc-kernels
#' @export
mcmc_simple_step_size_adaptation <- function(inner_kernel,
                                             num_adaptation_steps,
                                             target_accept_prob = 0.75,
                                             adaptation_rate = 0.01,
                                             step_size_setter_fn =
                                               tfp_mcmc_sssa()$`_hmc_like_step_size_setter_fn`,
                                             step_size_getter_fn =
                                               tfp_mcmc_sssa()$`_hmc_like_step_size_getter_fn`,
                                             log_accept_prob_getter_fn =
                                               tfp_mcmc_sssa()$`_hmc_like_log_accept_prob_getter_fn`,
                                             validate_args = FALSE,
                                             name = NULL) {
  args <- list(
    inner_kernel = inner_kernel,
    num_adaptation_steps = as.integer(num_adaptation_steps),
    target_accept_prob = target_accept_prob,
    adaptation_rate = adaptation_rate,
    step_size_setter_fn = step_size_setter_fn,
    step_size_getter_fn = step_size_getter_fn,
    log_accept_prob_getter_fn = log_accept_prob_getter_fn,
    validate_args = validate_args,
    name = name
  )

  do.call(tfp$mcmc$SimpleStepSizeAdaptation, args)
}
