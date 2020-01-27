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
#' `target_log_prob_fn(current_state)` should sum log-probabilities across all
#' event dimensions. Slices along the rightmost dimensions may have different
#' target distributions; for example, `current_state[0, :]` could have a
#' different target distribution from `current_state[1, :]`. These semantics are
#' governed by `target_log_prob_fn(current_state)`. (The number of independent
#' chains is `tf$size(target_log_prob_fn(current_state))`.)
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
#' `collections$namedtuple`) and returns updated step_size (`Tensor`s).
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
#' @return a Monte Carlo sampling kernel
#'
#' @family mcmc_kernels
#' @export
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

#' Runs one step of Uncalibrated Hamiltonian Monte Carlo
#'
#' Warning: this kernel will not result in a chain which converges to the
#' `target_log_prob`. To get a convergent MCMC, use `mcmc_hamiltonian_monte_carlo(...)`
#' or `mcmc_metropolis_hastings(mcmc_uncalibrated_hamiltonian_monte_carlo(...))`.
#' For more details on `UncalibratedHamiltonianMonteCarlo`, see `HamiltonianMonteCarlo`.
#'
#' @inherit mcmc_hamiltonian_monte_carlo return params
#' @family mcmc_kernels
#' @export
mcmc_uncalibrated_hamiltonian_monte_carlo <-
  function(target_log_prob_fn,
           step_size,
           num_leapfrog_steps,
           state_gradients_are_stopped = FALSE,
           seed = NULL,
           store_parameters_in_results = FALSE,
           name = NULL) {
    args <- list(
      target_log_prob_fn = target_log_prob_fn,
      step_size = step_size,
      num_leapfrog_steps = as.integer(num_leapfrog_steps),
      state_gradients_are_stopped = state_gradients_are_stopped,
      seed = seed,
      store_parameters_in_results = store_parameters_in_results,
      name = name
    )

    do.call(tfp$mcmc$UncalibratedHamiltonianMonteCarlo, args)
  }


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
#' - [Betancourt, M. J., Byrne, S., & Girolami, M. (2014). _Optimizing The Integrator Step Size for Hamiltonian Monte Carlo_.](http://arxiv.org/abs/1411.6669)
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
#' @inherit mcmc_hamiltonian_monte_carlo return params
#'
#' @family mcmc_kernels
#' @examples
#' \donttest{
#'   target_log_prob_fn <- tfd_normal(loc = 0, scale = 1)$log_prob
#'   num_burnin_steps <- 500
#'   num_results <- 500
#'   num_chains <- 64L
#'   step_size <- tf$fill(list(num_chains), 0.1)
#'
#'   kernel <- mcmc_hamiltonian_monte_carlo(
#'     target_log_prob_fn = target_log_prob_fn,
#'     num_leapfrog_steps = 2,
#'     step_size = step_size
#'   ) %>%
#'     mcmc_simple_step_size_adaptation(num_adaptation_steps = round(num_burnin_steps * 0.8))
#'
#'   res <- kernel %>% mcmc_sample_chain(
#'     num_results = num_results,
#'     num_burnin_steps = num_burnin_steps,
#'     current_state = rep(0, num_chains),
#'     trace_fn = function(x, pkr) {
#'       list (
#'         pkr$inner_results$accepted_results$step_size,
#'         pkr$inner_results$log_accept_ratio
#'       )
#'     }
#'   )
#'
#'   samples <- res$all_states
#'   step_size <- res$trace[[1]]
#'   log_accept_ratio <- res$trace[[2]]
#' }
#'
#' @export
mcmc_simple_step_size_adaptation <- function(inner_kernel,
                                             num_adaptation_steps,
                                             target_accept_prob = 0.75,
                                             adaptation_rate = 0.01,
                                             step_size_setter_fn = NULL,
                                             step_size_getter_fn = NULL,
                                             log_accept_prob_getter_fn = NULL,
                                             validate_args = FALSE,
                                             name = NULL) {
  args <- list(
    inner_kernel = inner_kernel,
    num_adaptation_steps = as.integer(num_adaptation_steps),
    target_accept_prob = target_accept_prob,
    adaptation_rate = adaptation_rate,
    validate_args = validate_args,
    name = name
  )

  default_hmc_like_step_size_setter_fn <-
    if (tfp_version() < "0.9")
      tfp$mcmc$simple_step_size_adaptation$`_hmc_like_step_size_setter_fn`
  else
    tfp$mcmc$simple_step_size_adaptation$hmc_like_step_size_setter_fn
  default_hmc_like_step_size_getter_fn <-
    if (tfp_version() < "0.9")
      tfp$mcmc$simple_step_size_adaptation$`_hmc_like_step_size_getter_fn`
  else
    tfp$mcmc$simple_step_size_adaptation$hmc_like_step_size_getter_fn
  default_hmc_like_log_accept_prob_getter_fn <-
    if (tfp_version() < "0.9")
      tfp$mcmc$simple_step_size_adaptation$`_hmc_like_log_accept_prob_getter_fn`
  else
    tfp$mcmc$simple_step_size_adaptation$hmc_like_log_accept_prob_getter_fn


  # see https://github.com/r-lib/pkgdown/issues/330
  args$step_size_setter_fn <-
    if (!is.null(step_size_setter_fn))
      step_size_setter_fn
  else
    default_hmc_like_step_size_setter_fn
  args$step_size_getter_fn <-
    if (!is.null(step_size_getter_fn))
      step_size_getter_fn
  else
    default_hmc_like_step_size_getter_fn
  args$log_accept_prob_getter_fn <-
    if (!is.null(log_accept_prob_getter_fn))
      log_accept_prob_getter_fn
  else
    default_hmc_like_log_accept_prob_getter_fn

  do.call(tfp$mcmc$SimpleStepSizeAdaptation, args)
}

#' Runs one step of the Metropolis-Hastings algorithm.
#'
#' The Metropolis-Hastings algorithm is a Markov chain Monte Carlo (MCMC) technique which uses a proposal distribution
#' to eventually sample from a target distribution.
#'
#' Note: `inner_kernel$one_step` must return `kernel_results` as a `collections$namedtuple` which must:
#' - have a `target_log_prob` field,
#' - optionally have a `log_acceptance_correction` field, and,
#' - have only fields which are `Tensor`-valued.
#'
#' The Metropolis-Hastings log acceptance-probability is computed as:
#'
#' ```
#' log_accept_ratio = (current_kernel_results.target_log_prob
#'                    - previous_kernel_results.target_log_prob
#'                    + current_kernel_results.log_acceptance_correction)
#' ```
#'
#' If `current_kernel_results$log_acceptance_correction` does not exist, it is
#' presumed `0` (i.e., that the proposal distribution is symmetric).
#' The most common use-case for `log_acceptance_correction` is in the
#' Metropolis-Hastings algorithm, i.e.,
#'
#' ```
#' accept_prob(x' | x) = p(x') / p(x) (g(x|x') / g(x'|x))
#' where,
#' p  represents the target distribution,
#' g  represents the proposal (conditional) distribution,
#' x' is the proposed state, and,
#' x  is current state
#' ```
#' The log of the parenthetical term is the `log_acceptance_correction`.
#' The `log_acceptance_correction` may not necessarily correspond to the ratio of
#' proposal distributions, e.g, `log_acceptance_correction` has a different
#' interpretation in Hamiltonian Monte Carlo.
#' @param inner_kernel `TransitionKernel`-like object which has `collections$namedtuple`
#' `kernel_results` and which contains a `target_log_prob` member and optionally a `log_acceptance_correction` member.
#' @param name string prefixed to Ops created by this function. Default value: `NULL` (i.e., "mh_kernel").
#'
#' @inherit mcmc_hamiltonian_monte_carlo return params
#' @family mcmc_kernels
#' @export
mcmc_metropolis_hastings <- function(inner_kernel,
                                     seed = NULL,
                                     name = NULL) {
  args <- list(inner_kernel = inner_kernel,
               seed = seed,
               name = name)

  do.call(tfp$mcmc$MetropolisHastings, args)
}

#' Runs one step of the RWM algorithm with symmetric proposal.
#'
#' Random Walk Metropolis is a gradient-free Markov chain Monte Carlo
#' (MCMC) algorithm. The algorithm involves a proposal generating step
#' `proposal_state = current_state + perturb` by a random
#' perturbation, followed by Metropolis-Hastings accept/reject step. For more
#' details see [Section 2.1 of Roberts and Rosenthal (2004)](http://emis.ams.org/journals/PS/images/getdoc510c.pdf?id=35&article=15&mode=pdf).
#'
#' The current class implements RWM for normal and uniform proposals. Alternatively,
#' the user can supply any custom proposal generating function.
#' The function `one_step` can update multiple chains in parallel. It assumes
#' that all leftmost dimensions of `current_state` index independent chain states
#' (and are therefore updated independently). The output of
#' `target_log_prob_fn(current_state)` should sum log-probabilities across all
#' event dimensions. Slices along the rightmost dimensions may have different
#' target distributions; for example, `current_state[0, :]` could have a
#' different target distribution from `current_state[1, :]`. These semantics
#' are governed by `target_log_prob_fn(current_state)`. (The number of
#' independent chains is `tf$size(target_log_prob_fn(current_state))`.)
#'
#' @param target_log_prob_fn Function which takes an argument like
#' `current_state` ((if it's a list `current_state` will be unpacked) and returns its
#' (possibly unnormalized) log-density under the target distribution.
#' @param new_state_fn Function which takes a list of state parts and a
#' seed; returns a same-type `list` of `Tensor`s, each being a perturbation
#' of the input state parts. The perturbation distribution is assumed to be
#' a symmetric distribution centered at the input state part.
#' Default value: `NULL` which is mapped to `tfp$mcmc$random_walk_normal_fn()`.
#' @param name String name prefixed to Ops created by this function.
#' Default value: `NULL` (i.e., 'rwm_kernel').
#'
#' @inherit mcmc_hamiltonian_monte_carlo return params
#' @family mcmc_kernels
#' @export
mcmc_random_walk_metropolis <- function(target_log_prob_fn,
                                        new_state_fn = NULL,
                                        seed = NULL,
                                        name = NULL) {
  args <- list(
    target_log_prob_fn = target_log_prob_fn,
    new_state_fn = new_state_fn,
    seed = seed,
    name = name
  )

  do.call(tfp$mcmc$RandomWalkMetropolis, args)
}

#' Applies a bijector to the MCMC's state space
#'
#' The transformed transition kernel enables fitting
#' a bijector which serves to decorrelate the Markov chain Monte Carlo (MCMC)
#' event dimensions thus making the chain mix faster. This is
#' particularly useful when the geometry of the target distribution is
#' unfavorable. In such cases it may take many evaluations of the
#' `target_log_prob_fn` for the chain to mix between faraway states.
#'
#' The idea of training an affine function to decorrelate chain event dims was
#' presented in Parno and Marzouk (2014). Used in conjunction with the
#' Hamiltonian Monte Carlo transition kernel, the Parno and Marzouk (2014)
#' idea is an instance of Riemannian manifold HMC (Girolami and Calderhead, 2011).
#'
#' The transformed transition kernel enables arbitrary bijective transformations
#' of arbitrary transition kernels, e.g., one could use bijectors
#' `tfb_affine`, `tfb_real_nvp`, etc.
#' with transition kernels `mcmc_hamiltonian_monte_carlo`, `mcmc_random_walk_metropolis`, etc.
#'
#' @section References:
#' - [Matthew Parno and Youssef Marzouk. Transport map accelerated Markov chain Monte Carlo. _arXiv preprint arXiv:1412.5492_, 2014.](https://arxiv.org/abs/1412.5492)
#' - [Mark Girolami and Ben Calderhead. Riemann manifold langevin and hamiltonian monte carlo methods. In _Journal of the Royal Statistical Society_, 2011.](https://doi.org/10.1111/j.1467-9868.2010.00765.x)
#'
#' @param inner_kernel `TransitionKernel`-like object which has a `target_log_prob_fn` argument.
#' @param bijector bijector or list of bijectors. These bijectors use `forward` to map the
#' `inner_kernel` state space to the state expected by `inner_kernel$target_log_prob_fn`.
#' @param  name string prefixed to Ops created by this function.
#' Default value: `NULL` (i.e., "transformed_kernel").
#'
#' @inherit mcmc_hamiltonian_monte_carlo return params
#'
#' @family mcmc_kernels
#' @export
mcmc_transformed_transition_kernel <- function(inner_kernel,
                                               bijector,
                                               name = NULL) {
  args <- list(inner_kernel = inner_kernel,
               bijector = bijector,
               name = name)

  do.call(tfp$mcmc$TransformedTransitionKernel, args)
}

#' Adapts the inner kernel's `step_size` based on `log_accept_prob`.
#'
#' The dual averaging policy uses a noisy step size for exploration, while
#' averaging over tuning steps to provide a smoothed estimate of an optimal
#' value. It is based on section 3.2 of Hoffman and Gelman (2013), which
#' modifies the [stochastic convex optimization scheme of Nesterov (2009).
#' The modified algorithm applies extra weight to recent iterations while
#' keeping the convergence guarantees of Robbins-Monro, and takes care not
#' to make the step size too small too quickly when maintaining a constant
#' trajectory length, to avoid expensive early iterations. A good target
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
#' The step size is assumed to broadcast with the chain state, potentially having
#' leading dimensions corresponding to multiple chains. When there are fewer of
#' those leading dimensions than there are chain dimensions, the corresponding
#' dimensions in the `log_accept_prob` are averaged (in the direct space, rather
#' than the log space) before being used to adjust the step size. This means that
#' this kernel can do both cross-chain adaptation, or per-chain step size
#' adaptation, depending on the shape of the step size.
#' For example, if your problem has a state with shape `[S]`, your chain state
#' has shape `[C0, C1, S]` (meaning that there are `C0 * C1` total chains) and
#' `log_accept_prob` has shape `[C0, C1]` (one acceptance probability per chain),
#' then depending on the shape of the step size, the following will happen:
#'
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
#' - [Matthew D. Hoffman, Andrew Gelman. The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. In _Journal of Machine Learning Research_, 15(1):1593-1623, 2014.](http://jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf)
#' - [Yurii Nesterov. Primal-dual subgradient methods for convex problems. Mathematical programming 120.1 (2009): 221-259](https://link.springer.com/article/10.1007/s10107-007-0149-x)
#' - [http://andrewgelman.com/2017/12/15/burn-vs-warm-iterative-simulation-algorithms/#comment-627745](http://andrewgelman.com/2017/12/15/burn-vs-warm-iterative-simulation-algorithms/#comment-627745)
#'
#' @param inner_kernel `TransitionKernel`-like object.
#' @param num_adaptation_steps Scalar `integer` `Tensor` number of initial steps to
#' during which to adjust the step size. This may be greater, less than, or
#' equal to the number of burnin steps.
#' @param target_accept_prob A floating point `Tensor` representing desired
#' acceptance probability. Must be a positive number less than 1. This can
#' either be a scalar, or have shape `[num_chains]`. Default value: `0.75`
#' (the center of asymptotically optimal rate for HMC).
#' @param exploration_shrinkage Floating point scalar `Tensor`. How strongly the
#' exploration rate is biased towards the shrinkage target.
#' @param step_count_smoothing Int32 scalar `Tensor`. Number of "pseudo-steps"
#' added to the number of steps taken to prevents noisy exploration during
#' the early samples.
#' @param decay_rate Floating point scalar `Tensor`. How much to favor recent
#' iterations over earlier ones. A value of 1 gives equal weight to all
#' history.
#' @param step_size_setter_fn A function with the signature
#' `(kernel_results, new_step_size) -> new_kernel_results` where `kernel_results` are the
#' results of the `inner_kernel`, `new_step_size` is a `Tensor` or a nested
#' collection of `Tensor`s with the same structure as returned by the
#' `step_size_getter_fn`, and `new_kernel_results` are a copy of
#' `kernel_results` with the step size(s) set.
#' @param step_size_getter_fn A callable with the signature
#' `(kernel_results) -> step_size` where `kernel_results` are the results of the `inner_kernel`,
#' and `step_size` is a floating point `Tensor` or a nested collection of
#' such `Tensor`s.
#' @param log_accept_prob_getter_fn A callable with the signature
#' `(kernel_results) -> log_accept_prob` where `kernel_results` are the results of the
#' `inner_kernel`, and `log_accept_prob` is a floating point `Tensor`.
#' `log_accept_prob` can either be a scalar, or have shape `[num_chains]`. If
#' it's the latter, `step_size` should also have the same leading
#' dimension.
#' @param validate_args `logical`. When `TRUE` kernel parameters are checked
#' for validity. When `FALSE` invalid inputs may silently render incorrect
#' outputs.
#' @param name name prefixed to Ops created by this function.
#' Default value: `NULL` (i.e., 'dual_averaging_step_size_adaptation').
#'
#' @inherit mcmc_hamiltonian_monte_carlo return params
#'
#' @family mcmc_kernels
#' @seealso For an example how to use see [mcmc_no_u_turn_sampler()].
#' @export
mcmc_dual_averaging_step_size_adaptation <- function(inner_kernel,
                                                     num_adaptation_steps,
                                                     target_accept_prob = 0.75,
                                                     exploration_shrinkage = 0.05,
                                                     step_count_smoothing = 10,
                                                     decay_rate = 0.75,
                                                     step_size_setter_fn = NULL,
                                                     step_size_getter_fn = NULL,
                                                     log_accept_prob_getter_fn = NULL,
                                                     validate_args = FALSE,
                                                     name = NULL) {
  args <- list(
    inner_kernel = inner_kernel,
    num_adaptation_steps = as.integer(num_adaptation_steps),
    target_accept_prob = target_accept_prob,
    exploration_shrinkage = exploration_shrinkage,
    step_count_smoothing = as.integer(step_count_smoothing),
    decay_rate = decay_rate,
    validate_args = validate_args,
    name = name
  )

  default_hmc_like_step_size_setter_fn <-
    if (tfp_version() < "0.9")
      tfp$mcmc$dual_averaging_step_size_adaptation$`_hmc_like_step_size_setter_fn`
  else
    tfp$mcmc$dual_averaging_step_size_adaptation$hmc_like_step_size_setter_fn
  default_hmc_like_step_size_getter_fn <-
    if (tfp_version() < "0.9")
      tfp$mcmc$dual_averaging_step_size_adaptation$`_hmc_like_step_size_getter_fn`
  else
    tfp$mcmc$dual_averaging_step_size_adaptation$hmc_like_step_size_getter_fn
  default_hmc_like_log_accept_prob_getter_fn <-
    if (tfp_version() < "0.9")
      tfp$mcmc$dual_averaging_step_size_adaptation$`_hmc_like_log_accept_prob_getter_fn`
  else
    tfp$mcmc$dual_averaging_step_size_adaptation$hmc_like_log_accept_prob_getter_fn


  # see https://github.com/r-lib/pkgdown/issues/330
  args$step_size_setter_fn <-
    if (!is.null(step_size_setter_fn))
      step_size_setter_fn
  else
    default_hmc_like_step_size_setter_fn
  args$step_size_getter_fn <-
    if (!is.null(step_size_getter_fn))
      step_size_getter_fn
  else
    default_hmc_like_step_size_getter_fn
  args$log_accept_prob_getter_fn <-
    if (!is.null(log_accept_prob_getter_fn))
      log_accept_prob_getter_fn
  else
    default_hmc_like_log_accept_prob_getter_fn

  do.call(tfp$mcmc$DualAveragingStepSizeAdaptation, args)
}

#' Runs one step of the No U-Turn Sampler
#'
#' The No U-Turn Sampler (NUTS) is an adaptive variant of the Hamiltonian Monte
#' Carlo (HMC) method for MCMC.  NUTS adapts the distance traveled in response to
#' the curvature of the target density.  Conceptually, one proposal consists of
#' reversibly evolving a trajectory through the sample space, continuing until
#' that trajectory turns back on itself (hence the name, 'No U-Turn').
#' This class implements one random NUTS step from a given
#' `current_state`.  Mathematical details and derivations can be found in
#' Hoffman & Gelman (2011).
#'
#' The `one_step` function can update multiple chains in parallel. It assumes
#' that a prefix of leftmost dimensions of `current_state` index independent
#' chain states (and are therefore updated independently).  The output of
#' `target_log_prob_fn(current_state)` should sum log-probabilities across all
#' event dimensions.  Slices along the rightmost dimensions may have different
#' target distributions; for example, `current_state[0][0, ...]` could have a
#' different target distribution from `current_state[0][1, ...]`.  These
#' semantics are governed by `target_log_prob_fn(*current_state)`.
#' (The number of independent chains is `tf$size(target_log_prob_fn(current_state))`.)
#'
#' @section References:
#' - [Matthew D. Hoffman, Andrew Gelman.  The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo.  2011.](https://arxiv.org/pdf/1111.4246.pdf)
#'
#' @param target_log_prob_fn function which takes an argument like
#' `current_state` and returns its (possibly unnormalized) log-density under the target
#' distribution.
#' @param step_size `Tensor` or `list` of `Tensor`s representing the step
#' size for the leapfrog integrator. Must broadcast with the shape of
#' `current_state`. Larger step sizes lead to faster progress, but
#' too-large step sizes make rejection exponentially more likely. When
#' possible, it's often helpful to match per-variable step sizes to the
#' standard deviations of the target distribution in each variable.
#' @param max_tree_depth Maximum depth of the tree implicitly built by NUTS. The
#' maximum number of leapfrog steps is bounded by `2**max_tree_depth` i.e.
#' the number of nodes in a binary tree `max_tree_depth` nodes deep. The
#' default setting of 10 takes up to 1024 leapfrog steps.
#' @param max_energy_diff Scaler threshold of energy differences at each leapfrog,
#' divergence samples are defined as leapfrog steps that exceed this
#' threshold. Default to 1000.
#' @param unrolled_leapfrog_steps The number of leapfrogs to unroll per tree
#' expansion step. Applies a direct linear multipler to the maximum
#' trajectory length implied by max_tree_depth. Defaults to 1.
#' @param seed integer to seed the random number generator.
#' @param name name prefixed to Ops created by this function.
#' Default value: `NULL` (i.e., 'nuts_kernel').
#'
#' @inherit mcmc_hamiltonian_monte_carlo return params
#'
#' @family mcmc_kernels
#' @examples
#' \donttest{
#' predictors <- tf$cast( c(201,244, 47,287,203,58,210,202,198,158,165,201,157,
#'   131,166,160,186,125,218,146),tf$float32)
#' obs <- tf$cast(c(592,401,583,402,495,173,479,504,510,416,393,442,317,311,400,
#'   337,423,334,533,344),tf$float32)
#' y_sigma <- tf$cast(c(61,25,38,15,21,15,27,14,30,16,14,25,52,16,34,31,42,26,
#'   16,22),tf$float32)
#'
#' # Robust linear regression model
#' robust_lm <- tfd_joint_distribution_sequential(list(
#'   tfd_normal(loc = 0, scale = 1),
#'   # b0
#'   tfd_normal(loc = 0, scale = 1),
#'   # b1
#'   tfd_half_normal(5),
#'   # df
#'   function(df, b1, b0)
#'     tfd_independent(
#'       tfd_student_t(
#'         # Likelihood
#'         df = df[, NULL],
#'         loc = b0[, NULL] + b1[, NULL] * predictors[NULL,],
#'         scale = y_sigma[NULL,]
#'       )
#'    )
#'  ),  validate_args = TRUE)
#'
#'  log_prob <-function(b0, b1, df) {robust_lm %>%
#'    tfd_log_prob(list(b0, b1, df, obs))}
#'  step_size0 <- Map(function(x) tf$cast(x, tf$float32), c(1, .2, .5))
#'
#'  number_of_steps <- 100
#'  burnin <- 50
#'  nchain <- 50
#'
#'  run_chain <- function() {
#'  # random initialization of the starting postion of each chain
#'  samples <- robust_lm %>% tfd_sample(nchain)
#'  b0 <- samples[[1]]
#'  b1 <- samples[[2]]
#'  df <- samples[[3]]
#'
#'  # bijector to map contrained parameters to real
#'  unconstraining_bijectors <- list(
#'    tfb_identity(), tfb_identity(), tfb_identity(), tfb_exp())
#'
#'  trace_fn <- function(x, pkr) {
#'    list(pkr$inner_results$inner_results$step_size,
#'      pkr$inner_results$inner_results$log_accept_ratio)
#'  }
#'
#'  nuts <- mcmc_no_u_turn_sampler(
#'    target_log_prob_fn = log_prob,
#'    step_size = step_size0
#'    ) %>%
#'    mcmc_transformed_transition_kernel(bijector = unconstraining_bijectors) %>%
#'    mcmc_dual_averaging_step_size_adaptation(
#'      num_adaptation_steps = burnin,
#'      step_size_setter_fn = function(pkr, new_step_size)
#'        pkr$`_replace`(
#'          inner_results = pkr$inner_results$`_replace`(step_size = new_step_size)),
#'      step_size_getter_fn = function(pkr) pkr$inner_results$step_size,
#'      log_accept_prob_getter_fn = function(pkr) pkr$inner_results$log_accept_ratio
#'      )
#'
#'    nuts %>% mcmc_sample_chain(
#'      num_results = number_of_steps,
#'      num_burnin_steps = burnin,
#'      current_state = list(b0, b1, df),
#'      trace_fn = trace_fn)
#'    }
#'
#'    run_chain <- tensorflow::tf_function(run_chain)
#'    res <- run_chain()
#' }
#' @export
mcmc_no_u_turn_sampler <- function(target_log_prob_fn,
                                   step_size,
                                   max_tree_depth = 10,
                                   max_energy_diff = 1000,
                                   unrolled_leapfrog_steps = 1,
                                   seed = NULL,
                                   name = NULL) {
  args <- list(
    target_log_prob_fn = target_log_prob_fn,
    step_size = step_size,
    max_tree_depth = as.integer(max_tree_depth),
    max_energy_diff = max_energy_diff,
    unrolled_leapfrog_steps = as.integer(unrolled_leapfrog_steps),
    seed = seed,
    name = name
  )

  do.call(tfp$mcmc$NoUTurnSampler, args)
}

#' Runs one step of Uncalibrated Langevin discretized diffusion.
#'
#' The class generates a Langevin proposal using `_euler_method` function and
#' also computes helper `UncalibratedLangevinKernelResults` for the next
#' iteration.
#' Warning: this kernel will not result in a chain which converges to the
#' `target_log_prob`. To get a convergent MCMC, use
#' `MetropolisAdjustedLangevinAlgorithm(...)` or `MetropolisHastings(UncalibratedLangevin(...))`.
#'
#' @param volatility_fn function which takes an argument like
#' `current_state` (or `*current_state` if it's a list) and returns
#' volatility value at `current_state`. Should return a `Tensor` or
#' `list` of `Tensor`s that must broadcast with the shape of
#' `current_state`. Defaults to the identity function.
#' @param parallel_iterations the number of coordinates for which the gradients of
#' the volatility matrix `volatility_fn` can be computed in parallel.
#' @param compute_acceptance logical indicating whether to compute the
#' Metropolis log-acceptance ratio used to construct `MetropolisAdjustedLangevinAlgorithm` kernel.
#' @param name String prefixed to Ops created by this function.
#' Default value: `NULL` (i.e., 'mala_kernel').
#'
#' @return list of
#' `next_state` (Tensor or Python list of `Tensor`s representing the state(s)
#' of the Markov chain(s) at each result step. Has same shape as
#' and `current_state`.) and
#' `kernel_results` (`collections$namedtuple` of internal calculations used to
#' 'advance the chain).
#'
#' @inherit mcmc_hamiltonian_monte_carlo return params
#' @family mcmc_kernels
#' @export
mcmc_uncalibrated_langevin <- function(target_log_prob_fn,
                                       step_size,
                                       volatility_fn = NULL,
                                       parallel_iterations = 10,
                                       compute_acceptance = TRUE,
                                       seed = NULL,
                                       name = NULL) {
  args <- list(
    target_log_prob_fn = target_log_prob_fn,
    step_size = step_size,
    volatility_fn = volatility_fn,
    parallel_iterations = as.integer(parallel_iterations),
    compute_acceptance = compute_acceptance,
    seed = seed,
    name = name
  )

  do.call(tfp$mcmc$UncalibratedLangevin, args)
}

#' Runs one step of Metropolis-adjusted Langevin algorithm.
#'
#' Metropolis-adjusted Langevin algorithm (MALA) is a Markov chain Monte Carlo
#' (MCMC) algorithm that takes a step of a discretised Langevin diffusion as a
#' proposal. This class implements one step of MALA using Euler-Maruyama method
#' for a given `current_state` and diagonal preconditioning `volatility` matrix.
#'
#' Mathematical details and derivations can be found in
#' Roberts and Rosenthal (1998) and Xifara et al. (2013).
#'
#' The `one_step` function can update multiple chains in parallel. It assumes
#' that all leftmost dimensions of `current_state` index independent chain states
#' (and are therefore updated independently). The output of
#' `target_log_prob_fn(current_state)` should reduce log-probabilities across
#' all event dimensions. Slices along the rightmost dimensions may have different
#' target distributions; for example, `current_state[0, :]` could have a
#' different target distribution from `current_state[1, :]`. These semantics are
#' governed by `target_log_prob_fn(current_state)`. (The number of independent
#' chains is `tf.size(target_log_prob_fn(current_state))`.)
#'
#' @section References:
#' - [Gareth Roberts and Jeffrey Rosenthal. Optimal Scaling of Discrete Approximations to Langevin Diffusions. _Journal of the Royal Statistical Society: Series B (Statistical Methodology)_, 60: 255-268, 1998.](https://doi.org/10.1111/1467-9868.00123)
#' - [T. Xifara et al. Langevin diffusions and the Metropolis-adjusted Langevin algorithm. _arXiv preprint arXiv:1309.2983_, 2013.](https://arxiv.org/abs/1309.2983)
#'
#' @inheritParams mcmc_uncalibrated_langevin
#' @family mcmc_kernels
#' @export
mcmc_metropolis_adjusted_langevin_algorithm <-
  function(target_log_prob_fn,
           step_size,
           volatility_fn = NULL,
           seed = NULL,
           parallel_iterations = 10,
           name = NULL) {
    args <- list(
      target_log_prob_fn = target_log_prob_fn,
      step_size = step_size,
      volatility_fn = volatility_fn,
      parallel_iterations = as.integer(parallel_iterations),
      seed = seed,
      name = name
    )
    do.call(tfp$mcmc$MetropolisAdjustedLangevinAlgorithm, args)
  }

#' Runs one step of the Replica Exchange Monte Carlo
#'
#' [Replica Exchange Monte Carlo](https://en.wikipedia.org/wiki/Parallel_tempering)
#' is a Markov chain Monte Carlo (MCMC) algorithm that is also known as Parallel Tempering.
#' This algorithm performs multiple sampling with different temperatures in parallel,
#' and exchanges those samplings according to the Metropolis-Hastings criterion.
#' The `K` replicas are parameterized in terms of `inverse_temperature`'s,
#' `(beta[0], beta[1], ..., beta[K-1])`.  If the target distribution has
#' probability density `p(x)`, the `kth` replica has density `p(x)**beta_k`.
#'
#' Typically `beta[0] = 1.0`, and `1.0 > beta[1] > beta[2] > ... > 0.0`.
#'
#' * `beta[0] == 1` ==> First replicas samples from the target density, `p`.
#' * `beta[k] < 1`, for `k = 1, ..., K-1` ==> Other replicas sample from
#' "flattened" versions of `p` (peak is less high, valley less low).  These
#' distributions are somewhat closer to a uniform on the support of `p`.
#' Samples from adjacent replicas `i`, `i + 1` are used as proposals for each
#' other in a Metropolis step.  This allows the lower `beta` samples, which
#' explore less dense areas of `p`, to occasionally be used to help the
#' `beta == 1` chain explore new regions of the support.
#' Samples from replica 0 are returned, and the others are discarded.
#'
#' @param  inverse_temperatures `1D` Tensor of inverse temperatures to perform
#' samplings with each replica. Must have statically known `shape`.
#' `inverse_temperatures[0]` produces the states returned by samplers,
#' and is typically == 1.
#' @param make_kernel_fn Function which takes target_log_prob_fn and seed
#' args and returns a TransitionKernel instance.
#' @param swap_proposal_fn function which take a number of replicas, and
#' return combinations of replicas for exchange.
#' @param name string prefixed to Ops created by this function.
#' Default value: `NULL` (i.e., "remc_kernel").
#'
#' @return list of
#' `next_state` (Tensor or Python list of `Tensor`s representing the state(s)
#' of the Markov chain(s) at each result step. Has same shape as
#' and `current_state`.) and
#' `kernel_results` (`collections$namedtuple` of internal calculations used to
#' 'advance the chain).
#'
#' @inherit mcmc_hamiltonian_monte_carlo return params
#' @family mcmc_kernels
#' @export
mcmc_replica_exchange_mc <- function(target_log_prob_fn,
                                     inverse_temperatures,
                                     make_kernel_fn,
                                     swap_proposal_fn = tfp$mcmc$replica_exchange_mc$default_swap_proposal_fn(1),
                                     seed = NULL,
                                     name = NULL) {
  args <- list(
    target_log_prob_fn = target_log_prob_fn,
    inverse_temperatures = inverse_temperatures,
    make_kernel_fn = make_kernel_fn,
    swap_proposal_fn = swap_proposal_fn,
    seed = seed,
    name = name
  )
  do.call(tfp$mcmc$ReplicaExchangeMC, args)
}

#' Runs one step of the slice sampler using a hit and run approach
#'
#' Slice Sampling is a Markov Chain Monte Carlo (MCMC) algorithm based, as stated
#' by Neal (2003), on the observation that "...one can sample from a
#' distribution by sampling uniformly from the region under the plot of its
#' density function. A Markov chain that converges to this uniform distribution
#' can be constructed by alternately uniform sampling in the vertical direction
#' with uniform sampling from the horizontal `slice` defined by the current
#' vertical position, or more generally, with some update that leaves the uniform
#' distribution over this slice invariant". Mathematical details and derivations
#' can be found in Neal (2003). The one dimensional slice sampler is
#' extended to n-dimensions through use of a hit-and-run approach: choose a
#' random direction in n-dimensional space and take a step, as determined by the
#' one-dimensional slice sampling algorithm, along that direction
#' (Belisle at al. 1993).
#'
#' The `one_step` function can update multiple chains in parallel. It assumes
#' that all leftmost dimensions of `current_state` index independent chain states
#' (and are therefore updated independently). The output of
#' `target_log_prob_fn(*current_state)` should sum log-probabilities across all
#' event dimensions. Slices along the rightmost dimensions may have different
#' target distributions; for example, `current_state[0, :]` could have a
#' different target distribution from `current_state[1, :]`. These semantics are
#' governed by `target_log_prob_fn(*current_state)`. (The number of independent
#' chains is `tf$size(target_log_prob_fn(*current_state))`.)
#'
#' Note that the sampler only supports states where all components have a common
#' dtype.
#'
#' @param max_doublings Scalar positive int32 `tf$Tensor`. The maximum number of
#' doublings to consider.
#' @param name string prefixed to Ops created by this function.
#' Default value: `NULL` (i.e., 'slice_sampler_kernel').
#'
#' @return list of
#' `next_state` (Tensor or Python list of `Tensor`s representing the state(s)
#' of the Markov chain(s) at each result step. Has same shape as
#' and `current_state`.) and
#' `kernel_results` (`collections$namedtuple` of internal calculations used to
#' 'advance the chain).
#'
#' @section References:
#' - [Radford M. Neal. Slice Sampling. The Annals of Statistics. 2003, Vol 31, No. 3 , 705-767.](https://projecteuclid.org/download/pdf_1/euclid.aos/1056562461)
#' - C.J.P. Belisle, H.E. Romeijn, R.L. Smith. \cite{Hit-and-run algorithms for generating multivariate distributions. Math. Oper. Res., 18(1993), 225-266.}
#'
#' @inherit mcmc_hamiltonian_monte_carlo return params
#' @family mcmc_kernels
#' @export
mcmc_slice_sampler <- function(target_log_prob_fn,
                               step_size,
                               max_doublings,
                               seed = NULL,
                               name = NULL) {
  args <- list(
    target_log_prob_fn = target_log_prob_fn,
    step_size = step_size,
    max_doublings = as.integer(max_doublings),
    seed = seed,
    name = name
  )
  do.call(tfp$mcmc$SliceSampler, args)
}

#' Generate proposal for the Random Walk Metropolis algorithm.
#'
#' Warning: this kernel will not result in a chain which converges to the
#' `target_log_prob`. To get a convergent MCMC, use
#' `mcmc_random_walk_metropolis(...)` or
#' `mcmc_metropolis_hastings(mcmc_uncalibrated_random_walk(...))`.
#'
#' @inherit mcmc_random_walk_metropolis return params
#' @family mcmc_kernels
#' @export
mcmc_uncalibrated_random_walk <- function(target_log_prob_fn,
                                        new_state_fn = NULL,
                                        seed = NULL,
                                        name = NULL) {
  args <- list(
    target_log_prob_fn = target_log_prob_fn,
    new_state_fn = new_state_fn,
    seed = seed,
    name = name
  )

  do.call(tfp$mcmc$UncalibratedRandomWalk, args)
}

