#' Implements Markov chain Monte Carlo via repeated `TransitionKernel` steps.
#'
#' This function samples from an Markov chain at `current_state` and whose
#' stationary distribution is governed by the supplied `TransitionKernel`
#' instance (`kernel`).
#'
#' This function can sample from multiple chains, in parallel. (Whether or not
#' there are multiple chains is dictated by the `kernel`.)
#'
#' The `current_state` can be represented as a single `Tensor` or a `list` of
#' `Tensors` which collectively represent the current state.
#' Since MCMC states are correlated, it is sometimes desirable to produce
#' additional intermediate states, and then discard them, ending up with a set of
#' states with decreased autocorrelation.  See Owen (2017). Such "thinning"
#' is made possible by setting `num_steps_between_results > 0`. The chain then
#' takes `num_steps_between_results` extra steps between the steps that make it
#' into the results. The extra steps are never materialized (in calls to
#' `sess$run`), and thus do not increase memory requirements.
#'
#' Warning: when setting a `seed` in the `kernel`, ensure that `sample_chain`'s
#' `parallel_iterations=1`, otherwise results will not be reproducible.
#' In addition to returning the chain state, this function supports tracing of
#' auxiliary variables used by the kernel. The traced values are selected by
#' specifying `trace_fn`. By default, all kernel results are traced but in the
#' future the default will be changed to no results being traced, so plan
#' accordingly. See below for some examples of this feature.
#'
#' @section References:
#' - [Art B. Owen. Statistically efficient thinning of a Markov chain sampler. _Technical Report_, 2017.](http://statweb.stanford.edu/~owen/reports/bestthinning.pdf)
#'
#' @param kernel An instance of `tfp$mcmc$TransitionKernel` which implements one step
#' of the Markov chain.
#' @param num_results Integer number of Markov chain draws.
#' @param current_state `Tensor` or `list` of `Tensor`s representing the
#' current state(s) of the Markov chain(s).
#' @param previous_kernel_results A `Tensor` or a nested collection of `Tensor`s
#' representing internal calculations made within the previous call to this
#' function (or as returned by `bootstrap_results`).
#' @param num_burnin_steps Integer number of chain steps to take before starting to
#' collect results. Default value: 0 (i.e., no burn-in).
#' @param num_steps_between_results Integer number of chain steps between collecting
#' a result. Only one out of every `num_steps_between_samples + 1` steps is
#' included in the returned results.  The number of returned chain states is
#' still equal to `num_results`.  Default value: 0 (i.e., no thinning).
#' @param trace_fn A function that takes in the current chain state and the previous
#' kernel results and return a `Tensor` or a nested collection of `Tensor`s
#' that is then traced along with the chain state.
#' @param return_final_kernel_results If `TRUE`, then the final kernel results are
#' returned alongside the chain state and the trace specified by the `trace_fn`.
#' @param parallel_iterations The number of iterations allowed to run in parallel. It
#' must be a positive integer. See `tf$while_loop` for more details.
#' @param name string prefixed to Ops created by this function. Default value: `NULL`,
#' (i.e., "mcmc_sample_chain").
#'
#' @family mcmc_functions
#' @export
mcmc_sample_chain <- function(kernel = NULL,
                              num_results,
                              current_state,
                              previous_kernel_results = NULL,
                              num_burnin_steps = 0,
                              num_steps_between_results = 0,
                              trace_fn = NULL,
                              return_final_kernel_results = FALSE,
                              parallel_iterations = 10,
                              name = NULL) {
  # need to make sure we keep trace_fn here, even if NULL, so no do.call
  if (tfp_version() >= "0.7") {
    tfp$mcmc$sample_chain(
      num_results = as.integer(num_results),
      current_state = as_tensor(current_state),
      previous_kernel_results = previous_kernel_results,
      kernel = kernel,
      num_burnin_steps = as.integer(num_burnin_steps),
      num_steps_between_results = as.integer(num_steps_between_results),
      return_final_kernel_results = return_final_kernel_results,
      trace_fn = trace_fn,
      parallel_iterations = as.integer(parallel_iterations),
      name = name
    )
  } else {
    tfp$mcmc$sample_chain(
      num_results = as.integer(num_results),
      current_state = tf$convert_to_tensor(current_state),
      previous_kernel_results = previous_kernel_results,
      kernel = kernel,
      num_burnin_steps = as.integer(num_burnin_steps),
      num_steps_between_results = as.integer(num_steps_between_results),
      parallel_iterations = as.integer(parallel_iterations),
      name = name
    )
  }
}

#' Estimate a lower bound on effective sample size for each independent chain.
#'
#' Roughly speaking, "effective sample size" (ESS) is the size of an iid sample
#' with the same variance as `state`.
#'
#' More precisely, given a stationary sequence of possibly correlated random
#' variables `X_1, X_2,...,X_N`, each identically distributed ESS is the number
#' such that
#' ```Variance{ N**-1 * Sum{X_i} } = ESS**-1 * Variance{ X_1 }.```
#'
#' If the sequence is uncorrelated, `ESS = N`.  In general, one should expect
#' `ESS <= N`, with more highly correlated sequences having smaller `ESS`.
#'
#' @param states  `Tensor` or list of `Tensor` objects.  Dimension zero should index
#' identically distributed states.
#' @param filter_threshold  `Tensor` or list of `Tensor` objects.
#' Must broadcast with `state`.  The auto-correlation sequence is truncated
#' after the first appearance of a term less than `filter_threshold`.
#' Setting to `NULL` means we use no threshold filter.  Since `|R_k| <= 1`,
#' setting to any number less than `-1` has the same effect.
#' @param filter_beyond_lag  `Tensor` or list of `Tensor` objects.  Must be
#' `int`-like and scalar valued.  The auto-correlation sequence is truncated
#' to this length.  Setting to `NULL` means we do not filter based on number of lags.
#' @param name name to prepend to created ops.
#'
#' @family mcmc_functions
#' @export
mcmc_effective_sample_size <- function(states,
                              filter_threshold = 0,
                              filter_beyond_lag = NULL,
                              name = NULL) {
 tfp$mcmc$effective_sample_size(
      states,
      filter_threshold,
      as_nullable_integer(filter_beyond_lag),
      name
    )
}

#' Gelman and Rubin (1992)'s potential scale reduction for chain convergence.
#'
#' Given `N > 1` states from each of `C > 1` independent chains, the potential
#' scale reduction factor, commonly referred to as R-hat, measures convergence of
#' the chains (to the same target) by testing for equality of means.
#'
#' Specifically, R-hat measures the degree to which variance (of the means)
#' between chains exceeds what one would expect if the chains were identically
#' distributed. See Gelman and Rubin (1992), Brooks and Gelman (1998)].
#'
#' Some guidelines:
#' * The initial state of the chains should be drawn from a distribution overdispersed with respect to the target.
#' * If all chains converge to the target, then as `N --> infinity`, R-hat --> 1.
#'   Before that, R-hat > 1 (except in pathological cases, e.g. if the chain paths were identical).
#' * The above holds for any number of chains `C > 1`.  Increasing `C` improves effectiveness of the diagnostic.
#' * Sometimes, R-hat < 1.2 is used to indicate approximate convergence, but of
#' course this is problem dependent. See Brooks and Gelman (1998).
#' * R-hat only measures non-convergence of the mean. If higher moments, or
#' other statistics are desired, a different diagnostic should be used. See Brooks and Gelman (1998).
#'
#' To see why R-hat is reasonable, let `X` be a random variable drawn uniformly
#' from the combined states (combined over all chains).  Then, in the limit
#' `N, C --> infinity`, with `E`, `Var` denoting expectation and variance,
#' ```R-hat = ( E[Var[X | chain]] + Var[E[X | chain]] ) / E[Var[X | chain]].```
#' Using the law of total variance, the numerator is the variance of the combined
#' states, and the denominator is the total variance minus the variance of the
#' the individual chain means.  If the chains are all drawing from the same
#' distribution, they will have the same mean, and thus the ratio should be one.
#'
#' @section References:
#' - Stephen P. Brooks and Andrew Gelman. General Methods for Monitoring Convergence of Iterative Simulations.
#'  _Journal of Computational and Graphical Statistics_, 7(4), 1998.
#' - Andrew Gelman and Donald B. Rubin. Inference from Iterative Simulation Using Multiple Sequences.
#'  _Statistical Science_, 7(4):457-472, 1992.
#' @param chains_states  `Tensor` or `list` of `Tensor`s representing the
#' state(s) of a Markov Chain at each result step.  The `ith` state is
#' assumed to have shape `[Ni, Ci1, Ci2,...,CiD] + A`.
#' Dimension `0` indexes the `Ni > 1` result steps of the Markov Chain.
#' Dimensions `1` through `D` index the `Ci1 x ... x CiD` independent
#' chains to be tested for convergence to the same target.
#' The remaining dimensions, `A`, can have any shape (even empty).
#' @param independent_chain_ndims Integer type `Tensor` with value `>= 1` giving the
#' number of giving the number of dimensions, from `dim = 1` to `dim = D`,
#' holding independent chain results to be tested for convergence.
#' @param name name to prepend to created tf.  Default: `potential_scale_reduction`.
#' @family mcmc_functions
#' @export
mcmc_potential_scale_reduction <- function(chains_states,
                                           independent_chain_ndims = 1,
                                           name = NULL) {
  tfp$mcmc$potential_scale_reduction(
    chains_states,
    as.integer(independent_chain_ndims),
    name
  )
}
