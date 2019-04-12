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
      current_state = current_state,
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
      current_state = current_state,
      previous_kernel_results = previous_kernel_results,
      kernel = kernel,
      num_burnin_steps = as.integer(num_burnin_steps),
      num_steps_between_results = as.integer(num_steps_between_results),
      parallel_iterations = as.integer(parallel_iterations),
      name = name
    )
  }
}
