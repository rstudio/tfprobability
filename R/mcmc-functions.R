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
#' @return list of:
#' - checkpointable_states_and_trace: if `return_final_kernel_results` is
#' `TRUE`. The return value is an instance of `CheckpointableStatesAndTrace`.
#' - all_states: if `return_final_kernel_results` is `FALSE` and `trace_fn` is
#' `NULL`. The return value is a `Tensor` or Python list of `Tensor`s
#' representing the state(s) of the Markov chain(s) at each result step. Has
#' same shape as input `current_state` but with a prepended
#' `num_results`-size dimension.
#' - states_and_trace: if `return_final_kernel_results` is `FALSE` and
#' `trace_fn` is not `NULL`. The return value is an instance of
#' `StatesAndTrace`.
#'
#' @family mcmc_functions
#' @examples
#' \donttest{
#'   dims <- 10
#'   true_stddev <- sqrt(seq(1, 3, length.out = dims))
#'   likelihood <- tfd_multivariate_normal_diag(scale_diag = true_stddev)
#'
#'   kernel <- mcmc_hamiltonian_monte_carlo(
#'     target_log_prob_fn = likelihood$log_prob,
#'     step_size = 0.5,
#'     num_leapfrog_steps = 2
#'   )
#'
#'   states <- kernel %>% mcmc_sample_chain(
#'     num_results = 1000,
#'     num_burnin_steps = 500,
#'     current_state = rep(0, dims),
#'     trace_fn = NULL
#'   )
#'
#'   sample_mean <- tf$reduce_mean(states, axis = 0L)
#'   sample_stddev <- tf$sqrt(
#'     tf$reduce_mean(tf$math$squared_difference(states, sample_mean), axis = 0L))
#' }
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
#' @return `Tensor` or list of `Tensor` objects.  The effective sample size of
#' each component of `states`.  Shape will be `states$shape[1:]`.
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
#' @return `Tensor` or `list` of `Tensor`s representing the R-hat statistic for
#' the state(s).  Same `dtype` as `state`, and shape equal to
#' `state$shape[1 + independent_chain_ndims:]`.
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

#' Runs annealed importance sampling (AIS) to estimate normalizing constants.
#'
#' This function uses an MCMC transition operator (e.g., Hamiltonian Monte Carlo)
#' to sample from a series of distributions that slowly interpolates between
#' an initial "proposal" distribution:
#' `exp(proposal_log_prob_fn(x) - proposal_log_normalizer)`
#' and the target distribution:
#' `exp(target_log_prob_fn(x) - target_log_normalizer)`,
#' accumulating importance weights along the way. The product of these
#' importance weights gives an unbiased estimate of the ratio of the
#' normalizing constants of the initial distribution and the target
#' distribution:
#' `E[exp(ais_weights)] = exp(target_log_normalizer - proposal_log_normalizer)`.
#'
#' Note: When running in graph mode, `proposal_log_prob_fn` and
#' `target_log_prob_fn` are called exactly three times (although this may be
#' reduced to two times in the future).
#'
#' @param num_steps Integer number of Markov chain updates to run. More
#' iterations means more expense, but smoother annealing between q
#' and p, which in turn means exponentially lower variance for the
#' normalizing constant estimator.
#' @param proposal_log_prob_fn function that returns the log density of the
#' initial distribution.
#' @param target_log_prob_fn function which takes an argument like
#' `current_state` and returns its
#' (possibly unnormalized) log-density under the target distribution.
#' @param current_state `Tensor` or `list` of `Tensor`s representing the
#' current state(s) of the Markov chain(s). The first `r` dimensions index
#' independent chains, `r` = `tf$rank(target_log_prob_fn(current_state))`.
#' @param make_kernel_fn function which returns a `TransitionKernel`-like
#' object. Must take one argument representing the `TransitionKernel`'s
#' `target_log_prob_fn`. The `target_log_prob_fn` argument represents the
#' `TransitionKernel`'s target log distribution.  Note:
#' `sample_annealed_importance_chain` creates a new `target_log_prob_fn`
#' which is an interpolation between the supplied `target_log_prob_fn` and
#' `proposal_log_prob_fn`; it is this interpolated function which is used as
#' an argument to `make_kernel_fn`.
#' @param parallel_iterations The number of iterations allowed to run in parallel.
#' It must be a positive integer. See `tf$while_loop` for more details.
#' @param name string prefixed to Ops created by this function.
#' Default value: `NULL` (i.e., "sample_annealed_importance_chain").
#'
#' @return  list of
#' `next_state` (`Tensor` or Python list of `Tensor`s representing the
#' state(s) of the Markov chain(s) at the final iteration. Has same shape as
#' input `current_state`),
#' `ais_weights` (Tensor with the estimated weight(s). Has shape matching
#' `target_log_prob_fn(current_state)`), and
#' `kernel_results` (`collections.namedtuple` of internal calculations used to
#' advance the chain).
#'
#' @family mcmc_functions
#' @seealso For an example how to use see [mcmc_sample_chain()].

#' @export
mcmc_sample_annealed_importance_chain <- function(num_steps,
                                                  proposal_log_prob_fn,
                                                  target_log_prob_fn,
                                                  current_state,
                                                  make_kernel_fn,
                                                  parallel_iterations = 10,
                                                  name = NULL) {
  tfp$mcmc$sample_annealed_importance_chain(
    num_steps = as.integer(num_steps),
    proposal_log_prob_fn = proposal_log_prob_fn,
    target_log_prob_fn = target_log_prob_fn,
    current_state = current_state,
    make_kernel_fn = make_kernel_fn,
    parallel_iterations = as.integer(parallel_iterations),
    name = name
  )
}

#' Returns a sample from the `dim` dimensional Halton sequence.
#'
#' Warning: The sequence elements take values only between 0 and 1. Care must be
#' taken to appropriately transform the domain of a function if it differs from
#' the unit cube before evaluating integrals using Halton samples. It is also
#' important to remember that quasi-random numbers without randomization are not
#' a replacement for pseudo-random numbers in every context. Quasi random numbers
#' are completely deterministic and typically have significant negative
#' autocorrelation unless randomization is used.
#'
#' Computes the members of the low discrepancy Halton sequence in dimension
#' `dim`. The `dim`-dimensional sequence takes values in the unit hypercube in
#' `dim` dimensions. Currently, only dimensions up to 1000 are supported. The
#' prime base for the k-th axes is the k-th prime starting from 2. For example,
#' if `dim` = 3, then the bases will be `[2, 3, 5]` respectively and the first
#' element of the non-randomized sequence will be: `[0.5, 0.333, 0.2]`. For a more
#' complete description of the Halton sequences see
#' [here](https://en.wikipedia.org/wiki/Halton_sequence). For low discrepancy
#' sequences and their applications see
#' [here](https://en.wikipedia.org/wiki/Low-discrepancy_sequence).
#'
#' If `randomized` is true, this function produces a scrambled version of the
#' Halton sequence introduced by Owen (2017). For the advantages of
#' randomization of low discrepancy sequences see
#' [here](https://en.wikipedia.org/wiki/Quasi-Monte_Carlo_method#Randomization_of_quasi-Monte_Carlo).
#'
#' The number of samples produced is controlled by the `num_results` and
#' `sequence_indices` parameters. The user must supply either `num_results` or
#' `sequence_indices` but not both.
#' The former is the number of samples to produce starting from the first
#' element. If `sequence_indices` is given instead, the specified elements of
#' the sequence are generated. For example, sequence_indices=tf$range(10) is
#' equivalent to specifying n=10.
#'
#' @param dim Positive `integer` representing each sample's `event_size.` Must
#' not be greater than 1000.
#' @param num_results (Optional) Positive scalar `Tensor` of dtype int32. The number
#' of samples to generate. Either this parameter or sequence_indices must
#' be specified but not both. If this parameter is None, then the behaviour
#' is determined by the `sequence_indices`. Default value: `NULL`.
#' @param sequence_indices (Optional) `Tensor` of dtype int32 and rank 1. The
#' elements of the sequence to compute specified by their position in the
#' sequence. The entries index into the Halton sequence starting with 0 and
#' hence, must be whole numbers. For example, sequence_indices=`[0, 5, 6]` will
#' produce the first, sixth and seventh elements of the sequence. If this
#' parameter is None, then the `num_results` parameter must be specified
#' which gives the number of desired samples starting from the first sample.
#' Default value: `NULL`.
#' @param dtype (Optional) The dtype of the sample. One of: `float16`, `float32` or
#' `float64`. Default value: `tf$float32`.
#' @param randomized (Optional) bool indicating whether to produce a randomized
#' Halton sequence. If TRUE, applies the randomization described in
#' Owen (2017). Default value: `TRUE`.
#' @param seed (Optional) integer to seed the random number generator. Only
#' used if `randomized` is TRUE. If not supplied and `randomized` is TRUE,
#' no seed is set. Default value: `NULL`.
#' @param name  (Optional) string describing ops managed by this function. If
#' not supplied the name of this function is used. Default value: "sample_halton_sequence".
#'
#' @return halton_elements Elements of the Halton sequence. `Tensor` of supplied dtype
#' and `shape` `[num_results, dim]` if `num_results` was specified or shape
#' `[s, dim]` where s is the size of `sequence_indices` if `sequence_indices`
#' were specified.
#'
#' @section References:
#' - [Art B. Owen. A randomized Halton algorithm in R. _arXiv preprint arXiv:1706.02808_, 2017.](https://arxiv.org/abs/1706.02808)
#' @family mcmc_functions
#' @seealso For an example how to use see [mcmc_sample_chain()].
#' @export
mcmc_sample_halton_sequence <- function(dim,
                                        num_results = NULL,
                                        sequence_indices = NULL,
                                        dtype = tf$float32,
                                        randomized = TRUE,
                                        seed = NULL,
                                        name = NULL) {

  args <- list(
    dim = as.integer(dim),
    dtype = dtype,
    randomized = randomized,
    seed = seed,
    name = name
  )

  if (!is.null(num_results)) args$num_results <- as.integer(num_results)
  if (!is.null(sequence_indices)) args$sequence_indices <- as_integer_list(sequence_indices)

  do.call(tfp$mcmc$sample_halton_sequence, args)
}

