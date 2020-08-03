#' Fit a surrogate posterior to a target (unnormalized) log density
#'
#' The default behavior constructs and minimizes the negative variational
#' evidence lower bound (ELBO), given by
#' ```
#' q_samples <- surrogate_posterior$sample(num_draws)
#' elbo_loss <- -tf$reduce_mean(target_log_prob_fn(q_samples) - surrogate_posterior$log_prob(q_samples))
#' ```
#'
#' This corresponds to minimizing the 'reverse' Kullback-Liebler divergence
#' (`KL[q||p]`) between the variational distribution and the unnormalized
#' `target_log_prob_fn`, and  defines a lower bound on the marginal log
#' likelihood, `log p(x) >= -elbo_loss`.
#'
#' More generally, this function supports fitting variational distributions that
#' minimize any [Csiszar f-divergence](https://en.wikipedia.org/wiki/F-divergence).
#'
#' @param target_log_prob_fn function that takes a set of `Tensor` arguments
#' and returns a `Tensor` log-density. Given
#' `q_sample <- surrogate_posterior$sample(sample_size)`, this
#' will be (in Python) called as `target_log_prob_fn(q_sample)` if `q_sample` is a list
#' or a tuple, `target_log_prob_fn(**q_sample)` if `q_sample` is a
#' dictionary, or `target_log_prob_fn(q_sample)` if `q_sample` is a `Tensor`.
#' It should support batched evaluation, i.e., should return a result of
#' shape `[sample_size]`.
#' @param surrogate_posterior A `tfp$distributions$Distribution`
#' instance defining a variational posterior (could be a
#' `tfp$distributions$JointDistribution`). Crucially, the distribution's `log_prob` and
#' (if reparameterized) `sample` methods must directly invoke all ops
#' that generate gradients to the underlying variables. One way to ensure
#' this is to use `tfp$util$DeferredTensor` to represent any parameters
#' defined as transformations of unconstrained variables, so that the
#' transformations execute at runtime instead of at distribution creation.
#' @param optimizer Optimizer instance to use. This may be a TF1-style
#' `tf$train$Optimizer`, TF2-style `tf$optimizers$Optimizer`, or any Python-compatible
#' object that implements `optimizer$apply_gradients(grads_and_vars)`.
#' @param num_steps `integer` number of steps to run the optimizer.
#' @param convergence_criterion Optional instance of
#' `tfp$optimizer$convergence_criteria$ConvergenceCriterion`
#' representing a criterion for detecting convergence. If `NULL`,
#' the optimization will run for `num_steps` steps, otherwise, it will run
#' for at *most* `num_steps` steps, as determined by the provided criterion.
#' Default value: `NULL`.
#' @param trace_fn function with signature `state = trace_fn(loss, grads, variables)`,
#' where `state` may be a `Tensor` or nested structure of `Tensor`s.
#' The state values are accumulated (by `tf$scan`)
#' and returned. The default `trace_fn` simply returns the loss, but in
#' general can depend on the gradients and variables (if
#' `trainable_variables` is not `NULL` then `variables==trainable_variables`;
#' otherwise it is the list of all variables accessed during execution of
#' `loss_fn()`), as well as any other quantities captured in the closure of
#' `trace_fn`, for example, statistics of a variational distribution.
#' Default value: `function(loss, grads, variables) loss`.
#' @param variational_loss_fn function with signature
#' `loss <- variational_loss_fn(target_log_prob_fn, surrogate_posterior, sample_size, seed)`
#' defining a variational loss function. The default is
#' a Monte Carlo approximation to the standard evidence lower bound (ELBO),
#' equivalent to minimizing the 'reverse' `KL[q||p]` divergence between the
#' surrogate `q` and true posterior `p`.
#' Default value: `functools.partial(tfp.vi.monte_carlo_variational_loss, discrepancy_fn=tfp.vi.kl_reverse, use_reparameterization=True)`.
#' @param sample_size `integer` number of Monte Carlo samples to use
#' in estimating the variational divergence. Larger values may stabilize
#' the optimization, but at higher cost per step in time and memory.
#' Default value: `1`.
#' @param trainable_variables Optional list of `tf$Variable` instances to optimize
#' with respect to. If `NULL`, defaults to the set of all variables accessed
#' during the computation of the variational bound, i.e., those defining
#' `surrogate_posterior` and the model `target_log_prob_fn`. Default value: `NULL`.
#' @param seed integer to seed the random number generator.
#' @param name name prefixed to ops created by this function. Default value: 'fit_surrogate_posterior'.
#'
#' @return results `Tensor` or nested structure of `Tensor`s, according to the
#' return type of `result_fn`. Each `Tensor` has an added leading dimension
#' of size `num_steps`, packing the trajectory of the result over the course of the optimization.
#'
#' @family vi-functions
#'
#' @export
vi_fit_surrogate_posterior <-
  function(target_log_prob_fn,
           surrogate_posterior,
           optimizer,
           num_steps,
           convergence_criterion = NULL,
           trace_fn = tfp$vi$optimization$`_trace_loss`,
           variational_loss_fn = tfp$vi$optimization$`_reparameterized_elbo`,
           sample_size = 1,
           trainable_variables = NULL,
           seed = NULL,
           name = "fit_surrogate_posterior") {

    args <- list(
      target_log_prob_fn = target_log_prob_fn,
      surrogate_posterior = surrogate_posterior,
      optimizer = optimizer,
      num_steps = as.integer(num_steps),
      trace_fn = trace_fn,
      variational_loss_fn = variational_loss_fn,
      sample_size = as.integer(sample_size),
      trainable_variables = trainable_variables,
      seed = as_nullable_integer(seed),
      name = name
    )

    if (tfp_version() > "0.10") args$convergence_criterion = convergence_criterion

    do.call(tfp$vi$fit_surrogate_posterior, args)
  }
