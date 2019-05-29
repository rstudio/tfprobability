
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


