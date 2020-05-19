## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  eval = FALSE
)

## -----------------------------------------------------------------------------
#  # assume it's version 1.14, with eager not yet being the default
#  library(tensorflow)
#  tf$enable_v2_behavior()
#  
#  library(tfprobability)
#  library(rethinking)
#  library(zeallot)
#  library(purrr)
#  
#  data("reedfrogs")
#  d <- reedfrogs
#  str(d)

## -----------------------------------------------------------------------------
#  n_tadpole_tanks <- nrow(d)
#  n_surviving <- d$surv
#  n_start <- d$density
#  
#  model <- tfd_joint_distribution_sequential(
#    list(
#      # a_bar, the prior for the mean of the normal distribution of per-tank logits
#      tfd_normal(loc = 0, scale = 1.5),
#      # sigma, the prior for the variance of the normal distribution of per-tank logits
#      tfd_exponential(rate = 1),
#      # normal distribution of per-tank logits
#      # parameters sigma and a_bar refer to the outputs of the above two distributions
#      function(sigma, a_bar)
#        tfd_sample_distribution(
#          tfd_normal(loc = a_bar, scale = sigma),
#          sample_shape = list(n_tadpole_tanks)
#        ),
#      # binomial distribution of survival counts
#      # parameter l refers to the output of the normal distribution immediately above
#      function(l)
#        tfd_independent(
#          tfd_binomial(total_count = n_start, logits = l),
#          reinterpreted_batch_ndims = 1
#        )
#    )
#  )

## -----------------------------------------------------------------------------
#  s <- model %>% tfd_sample(2)
#  s

## -----------------------------------------------------------------------------
#  model %>% tfd_log_prob(s)

## -----------------------------------------------------------------------------
#  logprob <- function(a, s, l)
#    model %>% tfd_log_prob(list(a, s, l, n_surviving))

## -----------------------------------------------------------------------------
#  # number of steps after burnin
#  n_steps <- 500
#  # number of chains
#  n_chain <- 4
#  # number of burnin steps
#  n_burnin <- 500
#  
#  hmc <- mcmc_hamiltonian_monte_carlo(
#    target_log_prob_fn = logprob,
#    num_leapfrog_steps = 3,
#    # one step size for each parameter
#    step_size = list(0.1, 0.1, 0.1),
#  ) %>%
#    mcmc_simple_step_size_adaptation(target_accept_prob = 0.8,
#                                     num_adaptation_steps = n_burnin)
#  

## -----------------------------------------------------------------------------
#  # initial values to start the sampler
#  c(initial_a, initial_s, initial_logits, .) %<-% (model %>% tfd_sample(n_chain))
#  
#  # optionally retrieve metadata such as acceptance ratio and step size
#  trace_fn <- function(state, pkr) {
#    list(pkr$inner_results$is_accepted,
#         pkr$inner_results$accepted_results$step_size)
#  }
#  
#  run_mcmc <- function(kernel) {
#    kernel %>% mcmc_sample_chain(
#      num_results = n_steps,
#      num_burnin_steps = n_burnin,
#      current_state = list(initial_a, tf$ones_like(initial_s), initial_logits),
#      trace_fn = trace_fn
#    )
#  }
#  
#  run_mcmc <- tf_function(run_mcmc)
#  res <- run_mcmc(hmc)

## -----------------------------------------------------------------------------
#  mcmc_trace <- res$all_states

## -----------------------------------------------------------------------------
#  map(mcmc_trace, ~ compose(dim, as.array)(.x))

## -----------------------------------------------------------------------------
#  mcmc_potential_scale_reduction(mcmc_trace)
#  mcmc_effective_sample_size(mcmc_trace)

