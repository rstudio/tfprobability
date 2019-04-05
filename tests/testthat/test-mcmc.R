context("mcmc")

source("utils.R")

test_succeeds("sampling from chain works", {

  dims <- 10
  true_stddev <- sqrt(seq(1, 3, length.out = dims))
  likelihood <- tfd_multivariate_normal_diag(loc = 0, scale_diag = true_stddev)

  kernel <- mcmc_hamiltonian_monte_carlo(
    target_log_prob_fn = likelihood$log_prob,
    step_size = 0.5,
    num_leapfrog_steps = 2
  )
  states <- kernel %>% mcmc_sample_chain(
    num_results = 1000,
    num_burnin_steps = 500,
    current_state = tf$zeros(dims),
    trace_fn = NULL)

  sample_mean <- tf$reduce_mean(states, axis = 0L)
  sample_stddev <- tf$sqrt(tf$reduce_mean(tf$squared_difference(states, sample_mean), axis = 0L))

  expect_equal(sample_stddev %>% tensor_value() %>% mean(), mean(true_stddev), tol = 1e-1)

})

test_succeeds("HamiltonianMonteCarlo with SimpleStepSizeAdaptation works", {

  target_log_prob_fn <- tfd_normal(loc = 0, scale = 1)$log_prob
  num_burnin_steps <- 500
  num_results <- 500
  num_chains <- 64L
  step_size <- 0.1
  #step_size <- tf$fill(list(num_chains), 0.1)

  kernel <- mcmc_hamiltonian_monte_carlo(
     target_log_prob_fn = target_log_prob_fn,
     num_leapfrog_steps = 2,
     step_size = step_size) %>%
    mcmc_simple_step_size_adaptation(
      num_adaptation_steps = round(num_burnin_steps * 0.8))

    # _hmc_like_step_size_setter_fn() missing 1 required positional argument: 'new_step_size'
    zeallot::`%<-%`(c(samples, c(step_size, log_accept_ratio)), kernel %>% mcmc_sample_chain(
      num_results = num_results,
      num_burnin_steps = num_burnin_steps,
      current_state = tf$zeros(num_chains),
      trace_fn = function(x, pkr) {list (pkr$inner_results$accepted_results$step_size,
                                         pkr$inner_results$log_accept_ratio)}))


  # # ~0.75
  # p_accept = tf.reduce_mean(tf.exp(tf.minimum(log_accept_ratio, 0.)))


})
