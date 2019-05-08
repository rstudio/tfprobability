context("mcmc")

source("utils.R")

test_succeeds("sampling from chain works", {

  dims <- 10
  true_stddev <- sqrt(seq(1, 3, length.out = dims))
  likelihood <- tfd_multivariate_normal_diag(scale_diag = true_stddev)

  kernel <- mcmc_hamiltonian_monte_carlo(
    target_log_prob_fn = likelihood$log_prob,
    step_size = 0.5,
    num_leapfrog_steps = 2
  )

  states_and_results <- kernel %>% mcmc_sample_chain(
    num_results = 1000,
    num_burnin_steps = 500,
    current_state = rep(0, dims),
    trace_fn = NULL)

  if(tfp_version() < "0.7") {
    states <- states_and_results[[1]]
  } else {
    states <- states_and_results
  }

  sample_mean <- tf$reduce_mean(states, axis = 0L)
  sample_stddev <- tf$sqrt(tf$reduce_mean(tf$squared_difference(states, sample_mean), axis = 0L))

  expect_equal(sample_stddev %>% tensor_value() %>% mean(), mean(true_stddev), tol = 1e-1)
})

test_succeeds("HamiltonianMonteCarlo with SimpleStepSizeAdaptation works", {

  skip_if_tfp_below("0.7")

  target_log_prob_fn <- tfd_normal(loc = 0, scale = 1)$log_prob
  num_burnin_steps <- 500
  num_results <- 500
  num_chains <- 64L
  step_size <- tf$fill(list(num_chains), 0.1)

  kernel <- mcmc_hamiltonian_monte_carlo(
     target_log_prob_fn = target_log_prob_fn,
     num_leapfrog_steps = 2,
     step_size = step_size) %>%
    mcmc_simple_step_size_adaptation(
      num_adaptation_steps = round(num_burnin_steps * 0.8))

   res <- kernel %>% mcmc_sample_chain(
      num_results = num_results,
      num_burnin_steps = num_burnin_steps,
      current_state = rep(0, num_chains),
      trace_fn = function(x, pkr) {list (pkr$inner_results$accepted_results$step_size,
                                         pkr$inner_results$log_accept_ratio)})

  # check for nicer unpacking
  # python: samples, [step_size, log_accept_ratio]

  samples <- res$all_states
  step_size <- res$trace[[1]]
  log_accept_ratio <- res$trace[[2]]
  #  ~0.75
  p_accept <- tf$reduce_mean(tf$exp(tf$minimum(log_accept_ratio, 0)))
  expect_lt(p_accept %>% tensor_value(), 1)

})

test_succeeds("MetropolisHastings works", {

  kernel <- mcmc_metropolis_hastings(
    mcmc_uncalibrated_hamiltonian_monte_carlo(
      target_log_prob_fn = function(x) -x - x^2,
      step_size = 0.1,
      num_leapfrog_steps = 3))

  states_and_results <- kernel %>% mcmc_sample_chain(
    num_results = 100,
    current_state = 1
    )

  if(tfp_version() < "0.7") {
    states <- states_and_results[[1]]
  } else {
    states <- states_and_results
  }

  expect_equal(states$get_shape() %>% length(), 2)
})

test_succeeds("RandomWalkMetropolis works", {

  kernel <- mcmc_random_walk_metropolis(
    target_log_prob_fn = function(x)
      - x - x ^ 2
  )

  states_and_results <- kernel %>% mcmc_sample_chain(num_results = 100,
                                                     current_state = 1)

  if (tfp_version() < "0.7") {
    states <- states_and_results[[1]]
  } else {
    states <- states_and_results
  }

  expect_equal(states$get_shape() %>% length(), 2)
})

test_succeeds("Can write summaries from trace_fn", {

  skip_if_tfp_below("0.7")

  d <- tfd_normal(0, 1)
  kernel <- mcmc_hamiltonian_monte_carlo(d$log_prob, step_size = 0.1, num_leapfrog_steps = 3) %>%
    mcmc_simple_step_size_adaptation(num_adaptation_steps=100)

  path <- "/tmp/summary_chain"
  summary_writer <- tf$compat$v2$summary$create_file_writer(path, flush_millis = 10000L)

  trace_fn <- function(state, results) {
    with(tf$compat$v2$summary$record_if(tf$equal(results$step %% 10L, 1L)), {
      tf$compat$v2$summary$scalar("state", state, step = tf$cast(results$step, tf$int64))
    })
  }

  with (summary_writer$as_default(), {
    chain_and_results <- kernel %>% mcmc_sample_chain(current_state = 0, num_results = 200, trace_fn = trace_fn)
  })
  summary_writer$close()

  # does not work on today's master (as opposed to y'day) ... keep checking
  #expect_equal(list.files(path) %>% length, 1)
})

test_succeeds("mcmc_effective_sample_size works", {

  target <- tfd_multivariate_normal_diag(scale_diag = c(1, 2))

  states_and_results <- mcmc_hamiltonian_monte_carlo(
    target_log_prob_fn = target$log_prob,
    step_size = 0.05,
    num_leapfrog_steps = 20)  %>%
    mcmc_sample_chain(num_burnin_steps = 20,
                      num_results = 100,
                      current_state = c(0, 0))

  if(tfp_version() < "0.7") {
    states <- states_and_results[[1]]
  } else {
    states <- states_and_results
  }

  ess <- mcmc_effective_sample_size(states)
  variance <- tf$nn$moments(states, axes = 0L)[[2]]
  standard_error <- tf$sqrt(variance / ess)
  expect_equal(standard_error$get_shape() %>% length(), 2)
})

test_succeeds("mcmc_potential_scale_reduction works", {

  target <- tfd_multivariate_normal_diag(scale_diag = c(1, 2))

  initial_state <- target %>% tfd_sample(10) * 2

  states_and_results <- mcmc_hamiltonian_monte_carlo(
    target_log_prob_fn = target$log_prob,
    step_size = 0.05,
    num_leapfrog_steps = 20)  %>%
    mcmc_sample_chain(num_burnin_steps = 20,
                      num_results = 100,
                      current_state = initial_state)

  if(tfp_version() < "0.7") {
    states <- states_and_results[[1]]
  } else {
    states <- states_and_results
  }

  rhat <- mcmc_potential_scale_reduction(states)
  expect_equal(rhat$get_shape() %>% length(), 2)
})

test_succeeds("mcmc_potential_scale_reduction works", {

 make_likelihood <- function(true_variances) tfd_multivariate_normal_diag(
   scale_diag = sqrt(true_variances))

 dims <- 10
 true_variances <- seq(1, 3, length.out = dims)
 likelihood <- make_likelihood(true_variances)

 realnvp_hmc <- mcmc_hamiltonian_monte_carlo(
   target_log_prob_fn = likelihood$log_prob,
   step_size = 0.5,
   num_leapfrog_steps = 2) %>%
   mcmc_transformed_transition_kernel(
     bijector = tfb_real_nvp(
       num_masked = 2,
       shift_and_log_scale_fn = tfb_real_nvp_default_template(
         hidden_layers = list(512, 512))))

 states_and_results <- realnvp_hmc %>% mcmc_sample_chain(
   num_results = 10,
   current_state = rep(0, dims),
   num_burnin_steps = 5)

 if(tfp_version() < "0.7") {
   states <- states_and_results[[1]]
 } else {
   states <- states_and_results
 }

 expect_equal(states$get_shape() %>% length(), 2)

})
