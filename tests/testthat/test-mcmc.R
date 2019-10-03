context("mcmc")

source("utils.R")

test_succeeds("sampling from chain works", {
  dims <- 10
  true_stddev <- sqrt(seq(1, 3, length.out = dims))
  likelihood <-
    tfd_multivariate_normal_diag(scale_diag = true_stddev)

  kernel <- mcmc_hamiltonian_monte_carlo(
    target_log_prob_fn = likelihood$log_prob,
    step_size = 0.5,
    num_leapfrog_steps = 2
  )

  states <- kernel %>% mcmc_sample_chain(
    num_results = 1000,
    num_burnin_steps = 500,
    current_state = rep(0, dims),
    trace_fn = NULL
  )

  sample_mean <- tf$reduce_mean(states, axis = 0L)
  sample_stddev <-
    tf$sqrt(tf$reduce_mean(tf$math$squared_difference(states, sample_mean), axis = 0L))

  expect_equal(sample_stddev %>% tensor_value() %>% mean(),
               mean(true_stddev),
               tol = 1e-1)
})

test_succeeds("HamiltonianMonteCarlo with SimpleStepSizeAdaptation works", {

  target_log_prob_fn <- tfd_normal(loc = 0, scale = 1)$log_prob
  num_burnin_steps <- 500
  num_results <- 500
  num_chains <- 64L
  step_size <- tf$fill(list(num_chains), 0.1)

  kernel <- mcmc_hamiltonian_monte_carlo(
    target_log_prob_fn = target_log_prob_fn,
    num_leapfrog_steps = 2,
    step_size = step_size
  ) %>%
    mcmc_simple_step_size_adaptation(num_adaptation_steps = round(num_burnin_steps * 0.8))

  res <- kernel %>% mcmc_sample_chain(
    num_results = num_results,
    num_burnin_steps = num_burnin_steps,
    current_state = rep(0, num_chains),
    trace_fn = function(x, pkr) {
      list (
        pkr$inner_results$accepted_results$step_size,
        pkr$inner_results$log_accept_ratio
      )
    }
  )

  # check for nicer unpacking
  # python: samples, [step_size, log_accept_ratio]

  samples <- res$all_states
  step_size <- res$trace[[1]]
  log_accept_ratio <- res$trace[[2]]
  #  ~0.75
  p_accept <-
    tf$reduce_mean(tf$exp(tf$minimum(log_accept_ratio, 0)))
  expect_lt(p_accept %>% tensor_value(), 1)

})

test_succeeds("MetropolisHastings works", {
  kernel <- mcmc_metropolis_hastings(
    mcmc_uncalibrated_hamiltonian_monte_carlo(
      target_log_prob_fn = function(x)
        - x - x ^ 2,
      step_size = 0.1,
      num_leapfrog_steps = 3
    )
  )

  states <- kernel %>% mcmc_sample_chain(num_results = 100,
                                                     current_state = 1)

  expect_equal(states$get_shape() %>% length(), 2)
})

test_succeeds("RandomWalkMetropolis works", {
  kernel <- mcmc_random_walk_metropolis(
    target_log_prob_fn = function(x)
      - x - x ^ 2
  )

  states <-
    kernel %>% mcmc_sample_chain(num_results = 100,
                                 current_state = 1)

  expect_equal(states$get_shape() %>% length(), 2)
})

test_succeeds("Can write summaries from trace_fn", {

  d <- tfd_normal(0, 1)
  kernel <-
    mcmc_hamiltonian_monte_carlo(d$log_prob,
                                 step_size = 0.1,
                                 num_leapfrog_steps = 3) %>%
    mcmc_simple_step_size_adaptation(num_adaptation_steps = 100)

  path <- "/tmp/summary_chain"
  summary_writer <-
    tf$compat$v2$summary$create_file_writer(path, flush_millis = 10000L)

  trace_fn <- function(state, results) {
    with(tf$compat$v2$summary$record_if(tf$equal(tf$math$mod(
      results$step, 10L
    ), 1L)), {
      tf$compat$v2$summary$scalar("state", state, step = tf$cast(results$step, tf$int64))
    })
  }

  with (summary_writer$as_default(), {
    chain_and_results <-
      kernel %>% mcmc_sample_chain(
        current_state = 0,
        num_results = 200,
        trace_fn = trace_fn
      )
  })
  summary_writer$close()

  # does not work on today's master (as opposed to y'day) ... keep checking
  #expect_equal(list.files(path) %>% length, 1)
})

test_succeeds("mcmc_effective_sample_size works", {
  target <- tfd_multivariate_normal_diag(scale_diag = c(1, 2))

  states <- mcmc_hamiltonian_monte_carlo(
    target_log_prob_fn = target$log_prob,
    step_size = 0.05,
    num_leapfrog_steps = 20
  )  %>%
    mcmc_sample_chain(
      num_burnin_steps = 20,
      num_results = 100,
      current_state = c(0, 0)
    )

  ess <- mcmc_effective_sample_size(states)
  variance <- tf$nn$moments(states, axes = 0L)[[2]]
  standard_error <- tf$sqrt(variance / ess)
  expect_equal(standard_error$get_shape() %>% length(), 2)
})

test_succeeds("mcmc_potential_scale_reduction works", {
  target <- tfd_multivariate_normal_diag(scale_diag = c(1, 2))

  initial_state <- target %>% tfd_sample(10) * 2

  states <- mcmc_hamiltonian_monte_carlo(
    target_log_prob_fn = target$log_prob,
    step_size = 0.05,
    num_leapfrog_steps = 20
  )  %>%
    mcmc_sample_chain(
      num_burnin_steps = 20,
      num_results = 100,
      current_state = initial_state
    )

  rhat <- mcmc_potential_scale_reduction(states)
  expect_equal(rhat$get_shape() %>% length(), 2)
})

test_succeeds("mcmc_potential_scale_reduction works", {
  make_likelihood <-
    function(true_variances)
      tfd_multivariate_normal_diag(scale_diag = sqrt(true_variances))

  dims <- 10
  true_variances <- seq(1, 3, length.out = dims)
  likelihood <- make_likelihood(true_variances)

  realnvp_hmc <- mcmc_hamiltonian_monte_carlo(
    target_log_prob_fn = likelihood$log_prob,
    step_size = 0.5,
    num_leapfrog_steps = 2
  ) %>%
    mcmc_transformed_transition_kernel(
      bijector = tfb_real_nvp(
        num_masked = 2,
        shift_and_log_scale_fn = tfb_real_nvp_default_template(hidden_layers = list(512, 512))
      )
    )

  states <- realnvp_hmc %>% mcmc_sample_chain(
    num_results = 10,
    current_state = rep(0, dims),
    num_burnin_steps = 5
  )

  expect_equal(states$get_shape() %>% length(), 2)

})

test_succeeds("mcmc_dual_averaging_step_size_adaptation works", {
  skip_if_tfp_below("0.8")

  target_dist <- tfd_joint_distribution_sequential(list(
    tfd_normal(0, 1.5),
    tfd_independent(tfd_normal(
      tf$zeros(shape = shape(2, 5), dtype = tf$float32), 5
    ),
    reinterpreted_batch_ndims = 2)
  ))

  num_burnin_steps <- 5
  num_results <- 7
  num_chains <- 64

  kernel <- mcmc_hamiltonian_monte_carlo(
    target_log_prob_fn = function(...)
      target_dist %>% tfd_log_prob(list(...)),
    num_leapfrog_steps = 2,
    step_size = target_dist %>% tfd_stddev()
  ) %>%
    mcmc_dual_averaging_step_size_adaptation(num_adaptation_steps = floor(num_burnin_steps * 0.8))

  res <- kernel %>% mcmc_sample_chain(
    num_results = num_results,
    num_burnin_steps = num_burnin_steps,
    current_state = target_dist %>% tfd_sample(num_chains),
    trace_fn = function(xx, pkr)
      pkr$inner_results$log_accept_ratio
  )

  log_accept_ratio <- res[1]
  expect_equal(log_accept_ratio$get_shape() %>% length(), 2)

})

test_succeeds("mcmc_no_u_turn_sampler works", {

  skip_if_tfp_below("0.8")

  predictors <-
    tf$cast(
      c(
        201,
        244,
        47,
        287,
        203,
        58,
        210,
        202,
        198,
        158,
        165,
        201,
        157,
        131,
        166,
        160,
        186,
        125,
        218,
        146
      ),
      tf$float32
    )
  obs <-
    tf$cast(
      c(
        592,
        401,
        583,
        402,
        495,
        173,
        479,
        504,
        510,
        416,
        393,
        442,
        317,
        311,
        400,
        337,
        423,
        334,
        533,
        344
      ),
      tf$float32
    )
  y_sigma <-
    tf$cast(c(
      61,
      25,
      38,
      15,
      21,
      15,
      27,
      14,
      30,
      16,
      14,
      25,
      52,
      16,
      34,
      31,
      42,
      26,
      16,
      22
    ),
    tf$float32)

  # Robust linear regression model
  robust_lm <- tfd_joint_distribution_sequential(
    list(
      tfd_normal(loc = 0, scale = 1),
      # b0
      tfd_normal(loc = 0, scale = 1),
      # b1
      tfd_half_normal(5),
      # df
      function(df, b1, b0)
        tfd_independent(
          tfd_student_t(
            # Likelihood
            df = df[, NULL],
            loc = b0[, NULL] + b1[, NULL] * predictors[NULL,],
            scale = y_sigma[NULL,]
          )
        )
    ),
    validate_args = TRUE
  )

  log_prob <-function(b0, b1, df) {
    robust_lm %>% tfd_log_prob(list(b0, b1, df, obs))
  }

  step_size0 <- Map(function(x)
    tf$cast(x, tf$float32), c(1, .2, .5))

  number_of_steps <- 100
  burnin <- 50
  nchain <- 50

  run_chain <- function() {
    # random initialization of the starting postion of each chain

    samples <- robust_lm %>% tfd_sample(nchain)
    b0 <- samples[[1]]
    b1 <- samples[[2]]
    df <- samples[[3]]

    # bijector to map contrained parameters to real
    unconstraining_bijectors <- list(tfb_identity(),
                                     tfb_identity(),
                                     tfb_identity(),
                                     tfb_exp())

    trace_fn <- function(x, pkr) {
      list(
        pkr$inner_results$inner_results$step_size,
        pkr$inner_results$inner_results$log_accept_ratio
      )
    }

    nuts <- mcmc_no_u_turn_sampler(
      target_log_prob_fn = log_prob,
      step_size = step_size0
    ) %>%
      mcmc_transformed_transition_kernel(bijector = unconstraining_bijectors) %>%
      mcmc_dual_averaging_step_size_adaptation(
        num_adaptation_steps = burnin,
        step_size_setter_fn = function(pkr, new_step_size)
          pkr$`_replace`(
            inner_results = pkr$inner_results$`_replace`(step_size = new_step_size)
          ),
        step_size_getter_fn = function(pkr)
          pkr$inner_results$step_size,
        log_accept_prob_getter_fn = function(pkr)
          pkr$inner_results$log_accept_ratio
      )

    nuts %>% mcmc_sample_chain(
      num_results = number_of_steps,
      num_burnin_steps = burnin,
      current_state = list(b0, b1, df),
      trace_fn = trace_fn
    )
  }


  run_chain <- tensorflow::tf_function(run_chain)

  # mcmc_trace, (step_size, log_accept_ratio)
  res <- run_chain()

  log_accept_ratio <- res[1][[2]]
  expect_equal(log_accept_ratio$get_shape() %>% length(), 2)
})

test_succeeds("MetropolisAdjustedLangevinAlgorithm works", {

  kernel <- mcmc_metropolis_adjusted_langevin_algorithm(
      target_log_prob_fn = function(x)
        - x - x ^ 2,
      step_size = 0.75
      )

  states <- kernel %>% mcmc_sample_chain(num_results = 100,
                                                     current_state = c(1, 1))

  expect_equal(states$get_shape() %>% length(), 2)
})

test_succeeds("mcmc_sample_annealed_importance_chain works", {

  make_prior <- function(dims, dtype) {
    tfd_multivariate_normal_diag(
      loc = tf$zeros(dims, dtype))
  }

  make_likelihood <- function(weights, x) {
    tfd_multivariate_normal_diag(
      loc = tf$linalg$matvec(x, weights))
  }

  num_chains <- 7
  dims <- 5
  dtype <- tf$float32

  x <- matrix(rnorm(num_chains * dims), nrow = num_chains, ncol = dims) %>% tf$cast(dtype)
  true_weights <- rnorm(dims) %>% tf$cast(dtype)
  y <- tf$linalg$matvec(x, true_weights) + rnorm(num_chains) %>% tf$cast(dtype)

  prior <- make_prior(dims, dtype)

  target_log_prob_fn <- function(weights) {
    prior$log_prob(weights) + make_likelihood(weights, x)$log_prob(y)
  }

  proposal <- tfd_multivariate_normal_diag(loc = tf$zeros(dims, dtype))

  res <- mcmc_sample_annealed_importance_chain(
      num_steps = 6,
      proposal_log_prob_fn = proposal$log_prob,
      target_log_prob_fn = target_log_prob_fn,
      current_state = tf$zeros(list(num_chains, dims), dtype),
      make_kernel_fn = function(tlp_fn) mcmc_hamiltonian_monte_carlo(
        target_log_prob_fn = tlp_fn,
        step_size = 0.1,
        num_leapfrog_steps = 2))

  weight_samples <- res[[1]]
  ais_weights <- res[[2]]
  kernel_results <- res[[3]]

  log_normalizer_estimate <- tf$reduce_logsumexp(ais_weights) - log(num_chains)

  expect_equal(log_normalizer_estimate$get_shape()$as_list() %>% length(), 0)
})

test_succeeds("mcmc_replica_exchange_mc works", {

  target <- tfd_normal(loc = 0, scale = 1)
  make_kernel_fn <- function(target_log_prob_fn, seed) {
    mcmc_hamiltonian_monte_carlo(
      target_log_prob_fn = target_log_prob_fn,
      seed = seed,
      step_size = 1,
      num_leapfrog_steps = 3)
  }

  remc <- mcmc_replica_exchange_mc(
    target_log_prob_fn = target$log_prob,
    inverse_temperatures = list(1., 0.3, 0.1, 0.03),
    make_kernel_fn = make_kernel_fn)

  res <- remc %>% mcmc_sample_chain(
    num_results = 10,
    current_state = 1,
    num_burnin_steps = 5,
    parallel_iterations = 1)

  expect_equal(res$get_shape()$as_list() %>% length(), 1)

})

test_succeeds("mcmc_slice_sampler works", {

  target <- tfd_normal(loc = 0, scale = 1)

  kernel <- mcmc_slice_sampler(
    target_log_prob_fn = target$log_prob,
    step_size = 0.1,
    max_doublings = 5)

  res <- kernel %>% mcmc_sample_chain(
    num_results = 10,
    current_state = 1,
    num_burnin_steps = 5,
    parallel_iterations = 1)

  expect_equal(res$get_shape()$as_list() %>% length(), 1)
})

test_succeeds("mcmc_sample_halton_sequence works", {

  num_results <- 10
  dim <- 3
  sample <- mcmc_sample_halton_sequence(
    dim,
    num_results = num_results,
    seed = 127)

  expect_equal(dim(sample) %>% length(), 2)
})
