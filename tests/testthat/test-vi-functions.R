context("vi-functions")

source("utils.R")

test_succeeds("vi_amari_alpha works", {
  u <- 2
  logu <- log(u)
  expect_equal(vi_amari_alpha(logu, self_normalized = TRUE) %>% tensor_value(),  u * logu - (u - 1))
})

test_succeeds("vi_kl_reverse works", {
  u <- 2
  logu <- log(u)
  expect_equal(vi_kl_reverse(logu, self_normalized = TRUE) %>% tensor_value(),  -logu + (u - 1))
})

test_succeeds("vi_kl_forward works", {
  u <- 2
  logu <- log(u)
  expect_equal(vi_kl_forward(logu, self_normalized = TRUE) %>% tensor_value(),  u * log(u) - (u - 1))
})

test_succeeds("vi_monte_carlo_variational_loss works", {

  skip_if_tfp_below("0.8")

  q <- tfd_normal(loc = 1,
                  scale = c(0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
  )

  p <- tfd_normal(loc = q$loc + 0.1, scale = q$scale - 0.2)

  approx_kl <- vi_monte_carlo_variational_loss(
    target_log_prob_fn = p$log_prob,
    surrogate_posterior = q,
    discrepancy_fn = vi_kl_reverse,
    sample_size = 4.5e5,
    seed = 777)

  exact_kl <- tfd_kl_divergence(q, p)

  expect_equal(exact_kl %>% tensor_value(), approx_kl %>% tensor_value(), tolerance = 0.01)
})

test_succeeds("vi_jensen_shannon works", {
  u <- 2
  logu <- log(u)
  expect_equal(vi_jensen_shannon(logu, self_normalized = TRUE) %>% tensor_value(),
               u * log(u) - (1 + u) * log(1 + u) + (u + 1) * log(2),
               tolerance = 1e-6)
})

test_succeeds("vi_arithmetic_geometric works", {
  u <- 2
  logu <- log(u)
  expect_equal(vi_arithmetic_geometric(logu, self_normalized = TRUE) %>% tensor_value(),
               (1 + u) * log( (1 + u) / sqrt(u) ) - (1 + u)  * log(2),
               tolerance = 1e-6)
})

test_succeeds("vi_total_variation works", {
  u <- 2
  logu <- log(u)
  expect_equal(vi_total_variation(logu) %>% tensor_value(),
               0.5 * abs(u - 1))
})


