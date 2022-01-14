context("vi-functions")

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

test_succeeds("vi_pearson works", {
  u <- 2
  logu <- log(u)
  expect_equal(vi_pearson(logu) %>% tensor_value(),
               (u - 1)^2)
})

test_succeeds("vi_squared_hellinger works", {
  u <- 2
  logu <- log(u)
  expect_equal(vi_squared_hellinger(logu) %>% tensor_value(),
               (sqrt(u) - 1)^2, tolerance = 1e-6)
})

test_succeeds("vi_triangular works", {
  u <- 2
  logu <- log(u)
  expect_equal(vi_triangular(logu) %>% tensor_value(),
               (u - 1)^2 / (1 + u))
})

test_succeeds("vi_t_power works", {
  u <- 2
  logu <- log(u)
  t <- 0.5
  expect_equal(vi_t_power(logu, t, self_normalized = TRUE) %>% tensor_value(),
               - (u^t - 1 - t * (u - 1)), tolerance = 1e-6)
})

test_succeeds("vi_log1p_abs works", {
  u <- 2
  logu <- log(u)
  expect_equal(vi_log1p_abs(logu) %>% tensor_value(),
               u^(sign(u - 1)) - 1)
})

test_succeeds("vi_jeffreys works", {
  u <- 2
  logu <- log(u)
  expect_equal(vi_jeffreys(logu) %>% tensor_value(),
               0.5 * (u * log(u) - log(u)))
})

test_succeeds("vi_chi_square works", {
  u <- 2
  logu <- log(u)
  expect_equal(vi_chi_square(logu) %>% tensor_value(),
               u^2 - 1)
})

test_succeeds("vi_modified_gan works", {
  u <- 2
  logu <- log(u)
  expect_equal(vi_modified_gan(logu, self_normalized = TRUE) %>% tensor_value(),
               log(1 + u) - log(u) + 0.5 * (u - 1),
               tolerance = 1e-6)
})

test_succeeds("vi_dual_csiszar_function works", {

  skip_if_tfp_below("0.8")
  u <- 2
  logu <- log(u)
  dual_csiszar_function <- vi_kl_forward
  expect_equal(vi_dual_csiszar_function(logu, dual_csiszar_function) %>% tensor_value(),
               vi_kl_reverse(logu) %>% tensor_value())
})

test_succeeds("vi_symmetrized_function works", {
  u <- 2
  logu <- log(u)
  other_csiszar_function <- function(logu)  2 * (exp(logu) * (logu - tf$nn$softplus(logu)))
  expect_equal(vi_symmetrized_csiszar_function(logu, other_csiszar_function) %>% tensor_value(),
               vi_jensen_shannon(logu) %>% tensor_value(),
               tolerance = 1e-6)
})

