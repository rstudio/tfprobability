context("vi-functions")

source("utils.R")

test_succeeds("amari_alpha works", {
  u <- 2
  logu <- log(u)
  expect_equal(vi_amari_alpha(logu, self_normalized = TRUE) %>% tensor_value(),  u * logu - (u - 1))
})

test_succeeds("kl_reverse works", {
  u <- 2
  logu <- log(u)
  expect_equal(vi_kl_reverse(logu, self_normalized = TRUE) %>% tensor_value(),  -logu + (u - 1))
})

test_succeeds("kl_forward works", {
  u <- 2
  logu <- log(u)
  expect_equal(vi_kl_forward(logu, self_normalized = TRUE) %>% tensor_value(),  u * log(u) - (u - 1))
})
