context("vi-functions")

source("utils.R")

test_succeeds("amari-alpha works", {
  u <- 1
  expect_equal(vi_amari_alpha(log(u)) %>% tensor_value(),  u * log(u) - (u - 1))
})
