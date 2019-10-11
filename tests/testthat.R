library(testthat)
library(tfprobability)

if (identical(Sys.getenv("NOT_CRAN"), "true")) {
  test_check("tfprobability")
}
