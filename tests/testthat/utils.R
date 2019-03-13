have_tfp <- function() {
  reticulate::py_module_available("tensorflow_probability")
}

skip_if_no_tfp <- function() {
  if (!have_tfp())
    skip("TensorFlow Probability not available for testing")
}

skip_if_tfp_below <- function(version) {
  if (tfp_version() < version) {
    skip(paste0("Skipped since this test requires TensorFlow Probability >= ", version))
  }
}

test_succeeds <- function(desc, expr) {
  test_that(desc, {
    skip_if_no_tfp()
    expect_error(force(expr), NA)
  })
}

tensor_value <- function(tensor) {
  if (tf$executing_eagerly()) {
    as.numeric(tensor)
  } else {
    sess <- tf$Session()
    sess$run(tensor)
  }
}
