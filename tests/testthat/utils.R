have_tf_probability <- function() {
  reticulate::py_module_available("tensorflow_probability")
}

skip_if_no_tf_probability <- function() {
  if (!have_tf_probability())
    skip("TensorFlow Probability not available for testing")
}

skip_if_tf_probability_below <- function(version) {
  if (tfp_version() < version) {
    skip(paste0("Skipped since this test requires TensorFlow Probability >= ", version))
  }
}

test_succeeds <- function(desc, expr) {
  test_that(desc, {
    skip_if_no_tf_probability()
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
