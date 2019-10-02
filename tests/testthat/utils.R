have_tfp <- function() {
  reticulate::py_module_available("tensorflow_probability")
}

skip_if_no_tfp <- function() {
  if (!have_tfp())
    skip("TensorFlow Probability not available for testing")
}

skip_if_tfp_below <- function(version) {
  if (tfprobability:::tfp_version() < version) {
    skip(paste0("Skipped since this test requires TensorFlow Probability >= ", version))
  }
}

skip_if_tfp_above <- function(version) {
  if (tfprobability:::tfp_version() > version) {
    skip(paste0("Skipped since this test requires TensorFlow Probability <= ", version))
  }
}

skip_if_tf_below <- function(version) {
  if (tensorflow:::tf_version() < version) {
    skip(paste0("Skipped since this test requires TensorFlow >= ", version))
  }
}

skip_if_tf_above <- function(version) {
  if (tensorflow:::tf_version() > version) {
    skip(paste0("Skipped since this test requires TensorFlow <= ", version))
  }
}

skip_if_not_eager <- function() {
  if (!tf$executing_eagerly())
    skip("This test requires eager execution")
}

skip_if_eager <- function() {
  if (tf$executing_eagerly())
    skip("This test requires graph execution")
}

test_succeeds <- function(desc, expr) {
  test_that(desc, {
    skip_if_no_tfp()
    expect_error(force(expr), NA)
  })
}

tensor_value <- function(tensor) {
  if (tf$executing_eagerly()) {
    as.array(tensor)
  } else {
    sess <- tf$compat$v1$Session()
    sess$run(tf$global_variables_initializer())
    sess$run(tensor)
  }
}

as_tensor <- tfprobability:::as_tensor
