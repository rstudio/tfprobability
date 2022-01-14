context("tensorflow probability keras initializers")

test_succeeds("initializer_blockwise works", {

  init <- initializer_blockwise(
    initializers = lapply(1:5, keras::initializer_constant),
    sizes = rep(1, 5)
  )

  layer <- keras::layer_dense(units = 5, input_shape = 1, kernel_initializer = init)
  layer$build(input_shape = 1L)

  expect_equivalent(as.numeric(keras::get_weights(layer)[[1]]), 1:5)
})
