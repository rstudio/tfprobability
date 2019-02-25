context("bijectors")

source("utils.R")

test_succeeds("Define an identity bijector", {
  b <- bijector_identity()
  x <- matrix(1:4, ncol = 2, byrow = TRUE)
  expect_equal(b %>% forward(x), b %>% inverse(x))

})

test_succeeds("Define a sigmoid bijector", {
  b <- bijector_sigmoid()
  x <- matrix(1.1:4.1, ncol = 2, byrow = TRUE)
  y <- b %>% forward(x)
  expect_equal((b %>% forward_log_det_jacobian(y, event_ndims = 0))$shape$ndims, 2)
})

test_succeeds("Define an exp bijector", {
  b <- bijector_exp()
  x <- matrix(1.1:4.1, ncol = 2, byrow = TRUE)
  y <- b %>% forward(x)
  expect_equal(b %>% inverse(y), x %>% tf$convert_to_tensor())
})

test_succeeds("Define an absolute value bijector", {
  b <- bijector_absolute_value()
  x <- -1.1
  y <- b %>% forward(x)
  expect_equal(b %>% inverse(y) %>% length(), 2)
})

test_succeeds("Define an affine bijector", {
  b <-
    bijector_affine(
      shift = c(0, 0),
      scale_tril = matrix(c(1.578, 0, 7.777, 0), nrow = 2, byrow = TRUE),
      dtype = tf$float32
    )
  x <- c(100, 1000)
  y <- b %>% forward(x)
  expect_equal(b %>% inverse(y) %>% length(), 2)
})

test_succeeds("Define an affine linear operator bijector", {
  b <-
    bijector_affine_linear_operator(shift = c(-1, 0, 1),
                                    scale = tf$linalg$LinearOperatorDiag(c(1, 2, 3)))
  x <- c(100, 1000, 10000)
  y <- b %>% forward(x)
  expect_equal(b %>% inverse(y) %>% length(), 3)
})

test_succeeds("Define an affine scalar bijector", {
  b <- bijector_affine_scalar(shift = 3.33)
  x <- c(100, 1000, 10000)
  y <- b %>% forward(x)
  expect_equal(b %>% inverse_log_det_jacobian(y, event_ndims = 0) %>% tensor_value(),
               0)
})

test_succeeds("Define a batch norm bijector", {
  dist <- distribution_transformed(distribution = distribution_normal(loc = 0, scale = 1),
                                   bijector = bijector_batch_normalization())
  y <-
    distribution_normal(loc = 1, scale = 2) %>% sample(100)  # ~ N(1, 2)
  # normalizes using the mean and standard deviation of the current minibatch.
  x <- dist$bijector %>% inverse(y)  # ~ N(0, 1)
  expect_equal(x$shape, y$shape)
})

test_succeeds("Define a blockwise bijector", {
  b <- bijector_blockwise(list(bijector_exp(), bijector_sigmoid()), block_sizes = list(2, 1))
  x <- matrix(rep(23, 5*3), ncol = 3)
  y <- b %>% forward(x)
  expect_equal(y$shape %>% length(), 2)
})


