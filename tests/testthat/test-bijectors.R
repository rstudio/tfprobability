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

test_succeeds("Define a chain of bijectors", {
  b <- bijector_chain(list(bijector_exp(), bijector_sigmoid()))
  expect_equal(b$bijectors %>% length(), 2)
})

test_succeeds("Define a Cholesky outer product bijector", {
  b <- bijector_cholesky_outer_product()
  x <- matrix(c(1, 0, 2, 1), ncol = 2, byrow = TRUE)
  y <- matrix(c(1, 2, 2, 5), byrow = TRUE, nrow = 2)
  expect_equal(b %>% forward(x) %>% tensor_value(), y)
  expect_equal(b %>% inverse(y) %>% tensor_value(), x)
})

test_succeeds("Define a Cholesky to inverse Cholesky bijector", {
  b <- bijector_cholesky_to_inv_cholesky()
  c <- bijector_chain(list(bijector_invert(bijector_cholesky_outer_product()),
                           bijector_matrix_inverse(),
                           bijector_cholesky_outer_product()))
  x <- matrix(c(1, 0, 2, 1), ncol = 2, byrow = TRUE)
  expect_equal(b %>% forward(x), c %>% forward(x))
})

test_succeeds("Define a discrete cosine transform bijector", {
  b <- bijector_discrete_cosine_transform()
  x <- matrix(runif(100)) %>% tf$cast(tf$float32)
  y <- b %>% forward(x)
  expect_equal(x %>% tensor_value(), b %>% inverse(y) %>% tensor_value())
})

test_succeeds("Define an expm1 bijector", {
  b <- bijector_expm1()
  c <- bijector_chain(list(bijector_affine_scalar(shift = -1), bijector_exp()))
  c <- bijector_chain(list(bijector_exp()))
  # need float32 due to affine_scalar.py calling the supertype _init_ with a dtype it determined
  # dtype = dtype_util.common_dtype([self._shift, self._scale]) which will be tf$float32
  # then _maybe_assert_dtype in bijector superclass checks against self$dtype
  # should the forward generic cast everything to float32??? q for JJ
  x <- matrix(1.1:4.1, ncol = 2, byrow = TRUE) %>% tf$cast(tf$float32)
  expect_equal(b %>% forward(x), c %>% forward(x))
})


# Create the Y=g(X)=expm1(X) transform.
# expm1 = Expm1()
# x = [[[1., 2],
#       [3, 4]],
#      [[5, 6],
#       [7, 8]]]
# expm1(x) == expm1.forward(x)
# log1p(x) == expm1.inverse(x)



