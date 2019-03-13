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
  skip_if_tfp_below("0.7")
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

# test_succeeds("Define a Cholesky to inverse Cholesky bijector", {
#   b <- bijector_cholesky_to_inv_cholesky()
#   c <- bijector_chain(list(bijector_invert(bijector_cholesky_outer_product()),
#                            bijector_matrix_inverse(),
#                            bijector_cholesky_outer_product()))
#   x <- matrix(c(1, 0, 2, 1), ncol = 2, byrow = TRUE)
#   expect_equal(b %>% forward(x), c %>% forward(x))
# })

test_succeeds("Define a discrete cosine transform bijector", {
  b <- bijector_discrete_cosine_transform()
  x <- matrix(runif(100))
  y <- b %>% forward(x)
  expect_equal(x, b %>% inverse(y) %>% tensor_value(), tolerance = 1e-6)
})

test_succeeds("Define an expm1 bijector", {
  b <- bijector_expm1()
  c <- bijector_chain(list(bijector_affine_scalar(shift = -1), bijector_exp()))
  c <- bijector_chain(list(bijector_exp()))
  x <- matrix(1.1:4.1, ncol = 2, byrow = TRUE)
  expect_equal(b %>% forward(x), c %>% forward(x))
})

test_succeeds("Define a fill_triangular bijector", {
  b <- bijector_fill_triangular()
  x <- 1:6
  expect_equal((b %>% forward(x))$shape$ndims, 2)
})

test_succeeds("Define a Gumbel bijector", {
  b <- bijector_gumbel()
  x <- runif(6)
  expect_equal(b %>% forward(x), tf$exp(-tf$exp(-x)))
})

test_succeeds("Define an inline bijector", {
  b <- bijector_inline(forward_fn = tf$exp,
                       inverse_fn = tf$log,
                       inverse_log_det_jacobian_fn = (function(y) -tf$reduce_sum(tf$log(y), axis = -1)),
                       forward_min_event_ndims = 0)
  x <- runif(6)
  expect_equal(b %>% forward(x), bijector_exp() %>% forward(x))
})

test_succeeds("Define an invert bijector", {
  inner <- bijector_identity()
  b <- bijector_invert(inner)
  x <- runif(6)
  expect_equal(b$inverse(x), inner %>% forward(x))
  # inverse() takes 2 positional arguments but 3 were given
  # "inverse" from inverse.tensorflow_probability.python.bijectors.bijector.Bijector
  # expect_equal(b %>% inverse(x), inner %>% forward(x))
})

test_succeeds("Define a kumaraswamy bijector", {
  b <- bijector_kumaraswamy(concentration1 = 2, concentration0 = 0.3)
  x <- runif(1)
  expect_lt(b %>% forward(x) %>% tensor_value(), 1)
})

test_succeeds("Use masked autoregressive flow with template", {
  # maf <- distribution_transformed(
  #   distribution = distribution_normal(loc=0., scale=1.),
  #   bijector = bijector_masked_autoregressive_flow(
  #     shift_and_log_scale_fn = tfb.masked_autoregressive_default_template(
  #       hidden_layers=[512, 512])),
  #   event_shape=[dims])
  # x = maf.sample()  # Expensive; uses `tf.while_loop`, no Bijector caching.
  # maf.log_prob(x)   # Almost free; uses Bijector caching.
  # maf.log_prob(0.)  # Cheap; no `tf.while_loop` despite no Bijector caching.

})


# Create the Y=g(X)=expm1(X) transform.
# expm1 = Expm1()
# x = [[[1., 2],
#       [3, 4]],
#      [[5, 6],
#       [7, 8]]]
# expm1(x) == expm1.forward(x)
# log1p(x) == expm1.inverse(x)



