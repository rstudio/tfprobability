context("bijectors")

# Flows -------------------------------------------------------------------


test_succeeds("Use tfb_masked_dense", {
  input <-
    tf$constant(matrix(1:40, ncol = 4, byrow = TRUE), dtype = tf$float32)
  m <-
    tfb_masked_dense(
      input = input,
      units = 22,
      num_blocks = 4,
      trainable = TRUE
    )
})

test_succeeds("Use tfb_real_nvp_default_template", {
  m <-
    tfb_masked_autoregressive_default_template(hidden_layers = 17,
                                           bias_initializer = tf$constant_initializer(0))
})

test_succeeds("Use tfb_masked_autoregressive_default_template", {
  m <-
    tfb_real_nvp_default_template(hidden_layers = 17, activity_regularizer = "l2")
})

test_succeeds("Use masked autoregressive flow with template", {
  skip_if_not_eager()
  skip_if_tfp_above("0.7")
  dims <- 5L
  maf <- tfd_transformed_distribution(
    distribution = tfd_normal(loc = 0, scale = 1),
    bijector = tfb_masked_autoregressive_flow(shift_and_log_scale_fn = tfb_masked_autoregressive_default_template(hidden_layers = c(7, 7))),
    event_shape = dims
  )
  target_dist <- tfd_normal(loc = 2.2, scale = 0.23)
  y  <-
    target_dist %>% tfd_sample(1000) %>% tf$reshape(shape = shape(200, 5))
  loss <- function()
    - tf$reduce_mean(maf %>% tfd_log_prob(y))
  optimizer <- tf$optimizers$Adam(1e-4)
  optimizer$minimize(loss)
  x <- maf %>% tfd_sample() %>% tensor_value()
})

test_succeeds("Use a tfb_inverse autoregressive flow", {
  skip_if_not_eager()
  skip_if_tfp_above("0.7")
  dims <- 5L
  iaf <- tfd_transformed_distribution(
    distribution = tfd_normal(loc = 0, scale = 1),
    bijector = tfb_invert(
      tfb_masked_autoregressive_flow(shift_and_log_scale_fn = tfb_masked_autoregressive_default_template(hidden_layers = c(7, 7)))
    ),
    event_shape = dims
  )
  target_dist <- tfd_normal(loc = 2.2, scale = 0.23)
  y  <-
    target_dist %>% tfd_sample(1000) %>% tf$reshape(shape = shape(200, 5))
  loss <- function() {
    - tf$reduce_mean(iaf %>% log_prob(y))
  }

  optimizer <- tf$optimizers$Adam(1e-4)
  optimizer$minimize(loss)
  x <- iaf %>% tfd_sample() %>% tensor_value()
})

test_succeeds("Use real NVP with template", {
  skip_if_not_eager()
  skip_if_tfp_above("0.7")
  dims <- 5L
  rnvp <- tfd_transformed_distribution(
    distribution = tfd_normal(loc = 0, scale = 1),
    bijector = tfb_real_nvp(
      num_masked = 1,
      shift_and_log_scale_fn = tfb_real_nvp_default_template(hidden_layers = c(7, 7))
    ),
    event_shape = dims
  )
  target_dist <- tfd_normal(loc = 2.2, scale = 0.23)
  y  <-
    target_dist %>% tfd_sample(1000) %>% tf$reshape(shape = shape(200, 5))
  loss <- function()
    - tf$reduce_mean(rnvp %>% log_prob(y))
  optimizer <- tf$optimizers$Adam(1e-4)
  optimizer$minimize(loss)
  x <- rnvp %>% sample() %>% tensor_value()
})

# Bijectors ---------------------------------------------------------------

test_succeeds("Define a reciprocal bijector", {
  b <- tfb_reciprocal()
  x <- c(1, 2, 3)
  y <- b %>% tfb_forward(x)
  expect_equivalent(b %>% tfb_inverse(y) %>% tensor_value(), x)
})

test_succeeds("Define a matvec_lu bijector", {

  if (!tf$compat$v1$resource_variables_enabled()) tf$compat$v1$enable_resource_variables()

  trainable_lu_factorization <- function(event_size,
                                         batch_shape = list(),
                                         seed = NULL,
                                         dtype = tf$float32,
                                         name = "matvec_lu_test") {
    with(tf$compat$v1$name_scope(name), {
      event_size <- tf$convert_to_tensor(event_size,
                                         dtype_hint = tf$int32,
                                         name = 'event_size')
      batch_shape <- tf$convert_to_tensor(batch_shape,
                                          dtype_hint = event_size$dtype,
                                          name = 'batch_shape')
      random_matrix <- tf$random$uniform(
        shape = tf$concat(list(
          batch_shape, list(event_size, event_size)
        ), axis = 0L),
        dtype = dtype,
        seed = seed
      )
      random_orthonormal <-
        tf$linalg$qr(random_matrix)[0] # qr returns tuple of tensors
      lu <- tf$linalg$lu(random_orthonormal)
      lower_upper <- lu[0]
      permutation <- lu[1]
      lower_upper <- tf$Variable(
        initial_value = lower_upper,
        trainable = TRUE,
        name = 'lower_upper'
      )
    })
    list(lower_upper, permutation)
  }

  channels <- 3L
  fact <- trainable_lu_factorization(channels)
  conv1x1 <- tfb_matvec_lu(fact[[1]],
                           fact[[2]],
                           validate_args = TRUE)
  x <- tf$random$uniform(shape = list(2L, 28L, 28L, channels))
  y <- conv1x1$forward(x)
  y_inv = conv1x1$inverse(y)
  # this is not the case, not even in the Python original!?!
  # expect_equal(x %>% tensor_value(), y_inv %>% tensor_value(), tol = 1e-6)
})

test_succeeds("Define a sinh_arcsinh bijector", {
  b <- tfb_sinh_arcsinh()
  x <- c(0, 1, 2)
  expect_equal(b %>% tfb_forward(x) %>% tensor_value(),
               tfb_identity() %>% tfb_forward(x) %>% tensor_value(),
               tol = 1e-6)
})

test_succeeds("Define a softmax_centered bijector", {
  b <- tfb_softmax_centered()
  x <- tf$math$log(c(2, 3, 4))
  y <- array(c(0.2, 0.3, 0.4, 0.1))
  expect_equal(b %>% tfb_forward(x) %>% tensor_value(), y, tolerance = 1e-7)
  expect_equal(b %>% tfb_inverse(y) %>% tensor_value(), x %>% tensor_value(), tolerance = 2e-7)
})

test_succeeds("Define a permute bijector", {
  b <- tfb_permute(permutation = list(1, 3, 0, 2))
  x <- seq(0, 1, by = 0.3)
  y <- b %>% tfb_forward(x)
  expect_equivalent(y %>% tensor_value(), c(0.3, 0.9, 0, 0.6), tol = 1e-6)

  b <- tfb_permute(permutation = list(2, 1, 0), axis = -2)
  x <- matrix(1:6, ncol = 2, byrow = TRUE)
  y <- b %>% tfb_forward(x)
  expect_equivalent(y %>% tensor_value(), rbind(x[3,], x[2,], x[1,]), tol = 1e-6)
})

test_succeeds("Define a power transform bijector", {
  power <- 2
  b <- tfb_power_transform(power = power)
  x <- c(1, 2, 3)
  y <- b %>% tfb_forward(x)
  expect_equivalent(y %>% tensor_value(), (1 + x * power) ^ (1 / power), tol = 1e-6)
})

test_succeeds("Define a normal_cdf bijector", {
  b <- tfb_normal_cdf()
  x <- matrix(c(1, 0, 2, 1), ncol = 2, byrow = TRUE)
  y <- b %>% tfb_forward(x)
  expect_equal(b %>% tfb_inverse(y) %>% tensor_value(), x)
})

test_succeeds("Define a tfb_inverse_tri_l bijector", {
  b <- tfb_matrix_inverse_tri_l()
  x <- matrix(c(1, 0, 2, 1), ncol = 2, byrow = TRUE)
  y <- matrix(c(1, 0,-2, 1), ncol = 2, byrow = TRUE)
  expect_equal(b %>% tfb_forward(x) %>% tensor_value(), y)
})

test_succeeds("Define an identity bijector", {
  b <- tfb_identity()
  x <- matrix(1:4, ncol = 2, byrow = TRUE)
  expect_equal(b %>% tfb_forward(x) %>% tensor_value(),
               b %>% tfb_inverse(x) %>% tensor_value())

})

test_succeeds("Define a sigmoid bijector", {
  b <- tfb_sigmoid()
  x <- matrix(1.1:4.1, ncol = 2, byrow = TRUE)
  y <- b %>% tfb_forward(x)
  expect_equal((b %>% tfb_forward_log_det_jacobian(y, event_ndims = 0))$shape$ndims, 2)
})

test_succeeds("Define an exp bijector", {
  b <- tfb_exp()
  x <- matrix(1.1:4.1, ncol = 2, byrow = TRUE)
  y <- b %>% tfb_forward(x)
  expect_equal(b %>% tfb_inverse(y), x %>% tf$convert_to_tensor())
})

test_succeeds("Define an absolute value bijector", {
  b <- tfb_absolute_value()
  x <- -1.1
  y <- b %>% tfb_forward(x)
  expect_equal(b %>% tfb_inverse(y) %>% length(), 2)
})

test_succeeds("Define an affine bijector", {
  skip("tfp$bijectors$Affine removed, wrapper needs fixing")
  # https://github.com/tensorflow/probability/commit/9afe4303054055959d631407cd2ef091a6c11fc0
  b <-
    tfb_affine(
      shift = c(0, 0),
      scale_tril = matrix(c(1.578, 0, 7.777, 0), nrow = 2, byrow = TRUE),
      dtype = tf$float32
    )
  x <- c(100, 1000)
  y <- b %>% tfb_forward(x)
  expect_equal(b %>% tfb_inverse(y) %>% length(), 2)
})

test_succeeds("Define an affine linear operator bijector", {
  skip("tfp$bijectors$Affine removed, wrapper needs fixing")
  # https://github.com/tensorflow/probability/commit/9afe4303054055959d631407cd2ef091a6c11fc0
  b <-
    tfb_affine_linear_operator(shift = c(-1, 0, 1),
                               scale = tf$linalg$LinearOperatorDiag(c(1, 2, 3)))
  x <- array(c(100, 1000, 10000))
  y <- b %>% tfb_forward(x)
  expect_equal(b %>% tfb_inverse(y) %>% length(), 3)
})

test_succeeds("Define an affine scalar bijector", {
  b <- tfb_shift(3.33)
  x <- c(100, 1000, 10000)
  y <- b %>% tfb_forward(x)
  skip("upstream bug in bijector.inverse_log_det_jacobian, needs reporting + fixing")
  expect_equal(b %>% tfb_inverse_log_det_jacobian(y, event_ndims = 0) %>% tensor_value(),
               0)
})

test_succeeds("Define a batch norm bijector", {
  dist <-
    tfd_transformed_distribution(distribution = tfd_normal(loc = 0, scale = 1),
                    bijector = tfb_batch_normalization())
  y <-
    tfd_normal(loc = 1, scale = 2) %>% tfd_sample(100)  # ~ N(1, 2)
  # normalizes using the mean and standard deviation of the current minibatch.
  x <- dist$bijector %>% tfb_inverse(y)  # ~ N(0, 1)
  expect_equal(x$shape, y$shape)
})

test_succeeds("Define a blockwise bijector", {

  b <-
    tfb_blockwise(list(tfb_exp(), tfb_sigmoid()), block_sizes = list(2, 1))
  x <- matrix(rep(23, 5 * 3), ncol = 3)
  y <- b %>% tfb_forward(x)
  expect_equal(y$shape %>% length(), 2)
})

test_succeeds("Define a chain of bijectors", {
  b <- tfb_chain(list(tfb_exp(), tfb_sigmoid()))
  expect_equal(b$bijectors %>% length(), 2)
})

test_succeeds("Define a Cholesky outer product bijector", {
  b <- tfb_cholesky_outer_product()
  x <- matrix(c(1, 0, 2, 1), ncol = 2, byrow = TRUE)
  y <- matrix(c(1, 2, 2, 5), byrow = TRUE, nrow = 2)
  expect_equal(b %>% tfb_forward(x) %>% tensor_value(), y)
  expect_equal(b %>% tfb_inverse(y) %>% tensor_value(), x)
})

test_succeeds("Define a Cholesky to inverse Cholesky bijector", {

  x <- matrix(c(1, 0, 2, 1), ncol = 2, byrow = TRUE)

  b <- tfb_cholesky_to_inv_cholesky()

  # since 0.12:
  # NotImplementedError: Subclasses without static min_event_ndims must override `inverse_event_ndims`
  # chain <- tfb_chain(list(
  #   tfb_invert(tfb_cholesky_outer_product()),
  #   tfb_inline(
  #     forward_fn = tf$linalg$inv,
  #     forward_min_event_ndims = 1
  #   ),
  #   tfb_cholesky_outer_product()
  # ))
  #
  # expect_equal(b %>% tfb_forward(x) %>% tensor_value(),
  #              chain %>% tfb_forward(x) %>% tensor_value())

  o <- tfb_cholesky_outer_product()
  i <- tfb_inline(
    forward_fn = tf$linalg$inv,
    forward_min_event_ndims = 1
  )
  io <- tfb_invert(tfb_cholesky_outer_product())
  # TypeError: forward() takes 2 positional arguments but 3 were given
  # c <- io %>% tfb_forward(i %>% tfb_forward(o %>% tfb_forward(x)))
  c <- io$forward(i %>% tfb_forward(o %>% tfb_forward(x)))
  expect_equal(b %>% tfb_forward(x) %>% tensor_value(),
               c %>% tensor_value())
})

test_succeeds("Define a discrete cosine transform bijector", {
  b <- tfb_discrete_cosine_transform()
  x <- matrix(runif(100))
  y <- b %>% tfb_forward(x)
  expect_equal(x, b %>% tfb_inverse(y) %>% tensor_value(), tolerance = 1e-6)
})

test_succeeds("Define an expm1 bijector", {
  b <- tfb_expm1()
  chain <- tfb_chain(list(tfb_shift(-1), tfb_exp()))
  x <- matrix(1.1:4.1, ncol = 2, byrow = TRUE)
  expect_equal(b %>% tfb_forward(x) %>% tensor_value(),
               chain %>% tfb_forward(x) %>% tensor_value(),
               tol = 1e-6)

  b <- tfb_expm1()
  x <- matrix(1:8, nrow = 4, byrow = TRUE)
  y <- b %>% tfb_forward(x)
  expect_equal(b %>% tfb_inverse(y), tf$math$log1p(tf$cast(x, tf$float32)))
})

test_succeeds("Define a fill_triangular bijector", {
  b <- tfb_fill_triangular()
  x <- 1:6
  expect_equal((b %>% tfb_forward(x))$shape$ndims, 2)
})

test_succeeds("Define a Gumbel bijector", {
  skip_if_tfp_above("0.9")
  b <- tfb_gumbel()
  x <- runif(6)
  expect_equal(b %>% tfb_forward(x) %>% tensor_value(),
               tf$exp(-tf$exp(-x)) %>% tensor_value())
})

test_succeeds("Define an inline bijector", {
  b <- tfb_inline(
    forward_fn = tf$math$exp,
    inverse_fn = tf$math$log,
    inverse_log_det_jacobian_fn = (function(y)
      - tf$reduce_sum(tf$math$log(y), axis = -1)),
    forward_min_event_ndims = 0
  )
  x <- runif(6)
  expect_equal(b %>% tfb_forward(x) %>% tensor_value(),
               tfb_exp() %>% tfb_forward(x) %>% tensor_value())
})

test_succeeds("Define an invert bijector", {
  inner <- tfb_identity()
  b <- tfb_invert(inner)
  x <- runif(6)
  expect_equal(b$inverse(x), inner %>% tfb_forward(x))
  # inverse() takes 2 positional arguments but 3 were given
  # if doing
  # expect_equal(b %>% tfb_inverse(x), inner %>% tfb_forward(x))
})

test_succeeds("Define a kumaraswamy bijector", {
  skip_if_tfp_above("0.8")
  b <- tfb_kumaraswamy(concentration1 = 2, concentration0 = 0.3)
  x <- runif(1)
  expect_lte(b %>% tfb_forward(x) %>% tensor_value(), 1)
})



test_succeeds("Define an ordered bijector", {
  skip_if_tfp_above("0.11")
  b <- tfb_ordered()
  x <- seq(0, 1, by = 0.1)
  y <- b %>% tfb_forward(x)
  expect_equivalent(b %>% tfb_inverse(y) %>% tensor_value(), x, tol = 1e-6)
})

test_succeeds("Define a softplus bijector", {
  # from the original docs:
  # " works only on Tensors with 1 batch ndim and 2 event ndims (i.e., vector of matrices)."
  b <- tfb_softplus()
  x <- matrix(1:8, ncol = 2, byrow = TRUE)
  y <- b %>% tfb_forward(x)
  rev_x <- b %>% tfb_inverse(x)
  expect_equivalent(y %>% tensor_value(), log(1 + exp(x)), tol = 1e-6)
  expect_equivalent(rev_x %>% tensor_value(), log(exp(x) - 1), tol = 1e-6)
})

test_succeeds("Define a softsign bijector", {
  b <- tfb_softsign()
  x <- matrix(1:8, ncol = 2, byrow = TRUE)
  expect_equivalent(b %>% tfb_forward(x) %>% tensor_value(), x / (1 + abs(x)), tol = 1e-6)
  expect_equivalent(b %>% tfb_inverse(x) %>% tensor_value(), x / (1 - abs(x)), tol = 1e-6)
})

test_succeeds("Define a square bijector", {
  b <- tfb_square()
  x <- matrix(1:8, ncol = 2, byrow = TRUE)
  expect_equivalent(b %>% tfb_forward(x) %>% tensor_value(), x ^ 2)
})

test_succeeds("Define a tanh bijector", {
  skip("tfp$bijectors$Affine removed, wrapper needs fixing")
  # https://github.com/tensorflow/probability/commit/9afe4303054055959d631407cd2ef091a6c11fc0
  b <- tfb_tanh()
  chain <-
    tfb_chain(list(
      tfb_affine(shift = -1, scale_identity_multiplier = 2),
      tfb_sigmoid(),
      tfb_affine(scale_identity_multiplier = 2)
    ))
  x <- matrix(1:8, ncol = 2, byrow = TRUE)
  expect_equal(b %>% tfb_forward(x) %>% tensor_value(),
               chain %>% tfb_forward(x) %>% tensor_value(),
               tol = 1e-6)
})

test_succeeds("Define a transform_diagonal bijector", {
  b <- tfb_transform_diagonal(tfb_exp())
  x <- diag(2)
  expect_equivalent(b %>% tfb_forward(x) %>% tensor_value(), exp(1) * x, tol = 1e-7)
})

test_succeeds("Define a transpose bijector", {
  b <- tfb_transpose(rightmost_transposed_ndims = 2)
  x <- matrix(1:8, ncol = 2, byrow = TRUE)
  expect_equivalent(b %>% tfb_forward(x) %>% tensor_value(),
                    matrix(1:8, ncol = 2, byrow = FALSE))
})

test_succeeds("Define a weibull bijector", {
  skip_if_tfp_above("0.9")
  b <- tfb_weibull(1.5, 2)
  x <- c(0, 0.1, 0.2)
  expect_equivalent(b %>% tfb_forward(x) %>% tensor_value(),
                    -tf$math$expm1(-((x / 1.5) ** 2)) %>% tensor_value())
})

test_succeeds("Define a reshape bijector", {
  b <- tfb_reshape(event_shape_out = c(1,-1))
  x <- c(1, 2)
  y <- b %>% tfb_forward(x)
  expect_equal(y$shape$as_list(), c(1, 2))

  x <- matrix(1:4, ncol = 2, byrow = TRUE)
  y <- b %>% tfb_forward(x)
  expect_equal(y$shape$as_list(), c(2, 1, 2))

  x <- matrix(3:4, ncol = 2, byrow = TRUE)
  y <- b %>% tfb_inverse(x)
  expect_equal(y$shape$as_list(), c(2))
})

test_succeeds("Define a correlation_cholesky bijector", {

  x <- c(2, 2, 1)
  b <- tfb_correlation_cholesky()
  y <- b %>% tfb_forward(x)
  # Result: [[ 1.        ,  0.        ,  0.        ],
  #          [ 0.70710678,  0.70710678,  0.        ],
  #          [ 0.66666667,  0.66666667,  0.33333333]]
  rev_x <- b %>% tfb_inverse(y)
  expect_equivalent(rev_x %>% tensor_value(), x)
})

test_succeeds("Define a cumsum bijector", {

  skip_if_tfp_below("0.8")

  x <- rep(1, 5)
  y <- cumsum(x)
  b <- tfb_cumsum()

  rev_x <- b %>% tfb_inverse(y)
  expect_equivalent(rev_x %>% tensor_value(), x)
})

test_succeeds("Define a iterated_sigmoid_centered bijector", {

  skip_if_tfp_below("0.8")

  x <- runif(10)
  b <- tfb_iterated_sigmoid_centered()

  f_x <- b %>% tfb_forward(x)
  rev_x <- b %>% tfb_inverse(f_x)
  expect_equivalent(rev_x %>% tensor_value(), x, tol = 1e-6)
})

test_succeeds("Define a shift bijector", {

  skip_if_tfp_below("0.9")

  b <- tfb_shift(0.77)
  x <- 0
  y <- b %>% tfb_forward(x)
  expect_equal(y %>% tensor_value(), 0.77, tolerance = 1e-7)
})

test_succeeds("Define a pad bijector", {

  skip_if_tfp_below("0.9")

  b <- tfb_pad()
  y <- b %>% tfb_forward(c(1, 2))
  expect_equal(y$shape$as_list(), c(3))
  y <- b %>% tfb_forward(list(c(1, 2), c(3, 4)))
  expect_equal(y$shape$as_list(), c(2, 3))

  b <- tfb_pad(axis = -2)
  y <- b %>% tfb_forward(list(list(1, 2)))
  expect_equal(y$shape$as_list(), c(2, 2))
  x <- b %>% tfb_inverse(list(list(1, 2), list(3, 4)))
  expect_equal(x$shape$as_list(), c(1, 2))

})

test_succeeds("Define a scale_matvec_diag bijector", {

  skip_if_tfp_below("0.9")

  scale_diag <- c(2, 20)
  b <- tfb_scale_matvec_diag(scale_diag)
  x <- matrix(1:4, ncol = 2) %>% tf$cast(tf$float32)
  y <- b %>% tfb_forward(x)
  expect_equal(y %>% tensor_value(),
               tf$matmul(x, tf$linalg$diag(scale_diag)) %>% tensor_value())

})

test_succeeds("Define a scale_matvec_linear_operator bijector", {

  skip_if_tfp_below("0.9")

  x <- c(1, 2, 3)
  diag <- c(1, 2, 3)
  scale <- tf$linalg$LinearOperatorDiag(diag)
  b <- tfb_scale_matvec_linear_operator(scale)
  y <- b %>% tfb_forward(x)
  expect_equal(y %>% tensor_value() %>% as.matrix(),
               tf$matmul(scale, as.matrix(x) %>% tf$cast(tf$float32)) %>%
                 tensor_value())

  tril <- list(c(1, 0, 0),
               c(2, 1, 0),
               c(3, 2, 1))
  scale <- tf$linalg$LinearOperatorLowerTriangular(tril)
  b <- tfb_scale_matvec_linear_operator(scale)
  y <- b %>% tfb_forward(x)
  expect_equal(y %>% tensor_value() %>% as.matrix(),
               tf$matmul(scale, as.matrix(x) %>% tf$cast(tf$float32)) %>%
                 tensor_value())
})

test_succeeds("Define a scale_matvec_lu bijector", {

  skip_if_tfp_below("0.9")

  trainable_lu_factorization <- function(event_size,
                                         batch_shape = list(),
                                         seed = NULL,
                                         dtype = tf$float32,
                                         name = "scale_matvec_lu_test") {
    with(tf$name_scope(name), {
      event_size <- tf$convert_to_tensor(event_size,
                                         dtype_hint = tf$int32,
                                         name = 'event_size')
      batch_shape <- tf$convert_to_tensor(batch_shape,
                                          dtype_hint = event_size$dtype,
                                          name = 'batch_shape')
      random_matrix <- tf$random$uniform(
        shape = tf$concat(list(
          batch_shape, list(event_size, event_size)
        ), axis = 0L),
        dtype = dtype,
        seed = seed
      )
      random_orthonormal <-
        tf$linalg$qr(random_matrix)[0] # qr returns tuple of tensors
      lu <- tf$linalg$lu(random_orthonormal)
      lower_upper <- lu[0]
      permutation <- lu[1]
      lower_upper <- tf$Variable(
        initial_value = lower_upper,
        trainable = TRUE,
        name = 'lower_upper'
      )
      # Initialize a non-trainable variable for the permutation indices so
      # that its value isn't re-sampled from run-to-run.
      permutation <- tf$Variable(
        initial_value = permutation,
        trainable = FALSE,
        name = 'permutation')
    })
    list(lower_upper, permutation)
  }

  channels <- 3L
  fact <- trainable_lu_factorization(channels)
  conv1x1 <- tfb_scale_matvec_lu(fact[[1]],
                           fact[[2]],
                           validate_args = TRUE)
  x <- tf$random$uniform(shape = list(2L, 28L, 28L, channels))
  y <- conv1x1$forward(x)
  y_inv = conv1x1$inverse(y)
  expect_equal(x %>% tensor_value(), y_inv %>% tensor_value(), tol = 1e-6)

})

test_succeeds("Define a scale_matvec_tri_l bijector", {

  skip_if_tfp_below("0.9")

  scale <- matrix(c(2, 0, 2, 2), ncol = 2, byrow = TRUE)  %>% tf$cast(tf$float32)
  b <- tfb_scale_matvec_tri_l(scale)
  x <- c(1, 2)
  y <- b %>% tfb_forward(x)
  expect_equal(y %>% tensor_value() %>% as.matrix(),
               tf$matmul(scale, as.matrix(x)%>% tf$cast(tf$float32)) %>% tensor_value())

})

test_succeeds("Define a GumbelCDF bijector", {

  skip_if_tfp_below("0.9")

  b <- tfb_gumbel_cdf()
  x <- runif(6)
  expect_equal(b %>% tfb_forward(x) %>% tensor_value(),
               tf$exp(-tf$exp(-x)) %>% tensor_value())
})

test_succeeds("Define a weibullCDF bijector", {

  skip_if_tfp_below("0.9")

  b <- tfb_weibull_cdf(1.5, 2)
  x <- c(0, 0.1, 0.2)
  expect_equivalent(b %>% tfb_forward(x) %>% tensor_value(),
                    -tf$math$expm1(-((x / 1.5) ** 2)) %>% tensor_value())
})


test_succeeds("Define a kumaraswamyCDF bijector", {

  skip_if_tfp_below("0.9")

  b <- tfb_kumaraswamy_cdf(concentration1 = 2, concentration0 = 0.3)
  x <- runif(1)
  expect_lt(b %>% tfb_forward(x) %>% tensor_value(), 1)
})

test_succeeds("Define a scale bijector", {

  skip_if_tfp_below("0.9")

  scale <- 1.5
  b <- tfb_scale(scale = scale)
  x <- matrix(1:4, ncol = 2) %>% tf$cast(tf$float32)
  y <- b %>% tfb_forward(x)
  expect_equal(y %>% tensor_value(),
               matrix(1:4, ncol = 2) * scale)

})

test_succeeds("Define a fill_scale_tri_l bijector", {

  skip_if_tfp_below("0.9")

  b <- tfb_fill_scale_tri_l(tfb_exp(), NULL)
  x <- c(0, 0, 0)
  expect_equal(b %>% tfb_forward(x) %>% tensor_value(), diag(2))
  y <- matrix(c(1, 0, .5, 2), byrow = TRUE, ncol = 2)
  expect_equivalent(b %>% tfb_inverse(y) %>% tensor_value(), c(log(2), .5, log(1)), tol = 1e-6)
})

test_succeeds("Define an FFJORD bijector", {

  skip_if_tfp_below("0.9")

  # state_time_derivative_fn: `Callable(time, state)` -> state_time_derivative
  move_ode_fn <- function(t, z) tf$ones_like(z)
  trace_augmentation_fn <- tfp$bijectors$ffjord$trace_jacobian_exact

  b <- tfb_ffjord(state_time_derivative_fn = move_ode_fn,
                        trace_augmentation_fn = trace_augmentation_fn)

  x <- matrix(rep(0, 10), ncol = 5)
  y <- matrix(rep(1, 10), ncol = 5)
  expected_log_det_jacobian <- array(c(0, 0), dim = 2)

  expect_equal(b %>% tfb_forward(x) %>% tensor_value(), y, tol = 1e-6)
  expect_equal(b %>% tfb_inverse(y) %>% tensor_value(), x, tol = 1e-6)
  expect_equal(
    b %>% tfb_inverse_log_det_jacobian(y, event_ndims = 1) %>% tensor_value(),
    expected_log_det_jacobian,
    tol = 1e-6
  )

})

test_succeeds("Define a LambertWTail bijector", {

  skip_if_tfp_below("0.10")

  scale <- 1
  shift <- 0
  tailweight <- 0
  b <- tfb_lambert_w_tail(shift, scale, tailweight)
  x <- matrix(1:4, ncol = 2) %>% tf$cast(tf$float32)
  y <- b %>% tfb_forward(x)
  expect_equal(y %>% tensor_value(), x %>% tensor_value())

})

test_succeeds("Define a Split bijector", {

  skip_if_tfp_below("0.11")

  b <- tfb_split(num_or_size_splits = c(3, 2, 1),
                 axis = -2)

  x <- array(0, dim = c(6, 6, 6))
  y <- b %>% tfb_forward(x)
  x_ <- b %>% tfb_inverse(y)
  expect_equal(x %>% tensor_value() %>% length(), x_ %>% tensor_value() %>% length())

})

test_succeeds("Define a GompertzCDF bijector", {

  skip_if_tfp_below("0.11")

  b <- tfb_gompertz_cdf(concentration = 1, rate = 1)

  x <- 1
  y <- b %>% tfb_forward(x)
  expect_equal(y %>% tensor_value(),  1 - exp(-1 * (exp(1 * x) - 1)), tol = 1e-6)

})

test_succeeds("Define a ShiftedGompertzCDF bijector", {

  skip_if_tfp_below("0.11")

  b <- tfb_shifted_gompertz_cdf(concentration = 1, rate = 1)

  x <- 1
  y <- b %>% tfb_forward(x)
  expect_equal(y %>% tensor_value(),  (1 - exp(-1 * x)) * exp(-1 * exp(-1 * x)), tol = 1e-6)

})

test_succeeds("Define a sinh bijector", {

  skip_if_tfp_below("0.11")

  b <- tfb_sinh()
  x <- 1
  expect_equal(b %>% tfb_forward(x) %>% tensor_value(),
               tf$sinh(x) %>% tensor_value(),
               tol = 1e-6)
})

test_succeeds("Define a RayleighCDF bijector", {

  skip_if_tfp_below("0.12")

  scale <- 0.5
  b <- tfb_rayleigh_cdf(scale = scale)
  x <- 0.77
  expect_equal(b %>% tfb_forward(x) %>% tensor_value(),
               1 - exp(-(x/scale)^2 / 2),
               tol = 1e-6)
})

test_succeeds("Define an ascending bijector", {
  skip_if_tfp_below("0.12")
  b <- tfb_ascending()
  y <- c(2, 3, 4)
  x <- c(2, 0, 0)
  expect_equivalent(b %>% tfb_inverse(y) %>% tensor_value(), x, tol = 1e-6)
})

test_succeeds("Define a Glow bijector", {
  skip_if_tfp_below("0.12")
  image_shape <- c(32, 32, 4)

  glow <- tfb_glow(
    output_shape = image_shape,
    coupling_bijector_fn = tfp$bijectors$GlowDefaultNetwork,
    exit_bijector_fn = tfp$bijectors$GlowDefaultExitNetwork
  )

  pz <- tfd_sample_distribution(tfd_normal(0, 1), prod(image_shape))
  px <- glow(pz)
  image <- px$sample(1L)

  expect_equal(dim(image), c(1, 32, 32, 4))
})







