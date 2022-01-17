
context("tensorflow probability distribution layers")

test_succeeds("can use layer_multivariate_normal_tri_l in a keras model", {
  library(keras)
  n <- as.integer(1e3)
  scale_tril <-
    matrix(c(1.6180, 0,-2.7183, 3.1416),
           ncol = 2,
           byrow = TRUE) %>% keras::k_cast_to_floatx()
  scale_noise <- 0.01
  x <- tfd_normal(loc = 0, scale = 1) %>% tfd_sample(c(n, 2L))
  eps <-
    tfd_normal(loc = 0, scale = scale_noise) %>% tfd_sample(c(n, 2L))
  y = tf$matmul(x, scale_tril) + eps
  d <- if (tf$executing_eagerly()) y$shape[[2]] else y$shape[[2]]$value

  model <- keras_model_sequential() %>%
    layer_dense(units = params_size_multivariate_normal_tri_l(d)) %>%
    layer_multivariate_normal_tri_l(event_size = d)

  log_loss <- function (y, model) - (model %>% tfd_log_prob(y))

  model %>% compile(optimizer = "adam",
                    loss = log_loss)

  model %>% fit(
    x,
    y,
    batch_size = 100,
    epochs = 1,
    steps_per_epoch = 10
  )
})

test_succeeds("can use layer_kl_divergence_add_loss in a keras model", {
  library(keras)
  encoded_size <- 2
  input_shape <- c(2L, 2L, 1L)
  train_size <- 100
  x_train <-
    array(runif(train_size * Reduce(`*`, input_shape)), dim = c(train_size, input_shape))

  encoder_model <- keras_model_sequential() %>%
    layer_flatten(input_shape = input_shape) %>%
    layer_dense(units = 10, activation = "relu") %>%
    layer_dense(units = params_size_multivariate_normal_tri_l(encoded_size)) %>%
    layer_multivariate_normal_tri_l(event_size = encoded_size) %>%
    layer_kl_divergence_add_loss(
      distribution = tfd_independent(
        tfd_normal(loc = c(0, 0), scale = 1),
        reinterpreted_batch_ndims = 1L
      ),
      weight = train_size
    )

  decoder_model <- keras_model_sequential() %>%
    layer_dense(units = 10,
                activation = 'relu',
                input_shape = encoded_size) %>%
    layer_dense(params_size_independent_bernoulli(input_shape)) %>%
    layer_independent_bernoulli(event_shape = input_shape,
                                convert_to_tensor_fn = tfp$distributions$Bernoulli$logits)

  vae_model <- keras_model(inputs = encoder_model$inputs,
                           outputs = decoder_model(encoder_model$outputs[1]))

  vae_loss <- function (x, rv_x)
    - (rv_x %>% tfd_log_prob(x))

  vae_model %>% compile(
    optimizer = tf$keras$optimizers$Adam(),
    loss = vae_loss
  )

  vae_model %>% fit(x_train,
                    x_train,
                    batch_size = 25,
                    epochs = 1)

})

test_succeeds("can use layer_independent_bernoulli in a keras model", {
  library(keras)
  n <- as.integer(1e3)
  scale_tril <-
    matrix(c(1.6180, 0.,-2.7183, 3.1416),
           ncol = 2,
           byrow = TRUE) %>% keras::k_cast_to_floatx()
  scale_noise <- 0.01
  x <- tfd_normal(loc = 0, scale = 1) %>% tfd_sample(c(n, 2L))
  eps <-
    tfd_normal(loc = 0, scale = scale_noise) %>% tfd_sample(c(1000L, 2L))
  y <-
    tfd_bernoulli(logits = tf$reshape(tf$matmul(x, scale_tril) + eps,
                                               shape = shape(n, 1L, 2L, 1L))) %>% tfd_sample()

  event_shape <- dim(y)[2:4]

  model <- keras_model_sequential() %>%
    layer_dense(units = params_size_independent_bernoulli(event_shape),
                input_shape = dim(x)[2]) %>%
    layer_independent_bernoulli(event_shape = event_shape)

  log_loss <- function (y, model)
    - (model %>% tfd_log_prob(y))

  model %>% compile(optimizer = "adam",
                    loss = log_loss)

  model %>% fit(
    x,
    y,
    batch_size = 100,
    epochs = 1,
    steps_per_epoch = n / 100
  )

})

test_succeeds("layer_distribution_lambda() works", {
  l <- layer_distribution_lambda(
    make_distribution_fn = function(x) tfd_relaxed_one_hot_categorical(temperature = 1e-5, logits = x)
  )

  s <- keras::k_constant(c(1e5, -1e5)) %>%
    l() %>%
    tfd_sample(10) %>%
    tensor_value()

  expect_equal(s, array(c(rep(1, 10), rep(0, 10)), dim = c(10, 2)))
})


test_succeeds("can use layer_one_hot_categorical in a keras model", {

  library(keras)

  d <- 3
  n <- 7L

  model <- keras_model_sequential(
    list(
      layer_dense(units = params_size_one_hot_categorical(d) - 1, input_shape = n),
      layer_lambda(f = function(x) tf$pad(x, paddings = list(list(0L, 0L), list(1L, 0L)))),
      layer_one_hot_categorical(event_size = d)
    )
  )

  # removed check in 0.12 because model$output_shape does not exist anymore
  # see https://github.com/tensorflow/probability/issues/1200
  # Message: AttributeError: 'UserRegisteredSpec' object has no attribute '_shape'
  # expect_equal(model$output_shape[[2]], d)

  m <- matrix(1:14, ncol = n)
  res <- model(m)
  expect_equal(res$event_shape[[1]], d)

})

test_succeeds("can use layer_categorical_mixture_of_one_hot_categorical in a keras model", {

  library(keras)

  k <- 3
  d <- 5
  n <- 7
  model <- keras_model_sequential(
    list(
      layer_dense(units = params_size_categorical_mixture_of_one_hot_categorical(d, k), input_shape = n),
      layer_categorical_mixture_of_one_hot_categorical(event_size = d, num_components = k)
    )
  )
  expect_equal(model$layers %>% length(), 2)

})

test_succeeds("can use layer_independent_poisson in a keras model", {

  library(keras)

  d <- 1
  n <- 2
  model <- keras_model_sequential(
    list(
      layer_dense(units = params_size_independent_poisson(d), input_shape = n),
      layer_independent_poisson(event_shape = d)
    )
  )

  # removed check in 0.12 because model$output_shape does not exist anymore
  # see https://github.com/tensorflow/probability/issues/1200
  # Message: AttributeError: 'UserRegisteredSpec' object has no attribute '_shape'
  # expect_equal(model$output_shape[[2]], d)

  m <- matrix(1:14, ncol = n)
  res <- model(m)
  expect_equal(res$event_shape[[1]], d)

})

test_succeeds("can use layer_independent_logistic in a keras model", {

  library(keras)

  input_shape <- c(28L, 28L, 1L)
  encoded_shape <- 2
  n <- 3

  model <- keras_model_sequential() %>%
      layer_flatten(input_shape = input_shape) %>%
      layer_dense(units = n) %>%
      layer_dense(units = params_size_independent_logistic(encoded_shape)) %>%
      layer_independent_logistic(event_shape = encoded_shape)

  # removed check in 0.12 because model$output_shape does not exist anymore
  # see https://github.com/tensorflow/probability/issues/1200
  # Message: AttributeError: 'UserRegisteredSpec' object has no attribute '_shape'
  # expect_equal(model$output_shape[[2]], encoded_shape)

  a <- array(28 * 28, dim = c(1L, input_shape))
  res <- model(a)
  expect_equal(res$event_shape[[1]], encoded_shape)


})

test_succeeds("can use layer_independent_normal in a keras model", {

  library(keras)

  input_shape <- c(28, 28, 1)
  encoded_shape <- 2
  n <- 2

  model <- keras_model_sequential() %>%
    layer_flatten(input_shape = input_shape) %>%
    layer_dense(units = n) %>%
    layer_dense(units = params_size_independent_normal(encoded_shape)) %>%
    layer_independent_normal(event_shape = encoded_shape)

  # removed check in 0.12 because model$output_shape does not exist anymore
  # see https://github.com/tensorflow/probability/issues/1200
  # Message: AttributeError: 'UserRegisteredSpec' object has no attribute '_shape'
  # expect_equal(model$output_shape[[2]], encoded_shape)

  a <- array(28 * 28, dim = c(1L, input_shape))
  res <- model(a)
  expect_equal(res$event_shape[[1]], encoded_shape)

})

test_succeeds("can use layer_mixture_same_family in a keras model", {

  library(keras)

  event_shape <- 1
  num_components <- 5
  params_size <- params_size_mixture_same_family(num_components,
                                                 component_params_size = params_size_independent_normal(event_shape))

  model <- keras_model_sequential(list(
    layer_dense(units = 12, activation = "relu", input_shape = list(2)),
    layer_dense(units = params_size),
    layer_mixture_same_family(
      num_components = num_components,
      component_layer = layer_independent_normal(event_shape = event_shape)
    )
  ))

  # removed check in 0.12 because model$output_shape does not exist anymore
  # see https://github.com/tensorflow/probability/issues/1200
  # Message: AttributeError: 'UserRegisteredSpec' object has no attribute '_shape'
  # expect_equal(model$output_shape[[2]], event_shape)

  m <- matrix(1:4, ncol = 2)
  res <- model(m)
  expect_equal(res$event_shape[[1]], event_shape)

})

test_succeeds("can use layer_mixture_normal in a keras model", {

  library(keras)

  event_shape <- 1
  num_components <- 5
  params_size <- params_size_mixture_normal(num_components, event_shape)

  model <- keras_model_sequential(list(
    layer_dense(units = 12, activation = "relu", input_shape = list(2)),
    layer_dense(units = params_size),
    layer_mixture_normal(
      num_components = num_components,
      event_shape = event_shape)
  ))

  # removed check in 0.12 because model$output_shape does not exist anymore
  # see https://github.com/tensorflow/probability/issues/1200
  # Message: AttributeError: 'UserRegisteredSpec' object has no attribute '_shape'
  # expect_equal(model$output_shape[[2]], event_shape)

  m <- matrix(1:4, ncol = 2)
  res <- model(m)
  expect_equal(res$event_shape[[1]], event_shape)
})

test_succeeds("can use layer_mixture_logistic in a keras model", {

  library(keras)

  event_shape <- 1
  num_components <- 5
  params_size <- params_size_mixture_logistic(num_components, event_shape)

  model <- keras_model_sequential(list(
    layer_dense(units = 12, activation = "relu", input_shape = list(2)),
    layer_dense(units = params_size),
    layer_mixture_normal(
      num_components = num_components,
      event_shape = event_shape)
  ))

  # removed check in 0.12 because model$output_shape does not exist anymore
  # see https://github.com/tensorflow/probability/issues/1200
  # Message: AttributeError: 'UserRegisteredSpec' object has no attribute '_shape'
  # expect_equal(model$output_shape[[2]], encoded_shape)

  m <- matrix(1:4, ncol = 2)
  res <- model(m)
  expect_equal(res$event_shape[[1]], event_shape)
})

test_succeeds("layer_variational_gaussian_process works", {

  skip_if_tf_below("2.0")

  library(keras)

  x <- runif(1000, 0, 60)
  y <- 5 + sin(2*x) + 0.04*x

  bt <- reticulate::import_builtins()

  KernelFn <- reticulate::PyClass(
    "KernelFn",
    inherit = tensorflow::tf$keras$layers$Layer,
    list(
      `__init__` = function(self, ...) {
        super()$`__init__`(...)
        self$`_amplitude` = self$add_variable(
          initializer = tf$initializers$constant(.54),
          dtype = 'float32',
          name = 'amplitude'
        )
        NULL
      },

      call = function(self, x, ...) {
        x
      },

      kernel = bt$property(
       reticulate::py_func(
          function(self)
            (get_psd_kernels())$ExponentiatedQuadratic(amplitude = tf$nn$softplus(self$`_amplitude`))
        )
      )
    )
  )

  KernelFn$`__module__` <- "KernelFn"

  model <- keras_model_sequential() %>%
    layer_dense(units = 10) %>%
    layer_variational_gaussian_process(
      num_inducing_points = 30,
      kernel_provider = KernelFn()
    )

  batch_size <- 64
  kl_weight <- batch_size/length(x)
  loss <- function(y, d) {
    d$variational_loss(y, kl_weight=as_tensors(kl_weight))
  }

  model %>%
    compile(optimizer = "adam", loss = loss)

  x <- as_tensors(x)
  y <- as_tensors(y)

  x <- tf$expand_dims(x, -1L)
  y <- tf$expand_dims(y, -1L)

  model %>%
    fit(x = x, y = y, batch_size = batch_size, verbose = 0L)
})










