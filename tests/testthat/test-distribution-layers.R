
context("tensorflow probability distribution layers")

source("utils.R")

test_succeeds("can use layer_multivariate_normal_tri_l in a keras model", {
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
  y = tf$matmul(x, scale_tril) + eps
  d <- y$shape[-1]$value

  model <- keras_model_sequential() %>%
    layer_dense(units = params_size_multivariate_normal_tri_l(d),
                input_shape = x$shape[-1]) %>%
    layer_multivariate_normal_tri_l(event_size = d)

  log_loss <- function (y, model)
    - (model %>% tfd_log_prob(x))

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
