
context("tensorflow probability keras layers")

source("utils.R")

test_succeeds("can use layer_autoregressive in a keras model", {

  skip_if_tfp_below("0.7")

  library(keras)
  n <- 2000
  x2 <- rnorm(n) %>% tf$cast(tf$float32) * 2
  x1 <- rnorm(n) %>% tf$cast(tf$float32) + (x2 * x2 / 4)
  data <- tf$stack(list(x1, x2), axis = -1L)

  made <- layer_autoregressive(params = 2, hidden_units = list(10, 10))
  distribution <- tfd_transformed_distribution(
    distribution = tfd_normal(loc = 0, scale = 1),
    bijector = tfb_masked_autoregressive_flow(
      function(x) tf$unstack(made(x), num = 2L, axis = -1L)),
    event_shape = list(2))

  x_ <- layer_input(shape = c(2), dtype = "float32")
  log_prob_ <- distribution$log_prob(x_)
  model <- keras_model(x_, log_prob_)
  loss <- function(x, log_prob) -log_prob
  model %>% compile(optimizer = "adam", loss = loss)

  model %>% fit(x = data,
            y = rep(0, n),
            batch_size = 25,
            epochs = 1,
            steps_per_epoch = 1,
            verbose = 0)

  expect_equal((distribution %>% tfd_sample(c(3, 1)))$get_shape()$as_list(), c(3, 1, 2))
  expect_equal((distribution %>% tfd_log_prob(matrix(rep(1, 3*2), ncol = 2)))$get_shape()$as_list(), c(3))
})


# `AutoregressiveLayer` can be used as a building block to achieve different
# autoregressive structures over rank-2+ tensors.  For example, suppose we want
# to build an autoregressive distribution over images with dimension
# `[weight, height, channels]` with `channels = 3`:
# We can parameterize a "fully autoregressive" distribution, with
# cross-channel and within-pixel autoregressivity:
#   ```
# r0    g0   b0     r0    g0   b0       r0   g0    b0
# ^   ^      ^         ^   ^   ^         ^      ^   ^
# |  /  ____/           \  |  /           \____  \  |
# | /__/                 \ | /                 \__\ |
# r1    g1   b1     r1 <- g1   b1       r1   g1 <- b1
# ^          |
#   \_________/

test_succeeds("can use layer_autoregressive to model rank-3 tensors with full autoregression", {

  skip_if_tfp_below("0.7")

  library(keras)

  n <- 1000L
  width <- 8L
  height <- 8L
  channels <- 3L
  images <-
    runif(n * height * width * channels) %>% array(dim = c(n, height, width, channels)) %>%
    tf$cast(tf$float32)

  # Reshape images to achieve desired autoregressivity.
  event_shape <- height * width * channels
  reshaped_images <- tf$reshape(images, c(n, event_shape))

  made <-
    layer_autoregressive(
      params = 2,
      event_shape = event_shape,
      hidden_units = list(20, 20),
      activation = "relu"
    )

  distribution <- tfd_transformed_distribution(
    distribution = tfd_normal(loc = 0, scale = 1),
    bijector = tfb_masked_autoregressive_flow(function (x)
      tf$unstack(
        made(x), num = 2, axis = -1L
      )),
    event_shape = event_shape
  )

  x_ <- layer_input(shape = event_shape, dtype = "float32")
  log_prob_ <- distribution %>% tfd_log_prob(x_)

  model <- keras_model(x_, log_prob_)
  loss <- function(x, log_prob)
    - log_prob
  model %>% compile(optimizer = "adam", loss = loss)

  model %>% fit(
    x = reshaped_images,
    y = rep(0, n),
    batch_size = 10,
    epochs = 1,
    steps_per_epoch = 1,
    verbose = 0
  )

  expect_equal((distribution %>% tfd_sample(c(3, 1)))$get_shape()$as_list(),
               c(3, 1, 192))
  })
