
<!-- README.md is generated from README.Rmd. Please edit that file -->
<!-- badges: start -->

[![CRAN_Status_Badge](https://www.r-pkg.org/badges/version/tfprobability)](https://cran.r-project.org/package=tfprobability)
[![R-CMD-check](https://github.com/rstudio/tfprobability/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/rstudio/tfprobability/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

# tfprobability: R interface to TensorFlow Probability

[TensorFlow Probability](https://www.tensorflow.org/probability/) is a
library for statistical computation and probabilistic modeling built on
top of TensorFlow.

Its building blocks include a vast range of distributions and invertible
transformations (*bijectors*), probabilistic layers that may be used in
`keras` models, and tools for probabilistic reasoning including
variational inference and Markov Chain Monte Carlo.

## Installation

Install the released version of `tfprobability` from CRAN:

``` r
install.packages("tfprobability")
```

To install `tfprobability` from github, do

    devtools::install_github("rstudio/tfprobability")

Then, use the `install_tfprobability()` function to install TensorFlow
and TensorFlow Probability python modules.

``` r
library(tfprobability)
install_tfprobability()
```

you will automatically get the current stable version of TensorFlow
Probability together with TensorFlow. Correspondingly, if you need
nightly builds,

``` r
install_tfprobability(version = "nightly")
```

will get you the nightly build of TensorFlow as well as TensorFlow
Probability.

## Usage

High-level application of `tfprobability` to tasks like

-   probabilistic (multi-level) modeling with MCMC and/or variational
    inference,
-   uncertainty estimation for neural networks,
-   time series modeling with state space models, or
-   density estimation with autoregressive flows

are described in the vignettes/articles and/or featured on the
[TensorFlow for R blog](https://blogs.rstudio.com/ai/).

This introductory text illustrates the lower-level building blocks:
distributions, bijectors, and probabilistic `keras` layers.

``` r
library(tfprobability)
library(tensorflow)
```

### Distributions

Distributions are objects with methods to compute summary statistics,
(log) probability, and (optionally) quantities like entropy and KL
divergence.

#### Example: Binomial distribution

``` r
# create a binomial distribution with n = 7 and p = 0.3
d <- tfd_binomial(total_count = 7, probs = 0.3)

# compute mean
d %>% tfd_mean()
#> tf.Tensor(2.1000001, shape=(), dtype=float32)
# compute variance
d %>% tfd_variance()
#> tf.Tensor(1.47, shape=(), dtype=float32)
# compute probability
d %>% tfd_prob(2.3)
#> tf.Tensor(0.303791, shape=(), dtype=float32)
```

#### Example: Hidden Markov Model

``` r
# Represent a cold day with 0 and a hot day with 1.
# Suppose the first day of a sequence has a 0.8 chance of being cold.
# We can model this using the categorical distribution:
initial_distribution <- tfd_categorical(probs = c(0.8, 0.2))
#> Loaded Tensorflow version 2.9.1
# Suppose a cold day has a 30% chance of being followed by a hot day
# and a hot day has a 20% chance of being followed by a cold day.
# We can model this as:
transition_distribution <- tfd_categorical(
  probs = matrix(c(0.7, 0.3, 0.2, 0.8), nrow = 2, byrow = TRUE) %>%
    tf$cast(tf$float32)
)
# Suppose additionally that on each day the temperature is
# normally distributed with mean and standard deviation 0 and 5 on
# a cold day and mean and standard deviation 15 and 10 on a hot day.
# We can model this with:
observation_distribution <- tfd_normal(loc = c(0, 15), scale = c(5, 10))
# We can combine these distributions into a single week long
# hidden Markov model with:
d <- tfd_hidden_markov_model(
  initial_distribution = initial_distribution,
  transition_distribution = transition_distribution,
  observation_distribution = observation_distribution,
  num_steps = 7
)
# The expected temperatures for each day are given by:
d %>% tfd_mean()  # shape [7], elements approach 9.0
#> tf.Tensor([3.        6.        7.4999995 8.249999  8.625001  8.812501  8.90625  ], shape=(7), dtype=float32)
# The log pdf of a week of temperature 0 is:
d %>% tfd_log_prob(rep(0, 7))
#> tf.Tensor(-19.855635, shape=(), dtype=float32)
```

### Bijectors

Bijectors are invertible transformations that allow to derive data
likelihood under the transformed distribution from that under the base
distribution. For an in-detail explanation, see [Getting into the flow:
Bijectors in TensorFlow
Probability](https://blogs.rstudio.com/ai/posts/2019-04-05-bijectors-flows/)
on the TensorFlow for R blog.

#### Affine bijector

``` r
# create an affine transformation that shifts by 3.33 and scales by 0.5
b <- tfb_shift(3.33)(tfb_scale(0.5))

# apply the transformation
x <- c(100, 1000, 10000)
b %>% tfb_forward(x)
#> tf.Tensor([  53.33  503.33 5003.33], shape=(3), dtype=float32)
```

#### Discrete cosine transform bijector

``` r
# create a bijector to that performs the discrete cosine transform (DCT)
b <- tfb_discrete_cosine_transform()

# run on sample data
x <- matrix(runif(3))
b %>% tfb_forward(x)
#> tf.Tensor(
#> [[0.5221709 ]
#>  [0.5336635 ]
#>  [0.06735111]], shape=(3, 1), dtype=float32)
```

### Keras layers

`tfprobality` wraps distributions in Keras layers so we can use them
seemlessly in a neural network, and work with tensors as targets as
usual. For example, we can use `layer_kl_divergence_add_loss` to have
the network take care of the KL loss automatically, and train a
variational autoencoder with just negative log likelihood only, like
this:

``` r
library(keras)

encoded_size <- 2
input_shape <- c(2L, 2L, 1L)
train_size <- 100
x_train <- array(runif(train_size * Reduce(`*`, input_shape)), dim = c(train_size, input_shape))

# encoder is a keras sequential model
encoder_model <- keras_model_sequential() %>%
  layer_flatten(input_shape = input_shape) %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = params_size_multivariate_normal_tri_l(encoded_size)) %>%
  layer_multivariate_normal_tri_l(event_size = encoded_size) %>%
  # last layer adds KL divergence loss
  layer_kl_divergence_add_loss(
      distribution = tfd_independent(
        tfd_normal(loc = c(0, 0), scale = 1),
        reinterpreted_batch_ndims = 1
      ),
      weight = train_size)

# decoder is a keras sequential model
decoder_model <- keras_model_sequential() %>%
  layer_dense(units = 10,
              activation = 'relu',
              input_shape = encoded_size) %>%
  layer_dense(params_size_independent_bernoulli(input_shape)) %>%
  layer_independent_bernoulli(event_shape = input_shape,
                              convert_to_tensor_fn = tfp$distributions$Bernoulli$logits)

# keras functional model uniting them both
vae_model <- keras_model(inputs = encoder_model$inputs,
                         outputs = decoder_model(encoder_model$outputs[1]))

# VAE loss now is just log probability of the data
vae_loss <- function (x, rv_x)
    - (rv_x %>% tfd_log_prob(x))

vae_model %>% compile(
  optimizer = "adam",
  loss = vae_loss
)

vae_model
#> Model: "model"
#> ________________________________________________________________________________
#>  Layer (type)                       Output Shape                    Param #     
#> ================================================================================
#>  flatten_input (InputLayer)         [(None, 2, 2, 1)]               0           
#>  flatten (Flatten)                  (None, 4)                       0           
#>  dense_1 (Dense)                    (None, 10)                      50          
#>  dense (Dense)                      (None, 5)                       55          
#>  multivariate_normal_tri_l (Multiva  ((None, 2),                    0           
#>  riateNormalTriL)                    (None, 2))                                 
#>  kl_divergence_add_loss (KLDivergen  (None, 2)                      0           
#>  ceAddLoss)                                                                     
#>  sequential_1 (Sequential)          (None, 2, 2, 1)                 74          
#> ================================================================================
#> Total params: 179
#> Trainable params: 179
#> Non-trainable params: 0
#> ________________________________________________________________________________

vae_model %>% fit(x_train, x_train, batch_size = 25, epochs = 1)
```
