
<!-- README.md is generated from README.Rmd. Please edit that file -->

<!-- badges: start -->

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://www.tidyverse.org/lifecycle/#experimental)
[![Build
Status](https://travis-ci.org/rstudio/tfprobability.svg?branch=master)](https://travis-ci.org/rstudio/tfprobability)
[![codecov](https://codecov.io/gh/rstudio/tfprobability/branch/master/graph/badge.svg)](https://codecov.io/gh/rstudio/tfprobability)
<!-- badges: end -->

# tfprobability: R interface to TensorFlow Probability

[TensorFlow Probability](https://www.tensorflow.org/probability/) is a
library for statistical analysis and probabilistic computation built on
top of TensorFlow.

Its building blocks include a vast range of distributions and invertible
transformations (*bijectors*), probabilistic layers that may be used in
`keras` models, and tools for probabilistic reasoning including
variational inference and Markov Chain Monte Carlo.

## Installation

To install `tfprobability` from this repository, do

    devtools::install_github("rstudio/tfprobability")

TensorFlow Probability depends on TensorFlow, and in the same way,
`tfprobability` depends on a working installation of the R packages
`tensorflow` and `keras`. To get the most up-to-date versions of these
packages, install them from github as well:

    devtools::install_github("rstudio/tensorflow")
    devtools::install_github("rstudio/keras")

As to the Python backend, if you do

    library(tensorflow)
    install_tensorflow()

you will automatically get the current stable version of TensorFlow
Probability. Correspondingly, if you need nightly builds,

    install_tensorflow(version = "nightly")

will get you the nightly build of TensorFlow Probability.

## Usage

Over time, vignettes will be added to the package explaining the usage
of the various modules. Also, the [TensorFlow for R
blog](https://blogs.rstudio.com/tensorflow/) will feature interesting
applications and provide conceptual background.

Here are a few examples using distributions, bijectors, and
probabilistic `keras` layers. We enable eager execution (not *yet* the
default in TensorFlow) to display values, not tensors.

``` r
library(tfprobability)
library(tensorflow)
tfe_enable_eager_execution()
```

### Distributions

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
#> tf.Tensor(0.30379143, shape=(), dtype=float32)
```

#### Example: Hidden Markov Model

``` r
# Represent a cold day with 0 and a hot day with 1.
# Suppose the first day of a sequence has a 0.8 chance of being cold.
# We can model this using the categorical distribution:
initial_distribution <- tfd_categorical(probs = c(0.8, 0.2))
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
#> tf.Tensor([2.9999998 5.9999995 7.4999995 8.25      8.625001  8.812501  8.90625  ], shape=(7,), dtype=float32)
# The log pdf of a week of temperature 0 is:
d %>% tfd_log_prob(rep(0, 7)) 
#> tf.Tensor(-20.120832, shape=(), dtype=float32)
```

### Bijectors

#### Discrete cosine transform bijector

``` r
# create a bijector to that performs the discrete cosine transform (DCT)
b <- tfb_discrete_cosine_transform()

# run on sample data
x <- matrix(runif(10))
b %>% tfb_forward(x)
#> tf.Tensor(
#> [[0.81395715]
#>  [0.91802233]
#>  [0.21701626]
#>  [0.7964485 ]
#>  [0.5835809 ]
#>  [0.36585388]
#>  [0.43929157]
#>  [0.73955095]
#>  [0.7171121 ]
#>  [0.32015073]], shape=(10, 1), dtype=float32)
```

#### 

## Package State

This project is under active development. As of this writing,
`distributions` and `bijectors` are covered comprehensively, `layers` in
part, and modules like `mcmc` and variational inference are upcoming.
