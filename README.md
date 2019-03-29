
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

Here are a few examples using distributions and bijectors:

### Distributions

    library(tfprobability)
    
    # create a binomial distribution with n = 7 and p = 0.3
    d <- tfd_binomial(total_count = 7, probs = 0.3)
    
    # compute mean
    d %>% tfd_mean()
    
    # compute variance
    d %>% tfd_variance()
    
    # compute probability
    d %>% tfd_prob(2.3)

### Bijectors

    library(tfprobability)
    
    # create a bijector to that performs the discrete cosine transform (DCT)
    b <- tfb_discrete_cosine_transform()
    
    # run on sample data
    x <- matrix(runif(10))
    b %>% tfb_forward(x)

## State

This project is under active development. As of this writing,
`distributions` and `bijectors` are covered comprehensively, `layers` in
part, and modules like `mcmc` and variational inference are upcoming.
