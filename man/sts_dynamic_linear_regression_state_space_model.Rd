% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/sts.R
\name{sts_dynamic_linear_regression_state_space_model}
\alias{sts_dynamic_linear_regression_state_space_model}
\title{State space model for a dynamic linear regression from provided covariates.}
\usage{
sts_dynamic_linear_regression_state_space_model(
  num_timesteps,
  design_matrix,
  drift_scale,
  initial_state_prior,
  observation_noise_scale = 0,
  initial_step = 0,
  validate_args = FALSE,
  allow_nan_stats = TRUE,
  name = NULL
)
}
\arguments{
\item{num_timesteps}{Scalar \code{integer} \code{tensor}, number of timesteps to model
with this distribution.}

\item{design_matrix}{float \code{tensor} of shape \code{tf$concat(list(batch_shape, list(num_timesteps, num_features)))}.
This may also optionally be an instance of \code{tf$linalg$LinearOperator}.}

\item{drift_scale}{Scalar (any additional dimensions are treated as batch
dimensions) \code{float} \code{tensor} indicating the standard deviation of the
latent state transitions.}

\item{initial_state_prior}{instance of \code{tfd_multivariate_normal} representing
the prior distribution on latent states.  Must have
event shape \code{list(num_features)}.}

\item{observation_noise_scale}{Scalar (any additional dimensions are
treated as batch dimensions) \code{float} \code{tensor} indicating the standard
deviation of the observation noise. Default value: \code{0}.}

\item{initial_step}{scalar \code{integer} \code{tensor} specifying the starting timestep.
Default value: \code{0}.}

\item{validate_args}{\code{logical}. Whether to validate input
with asserts. If \code{validate_args} is \code{FALSE}, and the inputs are
invalid, correct behavior is not guaranteed. Default value: \code{FALSE}.}

\item{allow_nan_stats}{\code{logical}. If \code{FALSE}, raise an
exception if a statistic (e.g. mean/mode/etc...) is undefined for any
batch member. If \code{TRUE}, batch members with valid parameters leading to
undefined statistics will return NaN for this statistic. Default value: \code{TRUE}.}

\item{name}{name prefixed to ops created by this class. Default value: 'DynamicLinearRegressionStateSpaceModel'.}
}
\value{
an instance of \code{LinearGaussianStateSpaceModel}.
}
\description{
A state space model (SSM) posits a set of latent (unobserved) variables that
evolve over time with dynamics specified by a probabilistic transition model
\code{p(z[t+1] | z[t])}. At each timestep, we observe a value sampled from an
observation model conditioned on the current state, \code{p(x[t] | z[t])}. The
special case where both the transition and observation models are Gaussians
with mean specified as a linear function of the inputs, is known as a linear
Gaussian state space model and supports tractable exact probabilistic
calculations; see \code{tfd_linear_gaussian_state_space_model} for details.
}
\details{
The dynamic linear regression model is a special case of a linear Gaussian SSM
and a generalization of typical (static) linear regression. The model
represents regression \code{weights} with a latent state which evolves via a
Gaussian random walk:
\code{weights[t] ~ Normal(weights[t-1], drift_scale)}

The latent state (the weights) has dimension \code{num_features}, while the
parameters \code{drift_scale} and \code{observation_noise_scale} are each (a batch of)
scalars. The batch shape of this \code{Distribution} is the broadcast batch shape
of these parameters, the \code{initial_state_prior}, and the
\code{design_matrix}. \code{num_features} is determined from the last dimension of
\code{design_matrix} (equivalent to the number of columns in the design matrix in
linear regression).

Mathematical Details

The dynamic linear regression model implements a
\code{tfd_linear_gaussian_state_space_model} with \code{latent_size = num_features} and
\code{observation_size = 1} following the transition model:

\if{html}{\out{<div class="sourceCode">}}\preformatted{transition_matrix = eye(num_features)
transition_noise ~ Normal(0, diag([drift_scale]))
}\if{html}{\out{</div>}}

which implements the evolution of \code{weights} described above. The observation
model is:

\if{html}{\out{<div class="sourceCode">}}\preformatted{observation_matrix[t] = design_matrix[t]
observation_noise ~ Normal(0, observation_noise_scale)
}\if{html}{\out{</div>}}
}
\seealso{
Other sts: 
\code{\link{sts_additive_state_space_model}()},
\code{\link{sts_autoregressive_state_space_model}()},
\code{\link{sts_autoregressive}()},
\code{\link{sts_constrained_seasonal_state_space_model}()},
\code{\link{sts_dynamic_linear_regression}()},
\code{\link{sts_linear_regression}()},
\code{\link{sts_local_level_state_space_model}()},
\code{\link{sts_local_level}()},
\code{\link{sts_local_linear_trend_state_space_model}()},
\code{\link{sts_local_linear_trend}()},
\code{\link{sts_seasonal_state_space_model}()},
\code{\link{sts_seasonal}()},
\code{\link{sts_semi_local_linear_trend_state_space_model}()},
\code{\link{sts_semi_local_linear_trend}()},
\code{\link{sts_smooth_seasonal_state_space_model}()},
\code{\link{sts_smooth_seasonal}()},
\code{\link{sts_sparse_linear_regression}()},
\code{\link{sts_sum}()}
}
\concept{sts}
