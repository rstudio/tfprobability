% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/distributions.R
\name{tfd_multivariate_normal_diag}
\alias{tfd_multivariate_normal_diag}
\title{Multivariate normal distribution on \code{R^k}}
\usage{
tfd_multivariate_normal_diag(
  loc = NULL,
  scale_diag = NULL,
  scale_identity_multiplier = NULL,
  validate_args = FALSE,
  allow_nan_stats = TRUE,
  name = "MultivariateNormalDiag"
)
}
\arguments{
\item{loc}{Floating-point Tensor. If this is set to NULL, loc is implicitly 0.
When specified, may have shape \verb{[B1, ..., Bb, k]} where b >= 0 and k is the event size.}

\item{scale_diag}{Non-zero, floating-point Tensor representing a diagonal matrix added to scale.
May have shape \verb{[B1, ..., Bb, k]}, b >= 0, and characterizes b-batches of \verb{k x k} diagonal matrices
added to scale. When both scale_identity_multiplier and scale_diag are NULL then scale
is the Identity.}

\item{scale_identity_multiplier}{Non-zero, floating-point Tensor representing a scaled-identity-matrix
added to scale. May have shape \verb{[B1, ..., Bb]}, b >= 0, and characterizes b-batches of scaled
\verb{k x k} identity matrices added to scale. When both scale_identity_multiplier and scale_diag
are NULL then scale is the Identity.}

\item{validate_args}{Logical, default FALSE. When TRUE distribution parameters are checked
for validity despite possibly degrading runtime performance. When FALSE invalid inputs may
silently render incorrect outputs. Default value: FALSE.}

\item{allow_nan_stats}{Logical, default TRUE. When TRUE, statistics (e.g., mean, mode, variance)
use the value NaN to indicate the result is undefined. When FALSE, an exception is raised if
one or more of the statistic's batch members are undefined.}

\item{name}{name prefixed to Ops created by this class.}
}
\value{
a distribution instance.
}
\description{
The Multivariate Normal distribution is defined over \verb{R^k`` and parameterized by a (batch of) length-k loc vector (aka "mu") and a (batch of) }k x k\verb{scale matrix;}covariance = scale @ scale.T\code{where}@` denotes
matrix-multiplication.
}
\details{
Mathematical Details

The probability density function (pdf) is,

\if{html}{\out{<div class="sourceCode">}}\preformatted{pdf(x; loc, scale) = exp(-0.5 ||y||**2) / Z
y = inv(scale) @ (x - loc)
Z = (2 pi)**(0.5 k) |det(scale)|
}\if{html}{\out{</div>}}

where:
\itemize{
\item \code{loc} is a vector in \code{R^k},
\item \code{scale} is a linear operator in \verb{R^\{k x k\}}, \code{cov = scale @ scale.T},
\item \code{Z} denotes the normalization constant, and,
\item \verb{||y||**2} denotes the squared Euclidean norm of \code{y}.
}

A (non-batch) \code{scale} matrix is:

\if{html}{\out{<div class="sourceCode">}}\preformatted{scale = diag(scale_diag + scale_identity_multiplier * ones(k))
}\if{html}{\out{</div>}}

where:
\itemize{
\item \verb{scale_diag.shape = [k]}, and,
\item \verb{scale_identity_multiplier.shape = []}.#'
}

Additional leading dimensions (if any) will index batches.

If both \code{scale_diag} and \code{scale_identity_multiplier} are \code{NULL}, then
\code{scale} is the Identity matrix.
The MultivariateNormal distribution is a member of the
\href{https://en.wikipedia.org/wiki/Location-scale_family}{location-scale family}, i.e., it can be
constructed as,

\if{html}{\out{<div class="sourceCode">}}\preformatted{X ~ MultivariateNormal(loc=0, scale=1)   # Identity scale, zero shift.
Y = scale @ X + loc
}\if{html}{\out{</div>}}
}
\seealso{
For usage examples see e.g. \code{\link[=tfd_sample]{tfd_sample()}}, \code{\link[=tfd_log_prob]{tfd_log_prob()}}, \code{\link[=tfd_mean]{tfd_mean()}}.

Other distributions: 
\code{\link{tfd_autoregressive}()},
\code{\link{tfd_batch_reshape}()},
\code{\link{tfd_bates}()},
\code{\link{tfd_bernoulli}()},
\code{\link{tfd_beta_binomial}()},
\code{\link{tfd_beta}()},
\code{\link{tfd_binomial}()},
\code{\link{tfd_categorical}()},
\code{\link{tfd_cauchy}()},
\code{\link{tfd_chi2}()},
\code{\link{tfd_chi}()},
\code{\link{tfd_cholesky_lkj}()},
\code{\link{tfd_continuous_bernoulli}()},
\code{\link{tfd_deterministic}()},
\code{\link{tfd_dirichlet_multinomial}()},
\code{\link{tfd_dirichlet}()},
\code{\link{tfd_empirical}()},
\code{\link{tfd_exp_gamma}()},
\code{\link{tfd_exp_inverse_gamma}()},
\code{\link{tfd_exponential}()},
\code{\link{tfd_gamma_gamma}()},
\code{\link{tfd_gamma}()},
\code{\link{tfd_gaussian_process_regression_model}()},
\code{\link{tfd_gaussian_process}()},
\code{\link{tfd_generalized_normal}()},
\code{\link{tfd_geometric}()},
\code{\link{tfd_gumbel}()},
\code{\link{tfd_half_cauchy}()},
\code{\link{tfd_half_normal}()},
\code{\link{tfd_hidden_markov_model}()},
\code{\link{tfd_horseshoe}()},
\code{\link{tfd_independent}()},
\code{\link{tfd_inverse_gamma}()},
\code{\link{tfd_inverse_gaussian}()},
\code{\link{tfd_johnson_s_u}()},
\code{\link{tfd_joint_distribution_named_auto_batched}()},
\code{\link{tfd_joint_distribution_named}()},
\code{\link{tfd_joint_distribution_sequential_auto_batched}()},
\code{\link{tfd_joint_distribution_sequential}()},
\code{\link{tfd_kumaraswamy}()},
\code{\link{tfd_laplace}()},
\code{\link{tfd_linear_gaussian_state_space_model}()},
\code{\link{tfd_lkj}()},
\code{\link{tfd_log_logistic}()},
\code{\link{tfd_log_normal}()},
\code{\link{tfd_logistic}()},
\code{\link{tfd_mixture_same_family}()},
\code{\link{tfd_mixture}()},
\code{\link{tfd_multinomial}()},
\code{\link{tfd_multivariate_normal_diag_plus_low_rank}()},
\code{\link{tfd_multivariate_normal_full_covariance}()},
\code{\link{tfd_multivariate_normal_linear_operator}()},
\code{\link{tfd_multivariate_normal_tri_l}()},
\code{\link{tfd_multivariate_student_t_linear_operator}()},
\code{\link{tfd_negative_binomial}()},
\code{\link{tfd_normal}()},
\code{\link{tfd_one_hot_categorical}()},
\code{\link{tfd_pareto}()},
\code{\link{tfd_pixel_cnn}()},
\code{\link{tfd_poisson_log_normal_quadrature_compound}()},
\code{\link{tfd_poisson}()},
\code{\link{tfd_power_spherical}()},
\code{\link{tfd_probit_bernoulli}()},
\code{\link{tfd_quantized}()},
\code{\link{tfd_relaxed_bernoulli}()},
\code{\link{tfd_relaxed_one_hot_categorical}()},
\code{\link{tfd_sample_distribution}()},
\code{\link{tfd_sinh_arcsinh}()},
\code{\link{tfd_skellam}()},
\code{\link{tfd_spherical_uniform}()},
\code{\link{tfd_student_t_process}()},
\code{\link{tfd_student_t}()},
\code{\link{tfd_transformed_distribution}()},
\code{\link{tfd_triangular}()},
\code{\link{tfd_truncated_cauchy}()},
\code{\link{tfd_truncated_normal}()},
\code{\link{tfd_uniform}()},
\code{\link{tfd_variational_gaussian_process}()},
\code{\link{tfd_vector_diffeomixture}()},
\code{\link{tfd_vector_exponential_diag}()},
\code{\link{tfd_vector_exponential_linear_operator}()},
\code{\link{tfd_vector_laplace_diag}()},
\code{\link{tfd_vector_laplace_linear_operator}()},
\code{\link{tfd_vector_sinh_arcsinh_diag}()},
\code{\link{tfd_von_mises_fisher}()},
\code{\link{tfd_von_mises}()},
\code{\link{tfd_weibull}()},
\code{\link{tfd_wishart_linear_operator}()},
\code{\link{tfd_wishart_tri_l}()},
\code{\link{tfd_wishart}()},
\code{\link{tfd_zipf}()}
}
\concept{distributions}
