% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/bijectors.R
\name{tfb_masked_autoregressive_flow}
\alias{tfb_masked_autoregressive_flow}
\title{Affine MaskedAutoregressiveFlow bijector}
\usage{
tfb_masked_autoregressive_flow(
  shift_and_log_scale_fn,
  is_constant_jacobian = FALSE,
  unroll_loop = FALSE,
  event_ndims = 1L,
  validate_args = FALSE,
  name = NULL
)
}
\arguments{
\item{shift_and_log_scale_fn}{Function which computes shift and log_scale from both the
forward domain (x) and the inverse domain (y).
Calculation must respect the "autoregressive property". Suggested default:
tfb_masked_autoregressive_default_template(hidden_layers=...).
Typically the function contains \code{tf$Variables} and is wrapped using \code{tf$make_template}.
Returning NULL for either (both) shift, log_scale is equivalent to (but more efficient than) returning zero.}

\item{is_constant_jacobian}{Logical, default: FALSE. When TRUE the implementation assumes log_scale
does not depend on the forward domain (x) or inverse domain (y) values.
(No validation is made; is_constant_jacobian=FALSE is always safe but possibly computationally inefficient.)}

\item{unroll_loop}{Logical indicating whether the \code{tf$while_loop} in _forward should be replaced with a
static for loop. Requires that the final dimension of x be known at graph construction time. Defaults to FALSE.}

\item{event_ndims}{integer, the intrinsic dimensionality of this bijector.
1 corresponds to a simple vector autoregressive bijector as implemented by the
\code{tfb_masked_autoregressive_default_template}, 2 might be useful for a 2D convolutional shift_and_log_scale_fn and so on.}

\item{validate_args}{Logical, default FALSE. Whether to validate input with asserts. If validate_args is
FALSE, and the inputs are invalid, correct behavior is not guaranteed.}

\item{name}{name prefixed to Ops created by this class.}
}
\value{
a bijector instance.
}
\description{
The affine autoregressive flow (Papamakarios et al., 2016) provides a
relatively simple framework for user-specified (deep) architectures to learn a
distribution over continuous events. Regarding terminology,
}
\details{
"Autoregressive models decompose the joint density as a product of
conditionals, and model each conditional in turn. Normalizing flows
transform a base density (e.g. a standard Gaussian) into the target density
by an invertible transformation with tractable Jacobian." (Papamakarios et al., 2016)

In other words, the "autoregressive property" is equivalent to the
decomposition, \verb{p(x) = prod\{ p(x[perm[i]] | x[perm[0:i]]) : i=0, ..., d \}}
where perm is some permutation of \verb{\{0, ..., d\}}. In the simple case where
the permutation is identity this reduces to:

\verb{p(x) = prod\{ p(x[i] | x[0:i]) : i=0, ..., d \}}. The provided
shift_and_log_scale_fn, tfb_masked_autoregressive_default_template, achieves
this property by zeroing out weights in its masked_dense layers.
In TensorFlow Probability, "normalizing flows" are implemented as
tfp.bijectors.Bijectors. The forward "autoregression" is implemented
using a tf.while_loop and a deep neural network (DNN) with masked weights
such that the autoregressive property is automatically met in the inverse.
A TransformedDistribution using MaskedAutoregressiveFlow(...) uses the
(expensive) forward-mode calculation to draw samples and the (cheap)
reverse-mode calculation to compute log-probabilities. Conversely, a
TransformedDistribution using Invert(MaskedAutoregressiveFlow(...)) uses
the (expensive) forward-mode calculation to compute log-probabilities and the
(cheap) reverse-mode calculation to compute samples.

Given a shift_and_log_scale_fn, the forward and inverse transformations are
(a sequence of) affine transformations. A "valid" shift_and_log_scale_fn
must compute each shift (aka loc or "mu" in Germain et al. (2015)])
and log(scale) (aka "alpha" in Germain et al. (2015)) such that ech
are broadcastable with the arguments to forward and inverse, i.e., such
that the calculations in forward, inverse below are possible.

For convenience, tfb_masked_autoregressive_default_template is offered as a
possible shift_and_log_scale_fn function. It implements the MADE
architecture (Germain et al., 2015). MADE is a feed-forward network that
computes a shift and log(scale) using masked_dense layers in a deep
neural network. Weights are masked to ensure the autoregressive property. It
is possible that this architecture is suboptimal for your task. To build
alternative networks, either change the arguments to
tfb_masked_autoregressive_default_template, use the masked_dense function to
roll-out your own, or use some other architecture, e.g., using tf.layers.
Warning: no attempt is made to validate that the shift_and_log_scale_fn
enforces the "autoregressive property".

Assuming shift_and_log_scale_fn has valid shape and autoregressive semantics,
the forward transformation is

\if{html}{\out{<div class="sourceCode">}}\preformatted{def forward(x):
   y = zeros_like(x)
   event_size = x.shape[-event_dims:].num_elements()
   for _ in range(event_size):
     shift, log_scale = shift_and_log_scale_fn(y)
     y = x * tf.exp(log_scale) + shift
   return y
}\if{html}{\out{</div>}}

and the inverse transformation is

\if{html}{\out{<div class="sourceCode">}}\preformatted{def inverse(y):
  shift, log_scale = shift_and_log_scale_fn(y)
  return (y - shift) / tf.exp(log_scale)
}\if{html}{\out{</div>}}

Notice that the inverse does not need a for-loop. This is because in the
forward pass each calculation of shift and log_scale is based on the y
calculated so far (not x). In the inverse, the y is fully known, thus is
equivalent to the scaling used in forward after event_size passes, i.e.,
the "last" y used to compute shift, log_scale.
(Roughly speaking, this also proves the transform is bijective.)
}
\section{References}{

\itemize{
\item \href{https://arxiv.org/abs/1502.03509}{Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE: Masked Autoencoder for Distribution Estimation. In \emph{International Conference on Machine Learning}, 2015.}
\item \href{https://arxiv.org/abs/1606.04934}{Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, and Max Welling. Improving Variational Inference with Inverse Autoregressive Flow. In \emph{Neural Information Processing Systems}, 2016.}
\item \href{https://arxiv.org/abs/1705.07057}{George Papamakarios, Theo Pavlakou, and Iain Murray. Masked Autoregressive Flow for Density Estimation. In \emph{Neural Information Processing Systems}, 2017.}
}
}

\seealso{
For usage examples see \code{\link[=tfb_forward]{tfb_forward()}}, \code{\link[=tfb_inverse]{tfb_inverse()}}, \code{\link[=tfb_inverse_log_det_jacobian]{tfb_inverse_log_det_jacobian()}}.

Other bijectors: 
\code{\link{tfb_absolute_value}()},
\code{\link{tfb_affine_linear_operator}()},
\code{\link{tfb_affine_scalar}()},
\code{\link{tfb_affine}()},
\code{\link{tfb_ascending}()},
\code{\link{tfb_batch_normalization}()},
\code{\link{tfb_blockwise}()},
\code{\link{tfb_chain}()},
\code{\link{tfb_cholesky_outer_product}()},
\code{\link{tfb_cholesky_to_inv_cholesky}()},
\code{\link{tfb_correlation_cholesky}()},
\code{\link{tfb_cumsum}()},
\code{\link{tfb_discrete_cosine_transform}()},
\code{\link{tfb_expm1}()},
\code{\link{tfb_exp}()},
\code{\link{tfb_ffjord}()},
\code{\link{tfb_fill_scale_tri_l}()},
\code{\link{tfb_fill_triangular}()},
\code{\link{tfb_glow}()},
\code{\link{tfb_gompertz_cdf}()},
\code{\link{tfb_gumbel_cdf}()},
\code{\link{tfb_gumbel}()},
\code{\link{tfb_identity}()},
\code{\link{tfb_inline}()},
\code{\link{tfb_invert}()},
\code{\link{tfb_iterated_sigmoid_centered}()},
\code{\link{tfb_kumaraswamy_cdf}()},
\code{\link{tfb_kumaraswamy}()},
\code{\link{tfb_lambert_w_tail}()},
\code{\link{tfb_masked_autoregressive_default_template}()},
\code{\link{tfb_masked_dense}()},
\code{\link{tfb_matrix_inverse_tri_l}()},
\code{\link{tfb_matvec_lu}()},
\code{\link{tfb_normal_cdf}()},
\code{\link{tfb_ordered}()},
\code{\link{tfb_pad}()},
\code{\link{tfb_permute}()},
\code{\link{tfb_power_transform}()},
\code{\link{tfb_rational_quadratic_spline}()},
\code{\link{tfb_rayleigh_cdf}()},
\code{\link{tfb_real_nvp_default_template}()},
\code{\link{tfb_real_nvp}()},
\code{\link{tfb_reciprocal}()},
\code{\link{tfb_reshape}()},
\code{\link{tfb_scale_matvec_diag}()},
\code{\link{tfb_scale_matvec_linear_operator}()},
\code{\link{tfb_scale_matvec_lu}()},
\code{\link{tfb_scale_matvec_tri_l}()},
\code{\link{tfb_scale_tri_l}()},
\code{\link{tfb_scale}()},
\code{\link{tfb_shifted_gompertz_cdf}()},
\code{\link{tfb_shift}()},
\code{\link{tfb_sigmoid}()},
\code{\link{tfb_sinh_arcsinh}()},
\code{\link{tfb_sinh}()},
\code{\link{tfb_softmax_centered}()},
\code{\link{tfb_softplus}()},
\code{\link{tfb_softsign}()},
\code{\link{tfb_split}()},
\code{\link{tfb_square}()},
\code{\link{tfb_tanh}()},
\code{\link{tfb_transform_diagonal}()},
\code{\link{tfb_transpose}()},
\code{\link{tfb_weibull_cdf}()},
\code{\link{tfb_weibull}()}
}
\concept{bijectors}
