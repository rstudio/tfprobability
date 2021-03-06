% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/bijectors.R
\name{tfb_inline}
\alias{tfb_inline}
\title{Bijector constructed from custom functions}
\usage{
tfb_inline(
  forward_fn = NULL,
  inverse_fn = NULL,
  inverse_log_det_jacobian_fn = NULL,
  forward_log_det_jacobian_fn = NULL,
  forward_event_shape_fn = NULL,
  forward_event_shape_tensor_fn = NULL,
  inverse_event_shape_fn = NULL,
  inverse_event_shape_tensor_fn = NULL,
  is_constant_jacobian = NULL,
  validate_args = FALSE,
  forward_min_event_ndims = NULL,
  inverse_min_event_ndims = NULL,
  name = "inline"
)
}
\arguments{
\item{forward_fn}{Function implementing the forward transformation.}

\item{inverse_fn}{Function implementing the inverse transformation.}

\item{inverse_log_det_jacobian_fn}{Function implementing the log_det_jacobian of the forward transformation.}

\item{forward_log_det_jacobian_fn}{Function implementing the log_det_jacobian of the inverse transformation.}

\item{forward_event_shape_fn}{Function implementing non-identical static event shape changes. Default: shape is assumed unchanged.}

\item{forward_event_shape_tensor_fn}{Function implementing non-identical event shape changes. Default: shape is assumed unchanged.}

\item{inverse_event_shape_fn}{Function implementing non-identical static event shape changes. Default: shape is assumed unchanged.}

\item{inverse_event_shape_tensor_fn}{Function implementing non-identical event shape changes. Default: shape is assumed unchanged.}

\item{is_constant_jacobian}{Logical indicating that the Jacobian is constant for all input arguments.}

\item{validate_args}{Logical, default FALSE. Whether to validate input with asserts. If validate_args is
FALSE, and the inputs are invalid, correct behavior is not guaranteed.}

\item{forward_min_event_ndims}{Integer indicating the minimal dimensionality this bijector acts on.}

\item{inverse_min_event_ndims}{Integer indicating the minimal dimensionality this bijector acts on.}

\item{name}{name prefixed to Ops created by this class.}
}
\value{
a bijector instance.
}
\description{
Bijector constructed from custom functions
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
\code{\link{tfb_invert}()},
\code{\link{tfb_iterated_sigmoid_centered}()},
\code{\link{tfb_kumaraswamy_cdf}()},
\code{\link{tfb_kumaraswamy}()},
\code{\link{tfb_lambert_w_tail}()},
\code{\link{tfb_masked_autoregressive_default_template}()},
\code{\link{tfb_masked_autoregressive_flow}()},
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
