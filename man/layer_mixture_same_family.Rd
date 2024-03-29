% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/distribution-layers.R
\name{layer_mixture_same_family}
\alias{layer_mixture_same_family}
\title{A mixture (same-family) Keras layer.}
\usage{
layer_mixture_same_family(
  object,
  num_components,
  component_layer,
  convert_to_tensor_fn = tfp$distributions$Distribution$sample,
  validate_args = FALSE,
  ...
)
}
\arguments{
\item{object}{What to compose the new \code{Layer} instance with. Typically a
Sequential model or a Tensor (e.g., as returned by \code{layer_input()}).
The return value depends on \code{object}. If \code{object} is:
\itemize{
\item missing or \code{NULL}, the \code{Layer} instance is returned.
\item a \code{Sequential} model, the model with an additional layer is returned.
\item a Tensor, the output tensor from \code{layer_instance(object)} is returned.
}}

\item{num_components}{Number of component distributions in the mixture distribution.}

\item{component_layer}{Function that, given a tensor of shape
\verb{batch_shape + [num_components, component_params_size]}, returns a
\code{tfd.Distribution}-like instance that implements the component
distribution (with batch shape \verb{batch_shape + [num_components]}) --
e.g., a TFP distribution layer.}

\item{convert_to_tensor_fn}{A callable that takes a tfd$Distribution instance and returns a
tf$Tensor-like object. Default value: \code{tfd$distributions$Distribution$sample}.}

\item{validate_args}{Logical, default FALSE. When TRUE distribution parameters are checked
for validity despite possibly degrading runtime performance. When FALSE invalid inputs may
silently render incorrect outputs. Default value: FALSE.
@param ... Additional arguments passed to \code{args} of \code{keras::create_layer}.}

\item{...}{Additional arguments passed to \code{args} of \code{keras::create_layer}.}
}
\value{
a Keras layer
}
\description{
A mixture (same-family) Keras layer.
}
\seealso{
For an example how to use in a Keras model, see \code{\link[=layer_independent_normal]{layer_independent_normal()}}.

Other distribution_layers: 
\code{\link{layer_categorical_mixture_of_one_hot_categorical}()},
\code{\link{layer_distribution_lambda}()},
\code{\link{layer_independent_bernoulli}()},
\code{\link{layer_independent_logistic}()},
\code{\link{layer_independent_normal}()},
\code{\link{layer_independent_poisson}()},
\code{\link{layer_kl_divergence_add_loss}()},
\code{\link{layer_kl_divergence_regularizer}()},
\code{\link{layer_mixture_logistic}()},
\code{\link{layer_mixture_normal}()},
\code{\link{layer_multivariate_normal_tri_l}()},
\code{\link{layer_one_hot_categorical}()}
}
\concept{distribution_layers}
