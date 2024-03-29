% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/vi-functions.R
\name{vi_monte_carlo_variational_loss}
\alias{vi_monte_carlo_variational_loss}
\title{Monte-Carlo approximation of an f-Divergence variational loss}
\usage{
vi_monte_carlo_variational_loss(
  target_log_prob_fn,
  surrogate_posterior,
  sample_size = 1L,
  importance_sample_size = 1L,
  discrepancy_fn = vi_kl_reverse,
  use_reparametrization = NULL,
  seed = NULL,
  name = NULL
)
}
\arguments{
\item{target_log_prob_fn}{function that takes a set of \code{Tensor} arguments
and returns a \code{Tensor} log-density. Given
\code{q_sample <- surrogate_posterior$sample(sample_size)}, this
will be (in Python) called as \code{target_log_prob_fn(q_sample)} if \code{q_sample} is a list
or a tuple, \verb{target_log_prob_fn(**q_sample)} if \code{q_sample} is a
dictionary, or \code{target_log_prob_fn(q_sample)} if \code{q_sample} is a \code{Tensor}.
It should support batched evaluation, i.e., should return a result of
shape \verb{[sample_size]}.}

\item{surrogate_posterior}{A \code{tfp$distributions$Distribution}
instance defining a variational posterior (could be a
\code{tfp$distributions$JointDistribution}). Crucially, the distribution's \code{log_prob} and
(if reparameterized) \code{sample} methods must directly invoke all ops
that generate gradients to the underlying variables. One way to ensure
this is to use \code{tfp$util$DeferredTensor} to represent any parameters
defined as transformations of unconstrained variables, so that the
transformations execute at runtime instead of at distribution creation.}

\item{sample_size}{\code{integer} number of Monte Carlo samples to use
in estimating the variational divergence. Larger values may stabilize
the optimization, but at higher cost per step in time and memory.
Default value: \code{1}.}

\item{importance_sample_size}{integer number of terms used to define an
importance-weighted divergence. If importance_sample_size > 1, then the
surrogate_posterior is optimized to function as an importance-sampling
proposal distribution. In this case it often makes sense to use importance
sampling to approximate posterior expectations (see
tfp.vi.fit_surrogate_posterior for an example). Default value: 1.}

\item{discrepancy_fn}{function representing a Csiszar \code{f} function in
in log-space. That is, \code{discrepancy_fn(log(u)) = f(u)}, where \code{f} is
convex in \code{u}.  Default value: \code{vi_kl_reverse}.}

\item{use_reparametrization}{\code{logical}. When \code{NULL} (the default),
automatically set to: \code{surrogate_posterior.reparameterization_type == tfp$distributions$FULLY_REPARAMETERIZED}.
When \code{TRUE} uses the standard Monte-Carlo average. When \code{FALSE} uses the score-gradient trick. (See above for
details.)  When \code{FALSE}, consider using \code{csiszar_vimco}.}

\item{seed}{\code{integer} seed for \code{surrogate_posterior$sample}.}

\item{name}{name prefixed to Ops created by this function.}
}
\value{
monte_carlo_variational_loss \code{float}-like \code{Tensor} Monte Carlo
approximation of the Csiszar f-Divergence.
}
\description{
Variational losses measure the divergence between an unnormalized target
distribution \code{p} (provided via \code{target_log_prob_fn}) and a surrogate
distribution \code{q} (provided as \code{surrogate_posterior}). When the
target distribution is an unnormalized posterior from conditioning a model on
data, minimizing the loss with respect to the parameters of
\code{surrogate_posterior} performs approximate posterior inference.
}
\details{
This function defines divergences of the form
\verb{E_q[discrepancy_fn(log p(z) - log q(z))]}, sometimes known as
\href{https://en.wikipedia.org/wiki/F-divergence}{f-divergences}.

In the special case \code{discrepancy_fn(logu) == -logu} (the default
\code{vi_kl_reverse}), this is the reverse Kullback-Liebler divergence
\code{KL[q||p]}, whose negation applied to an unnormalized \code{p} is the widely-used
evidence lower bound (ELBO). Other cases of interest available under
\code{tfp$vi} include the forward \code{KL[p||q]} (given by \code{vi_kl_forward(logu) == exp(logu) * logu}),
total variation distance, Amari alpha-divergences, and more.

Csiszar f-divergences

A Csiszar function \code{f} is a convex function from \verb{R^+} (the positive reals)
to \code{R}. The Csiszar f-Divergence is given by:

\if{html}{\out{<div class="sourceCode">}}\preformatted{D_f[p(X), q(X)] := E_\{q(X)\}[ f( p(X) / q(X) ) ]
~= m**-1 sum_j^m f( p(x_j) / q(x_j) ),
where x_j ~iid q(X)
}\if{html}{\out{</div>}}

For example, \verb{f = lambda u: -log(u)} recovers \code{KL[q||p]}, while \verb{f = lambda u: u * log(u)}
recovers the forward \code{KL[p||q]}. These and other functions are available in \code{tfp$vi}.

Tricks: Reparameterization and Score-Gradient

When q is "reparameterized", i.e., a diffeomorphic transformation of a
parameterless distribution (e.g., \verb{Normal(Y; m, s) <=> Y = sX + m, X ~ Normal(0,1)}),
we can swap gradient and expectation, i.e.,
\verb{grad[Avg\{ s_i : i=1...n \}] = Avg\{ grad[s_i] : i=1...n \}} where \verb{S_n=Avg\{s_i\}}
and \verb{s_i = f(x_i), x_i ~iid q(X)}.

However, if q is not reparameterized, TensorFlow's gradient will be incorrect
since the chain-rule stops at samples of unreparameterized distributions. In
this circumstance using the Score-Gradient trick results in an unbiased
gradient, i.e.,

\if{html}{\out{<div class="sourceCode">}}\preformatted{grad[ E_q[f(X)] ]
  = grad[ int dx q(x) f(x) ]
  = int dx grad[ q(x) f(x) ]
  = int dx [ q'(x) f(x) + q(x) f'(x) ]
  = int dx q(x) [q'(x) / q(x) f(x) + f'(x) ]
  = int dx q(x) grad[ f(x) q(x) / stop_grad[q(x)] ]
  = E_q[ grad[ f(x) q(x) / stop_grad[q(x)] ] ]
}\if{html}{\out{</div>}}

Unless \code{q.reparameterization_type != tfd.FULLY_REPARAMETERIZED} it is
usually preferable to set \code{use_reparametrization = True}.

Example Application:
The Csiszar f-Divergence is a useful framework for variational inference.
I.e., observe that,

\if{html}{\out{<div class="sourceCode">}}\preformatted{f(p(x)) =  f( E_\{q(Z | x)\}[ p(x, Z) / q(Z | x) ] )
        <= E_\{q(Z | x)\}[ f( p(x, Z) / q(Z | x) ) ]
        := D_f[p(x, Z), q(Z | x)]
}\if{html}{\out{</div>}}

The inequality follows from the fact that the "perspective" of \code{f}, i.e.,
\verb{(s, t) |-> t f(s / t))}, is convex in \verb{(s, t)} when \verb{s/t in domain(f)} and
\code{t} is a real. Since the above framework includes the popular Evidence Lower
BOund (ELBO) as a special case, i.e., \code{f(u) = -log(u)}, we call this framework
"Evidence Divergence Bound Optimization" (EDBO).
}
\section{References}{

\itemize{
\item Ali, Syed Mumtaz, and Samuel D. Silvey. "A general class of coefficients of divergence of one distribution from another."
Journal of the Royal Statistical Society: Series B (Methodological) 28.1 (1966): 131-142.
}
}

\seealso{
Other vi-functions: 
\code{\link{vi_amari_alpha}()},
\code{\link{vi_arithmetic_geometric}()},
\code{\link{vi_chi_square}()},
\code{\link{vi_csiszar_vimco}()},
\code{\link{vi_dual_csiszar_function}()},
\code{\link{vi_fit_surrogate_posterior}()},
\code{\link{vi_jeffreys}()},
\code{\link{vi_jensen_shannon}()},
\code{\link{vi_kl_forward}()},
\code{\link{vi_kl_reverse}()},
\code{\link{vi_log1p_abs}()},
\code{\link{vi_modified_gan}()},
\code{\link{vi_pearson}()},
\code{\link{vi_squared_hellinger}()},
\code{\link{vi_symmetrized_csiszar_function}()}
}
\concept{vi-functions}
