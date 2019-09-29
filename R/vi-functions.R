#' The Amari-alpha Csiszar-function in log-space
#'
#' A Csiszar-function is a member of ` F = { f:R_+ to R : f convex } `.
#'
#' When `self_normalized = TRUE`, the Amari-alpha Csiszar-function is:
#'
#' ```
#' f(u) = { -log(u) + (u - 1)},     alpha = 0
#'        { u log(u) - (u - 1)},    alpha = 1
#'        { [(u^alpha - 1) - alpha (u - 1)] / (alpha (alpha - 1))},    otherwise
#' ```
#'
#' When `self_normalized = FALSE` the `(u - 1)` terms are omitted.
#'
#' Warning: when `alpha != 0` and/or `self_normalized = True` this function makes
#' non-log-space calculations and may therefore be numerically unstable for
#' `|logu| >> 0`.
#'
#' @param  logu `float`-like `Tensor` representing `log(u)` from above.
#' @param alpha `float`-like scalar.
#' @param self_normalized `logical` indicating whether `f'(u=1)=0`. When
#' `f'(u=1)=0` the implied Csiszar f-Divergence remains non-negative even
#' when `p, q` are unnormalized measures.
#' @param name name prefixed to Ops created by this function.
#'
#' @section References:
#'  - A. Cichocki and S. Amari. "Families of Alpha-Beta-and GammaDivergences: Flexible and Robust Measures of Similarities." Entropy, vol. 12, no. 6, pp. 1532-1568, 2010.
#'
#' @family vi-functions
#'
#' @export
vi_amari_alpha <-
  function(logu,
           alpha = 1,
           self_normalized = FALSE,
           name = NULL) {
    tfp$vi$amari_alpha(logu, alpha, self_normalized, name)
  }


#' The reverse Kullback-Leibler Csiszar-function in log-space
#'
#' A Csiszar-function is a member of `F = { f:R_+ to R : f convex }`.
#'
#' When `self_normalized = TRUE`, the KL-reverse Csiszar-function is `f(u) = -log(u) + (u - 1)`.
#' When `self_normalized = FALSE` the `(u - 1)` term is omitted.
#' Observe that as an f-Divergence, this Csiszar-function implies: `D_f[p, q] = KL[q, p]`
#'
#' The KL is "reverse" because in maximum likelihood we think of minimizing `q` as in `KL[p, q]`.
#'
#' Warning: when self_normalized = True` this function makes non-log-space calculations and may
#' therefore be numerically unstable for `|logu| >> 0`.
#'
#' @param logu `float`-like `Tensor` representing `log(u)` from above.
#' @param self_normalized `logical` indicating whether `f'(u=1)=0`. When
#' `f'(u=1)=0` the implied Csiszar f-Divergence remains non-negative even
#' when `p, q` are unnormalized measures.
#' @param name name prefixed to Ops created by this function.
#'
#' @family vi-functions
#'
#' @export
vi_kl_reverse <-
  function(logu,
           self_normalized = FALSE,
           name = NULL) {
    tfp$vi$kl_reverse(logu, self_normalized, name)
           }


#' The forward Kullback-Leibler Csiszar-function in log-space.
#'
#' A Csiszar-function is a member of `F = { f:R_+ to R : f convex }`.
#'
#' When `self_normalized = TRUE`, the KL-reverse Csiszar-function is `f(u) = u log(u) - (u - 1)`.
#' When `self_normalized = FALSE` the `(u - 1)` term is omitted.
#' Observe that as an f-Divergence, this Csiszar-function implies: `D_f[p, q] = KL[q, p]`
#'
#' The KL is "forward" because in maximum likelihood we think of minimizing `q` as in `KL[p, q]`.
#'
#' Warning: when self_normalized = True` this function makes non-log-space calculations and may
#' therefore be numerically unstable for `|logu| >> 0`.
#'
#' @param logu `float`-like `Tensor` representing `log(u)` from above.
#' @param self_normalized `logical` indicating whether `f'(u=1)=0`. When
#' `f'(u=1)=0` the implied Csiszar f-Divergence remains non-negative even
#' when `p, q` are unnormalized measures.
#' @param name name prefixed to Ops created by this function.
#'
#' @family vi-functions
#'
#' @export
vi_kl_forward <-
  function(logu,
           self_normalized = FALSE,
           name = NULL) {
    tfp$vi$kl_forward(logu, self_normalized, name)
  }

#' Monte-Carlo approximation of an f-Divergence variational loss
#'
#' Variational losses measure the divergence between an unnormalized target
#' distribution `p` (provided via `target_log_prob_fn`) and a surrogate
#' distribution `q` (provided as `surrogate_posterior`). When the
#' target distribution is an unnormalized posterior from conditioning a model on
#' data, minimizing the loss with respect to the parameters of
#' `surrogate_posterior` performs approximate posterior inference.
#'
#' This function defines divergences of the form
#' `E_q[discrepancy_fn(log p(z) - log q(z))]`, sometimes known as
#' [f-divergences](https://en.wikipedia.org/wiki/F-divergence).
#'
#' In the special case `discrepancy_fn(logu) == -logu` (the default
#' `vi_kl_reverse`), this is the reverse Kullback-Liebler divergence
#' `KL[q||p]`, whose negation applied to an unnormalized `p` is the widely-used
#' evidence lower bound (ELBO). Other cases of interest available under
#' `tfp$vi` include the forward `KL[p||q]` (given by `vi_kl_forward(logu) == exp(logu) * logu`),
#' total variation distance, Amari alpha-divergences, and more.
#'
#' Csiszar f-divergences
#'
#' A Csiszar function `f` is a convex function from `R^+` (the positive reals)
#' to `R`. The Csiszar f-Divergence is given by:
#' ```
#' D_f[p(X), q(X)] := E_{q(X)}[ f( p(X) / q(X) ) ]
#' ~= m**-1 sum_j^m f( p(x_j) / q(x_j) ),
#' where x_j ~iid q(X)
#' ```
#'
#' For example, `f = lambda u: -log(u)` recovers `KL[q||p]`, while `f = lambda u: u * log(u)`
#' recovers the forward `KL[p||q]`. These and other functions are available in `tfp$vi`.
#'
#' Tricks: Reparameterization and Score-Gradient
#'
#' When q is "reparameterized", i.e., a diffeomorphic transformation of a
#' parameterless distribution (e.g., `Normal(Y; m, s) <=> Y = sX + m, X ~ Normal(0,1)`),
#' we can swap gradient and expectation, i.e.,
#' `grad[Avg{ s_i : i=1...n }] = Avg{ grad[s_i] : i=1...n }` where `S_n=Avg{s_i}`
#' and `s_i = f(x_i), x_i ~iid q(X)`.
#'
#' However, if q is not reparameterized, TensorFlow's gradient will be incorrect
#' since the chain-rule stops at samples of unreparameterized distributions. In
#' this circumstance using the Score-Gradient trick results in an unbiased
#' gradient, i.e.,
#' ```
#' grad[ E_q[f(X)] ]
#'   = grad[ int dx q(x) f(x) ]
#'   = int dx grad[ q(x) f(x) ]
#'   = int dx [ q'(x) f(x) + q(x) f'(x) ]
#'   = int dx q(x) [q'(x) / q(x) f(x) + f'(x) ]
#'   = int dx q(x) grad[ f(x) q(x) / stop_grad[q(x)] ]
#'   = E_q[ grad[ f(x) q(x) / stop_grad[q(x)] ] ]
#' ```
#' Unless `q.reparameterization_type != tfd.FULLY_REPARAMETERIZED` it is
#' usually preferable to set `use_reparametrization = True`.
#'
#' Example Application:
#' The Csiszar f-Divergence is a useful framework for variational inference.
#' I.e., observe that,
#' ```
#' f(p(x)) =  f( E_{q(Z | x)}[ p(x, Z) / q(Z | x) ] )
#'         <= E_{q(Z | x)}[ f( p(x, Z) / q(Z | x) ) ]
#'         := D_f[p(x, Z), q(Z | x)]
#' ```
#'
#' The inequality follows from the fact that the "perspective" of `f`, i.e.,
#' `(s, t) |-> t f(s / t))`, is convex in `(s, t)` when `s/t in domain(f)` and
#' `t` is a real. Since the above framework includes the popular Evidence Lower
#' BOund (ELBO) as a special case, i.e., `f(u) = -log(u)`, we call this framework
#' "Evidence Divergence Bound Optimization" (EDBO).
#'
#' @param target_log_prob_fn function that takes a set of `Tensor` arguments
#' and returns a `Tensor` log-density. Given
#' `q_sample <- surrogate_posterior$sample(sample_size)`, this
#' will be (in Python) called as `target_log_prob_fn(q_sample)` if `q_sample` is a list
#' or a tuple, `target_log_prob_fn(**q_sample)` if `q_sample` is a
#' dictionary, or `target_log_prob_fn(q_sample)` if `q_sample` is a `Tensor`.
#' It should support batched evaluation, i.e., should return a result of
#' shape `[sample_size]`.
#' @param surrogate_posterior A `tfp$distributions$Distribution`
#' instance defining a variational posterior (could be a
#' `tfp$distributions$JointDistribution`). Crucially, the distribution's `log_prob` and
#' (if reparameterized) `sample` methods must directly invoke all ops
#' that generate gradients to the underlying variables. One way to ensure
#' this is to use `tfp$util$DeferredTensor` to represent any parameters
#' defined as transformations of unconstrained variables, so that the
#' transformations execute at runtime instead of at distribution creation.
#' @param sample_size `integer` number of Monte Carlo samples to use
#' in estimating the variational divergence. Larger values may stabilize
#' the optimization, but at higher cost per step in time and memory.
#' Default value: `1`.
#' @param discrepancy_fn function representing a Csiszar `f` function in
#' in log-space. That is, `discrepancy_fn(log(u)) = f(u)`, where `f` is
#' convex in `u`.  Default value: `vi_kl_reverse`.
#' @param use_reparametrization `logical`. When `NULL` (the default),
#' automatically set to: `surrogate_posterior.reparameterization_type == tfp$distributions$FULLY_REPARAMETERIZED`.
#' When `TRUE` uses the standard Monte-Carlo average. When `FALSE` uses the score-gradient trick. (See above for
#' details.)  When `FALSE`, consider using `csiszar_vimco`.
#' @param seed `integer` seed for `surrogate_posterior$sample`.
#' @param name name prefixed to Ops created by this function.
#'
#' @return monte_carlo_variational_loss `float`-like `Tensor` Monte Carlo
#' approximation of the Csiszar f-Divergence.
#'
#' @section References:
#' - Ali, Syed Mumtaz, and Samuel D. Silvey. "A general class of coefficients of divergence of one distribution from another."
#' Journal of the Royal Statistical Society: Series B (Methodological) 28.1 (1966): 131-142.

#' @family vi-functions
#'
#' @export
vi_monte_carlo_variational_loss <-
  function(target_log_prob_fn,
           surrogate_posterior,
           sample_size = 1,
           discrepancy_fn = vi_kl_reverse,
           use_reparametrization = NULL,
           seed = NULL,
           name = NULL) {
    tfp$vi$monte_carlo_variational_loss(
      target_log_prob_fn,
      surrogate_posterior,
      as.integer(sample_size),
      discrepancy_fn,
      use_reparametrization,
      seed,
      name
    )
  }

#' The Jensen-Shannon Csiszar-function in log-space.
#'
#' A Csiszar-function is a member of `F = { f:R_+ to R : f convex }`.
#'
#' When `self_normalized = True`, the Jensen-Shannon Csiszar-function is:
#' ```
#' f(u) = u log(u) - (1 + u) log(1 + u) + (u + 1) log(2)
#' ```
#'
#' When `self_normalized = False` the `(u + 1) log(2)` term is omitted.
#'
#' Observe that as an f-Divergence, this Csiszar-function implies:
#'
#' ```
#' D_f[p, q] = KL[p, m] + KL[q, m]
#' m(x) = 0.5 p(x) + 0.5 q(x)
#' ```
#'
#' In a sense, this divergence is the "reverse" of the Arithmetic-Geometric
#' f-Divergence.
#'
#' This Csiszar-function induces a symmetric f-Divergence, i.e.,
#' `D_f[p, q] = D_f[q, p]`.
#'
#' Warning: this function makes non-log-space calculations and may therefore be
#' numerically unstable for `|logu| >> 0`.
#'
#' @section References:
#' - Lin, J. "Divergence measures based on the Shannon entropy." IEEE Trans.
#' Inf. Th., 37, 145-151, 1991.
#'
#' @inheritParams vi_amari_alpha
#'
#' @returns jensen_shannon_of_u, `float`-like `Tensor` of the Csiszar-function
#' evaluated at `u = exp(logu)`.
#'
#' @family vi-functions
#' @export
vi_jensen_shannon <-
  function(logu,
           self_normalized = FALSE,
           name = NULL) {
    tfp$vi$jensen_shannon(logu, self_normalized, name)
  }


#' The Arithmetic-Geometric Csiszar-function in log-space.
#'
#' A Csiszar-function is a member of `F = { f:R_+ to R : f convex }`.
#'
#' When `self_normalized = True` the Arithmetic-Geometric Csiszar-function is:
#' ```
#' f(u) = (1 + u) log( (1 + u) / sqrt(u) ) - (1 + u) log(2)
#' ```
#'
#' When `self_normalized = False` the `(1 + u) log(2)` term is omitted.
#'
#' Observe that as an f-Divergence, this Csiszar-function implies:
#'
#' ```
#' D_f[p, q] = KL[m, p] + KL[m, q]
#' m(x) = 0.5 p(x) + 0.5 q(x)
#' ```
#'
#' In a sense, this divergence is the "reverse" of the Jensen-Shannon
#' f-Divergence.
#' This Csiszar-function induces a symmetric f-Divergence, i.e.,
#' `D_f[p, q] = D_f[q, p]`.
#'
#' Warning: when self_normalized = True` this function makes non-log-space calculations and may
#' therefore be numerically unstable for `|logu| >> 0`.
#'
#' @inheritParams vi_amari_alpha
#'
#' @return arithmetic_geometric_of_u: `float`-like `Tensor` of the
#' Csiszar-function evaluated at `u = exp(logu)`.
#'
#' @family vi-functions#'
#' @export
vi_arithmetic_geometric <-
  function(logu,
           self_normalized = FALSE,
           name = NULL) {
    tfp$vi$arithmetic_geometric(logu, self_normalized, name)
  }


#' The forward Kullback-Leibler Csiszar-function in log-space.
#'
#' A Csiszar-function is a member of `F = { f:R_+ to R : f convex }`.
#'
#' When `self_normalized = TRUE`, the KL-reverse Csiszar-function is `f(u) = u log(u) - (u - 1)`.
#' When `self_normalized = FALSE` the `(u - 1)` term is omitted.
#' Observe that as an f-Divergence, this Csiszar-function implies: `D_f[p, q] = KL[q, p]`
#'
#' The KL is "forward" because in maximum likelihood we think of minimizing `q` as in `KL[p, q]`.
#'
#' Warning: when self_normalized = True` this function makes non-log-space calculations and may
#' therefore be numerically unstable for `|logu| >> 0`.
#'
#' @param logu `float`-like `Tensor` representing `log(u)` from above.
#' @param self_normalized `logical` indicating whether `f'(u=1)=0`. When
#' `f'(u=1)=0` the implied Csiszar f-Divergence remains non-negative even
#' when `p, q` are unnormalized measures.
#' @param name name prefixed to Ops created by this function.
#'
#' @family vi-functions
#'
#' @export
vi_kl_forward <-
  function(logu,
           self_normalized = FALSE,
           name = NULL) {
    tfp$vi$kl_forward(logu, self_normalized, name)
  }


#' The forward Kullback-Leibler Csiszar-function in log-space.
#'
#' A Csiszar-function is a member of `F = { f:R_+ to R : f convex }`.
#'
#' When `self_normalized = TRUE`, the KL-reverse Csiszar-function is `f(u) = u log(u) - (u - 1)`.
#' When `self_normalized = FALSE` the `(u - 1)` term is omitted.
#' Observe that as an f-Divergence, this Csiszar-function implies: `D_f[p, q] = KL[q, p]`
#'
#' The KL is "forward" because in maximum likelihood we think of minimizing `q` as in `KL[p, q]`.
#'
#' Warning: when self_normalized = True` this function makes non-log-space calculations and may
#' therefore be numerically unstable for `|logu| >> 0`.
#'
#' @param logu `float`-like `Tensor` representing `log(u)` from above.
#' @param self_normalized `logical` indicating whether `f'(u=1)=0`. When
#' `f'(u=1)=0` the implied Csiszar f-Divergence remains non-negative even
#' when `p, q` are unnormalized measures.
#' @param name name prefixed to Ops created by this function.
#'
#' @family vi-functions
#'
#' @export
vi_kl_forward <-
  function(logu,
           self_normalized = FALSE,
           name = NULL) {
    tfp$vi$kl_forward(logu, self_normalized, name)
  }


#' The forward Kullback-Leibler Csiszar-function in log-space.
#'
#' A Csiszar-function is a member of `F = { f:R_+ to R : f convex }`.
#'
#' When `self_normalized = TRUE`, the KL-reverse Csiszar-function is `f(u) = u log(u) - (u - 1)`.
#' When `self_normalized = FALSE` the `(u - 1)` term is omitted.
#' Observe that as an f-Divergence, this Csiszar-function implies: `D_f[p, q] = KL[q, p]`
#'
#' The KL is "forward" because in maximum likelihood we think of minimizing `q` as in `KL[p, q]`.
#'
#' Warning: when self_normalized = True` this function makes non-log-space calculations and may
#' therefore be numerically unstable for `|logu| >> 0`.
#'
#' @param logu `float`-like `Tensor` representing `log(u)` from above.
#' @param self_normalized `logical` indicating whether `f'(u=1)=0`. When
#' `f'(u=1)=0` the implied Csiszar f-Divergence remains non-negative even
#' when `p, q` are unnormalized measures.
#' @param name name prefixed to Ops created by this function.
#'
#' @family vi-functions
#'
#' @export
vi_kl_forward <-
  function(logu,
           self_normalized = FALSE,
           name = NULL) {
    tfp$vi$kl_forward(logu, self_normalized, name)
  }


#' The forward Kullback-Leibler Csiszar-function in log-space.
#'
#' A Csiszar-function is a member of `F = { f:R_+ to R : f convex }`.
#'
#' When `self_normalized = TRUE`, the KL-reverse Csiszar-function is `f(u) = u log(u) - (u - 1)`.
#' When `self_normalized = FALSE` the `(u - 1)` term is omitted.
#' Observe that as an f-Divergence, this Csiszar-function implies: `D_f[p, q] = KL[q, p]`
#'
#' The KL is "forward" because in maximum likelihood we think of minimizing `q` as in `KL[p, q]`.
#'
#' Warning: when self_normalized = True` this function makes non-log-space calculations and may
#' therefore be numerically unstable for `|logu| >> 0`.
#'
#' @param logu `float`-like `Tensor` representing `log(u)` from above.
#' @param self_normalized `logical` indicating whether `f'(u=1)=0`. When
#' `f'(u=1)=0` the implied Csiszar f-Divergence remains non-negative even
#' when `p, q` are unnormalized measures.
#' @param name name prefixed to Ops created by this function.
#'
#' @family vi-functions
#'
#' @export
vi_kl_forward <-
  function(logu,
           self_normalized = FALSE,
           name = NULL) {
    tfp$vi$kl_forward(logu, self_normalized, name)
  }


#' The forward Kullback-Leibler Csiszar-function in log-space.
#'
#' A Csiszar-function is a member of `F = { f:R_+ to R : f convex }`.
#'
#' When `self_normalized = TRUE`, the KL-reverse Csiszar-function is `f(u) = u log(u) - (u - 1)`.
#' When `self_normalized = FALSE` the `(u - 1)` term is omitted.
#' Observe that as an f-Divergence, this Csiszar-function implies: `D_f[p, q] = KL[q, p]`
#'
#' The KL is "forward" because in maximum likelihood we think of minimizing `q` as in `KL[p, q]`.
#'
#' Warning: when self_normalized = True` this function makes non-log-space calculations and may
#' therefore be numerically unstable for `|logu| >> 0`.
#'
#' @param logu `float`-like `Tensor` representing `log(u)` from above.
#' @param self_normalized `logical` indicating whether `f'(u=1)=0`. When
#' `f'(u=1)=0` the implied Csiszar f-Divergence remains non-negative even
#' when `p, q` are unnormalized measures.
#' @param name name prefixed to Ops created by this function.
#'
#' @family vi-functions
#'
#' @export
vi_kl_forward <-
  function(logu,
           self_normalized = FALSE,
           name = NULL) {
    tfp$vi$kl_forward(logu, self_normalized, name)
  }


#' The forward Kullback-Leibler Csiszar-function in log-space.
#'
#' A Csiszar-function is a member of `F = { f:R_+ to R : f convex }`.
#'
#' When `self_normalized = TRUE`, the KL-reverse Csiszar-function is `f(u) = u log(u) - (u - 1)`.
#' When `self_normalized = FALSE` the `(u - 1)` term is omitted.
#' Observe that as an f-Divergence, this Csiszar-function implies: `D_f[p, q] = KL[q, p]`
#'
#' The KL is "forward" because in maximum likelihood we think of minimizing `q` as in `KL[p, q]`.
#'
#' Warning: when self_normalized = True` this function makes non-log-space calculations and may
#' therefore be numerically unstable for `|logu| >> 0`.
#'
#' @param logu `float`-like `Tensor` representing `log(u)` from above.
#' @param self_normalized `logical` indicating whether `f'(u=1)=0`. When
#' `f'(u=1)=0` the implied Csiszar f-Divergence remains non-negative even
#' when `p, q` are unnormalized measures.
#' @param name name prefixed to Ops created by this function.
#'
#' @family vi-functions
#'
#' @export
vi_kl_forward <-
  function(logu,
           self_normalized = FALSE,
           name = NULL) {
    tfp$vi$kl_forward(logu, self_normalized, name)
  }

