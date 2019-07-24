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
