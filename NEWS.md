# tfprobability 0.15.1

- updated docs for compatibility with R 4.2 / HTML5

## 0.15.0

- updated `install_tfprobability()` to use latest release.
- `vi_mote_carlo_variational_loss()` gains a `importance_sample_size` arg.
- `sts_one_step_predictive()` gains a `timesteps_are_event_shape` arg.

- Deprecations:
  - `tfb_affine_scalar()`
  - `variational_loss_fn` arg in `vi_fit_surrogate_posterior`

- New maintainer Tomasz Kalinowski (@t-kalinowski)


## 0.12.0.0 (CRAN)

- new distributions:
  - tfd_skellam
  - tfd_exp_gamma
  - tfd_exp_inverse_gamma

- changes in distributions:
  - remove deprecated batch_shape and event_shape in tfd_transformed_distribution

- new bijectors:
  - tfb_glow
  - tfb_rayleigh_cdf
  - tfb_ascending

- changes in bijectors
  - add optional low parameter to tfb_softplus.
  - tfb_chain() takes new parameters validate_event_size and parameters



## 0.11.0.0 (CRAN)

- new distributions:
  - tfd_joint_distribution_sequential_auto_batched
  - tfd_joint_distribution_named_auto_batched
  - tfd_weibull
  - tfd_truncated_cauchy
  - tfd_spherical_uniform
  - tfd_power_spherical
  - tfd_log_logistic
  - tfd_bates
  - tfd_generalized_normal
  - tfd_johnson_s_u
  - tfd_continuous_bernoulli

- new bijectors:
  - tfb_split
  - tfb_gompertz_cdf
  - tfb_shifted_gompertz_cdf
  - tfb_sinh

- changes in bijectors:
  - add log_scale argument to tfb_scale

## 0.10.0.0 (CRAN)

- changes in distributions:
   - added: tfd_beta_binomial
   - parameter list changed: tfd_transformed_distribution

- changes in bijectors:
   - added: tfb_lambert_w_tail


## 0.9.0.0 (CRAN)

- new distributions:
  - tfd_generalized_pareto
  - tfd_doublesided_maxwell
  - tfd_placket_luce
  - tfd_discrete_finite
  - tfd_logit_normal
  - tfd_log_normal
  - tfd_pert
  - tfd_wishart_linear_operator
  - tfd_wishart_tri_l
  - tfd_pixel_cnn

- new bijectors
  - tfb_shift
  - tfb_pad
  - tfb_scale
  - tfb_scale_matvec_diag
  - tfb_scale_matvec_tri_l
  - tfb_scale_matvec_linear_operator
  - tfb_scale_matvec__diag_lu
  - tfb_rational_quadratic_spline
  - tfb_gumbel_cdf
  - tfb_weibull_cdf
  - tfb_kumaraswamy_cdf
  - tfb_fill_scale_triangular
  - tfb_ffjord

- new state space models
  - sts_smooth_seasonal
  - sts_smooth_seasonal_state_space_model


## 0.8.0.0 (Initial release, CRAN)
