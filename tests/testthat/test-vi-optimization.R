context("vi-optimization")

source("utils.R")

test_succeeds("vi_fit_surrogate_posterior works", {
  skip_if_tfp_below("0.9")

  if (!tf$compat$v1$resource_variables_enabled()) tf$compat$v1$enable_resource_variables()

  # 1: Normal-Normal model
  # We'll first consider a simple model `z ~ N(0, 1)`, `x ~ N(z, 1)`,
  # where we suppose we are interested in the posterior `p(z | x=5)`:

  log_prob <-
    function(z, x)
      tfd_normal(0, 1) %>% tfd_log_prob(z) + tfd_normal(z, 1) %>% tfd_log_prob(x)

  conditioned_log_prob <- function(z)
    log_prob(z, x = 5)

  # The posterior is itself normal by and can be computed analytically (it's `N(loc=5/2., scale=1/sqrt(2)`).
  # But suppose we don't want to bother doing the math: we can use variational inference instead!
  # Note that we ensure positive scale by using a softplus transformation of
  # the underlying variable, invoked via `DeferredTensor`. Deferring the
  # transformation causes it to be performed at runtime of the distribution's
  # methods, creating a gradient to the underlying variable. If we
  # had simply specified `scale=tf.nn.softplus(scale_var)` directly,
  # without the `DeferredTensor`, fitting would fail because calls to
  # `q.log_prob` and `q.sample` would never access the underlying variable. In
  # general, transformations of trainable parameters must be deferred to runtime,
  # using either `DeferredTensor` or by the callable mechanisms available in
  # joint distribution classes.
  q_z <- tfd_normal(
    loc = tf$Variable(0, name = 'q_z_loc'),
    scale = tfp$util$TransformedVariable(
      initial_value = 1,
      bijector = tfp$bijectors$Softplus(),
      name = 'q_z_scale'),
    name = 'q_z'
  )

  losses <- vi_fit_surrogate_posterior(
    target_log_prob_fn = conditioned_log_prob,
    surrogate_posterior = q_z,
    optimizer = tf$optimizers$Adam(learning_rate = 0.1),
    num_steps = 100
  )

  if (tf$executing_eagerly()) {

    optimized_mean <- q_z %>% tfd_mean()
    optimized_sd <- q_z %>% tfd_stddev()

  } else {
    with (tf$control_dependencies(list(losses)), {
      # tf$identity ensures we create a new op to capture the dependency
      optimized_mean <- tf$identity(q_z %>% tfd_mean())
      optimized_sd <- tf$identity(q_z %>% tfd_stddev())
    })
  }

  expect_equal(optimized_mean %>% tensor_value() %>% length(), 1)

  # 2: Custom loss function

  q_z2 <- tfd_normal(
    loc = tf$Variable(0., name = 'q_z2_loc'),
    scale = tfp$util$TransformedVariable(
      initial_value = 1,
      bijector = tfp$bijectors$Softplus(),
      name = 'q_z2_scale'),
    name = 'q_z2'
  )

  #forward_kl_loss <- purrr::partial(vi_monte_carlo_variational_loss, ... =, discrepancy_fn = vi_kl_forward)
  forward_kl_loss <- function(target_log_prob_fn,
                              surrogate_posterior,
                              sample_size = 1,
                              use_reparametrization = NULL,
                              seed = NULL,
                              name = NULL)
    vi_monte_carlo_variational_loss(
      target_log_prob_fn,
      surrogate_posterior,
      sample_size,
      discrepancy_fn = vi_kl_forward,
      use_reparametrization,
      seed,
      name
    )

  losses2 <- vi_fit_surrogate_posterior(
    target_log_prob_fn = conditioned_log_prob,
    surrogate_posterior = q_z2,
    optimizer = tf$optimizers$Adam(learning_rate = 0.1),
    num_steps = 100,
    variational_loss_fn = forward_kl_loss
  )

  if (tf$executing_eagerly()) {

    optimized_mean <- q_z2 %>% tfd_mean()
    optimized_sd <- q_z2 %>% tfd_stddev()

  } else {
    with (tf$control_dependencies(list(losses2)), {
      # tf$identity ensures we create a new op to capture the dependency
      optimized_mean <- tf$identity(q_z2 %>% tfd_mean())
      optimized_sd <- tf$identity(q_z2 %>% tfd_stddev())
    })
  }

  expect_equal(optimized_mean %>% tensor_value() %>% length(), 1)

  # 3: Inhomogeneous Poisson Process
  # TBD

})
