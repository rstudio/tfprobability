test_succeeds("glm_fit.tensorflow.tensor works", {

  x <- matrix(runif(100), ncol = 2)
  y <- rnorm(50, mean = rowSums(x), sd = 0.2)

  model <- glm_fit(x, y, model = tfp$glm$Normal())
  model_r <- glm(y ~ 0 + x[,1] + x[,2])

  expect_equivalent(as.numeric(model[[1]]), model_r$coefficients)
  expect_s3_class(model, "glm_fit")
})


