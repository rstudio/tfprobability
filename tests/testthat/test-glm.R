test_succeeds("glm_fit.tensorflow.tensor works", {

  x <- matrix(runif(100), ncol = 2)
  y <- rnorm(50, mean = rowSums(x), sd = 0.2)

  model <- glm_fit(x, y, model = tfp$glm$Normal())

})


