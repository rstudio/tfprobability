#' Blockwise Initializer
#'
#' Initializer which concats other intializers
#'
#' @param initializers list of Keras initializers, eg: [keras::initializer_glorot_uniform()]
#'  or [initializer_constant()].
#' @param sizes list of integers scalars representing the number of elements associated
#'  with each initializer in `initializers`.
#' @param validate_args bool indicating we should do (possibly expensive) graph-time
#'  assertions, if necessary.
#'
#'  @return Initializer which concats other intializers
#'
#' @export
initializer_blockwise <- function(initializers, sizes, validate_args = FALSE) {
  tfp$layers$BlockwiseInitializer(
    initializers = initializers,
    sizes = as_integer_list(sizes),
    validate_args = validate_args
  )
}
