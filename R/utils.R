# from keras
# Helper function to coerce shape arguments to tuple
normalize_shape <- function(shape) {
  # reflect NULL back
  if (is.null(shape))
    return(shape)

  # if it's a list or a numeric vector then convert to integer
  if (is.list(shape) || is.numeric(shape)) {
    shape <- lapply(shape, function(value) {
      if (!is.null(value))
        as.integer(value)
      else
        NULL
    })
  }

  # coerce to tuple so it's iterable
  reticulate::tuple(shape)
}

# from keras
as_nullable_integer <- function(x) {
  if (is.null(x))
    x
  else
    as.integer(x)
}

# from keras
as_axis <- function(axis) {
  if (length(axis) > 1) {
    sapply(axis, as_axis)
  } else {
    axis <- as_nullable_integer(axis)
    if (is.null(axis))
      axis
    else if (axis == -1L)
      axis
    else
      axis - 1L
  }
}

as_tf_float <- function(x) {
  tf$cast(x, tf$float32)
}

as_tensor <- function(x) {
  if (is.list(x)) {
    Map(tf$convert_to_tensor, x)
  } else {
    tf$convert_to_tensor(x)
  }
}

as_integer_list <- function(x) {
  lapply(x, as.integer)
}
