# from keras
# leave for now, decide later

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
  tuple(shape)
}

as_nullable_integer <- function(x) {
  if (is.null(x))
    x
  else
    as.integer(x)
}

