get_psd_kernels <- function()
  if (tfp_version() < "0.9") tfp$positive_semidefinite_kernels else tfp$math$psd_kernels
