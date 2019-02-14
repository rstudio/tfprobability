This is the start of what hopefully will become the `tfprobability` package.

Implemented so far:

- a few distribution layers with corresponding tests (tests are still heavily using `$` syntax)
- a few distributions corresponding to those layers (no generic `sample` and `log_prob` yet)


To be implemented (move around to reflect priority):

- distribution generics `sample` and `log_prob` (and more?)
- more distributions
- more distribution layers
- bijectors (the ones which are implemented in TFP)
- installation (to be discussed how)
- `tfp.layers` (different from the above distribution layers)
- ...
- ... 
- bijector R class (ideally just a closure, not R6 - low prio)


For modules not listed in this list (`edward2`, `monte_carlo`, `mcmc` ...) I suggest not implementing anything until we have more information about their state in TFP.
