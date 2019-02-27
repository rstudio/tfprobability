This is the start of what hopefully will become the `tfprobability` package.

Implemented so far (all with tests):

- a few distribution layers 
- a few distributions corresponding to those layers 
- generic distribution methods (`sample`, `log_prob` ...)
- installation (this is done in `tensorflow`)


To be implemented (move around to reflect priority):

- more distributions
- bijectors (the ones which are implemented in TFP)
- more distribution layers
- `tfp.layers` (different from the above distribution layers)
- `mcmc`
- `optimizer`
- ...
- ... 
- bijector R class (ideally just a closure, not R6 - low prio)

For modules not listed in this list (`edward2`, ...) I suggest not implementing anything until we have more information about their state in TFP (some stuff might be under heavy development or outdated).
