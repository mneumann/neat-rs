- [ ] Adaptive compatibility threshold.
- [ ] Automatic selection of a good `compatibility_threshold` by sampling
      the population. Split into `n` niches.
- [ ] Use configuration file for most parameter settings.
- [ ] Include node gene compatibility into the overall compatibility measure
      for speciation.
- [ ] Think about using a mutation method based on the crossover of matching
      genes, using a method to combine their weights.
- [x] Display the target graph and found graph as dot.
- [x] Use a single representation instead of trying to keep the node/link
      gene list and the network representation in sync. This will simply
      the code a lot.
- [x] Implement the mutation of weights.
- [x] Add DeleteConnection structural mutation
