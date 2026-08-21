from AlleePopulationClass import AlleePopulation

"""
Experiment 5:
Starting and ending in a scenario with mean-dependence
with high migration and sigma
"""

# Build an experiment whose starting point is the equilibrium of old_params.
old_params = {'r': 0.2, 'T': 3, 'Kc': 800, 'm': 20, 'phi1': 0.0001, 'phi2': 0.00001, 'S': 100}
new_params = {'r': 0.2, 'T': 3, 'Kc': 800, 'm': 20, 'phi1': 0.0002, 'phi2': 0.000009, 'S': 100}
o = AlleePopulation(old_params, new_params, 10.0)

# Run direct simulations and DME
o.runStochasticSimulations(dt = 0.01, t = 4)
o.runDME(dt = 0.01, t = 4, update='regularised')    # 'plain', 'regularised' or 'fixed_a4'

# Plot results
fig_moments = o.plot_moments()
fig_moments.show()

fig_forces = o.plot_forces()
fig_forces.show()

fig_distr = o.plot_distributions()
fig_distr.show()

"""
When <n> shrinks:
Can only handle small perturbations (e.g., Kc: 800 -> 790). Then, expectations still match. 
However, when perturbation is too large (E.g., Kc: 800 -> 700), DME fails. 
After an initial change, the effective forces remain constant, even though the expectations are incorrect.

When <n> grows:
Can handle largere perturbations while still matching expectations (e.g., Kc: 800 -> 900). 
However, forces do not converge to target values. 
"""