from AlleePopulationClass import AlleePopulation

"""
Experiment 1:
Starting and ending in the same equilibrium
"""

# Build an experiment whose starting point is the equilibrium of old_params.
old_params = {'r': 0.12, 'T': 3, 'Kc': 800, 'm': 10, 'phi1': 0, 'phi2': 0, 'S': 100}
new_params = {'r': 0.12, 'T': 3, 'Kc': 800, 'm': 10, 'phi1': 0, 'phi2': 0, 'S': 100}
o = AlleePopulation(old_params, new_params, 1.0)

# Run direct simulations and DME
o.runStochasticSimulations(dt = 0.01, t = 2)

o.runDME(dt = 0.01, t = 2, update='regularised')    # 'plain', 'regularised' or 'fixed_a4'

# Plot results
fig_moments = o.plot_moments()
fig_moments.show()

fig_forces = o.plot_forces()
fig_forces.show()

fig_distr = o.plot_distributions()
fig_distr.show()

