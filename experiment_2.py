from AlleePopulationClass import AlleePopulation

"""
Experiment 2:
Only changing Kc
"""

# Build an experiment whose starting point is the equilibrium of old_params.
old_params = {'r': 0.12, 'T': 3, 'Kc': 800, 'm': 10, 'phi1': 0, 'phi2': 0, 'S': 100}
new_params = {'r': 0.12, 'T': 3, 'Kc': 700, 'm': 10, 'phi1': 0, 'phi2': 0, 'S': 100}
o = AlleePopulation(old_params, new_params, 1.0)

# Run direct simulations and DME
o.runStochasticSimulations(dt = 0.01, t = 0.2)
o.runDME(dt = 0.01, t = 0.2, update='regularised')

# Plot results
fig_moments = o.plot_moments()
fig_moments.show()

fig_forces = o.plot_forces()
fig_forces.show()

fig_distr = o.plot_distributions()
fig_distr.show()


#######################################################################
###      Repeat with increased sigma and smaller step size          ###
#######################################################################

o = AlleePopulation(old_params, new_params, 6.0)
o.runStochasticSimulations(dt = 0.01, t = 0.5)
o.runDME(dt = 0.01, t = 0.5, update='regularised')          # 'plain', 'regularised' or 'fixed_a4'

fig_moments = o.plot_moments()
fig_moments.show()

fig_forces = o.plot_forces()
fig_forces.show()

fig_distr = o.plot_distributions()
fig_distr.show()
