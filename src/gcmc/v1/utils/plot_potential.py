import matplotlib.pyplot as plt
import numpy as np
import yaml

from gcmc.v1.potentials import initialize_potentials

# Load the configuration file
with open("input.yaml", "r") as file:
    config = yaml.safe_load(file)

# Initialize the pair potentials
pair_potentials = initialize_potentials(config)

# Define the distances to evaluate the potentials
r_distances = np.linspace(0.01, config["global_rc"], 1000)  # Avoid zero to prevent division by zero
beta = 1 / (config["kB"] * config["T"])
# Plot the pair potentials for all pairs
for pair, potential in pair_potentials.items():
    potential_values = potential.calculate(r_distances)
    plt.plot(r_distances, beta * potential_values, label=f"{pair}", lw=2.5)

plt.xlabel(r"$r$")
plt.ylabel(r"$\beta u(r)$")
plt.ylim(-5, 20)
plt.xlim(0, config["global_rc"] + 1)
plt.legend()
plt.show()
plt.tight_layout()
