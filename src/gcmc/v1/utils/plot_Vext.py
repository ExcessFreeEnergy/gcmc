import matplotlib.pyplot as plt
import numpy as np
import yaml

from gcmc.v1.external_potentials import initialize_external_potentials

# Load the configuration file
with open("input.yaml", "r") as file:
    config = yaml.safe_load(file)

# Initialize the external potentials
external_potentials = initialize_external_potentials(config)

# Define the positions to evaluate the potentials
x_positions = np.linspace(0, config["box_length_x"], 1000)
positions = np.array([x_positions])

# Plot the potentials for all particle types
beta = 1 / (config["kB"] * config["T"])
for particle_type, potential in external_potentials.items():
    potential_values = potential(positions) * beta
    plt.plot(x_positions, potential_values, label=f"{particle_type}", lw=2.5)


filtered_potential_values = potential_values[potential_values < 1e6]
# Find the maximum value in the filtered array
max_y = np.max(filtered_potential_values)
min_y = np.min(filtered_potential_values)

plt.ylim(min_y - 2, 10)
plt.xlabel("x")
plt.ylabel(r"$\beta V_{\mathrm{ext}}(x)$")
plt.legend()
plt.show()
plt.tight_layout()

# plt.savefig(output_file)
