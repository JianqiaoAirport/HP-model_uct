import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dist_3_10 = pd.read_csv("ave_energy.csv")
dist_5_10 = pd.read_csv("lowest_energy.csv")

x = dist_3_10['episode']
y = dist_3_10['energy']

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.plot(dist_3_10['episode'], -dist_3_10['energy'], 'o-', linewidth=1, label='Average')
ax.plot(dist_5_10['episode'], -dist_5_10['energy'], '*-', linewidth=1, label='Best')

# ax.plot(dist_10_10['episode'], dist_10_10['diversity'], linewidth=1, label='basic_GA_with_decreasing_population')
# ax.plot(dist_basic['episode'], dist_basic['diversity'], linewidth=1, label='basic_GA')



ax.legend()
ax.set_xlim([-1, 1200])
ax.set_ylim([0, 10])
ax.set_title('negative energy per 100 episodes')
ax.set_xlabel('episode')
ax.set_ylabel('-energy')

plt.show()
