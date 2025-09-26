import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Plot properties
sns.set(style='whitegrid', font_scale=1.2)
colors = sns.color_palette("deep", 4)
fig, axes = plt.subplots(1, 2, figsize=(10, 6))
markersize = 10
linewidth = 2
fontsize = 18

# Legends and mapping
legends = ['Bloch sim.', 'k-space sim.', 'Total sim.', 'Full script']
cmap = [colors[i] for i in range(len(legends))]
markers = {'Runtime': 'o', 'Precomputed': 'x'}
linestyles = {'Runtime': '-', 'Precomputed': '--'}

# Legend for first plot
p1_leg = [Line2D([0], [0], color=cmap[i], lw=linewidth, label=legends[i]) for i in range(len(legends))]
p1_leg.append(Line2D([0], [0], marker='o', color='k', label='Runtime',
                      markerfacecolor='k', markersize=markersize, linewidth=linewidth))
p1_leg.append(Line2D([0], [0], marker='x', color='k', linestyle='--',
                      label='Precomputed', markerfacecolor='k', markersize=markersize, linewidth=linewidth))

# Legend for second plot
p2_leg = [Line2D([0], [0], color=cmap[2], lw=linewidth, label=legends[2]),
          Line2D([0], [0], color=cmap[3], lw=linewidth, label=legends[3]),
          Line2D([0], [0], color='k', linestyle='--', lw=linewidth, label='Ideal linear scaling')]
p2_leg.append(Line2D([0], [0], marker='o', color='k', label='Runtime', markerfacecolor='k', markersize=markersize, linewidth=linewidth))
p2_leg.append(Line2D([0], [0], marker='x', color='k', linestyle='--', label='Precomputed', markerfacecolor='k', markersize=markersize, linewidth=linewidth))

# For storing handles to avoid duplicate labels
handles_plot1 = {}
handles_plot2 = {}

for matrix in ['Runtime', 'Precomputed']:
    cpus = [1, 2, 4, 6, 8]
    times = np.loadtxt(f'exp2_times_EPI_P2_{matrix}.txt')  # shape (5,8)
    df = pd.DataFrame(times.T, index=legends, columns=cpus).T  # shape (8, 5)

    # 1) Execution time per component
    for i, label in enumerate(legends):
        line, = axes[0].plot(
            cpus,
            df[label]/60.0,  # Convert to minutes
            color=cmap[i],
            marker=markers[matrix],
            linestyle=linestyles[matrix],
            linewidth=linewidth,
            markersize=markersize,
        )
        # Save handles without duplicating labels
        handles_plot1[(label, matrix)] = line

    # 2) Speedup (only for two components)
    speedup_script = df['Full script'].iloc[0] / df['Full script']
    speedup_sim = df['Total sim.'].iloc[0] / df['Total sim.']

    line1, = axes[1].plot(
        cpus, speedup_script,
        color=cmap[3],
        marker=markers[matrix],
        markersize=markersize,
        linestyle=linestyles[matrix],
        linewidth=linewidth
    )
    line2, = axes[1].plot(
        cpus, speedup_sim,
        color=cmap[2],
        marker=markers[matrix],
        markersize=markersize,
        linestyle=linestyles[matrix],
        linewidth=linewidth
    )

    handles_plot2[f"Script ({matrix})"] = line1
    handles_plot2[f"Simulation ({matrix})"] = line2

# Ideal scaling line
ideal_line, = axes[1].plot(cpus, cpus, 'k--', label='Ideal linear scaling')

# Labels and layout
axes[0].set_xlabel('CPU cores', fontsize=fontsize)
axes[0].set_ylabel('Execution time (min)', fontsize=fontsize)
axes[0].set_title('Execution Time vs CPU Cores', fontsize=fontsize)

axes[1].set_xlabel('CPU cores', fontsize=fontsize)
axes[1].set_ylabel('Speedup', fontsize=fontsize)
axes[1].set_title('Speedup vs CPU Cores', fontsize=fontsize)

# Change x-ticks and y-ticks font size
axes[0].tick_params(axis='both', which='major', labelsize=fontsize)
axes[1].tick_params(axis='both', which='major', labelsize=fontsize)

# Custom legends
axes[0].legend(handles=p1_leg, loc='upper right', fontsize=fontsize)
axes[1].legend(handles=p2_leg, loc='upper left', fontsize=fontsize)

plt.tight_layout()
plt.savefig('exp2_times_EPI_P2_elements.eps')
plt.savefig('exp2_times_EPI_P2_elements.pdf')
plt.show()
