import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Raw data: (block, power, energy, throughput)
data = np.array([
    [32, 100, 55253.4, 556.05],
    [32, 125, 55600.3, 553.74],
    [32, 150, 56286.8, 554.61],
    [32, 175, 55275.5, 555.12],
    [32, 200, 55837.3, 555.63],
    [32, 225, 54148.3, 555.63],
    [32, 250, 55721.8, 556.84],
    [64, 100, 53510.4, 568.05],
    [64, 125, 53313.3, 564.42],
    [64, 150, 54202.8, 564.99],
    [64, 175, 53911.8, 564.85],
    [64, 200, 52491.4, 565.62],
    [64, 225, 53067.3, 567.37],
    [64, 250, 53738.7, 564.70],
    [128,100, 54510.7, 557.64],
    [128,125, 53460.4, 557.49],
    [128,150, 53755.4, 553.69],
    [128,175, 54538.5, 552.89],
    [128,200, 54276.2, 551.95],
    [128,225, 53392.9, 554.82],
    [128,250, 53815.2, 556.78],
    [256,100, 54402.4, 541.16],
    [256,125, 54139.1, 538.75],
    [256,150, 55834.4, 536.64],
    [256,175, 54233.7, 536.55],
    [256,200, 54287.2, 539.28],
    [256,225, 54002.8, 536.31],
    [256,250, 55080.8, 538.51],
    [512,100, 51136.0, 585.02],
    [512,125, 51342.5, 578.71],
    [512,150, 51610.3, 578.01],
    [512,175, 52039.6, 580.71],
    [512,200, 51003.7, 581.17],
    [512,225, 51679.2, 579.61],
    [512,250, 50302.7, 581.67],
    [786,100, 51264.6, 578.17],
    [786,125, 50700.2, 571.54],
    [786,150, 51374.4, 574.79],
    [786,175, 52337.5, 574.59],
    [786,200, 51395.2, 575.85],
    [786,225, 51544.6, 575.04],
    [786,250, 51323.7, 576.55],
    [1024,100,45529.6, 645.40],
    [1024,125,45645.0, 641.57],
    [1024,150,44714.4, 638.28],
    [1024,175,44535.6, 636.90],
    [1024,200,45354.6, 639.89],
    [1024,225,45284.5, 637.29],
    [1024,250,44707.2, 645.06],
    [2048,100,126296.6,227.11],
    [2048,125,124570.6,225.02],
    [2048,150,126940.2,225.21],
    [2048,175,126938.8,224.14],
    [2048,200,127588.3,225.15],
    [2048,225,124816.5,224.29],
    [2048,250,127246.0,225.66],
    [4096,100,127013.3,226.40],
    [4096,125,128634.6,223.12],
    [4096,150,129011.5,222.94],
    [4096,175,127151.8,223.09],
    [4096,200,126904.7,225.51],
    [4096,225,125234.9,224.20],
    [4096,250,126013.7,225.31],
    [8192,100,126163.8,225.72],
    [8192,125,126339.9,223.40],
    [8192,150,126890.2,224.46],
    [8192,175,128266.7,223.98],
    [8192,200,125555.0,224.69],
    [8192,225,124729.3,224.22],
    [8192,250,128173.6,224.18],
])

energy = data[:, 2]  # Energy per token in μJ
throughput = data[:, 3]  # Throughput in tokens/s

# Identify Pareto optimal points: lower energy and higher throughput
is_pareto = np.ones(len(data), dtype=bool)
for i in range(len(data)):
    for j in range(len(data)):
        if (energy[j] <= energy[i] and throughput[j] >= throughput[i]
            and (energy[j] < energy[i] or throughput[j] > throughput[i])):
            is_pareto[i] = False
            break

pareto_data = data[is_pareto]
# Sort Pareto data by increasing energy
pareto_data = pareto_data[np.argsort(pareto_data[:,2])]
e_pareto = pareto_data[:,2]
t_pareto = pareto_data[:,3]

fig, ax = plt.subplots(figsize=(8,6))

# Background points
ax.scatter(energy, throughput, color='lightgray', s=30, edgecolor='gray')

# Highlight Pareto points
ax.scatter(e_pareto, t_pareto, color='green', s=150, edgecolor='black', label='Pareto points', zorder=3)

# Monotonic interpolation for smoothness
if len(e_pareto) >= 2:
    e_dense = np.linspace(e_pareto.min(), e_pareto.max(), 10000)
    # PCHIP preserves monotonicity
    pchip = PchipInterpolator(e_pareto, t_pareto)
    t_dense = pchip(e_dense)
    ax.plot(e_dense, t_dense, color='green', linewidth=2, label='Pareto front', zorder=2)

# Annotate each Pareto point
for block,power,en,thr in pareto_data:
    ax.text(en, thr, f"{int(block)}@{int(power)}", fontsize=10, va='bottom', ha='right')

# Labels with preference hints
ax.set_xlabel('Energy per Token (μJ) — lower is better', fontsize=16)
ax.set_ylabel('Throughput (tokens/s) — higher is better', fontsize=16)
# ax.set_title('Pareto Front: Energy vs. Throughput', fontsize=14)
ax.grid(True)
ax.legend(loc='lower left', fontsize=14)

# set x and y tick size
ax.tick_params(axis='both', which='major', labelsize=14)

# Inset zoom
x1, x2 = e_pareto.min()*0.98, e_pareto.max()*1.02
y1, y2 = t_pareto.min()*0.98, t_pareto.max()*1.02
axins = inset_axes(ax, width="40%", height="40%", loc='upper right')
axins.scatter(energy, throughput, color='lightgray', s=20, edgecolor='gray')
axins.scatter(e_pareto, t_pareto, color='green', s=100, edgecolor='black', zorder=3)
axins.plot(e_dense, t_dense, color='green', linewidth=2, zorder=2)
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.tick_params(axis='both', which='major', labelsize=14)
axins.grid(True)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.tight_layout()
plt.savefig('pareto_plot_smooth.png', dpi=300)
plt.show()