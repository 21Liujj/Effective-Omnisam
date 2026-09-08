import matplotlib.pyplot as plt
import numpy as np

# Data from paper table
methods = ['Baseline', 'Baseline+MTC+SMR', 'Baseline+MTC+SMR+SCM']
miou = [60.22, 60.00, 60.84]
macc = [70.45, 70.22, 71.15]
pixacc = [91.23, 91.15, 91.52]
fwiou = [83.67, 83.52, 84.23]
bfscore = [39.59, 39.23, 39.86]

# Setup figure
fig, ax = plt.subplots(figsize=(12, 6))

# Bar chart parameters
x = np.arange(len(methods))
width = 0.15

# Plot bars
bars1 = ax.bar(x - 2*width, miou, width, label='mIoU(%)', color='#2E86C1')
bars2 = ax.bar(x - width, macc, width, label='mAcc(%)', color='#28B463')
bars3 = ax.bar(x, pixacc, width, label='PixAcc(%)', color='#F39C12')
bars4 = ax.bar(x + width, fwiou, width, label='FWIoU(%)', color='#8E44AD')
bars5 = ax.bar(x + 2*width, bfscore, width, label='BFScore', color='#E74C3C')

# Axis settings
ax.set_xlabel('Methods', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Performance Comparison Across Methods', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10, rotation=15, ha='right')
ax.legend(loc='upper left', fontsize=10)
ax.set_ylim([30, 95])
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=7)

autolabel(bars1)
autolabel(bars2)
autolabel(bars3)
autolabel(bars4)
autolabel(bars5)

plt.tight_layout()
plt.savefig('figures/fig_metrics_bar_chart.png', dpi=300, bbox_inches='tight')
print('Figure 1 bar chart generated: figures/fig_metrics_bar_chart.png')
