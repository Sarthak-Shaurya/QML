import matplotlib.pyplot as plt
import numpy as np

# Data extracted from the provided text messages
labels = ['qDC', 'sqKSVM', 'qKSVM']
auc_8_features = [0.9206, 0.8802, 0.9389]
auc_16_features = [0.8998, 0.8631, 0.9350]

x = np.arange(len(labels))  # The label locations
width = 0.35                # The width of the bars

fig, ax = plt.subplots(figsize=(8, 5))

# Create the grouped bars (colors approximate the reference image)
rects1 = ax.bar(x - width/2, auc_8_features, width, label='8 Features (3 Qubits)', 
                color='#3b82b8', edgecolor='black', zorder=3)
rects2 = ax.bar(x + width/2, auc_16_features, width, label='16 Features (4 Qubits)', 
                color='#ff9130', edgecolor='black', zorder=3)

# Add y-axis label, title, and custom x-axis tick labels
ax.set_ylabel('Validation AUC', fontweight='bold', fontsize=12)
ax.set_title('Baseline Validation: Clinical Data Classification (Wisconsin Breast Cancer)', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontweight='bold', fontsize=12)

# Set y-axis limits exactly as shown in the reference graph
ax.set_ylim(0.750, 0.950)

# Add a dashed horizontal grid behind the bars
ax.yaxis.grid(True, linestyle='--', color='lightgray')
ax.set_axisbelow(True) # Puts the grid behind the bar objects

# Add the legend in the upper right
ax.legend(loc='upper right')

# Function to attach a text label above each bar
def autolabel(rects):
    """Attach a text label above each bar displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

# Display the plot
plt.show()
