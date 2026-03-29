import numpy as np
import matplotlib.pyplot as plt

print("Loading a sample from the Quantum Features...")
X = np.load('data/quantum_features/X_qcnn.npy')

# Grab the very first audio sample in your dataset
sample = X[0] # Shape is (4 channels, 8 height, 8 width)

fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))

for i in range(4):
    # Plot the 8x8 spatial grid for each qubit's measurement
    im = axes[i].imshow(sample[i], cmap='magma', interpolation='nearest')
    axes[i].set_title(f'Qubit {i} Output', fontsize=12)
    axes[i].axis('off')

# Add a colorbar to show the measurement expectation values (-1 to 1)
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
cbar.set_label('Expectation Value (Pauli-Z)', rotation=270, labelpad=15)

plt.suptitle('Quantum Feature Extraction: The 4-Channel "Quanvolutional" View', fontsize=14, y=1.05)
plt.savefig('quantum_vision.png', dpi=300, bbox_inches='tight')
print("Saved 'quantum_vision.png'. Look at the distinct patterns each qubit captured!")