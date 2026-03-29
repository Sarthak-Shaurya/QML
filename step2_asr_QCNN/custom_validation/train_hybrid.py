from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# --- 1. LOAD THE QUANTUM DATA ---
print("Loading Quantum Features...")
X = np.load('data/quantum_features/X_qcnn.npy')
y = np.load('data/quantum_features/y_qcnn.npy')

# Split into Training (80%) and Validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch Tensors
train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_data = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# --- 2. DEFINE THE CLASSICAL BACKEND MODEL ---
class ClassicalCNN(nn.Module):
    def __init__(self):
        super(ClassicalCNN, self).__init__()
        # Input shape: (Batch, 4 quantum channels, 8 height, 8 width)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Output: 16 channels, 4x4 spatial
        
        # Flattened size: 16 channels * 4 * 4 = 256
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 2) # 2 output classes: Left vs. Right

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = ClassicalCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nModel Parameter Count:")
print(f" -> Hybrid QCNN Trainable Parameters: {count_parameters(model):,}")
# --- 3. TRAINING LOOP ---
epochs = 30
train_losses = []
val_accuracies = []

print("\nStarting Hybrid Training (Quantum Features -> Classical CNN)...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    val_acc = 100 * correct / total
    val_accuracies.append(val_acc)
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_train_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

print("Training Complete!")

# --- 4. GENERATE THE PRESENTATION PLOT (SLIDE 6) ---
print("\nGenerating qcnn_results.png for your presentation...")
fig, ax1 = plt.subplots(figsize=(8, 5))

color = '#d62728'
ax1.set_xlabel('Training Epochs', fontsize=12)
ax1.set_ylabel('Training Loss', color=color, fontsize=12)
ax1.plot(range(1, epochs + 1), train_losses, color=color, linewidth=2, marker='o', markersize=4, label='QCNN Training Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()  
color = '#1f77b4'
ax2.set_ylabel('Validation Accuracy (%)', color=color, fontsize=12)  
ax2.plot(range(1, epochs + 1), val_accuracies, color=color, linewidth=2, marker='s', markersize=4, label='QCNN Validation Accuracy')
ax2.tick_params(axis='y', labelcolor=color)

# Set accuracy axis from roughly min to 100% for better scale
ax2.set_ylim([max(0, min(val_accuracies) - 10), 100])

plt.title('Hybrid QCNN: Audio Feature Classification Performance', fontsize=14, pad=15)

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

fig.tight_layout()
plt.savefig('qcnn_results.png', dpi=300)
# --- 5. GENERATE CONFUSION MATRIX ---
print("\nGenerating Confusion Matrix...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

fig3, ax3 = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, 
            xticklabels=['Left', 'Right'], yticklabels=['Left', 'Right'],
            annot_kws={"size": 14})

ax3.set_xlabel('Predicted Label', fontsize=12)
ax3.set_ylabel('True Label', fontsize=12)
ax3.set_title('QCNN Confusion Matrix', fontsize=14, pad=15)

plt.tight_layout()
plt.savefig('qcnn_confusion_matrix.png', dpi=300)
print("Saved 'qcnn_confusion_matrix.png'!")
print("Saved qcnn_results.png successfully! You can now upload this to Overleaf.")