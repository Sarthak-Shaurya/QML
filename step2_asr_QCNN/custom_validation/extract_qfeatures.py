import os
import numpy as np
import librosa
import pennylane as qml
import torch

# --- 1. CONFIGURATION ---
DATA_DIR = 'data/binary_speech'
OUTPUT_DIR = 'data/quantum_features'
CLASSES = ['left', 'right']
# Simulating quantum circuits is slow on laptops. 
# We limit to 400 samples per class (800 total) to ensure it finishes in ~30-45 mins.
MAX_SAMPLES_PER_CLASS = 400 

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. THE 4-QUBIT QUANTUM CIRCUIT (QUANVOLUTION) ---
# We use default.qubit because it runs significantly faster on Macs than aer simulators
dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def quanv_circuit(pixels):
    """Encodes 4 classical pixels into 4 qubits and entangles them."""
    # Data Encoding (Mapping classical pixels to quantum rotations)
    for j, pixel in enumerate(pixels):
        qml.RY(np.pi * pixel, wires=j)
    
    # Entanglement / Feature Extraction
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])
    
    # Measurement in the Pauli-Z basis
    return [qml.expval(qml.PauliZ(j)) for j in range(4)]

def apply_quanvolution(image):
    """Sweeps the 4-qubit circuit across the 16x16 spectrogram."""
    # Output will be 8x8 spatial dimensions with 4 quantum channels
    out = np.zeros((8, 8, 4))
    
    # Stride of 2
    for j in range(0, 16, 2):
        for k in range(0, 16, 2):
            # Extract 2x2 window (4 pixels)
            window = [
                image[j, k], image[j, k+1],
                image[j+1, k], image[j+1, k+1]
            ]
            # Feed to quantum circuit
            q_results = quanv_circuit(window)
            
            # Map results to output channels
            for c in range(4):
                out[j//2, k//2, c] = q_results[c]
    return out

# --- 3. AUDIO PROCESSING PIPELINE ---
X_quantum = []
y_labels = []

print(f"Starting Quantum Feature Extraction (Max {MAX_SAMPLES_PER_CLASS} per class)...")
print("This simulates thousands of quantum circuits. Grab a coffee, this will take a while!")

for label_idx, label_name in enumerate(CLASSES):
    class_dir = os.path.join(DATA_DIR, label_name)
    files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
    
    count = 0
    for file in files:
        if count >= MAX_SAMPLES_PER_CLASS:
            break
            
        file_path = os.path.join(class_dir, file)
        
        try:
            # 1. Load Audio (1 second at 16000 Hz)
            y, sr = librosa.load(file_path, sr=16000, duration=1.0)
            
            # Pad if audio is slightly less than 1 second
            if len(y) < 16000:
                y = np.pad(y, (0, 16000 - len(y)))
            else:
                y = y[:16000]
                
            # 2. Convert to 16x16 Mel-Spectrogram
            # hop_length=1024 over 16000 samples yields exactly 16 time frames
            melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=16, hop_length=1024)
            melspec = librosa.power_to_db(melspec, ref=np.max)
            
            # Normalize between 0 and 1 for the quantum rotation angles
            melspec = (melspec - np.min(melspec)) / (np.max(melspec) - np.min(melspec) + 1e-8)
            
            # 3. Apply Quantum Filter
            q_features = apply_quanvolution(melspec)
            
            X_quantum.append(q_features)
            y_labels.append(label_idx)
            
            count += 1
            if count % 50 == 0:
                print(f"  Processed {count}/{MAX_SAMPLES_PER_CLASS} for class '{label_name}'")
                
        except Exception as e:
            continue

print("\nExtraction Complete! Saving Tensors...")

# Convert to PyTorch friendly formats (Samples, Channels, Height, Width)
X_tensor = np.array(X_quantum).transpose(0, 3, 1, 2) 
y_tensor = np.array(y_labels)

# Save the arrays
np.save(os.path.join(OUTPUT_DIR, 'X_qcnn.npy'), X_tensor)
np.save(os.path.join(OUTPUT_DIR, 'y_qcnn.npy'), y_tensor)

print(f"Saved successfully to {OUTPUT_DIR}/!")
print(f"X shape: {X_tensor.shape} (800 samples, 4 channels, 8x8 spatial dims)")