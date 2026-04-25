import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# --- 1. DEFINING THE SEARCH SPACE & HAMILTONIAN ---
n_qubits = 4
wires = range(n_qubits)
dev = qml.device("default.qubit", wires=n_qubits)

coeffs = [1.0, -1.0, 1.0, -1.0, 0.8, 0.8]
obs = [
    qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(2), qml.PauliZ(3),
    qml.PauliZ(0) @ qml.PauliZ(1), 
    qml.PauliZ(2) @ qml.PauliZ(3)
]
cost_h = qml.Hamiltonian(coeffs, obs)
mixer_h = qml.qaoa.x_mixer(wires)

# --- 2. BUILDING THE QAOA CIRCUIT ---
def qaoa_layer(gamma, alpha):
    qml.qaoa.cost_layer(gamma, cost_h)
    qml.qaoa.mixer_layer(alpha, mixer_h)

depth = 3 

def circuit(params):
    for w in wires:
        qml.Hadamard(wires=w)
    for i in range(depth):
        qaoa_layer(params[0][i], params[1][i])

@qml.qnode(dev)
def cost_function(params):
    circuit(params)
    return qml.expval(cost_h)

@qml.qnode(dev)
def probability_circuit(params):
    circuit(params)
    return qml.probs(wires=wires)

# --- 3. CLASSICAL OPTIMIZATION LOOP ---
np.random.seed(42)
params = np.random.uniform(0, np.pi, (2, depth), requires_grad=True)
# Swapping to Adam for adaptive, smooth convergence over a slightly longer run
optimizer = qml.AdamOptimizer(stepsize=0.05)
steps = 80

# --> NEW: Track the cost history to plot convergence
cost_history = []

print("Starting QAOA Hyperparameter Optimization...")
for i in range(steps):
    params, cost = optimizer.step_and_cost(cost_function, params)
    cost_history.append(cost) # Save the cost at each step
    
    if (i + 1) % 10 == 0:
        print(f"Optimization Step {i+1:2d} | Cost (Energy): {cost:.4f}")

print("\nOptimization Complete.")

# --- 4. EXTRACTING RESULTS & PLOTTING ---
probs = probability_circuit(params)
bitstrings = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
best_idx = np.argmax(probs)
best_bitstring = bitstrings[best_idx]

lr_map = {"00": "0.01", "01": "0.005", "10": "0.001", "11": "0.0001"}
hn_map = {"00": "16", "01": "32", "10": "64", "11": "128"}
best_lr = lr_map[best_bitstring[:2]]
best_hn = hn_map[best_bitstring[2:]]

# --- PLOT 1: QAOA Convergence Curve (The New Plot) ---
print("\nGenerating 'qaoa_convergence.png'...")
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(range(1, steps + 1), cost_history, color='#2ca02c', linewidth=2.5, marker='o', markersize=3)
ax1.set_xlabel('Optimization Steps', fontsize=12)
ax1.set_ylabel('Expectation Value (Cost / Energy)', fontsize=12)
ax1.set_title('QAOA Optimization: Finding the Ground State', fontsize=14, pad=15)
ax1.grid(True, linestyle='--', alpha=0.6)
fig1.tight_layout()
plt.savefig('qaoa_convergence.png', dpi=300)

# --- PLOT 2: Probability Distribution (Your Original Plot) ---
print("Generating 'qaoa_results.png'...")
fig2, ax2 = plt.subplots(figsize=(10, 5))
bars = ax2.bar(bitstrings, probs, color='#1f77b4', edgecolor='black', alpha=0.8)
bars[best_idx].set_color('#d62728')
bars[best_idx].set_edgecolor('black')

ax2.set_ylabel('Measurement Probability', fontsize=12)
ax2.set_xlabel('Hyperparameter Configurations (Bitstrings)', fontsize=12)
ax2.set_title('QAOA Convergence on Optimal Model Parameters', fontsize=14, pad=15)
plt.xticks(rotation=45)
ax2.grid(axis='y', linestyle='--', alpha=0.6)

textstr = f"QAOA Output:\nBest Config: |{best_bitstring}>\nLR: {best_lr}\nNodes: {best_hn}"
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax2.text(0.02, 0.95, textstr, transform=ax2.transAxes, fontsize=11, verticalalignment='top', bbox=props)

fig2.tight_layout()
plt.savefig('qaoa_results.png', dpi=300)

print("\nSaved both plots successfully!")
