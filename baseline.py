#!pip install pennylane qiskit qiskit-aer scikit-learn numpy pandas matplotlib

import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from qiskit_aer.noise import NoiseModel, depolarizing_error
import warnings
warnings.filterwarnings('ignore')

# --- 1. SETUP NISQ NOISE MODEL ---
noise_model = NoiseModel()
error_1q = depolarizing_error(0.001, 1) # 0.1% error on 1-qubit gates
error_2q = depolarizing_error(0.01, 2)  # 1.0% error on 2-qubit gates
noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz'])
noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])

def create_quantum_environments(n_features):
    n_qubits = int(np.log2(n_features))
    
    # Device for qKSVM / sqKSVM 
    dev_kernel = qml.device(
        "qiskit.aer", 
        wires=n_qubits, 
        shots=8192, 
        noise_model=noise_model
    )
    
    # Device for qDC Hadamard Test (requires 1 extra ancilla qubit)
    dev_hadamard = qml.device(
        "qiskit.aer", 
        wires=n_qubits + 1, 
        shots=8192, 
        noise_model=noise_model
    )
    
    return dev_kernel, dev_hadamard, n_qubits

# --- 2. THE AUTHORS' EXACT QUANTUM KERNEL (sqKSVM & qKSVM) ---
def get_kernel_functions(dev_kernel, n_qubits):
    projector = np.zeros((2**n_qubits, 2**n_qubits))
    projector[0, 0] = 1

    @qml.qnode(dev_kernel)
    def compute_kernel(x1, x2):
        """The quantum kernel (Adjoint Method)."""
        qml.MottonenStatePreparation(x1, wires=range(n_qubits))
        qml.adjoint(qml.MottonenStatePreparation)(x2, wires=range(n_qubits))
        return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))

    def kernel_matrix(A, B):
        return np.array([[compute_kernel(a, b) for b in B] for a in A])
        
    return compute_kernel, kernel_matrix

# --- 3. THE AUTHORS' EXACT qDC CIRCUIT (Hadamard Test) ---
def get_qdc_circuit(dev_hadamard, n_qubits):
    
    def ops(X):
        qml.MottonenStatePreparation(X, wires=range(1, n_qubits + 1))
        
    ops1 = qml.ctrl(ops, control=0)

    @qml.qnode(dev_hadamard)
    def qdc_circuit(X1, X2):
        qml.Hadamard(wires=0)
        ops1(X1)
        qml.PauliX(wires=0)
        ops1(X2)
        qml.PauliX(wires=0)
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliZ(0))
        
    return qdc_circuit

# --- 4. PREDICTION ALGORITHMS ---
def qdc_predict(X_train, y_train, X_test, qdc_circuit):
    """Authors' qDC logic: Max-pooling the overlap per class."""
    x1_train = X_train[y_train == 1]
    x2_train = X_train[y_train == 0]
    
    D1 = np.array([[qdc_circuit(x1, x_t) for x_t in X_test] for x1 in x1_train])
    D2 = np.array([[qdc_circuit(x2, x_t) for x_t in X_test] for x2 in x2_train])
    
    d11 = np.max(D1, axis=0)
    d22 = np.max(D2, axis=0)
    
    return [1 if d11[i] >= d22[i] else 0 for i in range(len(X_test))]

def sqksvm_predict(X_train, y_train, X_test, kernel_matrix_fn):
    """Authors' sqKSVM logic: Manual dot product with IR weights."""
    ir = np.sum(y_train == 1) / len(y_train)
    y_train_new = np.array([1 - ir if y == 1 else -ir for y in y_train])
    
    sigma_12 = kernel_matrix_fn(X_train, X_test)
    mu = y_train_new @ sigma_12
    
    return [1 if i >= 0 else 0 for i in mu]

# --- 5. EXECUTION LOOP WITH CLASSICAL BASELINES ---
def run_experiment(k_features):
    data = load_breast_cancer()
    X = data.data
    y = data.target 

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    auc_qdc, auc_sq, auc_qk, auc_csvm, auc_knn = [], [], [], [], []

    dev_kernel, dev_hadamard, n_qubits = create_quantum_environments(k_features)
    compute_kernel, kernel_matrix_fn = get_kernel_functions(dev_kernel, n_qubits)
    qdc_circuit = get_qdc_circuit(dev_hadamard, n_qubits)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"  Running Fold {fold + 1}/10...")
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Feature Selection
        selector = SelectKBest(f_classif, k=k_features)
        X_train_sel = selector.fit_transform(X_train_raw, y_train)
        X_test_sel = selector.transform(X_test_raw)

        # EXACT AUTHORS' NORMALIZATION (Axis 0, then Axis 1)
        X_train_ax0 = normalize(X_train_sel, norm='l2', axis=0)
        X_test_ax0  = normalize(X_test_sel,  norm='l2', axis=0)
        
        X_train = normalize(X_train_ax0, norm='l2', axis=1)
        X_test  = normalize(X_test_ax0,  norm='l2', axis=1)

        # --- Quantum Models ---
        pred_qdc = qdc_predict(X_train, y_train, X_test, qdc_circuit)
        pred_sq = sqksvm_predict(X_train, y_train, X_test, kernel_matrix_fn)
        
        K_train = kernel_matrix_fn(X_train, X_train)
        K_test = kernel_matrix_fn(X_test, X_train) 
        svm_q = SVC(kernel='precomputed', class_weight='balanced').fit(K_train, y_train)
        pred_qk = svm_q.predict(K_test)

        # --- Classical Models (Baselines) ---
        svm_c = SVC(kernel='rbf', class_weight='balanced').fit(X_train, y_train)
        pred_csvm = svm_c.predict(X_test)
        
        knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
        pred_knn = knn.predict(X_test)

        # --- Score Collection ---
        auc_qdc.append(roc_auc_score(y_test, pred_qdc))
        auc_sq.append(roc_auc_score(y_test, pred_sq))
        auc_qk.append(roc_auc_score(y_test, pred_qk))
        auc_csvm.append(roc_auc_score(y_test, pred_csvm))
        auc_knn.append(roc_auc_score(y_test, pred_knn))

    return np.mean(auc_qdc), np.mean(auc_sq), np.mean(auc_qk), np.mean(auc_csvm), np.mean(auc_knn)

# --- 6. SCALING ACROSS FEATURE SIZES ---
feature_sizes = [4, 8, 16]
results_qdc, results_sqksvm, results_qksvm, results_csvm, results_knn = [], [], [], [], []

print("Starting Full Evaluation over Feature Sizes [4, 8, 16]...")

for k in feature_sizes:
    n_qubits = int(np.log2(k))
    print(f"\nEvaluating {k} Features ({n_qubits} Qubits):")
    qdc_score, sq_score, qk_score, csvm_score, knn_score = run_experiment(k)
    
    results_qdc.append(qdc_score)
    results_sqksvm.append(sq_score)
    results_qksvm.append(qk_score)
    results_csvm.append(csvm_score)
    results_knn.append(knn_score)
    
    print(f"  -> qDC AUC:       {qdc_score:.4f}")
    print(f"  -> sqKSVM AUC:    {sq_score:.4f}")
    print(f"  -> qKSVM AUC:     {qk_score:.4f}")
    print(f"  -> Classical SVM: {csvm_score:.4f}")
    print(f"  -> Classical KNN: {knn_score:.4f}")

# --- 7. PLOTTING THE RESULTS FOR BEAMER PRESENTATION ---
print("\nGenerating final plot...")

labels = ['4 Features\n(2 Qubits)', '8 Features\n(3 Qubits)', '16 Features\n(4 Qubits)']
x = np.arange(len(labels))
width = 0.15

fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - 2*width, results_qdc, width, label='qDC', color='#1f77b4')
rects2 = ax.bar(x - width, results_sqksvm, width, label='sqKSVM', color='#ff7f0e')
rects3 = ax.bar(x, results_qksvm, width, label='qKSVM', color='#d62728')
rects4 = ax.bar(x + width, results_csvm, width, label='Classical SVM', color='#2ca02c')
rects5 = ax.bar(x + 2*width, results_knn, width, label='Classical KNN', color='#9467bd')

ax.set_ylabel('Mean AUC Score', fontsize=12)
ax.set_title('Breast Cancer Dataset: Quantum vs Classical Classification Performance', fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim([0.4, 1.05])
ax.legend(loc='lower left')
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Format the labels above the bars
for rects in [rects1, rects2, rects3, rects4, rects5]:
    ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=8)

fig.tight_layout()
plt.savefig('sqksvm_results.png', dpi=300)
plt.show()

print("Execution complete. 'sqksvm_results.png' saved successfully.")
