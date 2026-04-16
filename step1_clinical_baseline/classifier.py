



import numpy as np

import pennylane as qml

from sklearn import metrics

from sklearn.svm import SVC

from sklearn.preprocessing import normalize

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.metrics import roc_auc_score

from qiskit_aer.noise import NoiseModel, depolarizing_error

import warnings

warnings.filterwarnings('ignore')



# --- 1. SETUP NISQ NOISE MODEL ---

# The authors used ibmq_quito. To replicate this without an IBM account, 

# we build a generic NISQ hardware noise model using Qiskit Aer.

noise_model = NoiseModel()

error_1q = depolarizing_error(0.001, 1) # 0.1% error on 1-qubit gates

error_2q = depolarizing_error(0.01, 2)  # 1.0% error on 2-qubit gates

noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz'])

noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])



def create_quantum_environments(n_features):

    n_qubits = int(np.log2(n_features))

    

    # Device for qKSVM / sqKSVM (3 qubits for 8 features)

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

        # Authors used qml.inv, which is deprecated in modern pennylane. Using adjoint.

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

    

    # Compute overlaps for Class 1

    D1 = np.array([[qdc_circuit(x1, x_t) for x_t in X_test] for x1 in x1_train])

    # Compute overlaps for Class 0

    D2 = np.array([[qdc_circuit(x2, x_t) for x_t in X_test] for x2 in x2_train])

    

    d11 = np.max(D1, axis=0)

    d22 = np.max(D2, axis=0)

    

    # Assign label based on highest max overlap

    return [1 if d11[i] >= d22[i] else 0 for i in range(len(X_test))]



def sqksvm_predict(X_train, y_train, X_test, kernel_matrix_fn):

    """Authors' sqKSVM logic: Manual dot product with IR weights."""

    # Calculate analytical weights

    ir = np.sum(y_train == 1) / len(y_train)

    y_train_new = np.array([1 - ir if y == 1 else -ir for y in y_train])

    

    # Σ12 = kernel_matrix(X1, X2) -> size: (N_train, N_test)

    sigma_12 = kernel_matrix_fn(X_train, X_test)

    

    # μ = y1 @ Σ12 

    mu = y_train_new @ sigma_12

    

    return [1 if i >= 0 else 0 for i in mu]



# --- 5. EXECUTION LOOP ---

def run_experiment(k_features):

    data = load_breast_cancer()

    X = data.data

    y = data.target # Keep as 0 and 1, matching authors' code



    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    auc_qdc, auc_sq, auc_qk = [], [], []



    dev_kernel, dev_hadamard, n_qubits = create_quantum_environments(k_features)

    compute_kernel, kernel_matrix_fn = get_kernel_functions(dev_kernel, n_qubits)

    qdc_circuit = get_qdc_circuit(dev_hadamard, n_qubits)



    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):

        print(f"Running Fold {fold + 1}/10...")

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



        print("  - Running qDC...")

        pred_qdc = qdc_predict(X_train, y_train, X_test, qdc_circuit)

        

        print("  - Running sqKSVM...")

        pred_sq = sqksvm_predict(X_train, y_train, X_test, kernel_matrix_fn)

        

        print("  - Running qKSVM...")

        # Authors used SVC(kernel=kernel_matrix, class_weight='balanced')

        K_train = kernel_matrix_fn(X_train, X_train)

        K_test = kernel_matrix_fn(X_test, X_train) 

        svm = SVC(kernel='precomputed', class_weight='balanced').fit(K_train, y_train)

        pred_qk = svm.predict(K_test)



        auc_qdc.append(roc_auc_score(y_test, pred_qdc))

        auc_sq.append(roc_auc_score(y_test, pred_sq))

        auc_qk.append(roc_auc_score(y_test, pred_qk))



        # Break early just to verify it runs

        if fold == 0:

            print("Fold 1 complete. (Remove the break statement to run full CV)")

            



    return np.mean(auc_qdc), np.mean(auc_sq), np.mean(auc_qk)



print("--- RUNNING 8 FEATURES (3 QUBITS) ---")

qdc8, sq8, qksvm8 = run_experiment(8)



print(f"\nqDC AUC:    {qdc8:.4f}")

print(f"sqKSVM AUC: {sq8:.4f}")

print(f"qKSVM AUC:  {qksvm8:.4f}")

