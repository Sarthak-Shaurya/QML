# Practical QML in the NISQ Era: Automated Hybrid Pipelines for Speech Processing

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14%2B-ff6f00)
![Qiskit](https://img.shields.io/badge/Qiskit-0.44%2B-6929c4)
![PennyLane](https://img.shields.io/badge/PennyLane-0.32%2B-000000)

**Authors:**
* Sarthak Shaurya (22116082)
* Aryan Chaudhary (22115031)
* Nitesh Thalor (22116062)

---

## 📌 Project Overview

Processing continuous, high-dimensional temporal signals (like Automatic Speech Recognition or Audio Deepfake Detection) natively on Noisy Intermediate-Scale Quantum (NISQ) devices is severely bottlenecked by limited qubit budgets and high gate error rates. 

This project engineers a solution by decentralizing the architecture. We built a fully automated, hybrid Quantum-Classical (CQ) pipeline that leverages a 4-qubit **Quantum Convolutional Neural Network (QCNN)** as a highly non-linear feature extractor (Quantum Sieve), offloading the heavy sequence learning to classical cloud-accelerated **Recurrent Neural Networks (Bi-LSTM)**.

### The 3-Step Engineering Pipeline:
1. **Feasibility (Baseline):** Validated NISQ encoding constraints by reproducing a clinical data classification baseline using a linear-time $\log_2N$ encoding technique and a simplified Quantum-Kernel Support Vector Machine (sqKSVM).
2. **Scalability (Hybrid ASR):** Transitioned to continuous audio processing. Extracted $30 \times 63 \times 4$ quantum spatial tensors from Mel-spectrograms using localized "Quanvolutional" filters. Replicated the ICASSP 2021 10-Class Speech Benchmark, achieving **90.60% Validation Accuracy**.
3. **Deployability (MLOps Automation):** Eliminated the manual CPU/QPU hyperparameter tuning bottleneck by deploying the **Quantum Approximate Optimization Algorithm (QAOA)**. Finally, repurposed the initial sqKSVM baseline to act as a real-time production drift monitor.

---

## 📂 Repository Architecture

```text
Quantum-Hybrid-ASR/
│
├── README.md                      # Project documentation
├── requirements.txt               # Environment dependencies
├── presentation_deck.pdf          # Final presentation slides
│
├── assets/                        # Visualizations and result graphs
│
├── step1_clinical_baseline/       # Step 1: Clinical Data Proof of Concept
│   └── classifier.py       # Linear-time encoding quantum kernel SVM
│
├── step2_hybrid_asr/              # Step 2: Continuous Audio & Speech Recognition
│   ├── authors_baseline/          # (Cloud Keras) 10-Class Sequence Training
│   │   ├── main_qsr.py            # Main training loop (Bi-LSTM / Softmax Attention)
│   │   ├── models.py               # Classical backend architectures
│   │   └── helper_q_tool.py       # Pennylane/Qiskit quantum circuit generator
│   │
│   └── custom_pytorch_validation/ # (Local PyTorch) Engineering Prototype
│       ├── download_data.py       # Download the required data
│       ├── train_hybrid.py        # Custom binary classification for local hardware test
│       └── plot_qvision.py        # Generates the 4-channel Pauli-Z feature maps
│
├── step3_mlops_automation/        # Step 3: Novelty & Pipeline Automation
│   ├── qaoa_hpo.py                # QUBO formulated hyperparameter tuning
│   └── qsvm_monitor.py            # Real-time out-of-distribution audio flagging
│
└── data_samples/                  # Sample data formats (Full dataset hosted externally)
    ├── sample_audio.wav           
    └── sample_tensor.npy
```
## 🚀 Installation & Setup

It is highly recommended to use a virtual environment (like `venv` or `conda`) to avoid conflicts between PyTorch and TensorFlow dependencies.

```bash
# 1. Clone the repository
git clone [https://github.com/Sarthak-Shaurya/QML.git](https://github.com/Sarthak-Shaurya/QML.git)
cd QML

# 2. Create and activate a virtual environment (Linux/Mac)
python3 -m venv qml_env
source qml_env/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt
```
## 💻 How to Run the Code

<b> Step 1: Execute the Clinical Baseline (sqKSVM)</b> To run the static baseline that proves the viability of the $\log_2N$ encoding on constrained hardware:
```bash
cd step1_clinical_baseline
python classifier.py
```
Expected Output: Generates classification boundaries achieving ~0.9074 AUC on the Wisconsin Breast Cancer dataset.

<b> Step 2: Execute the Hybrid ASR Pipeline</b> <br>
<b>A. Local Prototype (PyTorch Binary Validation) </b>To visualize the 4-channel quantum feature maps and run the lightweight local CNN:

```Bash
cd step2_hybrid_asr/custom_pytorch_validation
python download_data.py     # Downloads the required data
python plot_qvision.py      # Generates 'quantum_vision.png'
python train_hybrid.py      # Runs the Left vs. Right binary classification
```
<b> B. Full 10-Class Cloud Benchmark (Keras Bi-LSTM)</b> (Requires the full extracted tensor dataset placed in the data_quantum/ directory)

```Bash
cd step2_hybrid_asr/authors_baseline
python main_qsr.py --sr 16000 --mel 1
```
Expected Output: 94 steps/epoch, converging to >90.0% validation accuracy by Epoch 16.

<b> Step 3: MLOps Automation</b>
To run the QAOA hyperparameter optimization solver or test the QSVM drift monitor on sample audio streams:

```Bash
cd step3_mlops_automation
python qaoa_hpo.py
python qsvm_monitor.py
```

## 📚 References & Acknowledgements
S. Moradi, C. Brandner, et al., "Clinical data classification with noisy intermediate scale quantum computers," Scientific Reports, vol. 12, no. 1851, 2022.

C.-H. H. Yang, J. Qi, et al., "Decentralizing Feature Extraction with Quantum Convolutional Neural Network for Automatic Speech Recognition," arXiv:2010.13309, 2021.

E. Farhi, J. Goldstone, and S. Gutmann, "A Quantum Approximate Optimization Algorithm," arXiv:1411.4028, 2014.

We extend our gratitude to the authors of the original QCNN ASR repository for providing the Keras/Qiskit foundation upon which we built our decoupled pipeline and automation wrappers.
