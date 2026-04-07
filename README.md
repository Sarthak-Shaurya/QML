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
│   └── sqKSVM_classifier.py       # Linear-time encoding quantum kernel SVM
│
├── step2_hybrid_asr/              # Step 2: Continuous Audio & Speech Recognition
│   ├── authors_baseline/          # (Cloud Keras) 10-Class Sequence Training
│   │   ├── main_qsr.py            # Main training loop (Bi-LSTM / Softmax Attention)
│   │   ├── model.py               # Classical backend architectures
│   │   └── helper_q_tool.py       # Pennylane/Qiskit quantum circuit generator
│   │
│   └── custom_pytorch_validation/ # (Local PyTorch) Engineering Prototype
│       ├── train_hybrid.py        # Custom binary classification for local hardware test
│       └── plot_qvision.py        # Generates the 4-channel Pauli-Z feature maps
│
├── step3_mlops_automation/        # Step 3: Novelty & Pipeline Automation
│   ├── qaoa_optimizer.py          # QUBO formulated hyperparameter tuning
│   └── qsvm_drift_monitor.py      # Real-time out-of-distribution audio flagging
│
└── data_samples/                  # Sample data formats (Full dataset hosted externally)
    ├── sample_audio.wav           
    └── sample_tensor.npy


## Installation and Setup
