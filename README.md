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

# The Demonstration Video can be viewed **[here](https://drive.google.com/file/d/1tbqcg0HRf_b30LUEYCNaSG5H-EBfT90I/view?usp=sharing)**.


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
│   └── classifier.py              # Linear-time encoding quantum kernel SVM
│
├── step2_asr_QCNN/                # Step 2: Continuous Audio & Speech Recognition
│   ├── baseline/                  # (Cloud Keras) 10-Class Sequence Training
│   │   ├── main_qsr.py            # Main training loop (Bi-LSTM / Softmax Attention)
│   │   ├── models.py              # Classical backend architectures
│   │   └── helper_q_tool.py       # Pennylane/Qiskit quantum circuit generator
│   │
│   └── custom_validation/         # (Local PyTorch) Engineering Prototype
│       ├── download_data.py       # Download the required data
│       ├── extract_qfeatures.py   # Extracts features
│       ├── plot_qvision.py        # Generates the 4-channel Pauli-Z feature maps
│       └── train_hybrid.py        # Custom binary classification for local hardware test
│       
└── step3_mlops_automation/        # Step 3: Novelty & Pipeline Automation
       ├── qaoa_hpo.py                # QUBO formulated hyperparameter tuning
       └── qsvm_monitor.py            # Real-time out-of-distribution audio flagging

```

## 🚀 Installation & Setup

It is highly recommended to use a virtual environment (like `venv` or `conda`) to avoid conflicts between PyTorch and TensorFlow dependencies.

```bash
# 1. Clone the repository
git clone https://github.com/Sarthak-Shaurya/QML.git
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

This code will take a long time to run. For a faster version, you can run <b> classifier_gpu.py </b> on a GPU. However, be aware that even that could take much longer than expected because many quantum circuits are being executed.

<b> Step 2: Execute the Hybrid ASR Pipeline</b> <br>
### A. Local Prototype (PyTorch Binary Validation)
To visualize the 4-channel quantum feature maps and run the lightweight local CNN:

```Bash
cd step2_asr_QCNN/custom_validation
python download_data.py     # Downloads the required data
python extract_qfeatures.py # Extracts features
python train_hybrid.py      # Runs the Left vs. Right binary classification
python plot_qvision.py      # Generates 'quantum_vision.png'

```
### B. Running the 10-Class Cloud Benchmark 

This section replicates the core sequence-learning phase (Phase B) of the ICASSP 2021 benchmark. It utilizes a classical Recurrent Neural Network (Bi-LSTM with Softmax Attention) to classify the 4-channel quantum spatial tensors extracted during Phase A.

#### 1. Download the Preprocessed Quantum Tensors
To avoid exceeding GitHub's file size limits, the massive preprocessed quantum dataset (`.npy` files) is hosted externally.

* Download the preprocessed dataset archive from our Google Drive **[here](https://drive.google.com/file/d/1hdu2px3-bTp2C1JJ62irI62o60f99iLy/view?usp=sharing)**.
* Extract the archive and place the `.npy` files directly into the `step2_hybrid_asr/baseline/data_quantum/` directory.

#### 2. Patch the Legacy Code for Modern TensorFlow (2.14+)
The original authors' code was written for an older version of Keras. Before running the training pipeline, you must apply the following hotfixes to update the optimizer arguments and model-saving formats:

```bash
cd step2_asr_QCNN/baseline

# Fix the model saving format from legacy .hdf5 to modern .keras
sed -i 's/\.hdf5/\.keras/g' main_qsr.py

# Fix the deprecated learning rate argument in the optimizer
sed -i 's/lr=/learning_rate=/g' models.py
pip install tqdm # install if not already installed
# Run the model
python main_qsr.py
```


<b> Step 3: MLOps Automation</b>
To run the QAOA hyperparameter optimization solver or test the QSVM drift monitor on sample audio streams:

```Bash
cd step3_mlops_automation
python qaoa_hpo.py
# 1. Create a data directory in step 3
mkdir -p data

# 2. Copy the quantum_features folder from step 2 into step 3's data folder
cp -r ../step2_asr_QCNN/custom_validation/data/quantum_features data/
python qsvm_monitor.py
```

## 📚 References & Acknowledgements
S. Moradi, C. Brandner, et al., "Clinical data classification with noisy intermediate scale quantum computers," Scientific Reports, vol. 12, no. 1851, 2022.

C.-H. H. Yang, J. Qi, et al., "Decentralizing Feature Extraction with Quantum Convolutional Neural Network for Automatic Speech Recognition," arXiv:2010.13309, 2021.

E. Farhi, J. Goldstone, and S. Gutmann, "A Quantum Approximate Optimization Algorithm," arXiv:1411.4028, 2014.

We extend our gratitude to the authors of the original QCNN ASR repository for providing the Keras/Qiskit foundation upon which we built our decoupled pipeline and automation wrappers.
