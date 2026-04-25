import pickle
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
from tqdm import tqdm
import warnings
import qiskit
# from qiskit.providers.aer.noise.device import basic_device_noise_model

# Define the number of qubits (wires) for our quantum circuit. 
# We use 4 wires to process 2x2 image patches (4 pixels = 4 input parameters).
n_w = 4 # numbers of wires def 4
noise_mode = False # for running at QPU

# Initialize the quantum device. If testing on a real QPU or simulating realistic 
# quantum errors, we use Qiskit's Aer simulator with a noise model. 
# Otherwise, we default to PennyLane's standard state vector simulator for speed.
if  noise_mode == True:
    dev = qml.device('qiskit.aer', wires= n_w, noise_model=noise_model)
else:
    dev = qml.device("default.qubit", wires= n_w)

n_layers = 1

# Random circuit parameters
# Initialize random weights for the parameterized quantum circuit (PQC).
# These weights are uniformly distributed between 0 and 2pi (full rotation).
rand_params = np.random.uniform(high= 2 * np.pi, size=(n_layers, n_w)) # def 2, n_w = 4

@qml.qnode(dev)
def circuit(phi=None):
    # Encoding of 4 classical input values
    for j in range(n_w):
        # Encode the classical input data (pixel values) into quantum states.
        # We use RY (Rotation Y) gates to embed the classical data as rotation angles.
        qml.RY(np.pi * phi[j], wires=j)

    # Random quantum circuit
    # Apply a random circuit layer. In a full Quantum Neural Network (QNN), 
    # these parameters would be updated via gradient descent. Here, it acts 
    # as a fixed quantum feature extractor.
    RandomLayers(rand_params, wires=list(range(n_w)))

    # Measurement producing 4 classical output values
    # Measure the expectation value of the Pauli-Z observable for each qubit.
    # This collapses the quantum state and returns 4 classical values.
    return [qml.expval(qml.PauliZ(j)) for j in range(n_w)]

def quanv(image, kr=2):
    h_feat, w_feat, ch_n = image.shape
    """Convolves the input speech with many applications of the same quantum circuit."""
    # Initialize the output feature map. The spatial dimensions are halved 
    # because we are using a 2x2 kernel with a stride of 2 (non-overlapping).
    # The output channels equal the number of qubits (n_w).
    out = np.zeros((h_feat//kr, w_feat//kr, n_w))

    # Loop over the coordinates of the top-left pixel of 2X2 squares
    for j in range(0, h_feat, kr):
        for k in range(0, w_feat, kr):
            # Process a squared 2x2 region of the image with a quantum circuit
            # Extract a 2x2 patch from the input spectrogram.
            # Flatten the patch into a 1D array of 4 elements to feed into the circuit.
            q_results = circuit(
                # kernal 3 ## phi=[image[j, k, 0], image[j, k + 1, 0], image[j, k + 2, 0], image[j + 1, k, 0], 
                # image[j + 1, k + 1, 0], image[j + 1, k +2 , 0],image[j+2, k, 0], image[j+2, k+1, 0], image[j+2, k+2, 0]]
                phi=[image[j, k, 0], image[j, k + 1, 0], image[j + 1, k, 0], image[j + 1, k + 1, 0]]
            )
            # Assign expectation values to different channels of the output pixel (j/2, k/2)
            # Map the 4 quantum expectation values back into the 4 channels 
            # of the resulting output pixel.
            for c in range(n_w):
                out[j // kr, k // kr, c] = q_results[c]
    return out

def gen_qspeech(x_train, x_valid, kr): # kernal size = 2x2 or 3x3
    # Iterate through the entire training and validation datasets, 
    # applying the quantum convolution layer to every spectrogram.
    # This acts as an initial quantum feature extraction step before classical ML.
    q_train = []
    print("Quantum pre-processing of train Speech:")
    for idx, img in enumerate(x_train):
        print("{}/{}        ".format(idx + 1, len(x_train)), end="\r")
        q_train.append(quanv(img, kr))
    q_train = np.asarray(q_train)

    q_valid = []
    print("\nQuantum pre-processing of test Speech:")
    for idx, img in enumerate(x_valid):
        print("{}/{}        ".format(idx + 1, len(x_valid)), end="\r")
        q_valid.append(quanv(img, kr))
    q_valid = np.asarray(q_valid)
    
    return q_train, q_valid

import matplotlib.pyplot as plt
import librosa
import librosa.display

def plot_acc_loss(q_history, x_history, v_history, data_ix):
    # Plotting utility to compare three different model architectures:
    # 1. Baseline Attention Bi-LSTM
    # 2. Hybrid model with a Quantum Convolutional (Quanv) Layer
    # 3. Classical model with a standard Convolutional Layer

    plt.figure()
    plt.style.use("seaborn")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))

    ax1.plot(v_history.history["val_accuracy"], "-ok", label="Baseline Attn-BiLSTM")
    ax1.plot(q_history.history["val_accuracy"], "-ob", label="With Quanv Layer")
    ax1.plot(x_history.history["val_accuracy"], "-og", label="With Conv Layer")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim([0, 1])
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(v_history.history["val_loss"], "-ok", label="Baseline Attn-BiLSTM")
    ax2.plot(q_history.history["val_loss"], "-ob", label="With Quanv Layer")
    ax2.plot(x_history.history["val_loss"], "-og", label="With Conv Layer")
    ax2.set_ylabel("Loss")
    #ax2.set_ylim(top=5.5)
    ax2.set_xlabel("Epoch")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("images/"+ data_ix +"_conv_speech_loss.png")

def show_speech(x_train, q_train, use_ch, tmp = "tmp.png"):
    # Converts the power spectrogram (amplitude squared) to decibel (dB) units
    # for better visualization of the human hearing range.
    plt.figure()
    plt.subplot(5, 1, 1)
    if use_ch != True:
        librosa.display.specshow(librosa.power_to_db(x_train[0,:,:,0], ref=np.max))
    else:
        librosa.display.specshow(librosa.power_to_db(x_train[0,:,:], ref=np.max))
    plt.title('Input Speech')
    # Plot the 4 separate channels generated by the quantum circuit measurements.
    # This visualizes what features the quantum layer actually extracted.
    for i in range(4):
        plt.subplot(5, 1, i+2)
        librosa.display.specshow(librosa.power_to_db(q_train[0,:,:,i], ref=np.max))
        plt.title('Channel '+str(i+1)+': Quantum Compressed Speech')


    plt.tight_layout()
    plt.savefig("images/speech_encoder_" + tmp)

