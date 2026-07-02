# Interpretable Neuro-Fuzzy CNN for MNIST Image Classification

An advanced Soft Computing project combining standard Convolutional Neural Networks (CNNs) with Fuzzy Logic primitives to improve image classification interpretability and robustness against noise.

## Author
* **Tanmay Tyagi** 

## Project Objective
This project integrates a trainable **Gaussian Membership Function (GMF)** directly into a custom CNN layer pipeline. Fuzzifying crisp pixel intensities into localized activation degrees (0 to 1) allows the network to maintain high classification accuracy even under uncertain or noisy conditions.

## Core Methodology

### 1. Mathematical Framework
Instead of forwarding raw structural pixel inputs directly to the filters, the data passes through a parameterized Gaussian activation envelope:
$$\mu(x) = e^{-\frac{(x - \text{mean})^2}{2\sigma^2}}$$
This transformation bridges classical structural image data with soft approximate modes of reasoning.

### 2. Custom Fuzzy Convolutional Layer
Implemented as a native TensorFlow/Keras layer subclass, this custom module:
* Tracks trainable distribution anchors ($\text{mean}$ and $\sigma$) alongside global spatial filter kernels.
* Fuzzifies input tensors natively on the forward pass before standard 2D convolution maps are produced.
* Evaluates exact system errors via standard backpropagation to update spatial weights and fuzzy parameters simultaneously.

### 3. Model Architecture
* **Input Stage:** $28 \times 28 \times 1$ Normalized Digit Framework.
* **Feature Extraction Block 1:** 32-Filter Fuzzy Conv Layer ($3\times3$ Kernel) $\rightarrow$ Batch Normalization $\rightarrow$ Average Pooling ($2\times2$).
* **Feature Extraction Block 2:** 64-Filter Fuzzy Conv Layer ($3\times3$ Kernel) $\rightarrow$ Batch Normalization $\rightarrow$ Average Pooling ($2\times2$).
* **Classification Pipeline:** Flatten Operator $\rightarrow$ Dense Feature Projection (32 units, ReLU) $\rightarrow$ Regularization Dropout (0.3) $\rightarrow$ Softmax Output (10 Matrix Categories).

## Training Strategy & Evaluation
The model is optimized using Categorical Crossentropy over 10 training epochs with an Adam optimizer and a batch allocation size of 64.

* **Test Accuracy:** ~97.32%
* **Performance Analysis:** The evaluation pipeline maps model predictions against absolute targets using a standard classification performance matrix to isolate edge case digit anomalies.

## Environment Configuration
Install the required dependency stack to run the execution script locally:

```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn
```

## Repository Artifacts
* `nfuzz.py` — Complete script (Data cleaning, Custom Layers, Fitting).
* `CSE-458 Project.pdf` — Explanatory project deck detailing theoretical fuzzy definitions and training validation charts.

