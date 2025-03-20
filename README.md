# Simple Neural Network for MNIST Classification

A from-scratch implementation of a 2-layer neural network using NumPy for handwritten digit recognition on the MNIST dataset. This project demonstrates fundamental deep learning concepts and achieves **~95% accuracy** on the test set.

## Key Features
- Pure NumPy implementation (no deep learning frameworks)
- Two-layer neural network architecture
- ReLU activation in hidden layer
- Softmax output layer
- Cross-entropy loss function
- Stochastic Gradient Descent (SGD) with backpropagation
- Vectorized operations for efficient computation

## Neural Network Architecture

### Forward Pass Equations

**Layer 1 (Input → Hidden):**
```math
\[
\begin{aligned}
\mathbf{l}_1 &= \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1 \\
\mathbf{y}_1 &= \text{ReLU}(\mathbf{l}_1)
\end{aligned}
\]
```

**Layer 2 (Hidden → Output):**
\[
\begin{aligned}
\mathbf{l}_2 &= \mathbf{W}_2 \mathbf{y}_1 + \mathbf{b}_2 \\
\mathbf{y}_2 &= \text{softmax}(\mathbf{l}_2)
\end{aligned}
\]

### Backward Pass Equations

**Output Layer Gradients:**
\[
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_2} &= \frac{1}{m} (\mathbf{y}_2 - \mathbf{y}_{\text{true}}) \mathbf{y}_1^\top \\
\frac{\partial \mathcal{L}}{\partial \mathbf{b}_2} &= \frac{1}{m} \sum (\mathbf{y}_2 - \mathbf{y}_{\text{true}})
\end{aligned}
\]

**Hidden Layer Gradients:**
\[
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} &= \frac{1}{m} \left(\mathbf{W}_2^\top (\mathbf{y}_2 - \mathbf{y}_{\text{true}}) \odot \text{ReLU}'(\mathbf{l}_1)\right) \mathbf{x}^\top \\
\frac{\partial \mathcal{L}}{\partial \mathbf{b}_1} &= \frac{1}{m} \sum \left(\mathbf{W}_2^\top (\mathbf{y}_2 - \mathbf{y}_{\text{true}}) \odot \text{ReLU}'(\mathbf{l}_1)\right)
\end{aligned}
\]

**ReLU Derivative:**
\[
\text{ReLU}'(x) = \begin{cases} 
1 & \text{if } x > 0 \\
0 & \text{otherwise}
\end{cases}
\]

## Installation

1. Clone repository:
```bash
git clone https://github.com/csanri/SimpleNN
cd SimpleNN
```
2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Run the script:

```bash
python main.py
```
