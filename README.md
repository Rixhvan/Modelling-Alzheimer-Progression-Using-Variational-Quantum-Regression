# 🧠 Modeling Alzheimer's Progression using Variational Quantum Regression (VQR)

This project explores the use of **quantum machine learning** to predict Alzheimer's disease progression using a **Variational Quantum Regression (VQR)** model.

---

## 🚀 Overview

The goal of this project is to:
- Predict Alzheimer's severity (CDR score)
- Use **classical preprocessing + quantum modeling**
- Evaluate performance on:
  - Local quantum simulator
  - IBM Quantum hardware

---

## ⚙️ Workflow

1. **Data Preprocessing**
   - Clinical dataset (numerical features)
   - Feature scaling using MinMaxScaler
   - Dimensionality reduction using PCA (→ 4 features)

2. **Quantum Model (VQR)**
   - Angle encoding using RY rotations
   - Variational circuit with entanglement
   - Measurement using Pauli-Z expectation

3. **Training**
   - Optimized using Adam optimizer
   - Loss: Mean Squared Error

4. **Testing**
   - Local simulation
   - Real IBM Quantum hardware execution

---

## ⚛️ Why Quantum?

- Explore feasibility of quantum models for healthcare prediction
- Study performance under real hardware noise
- Compare with classical regression

---

## 📊 Results

- Classical model outperforms quantum model
- IBM hardware introduces noise but shows realistic behavior
- Demonstrates limitations of current NISQ devices

---

## 🧠 Key Concepts Used

- Principal Component Analysis (PCA)
- Variational Quantum Circuits
- Quantum Measurement (Pauli-Z)
- Adam Optimization
- Hybrid Quantum-Classical Learning

---

## 🛠️ Tech Stack

- Python
- PennyLane
- Qiskit (IBM Quantum)
- Scikit-learn
- NumPy, Matplotlib

---

## ⚠️ Limitations

- Small dataset
- Limited qubits (4)
- Noise in real quantum hardware
- Shallow circuit depth

---

## 📌 Conclusion

This project demonstrates a hybrid approach combining classical and quantum techniques, highlighting both the potential and current limitations of quantum machine learning.

---

## 👤 Author

Rishvan Kumar. R
