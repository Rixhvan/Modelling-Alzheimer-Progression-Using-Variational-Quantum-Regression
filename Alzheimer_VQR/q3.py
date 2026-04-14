import pandas as pd
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from qiskit_ibm_runtime import QiskitRuntimeService

#ibm connection account
QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="y9YtKmsBzH2qtNQDj0QrpUI6VikX4toxUjHU53lBXB4o",
    overwrite=True
)

service = QiskitRuntimeService()

train_df = pd.read_csv("train_data.csv")
test_df  = pd.read_csv("test_data.csv")

X_train = train_df.drop("CDR", axis=1).values
y_train = train_df["CDR"].values

X_test = test_df.drop("CDR", axis=1).values
y_test = test_df["CDR"].values

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

pca = PCA(n_components=4)
X_train = pca.fit_transform(X_train)
X_test  = pca.transform(X_test)

X_train = X_train * np.pi #[0,1] -> [0,pi]
X_test  = X_test  * np.pi

n_qubits = X_train.shape[1]
print("Number of qubits:", n_qubits)

y_min, y_max = y_train.min(), y_train.max()

def scale_output(raw): 
    return (raw + 1) / 2 * (y_max - y_min) + y_min

def unscale_output(val):
    return 2 * (val - y_min) / (y_max - y_min) - 1

y_train_scaled = unscale_output(y_train)
y_test_scaled  = unscale_output(y_test)

#local stimulator
dev = qml.device("default.qubit", wires=n_qubits)
weights = pnp.random.randn(1, n_qubits, requires_grad=True)

#circuit
@qml.qnode(dev)
def circuit(x, weights):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

def loss(weights, X, y_scaled):
    preds = pnp.array([circuit(X[i], weights) for i in range(len(X))])
    return pnp.mean((preds - y_scaled) ** 2)

#training
opt    = qml.AdamOptimizer(stepsize=0.05)   
epochs = 100

loss_history = []
print("\nTraining started...")

for epoch in range(epochs):
    weights, current_loss = opt.step_and_cost(
        lambda w: loss(w, X_train, y_train_scaled), weights
    )
    loss_history.append(float(current_loss))
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:>3}/{epochs}  Loss = {current_loss:.6f}")

print("Training completed!")

#classical
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_classical = lr.predict(X_test)

print("\n--- CLASSICAL MODEL ---")
print("MSE:     ", mean_squared_error(y_test, y_pred_classical))
print("MAE:     ", mean_absolute_error(y_test, y_pred_classical))
print("R2 Score:", r2_score(y_test, y_pred_classical))


#ibm set up
print("\nConnecting to IBM Quantum...")

backends = service.backends(simulator=False, operational=True) #hardware devices
backend  = min(backends, key=lambda b: b.status().pending_jobs)
print("\nUsing backend:", backend)

dev_ibm = qml.device(
    "qiskit.remote",
    wires=n_qubits,
    backend=backend,
    shots=1024
)

@qml.qnode(dev_ibm)
def circuit_ibm(x, weights):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return qml.sample(qml.PauliZ(0)) #[-1,+1,-1,-1,+1....1024]

#testing
print("\nTesting on IBM hardware...")

MAX_SAMPLES  = 10
X_test_ibm   = X_test[:MAX_SAMPLES]
y_test_ibm   = y_test[:MAX_SAMPLES]

preds_ibm = []
for i in range(MAX_SAMPLES):
    print(f"  Running sample {i+1}/{MAX_SAMPLES}")
    xi      = pnp.array(X_test_ibm[i])
    samples = circuit_ibm(xi, weights)
    raw     = float(np.mean(samples))   #[-1,1] 
    preds_ibm.append(float(scale_output(raw))) #[0,1]

preds_ibm = np.array(preds_ibm)

print("\n--- IBM QUANTUM MODEL ---")
print("MSE:     ", mean_squared_error(y_test_ibm, preds_ibm))
print("MAE:     ", mean_absolute_error(y_test_ibm, preds_ibm))
print("R2 Score:", r2_score(y_test_ibm, preds_ibm))

#plot
y_true_subset = y_test_ibm
preds_class_subset = y_pred_classical[:MAX_SAMPLES]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Regression Results — Actual vs Predicted CDR", fontsize=14, fontweight="bold")

models = [
    ("Classical (Linear Regression)", preds_class_subset, "#2196F3"),
    ("IBM Quantum Hardware", preds_ibm,"#FF5722"),
]

for ax, (title, preds, color) in zip(axes, models):
    ax.scatter(y_true_subset, preds, color=color, edgecolors="white",
               linewidths=0.6, s=70, alpha=0.9, label="Predictions")

    lims = [
        min(y_true_subset.min(), preds.min()) - 0.05,
        max(y_true_subset.max(), preds.max()) + 0.05,
    ]

    m, b    = np.polyfit(y_true_subset, preds, 1)
    x_line  = np.linspace(lims[0], lims[1], 100)
    ax.plot(x_line, m * x_line + b, color=color, linewidth=1.8,
            linestyle="-", alpha=0.7, label="Best fit")

    mse = mean_squared_error(y_true_subset, preds)
    mae = mean_absolute_error(y_true_subset, preds)
    r2  = r2_score(y_true_subset, preds)
    ax.text(0.05, 0.95,
            f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nR²:  {r2:.4f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=color, alpha=0.85))

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Actual CDR")
    ax.set_ylabel("Predicted CDR")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.show()