import pandas as pd
import numpy as np
import os
import pickle

from model.logistic_regression import LogisticRegressionFromScratch

# paths
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "health_model.pkl")

# load data
df = pd.read_csv(DATA_PATH)

X = df.drop("healthy", axis=1).values
y = df["healthy"].values

# normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# train / test split
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# init model (FROM SCRATCH)
model = LogisticRegressionFromScratch(
    lr=0.01,        # ✅ الاسم الصح
    epochs=1500
)

# train
model.fit(X_train, y_train)

# evaluate
preds = model.predict(X_test)
accuracy = (preds == y_test).mean()
print(f"Test Accuracy: {accuracy:.4f}")

# save model
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("Model saved to:", MODEL_PATH)

import matplotlib.pyplot as plt

# plot loss curve
plt.figure(figsize=(8, 5))
plt.plot(model.losses)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

# save plot
PLOT_PATH = os.path.join(BASE_DIR, "..", "model", "loss_curve.png")
plt.savefig(PLOT_PATH)
plt.close()

print("Loss curve saved to:", PLOT_PATH)

