import random

# -----------------------------
# Training data (BIG DATA idea)
# x = count of word "good"
# y = human given rating (with Gaussian noise)
# -----------------------------

data = [
    (1, 2.2),
    (2, 4.1),
    (3, 6.0),
    (4, 7.9),
    (5, 10.2)
]

# -----------------------------
# Step 1: Initialize weight θ
# -----------------------------
theta = 0.0   # starting guess
alpha = 0.01  # learning rate (step size)

# -----------------------------
# Step 2: Training loop
# -----------------------------
for epoch in range(10):  # repeat many times
    total_loss = 0

    for x, y in data:
        # -----------------------------
        # Prediction
        # y_hat = θ * x
        # -----------------------------
        y_hat = theta * x

        # -----------------------------
        # Error (actual - predicted)
        # -----------------------------
        error = y - y_hat

        # -----------------------------
        # Squared loss (Gaussian loss)
        # -----------------------------
        loss = error ** 2
        total_loss += loss

        # -----------------------------
        # Gradient of loss w.r.t θ
        # dL/dθ = -2 * x * error
        # -----------------------------
        gradient = -2 * x * error

        # -----------------------------
        # Gradient Descent update
        # θ = θ - α * gradient
        # -----------------------------
        theta = theta - alpha * gradient

    print(f"Epoch {epoch+1}: θ = {theta:.4f}, Loss = {total_loss:.4f}")
