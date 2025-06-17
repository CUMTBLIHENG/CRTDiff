# train_crtdiff.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import *
from model import ResDenoiseNet
from data_utils import setup_condition_encoding, prepare_dataloader, plot_condition_distribution

# Load and preprocess data
df = pd.read_excel(FILE_PATH)
df["PPV"] = np.log1p(df["PPV"])
encode_conditions, all_onehot_cols, FEATURES_INPUT = setup_condition_encoding(df, COND_COLS)
loader, scaler, encode_conditions = prepare_dataloader(df, COND_COLS, FEATURES_INPUT, BATCH_SIZE)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Plot condition frequency distribution
joint_freq = plot_condition_distribution(df, COND_COLS, COND_DIST_PLOT, COND_DIST_FILE)

# DDPM parameters
T = 1000
beta = torch.linspace(1e-4, 0.02, T).to(device)
alpha = 1 - beta
alpha_hat = torch.cumprod(alpha, dim=0)

# Model
model = ResDenoiseNet(len(FEATURES_INPUT), len(all_onehot_cols), hidden=128).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
loss_fn = nn.SmoothL1Loss()

# Add noise
def add_noise(x0, t):
    noise = torch.randn_like(x0)
    alpha_hat_t = alpha_hat[t].unsqueeze(1)
    xt = torch.sqrt(alpha_hat_t) * x0 + torch.sqrt(1 - alpha_hat_t) * noise
    return xt, noise

# Training loop
loss_history = []
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    model.train()
    for x0, cond in loader:
        x0, cond = x0.to(device), cond.to(device)
        t = torch.randint(0, T, (x0.size(0),), device=device)
        xt, noise = add_noise(x0, t)
        x_cat = torch.cat([xt, cond], dim=1)
        pred = model(x_cat, t)
        loss = loss_fn(pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    avg_loss = epoch_loss / len(loader)
    loss_history.append(avg_loss)
    if epoch % 100 == 0 or epoch == EPOCHS - 1:
        print(f"[Epoch {epoch}/{EPOCHS}] Loss: {avg_loss:.6f}")

# Save loss curve
plt.figure(figsize=(6, 3))
plt.plot(loss_history, label="Training Loss", color="tab:blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("训练损失曲线.png")
plt.show()

# Save model
torch.save(model.state_dict(), "crtdiff_model.pth")
