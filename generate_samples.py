# generate_samples.py

import torch
import numpy as np
import pandas as pd
from config import *
from model import ResDenoiseNet
from data_utils import setup_condition_encoding, sample_conditions
from sklearn.preprocessing import MinMaxScaler

# Load data and condition encoding
df = pd.read_excel(FILE_PATH)
df["PPV"] = np.log1p(df["PPV"])
encode_conditions, _, FEATURES_INPUT = setup_condition_encoding(df, COND_COLS)
joint_freq = pd.read_excel(COND_DIST_FILE)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained model
model = ResDenoiseNet(len(FEATURES_INPUT), len(encode_conditions(df).columns), hidden=128).to(device)
model.load_state_dict(torch.load("crtdiff_model.pth", map_location=device))
model.eval()

# DDPM noise schedule
T = 1000
beta = torch.linspace(1e-4, 0.02, T).to(device)
alpha = 1 - beta
alpha_hat = torch.cumprod(alpha, dim=0)

# Fit scaler on original data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[FEATURES_INPUT].values)

@torch.no_grad()
def generate_with_sampled_conditions(n_samples):
    cond_values_df = sample_conditions(joint_freq, n_samples, COND_COLS)
    cond_encoded = torch.tensor(encode_conditions(cond_values_df).values, dtype=torch.float32).to(device)
    x = torch.randn(n_samples, len(FEATURES_INPUT)).to(device)
    for t in reversed(range(T)):
        t_tensor = torch.full((n_samples,), t, device=device)
        x_cat = torch.cat([x, cond_encoded], dim=1)
        pred_noise = model(x_cat, t_tensor)
        alpha_t = alpha[t]
        alpha_hat_t = alpha_hat[t]
        beta_t = beta[t]
        z = torch.randn_like(x) if t > 0 else 0
        x = (1 / torch.sqrt(alpha_t)) * (
            x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * pred_noise
        ) + torch.sqrt(beta_t) * z
    samples = scaler.inverse_transform(x.cpu().numpy())
    df_gen = pd.DataFrame(samples, columns=FEATURES_INPUT)
    for col in COND_COLS:
        df_gen[col] = cond_values_df[col].values
    if "PPV" in df_gen.columns:
        df_gen["PPV"] = np.expm1(df_gen["PPV"])
    return df_gen

# Generate and save data
n_gen = int(len(df) * GEN_MULTIPLIER)
df_gen = generate_with_sampled_conditions(n_gen)
desired_order = [f"X{i}" for i in range(1, 12)] + ["PPV"]
df_gen = df_gen[desired_order]
df_gen.to_excel(SAVE_FILE, index=False)
print(f"✅ 已生成 {n_gen} 条样本，并保存至 {SAVE_FILE}")
