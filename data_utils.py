# data_utils.py

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt

def setup_condition_encoding(df, cond_cols):
    dummies = pd.get_dummies(df[cond_cols], columns=cond_cols)
    onehot_cols = dummies.columns.tolist()
    def encoder(df_):
        encoded = pd.get_dummies(df_[cond_cols], columns=cond_cols)
        for col in onehot_cols:
            if col not in encoded:
                encoded[col] = 0
        return encoded[onehot_cols].astype(np.float32)
    features_input = [f for f in df.columns if f not in cond_cols]
    return encoder, onehot_cols, features_input

def prepare_dataloader(df, cond_cols, features_input, batch_size):
    encode_conditions, _, _ = setup_condition_encoding(df, cond_cols)
    cond_encoded = encode_conditions(df)
    cond_tensor = torch.tensor(cond_encoded.values, dtype=torch.float32)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[features_input].values)
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
    loader = DataLoader(TensorDataset(data_tensor, cond_tensor), batch_size=batch_size, shuffle=True)
    return loader, scaler, encode_conditions

def plot_condition_distribution(df, cond_cols, save_path, xlsx_path):
    joint_freq = df.groupby(cond_cols).size().reset_index(name="count")
    joint_freq["prob"] = joint_freq["count"] / joint_freq["count"].sum()
    joint_freq.to_excel(xlsx_path, index=False)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.histplot(joint_freq["count"], bins=30, kde=True)
    plt.title("Joint Condition Frequency Distribution")
    plt.xlabel("Sample Count per Condition Combination")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return joint_freq

def sample_conditions(joint_freq, n, cond_cols):
    return joint_freq.sample(n, replace=True, weights=joint_freq["prob"]).reset_index(drop=True)[cond_cols]
