# model_trainer.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

class DualInputANN(nn.Module):
    def __init__(self, num_numeric_features, num_text_features, hidden_dim=32):
        super().__init__()
        self.num_net = nn.Sequential(
            nn.Linear(num_numeric_features, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU()
        )
        self.text_net = nn.Sequential(
            nn.Linear(num_text_features, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dim, 3)

    def forward(self, numeric_x, text_x):
        num_feat = self.num_net(numeric_x)
        text_feat = self.text_net(text_x)
        combined = torch.cat([num_feat, text_feat], dim=1)
        return self.classifier(combined)

def run_training():
    label_map = {"Bullish": 0, "Neutral": 1, "Bearish": 2}
    df = pd.read_csv("nasdaq_labeled_news.csv", parse_dates=["date"])

    df = df.dropna(subset=['price_dir'])  # drop rows where label missing
    df["price_dir_int"] = df["price_dir"].map(label_map)
    df["text_dir_int"]  = df["text_dir"].map(label_map)

    numeric_cols = ["pct_ret", "prev_close", "next_close", "txt_len", "has_excl"]
    if "ATR_14" in df.columns:
        numeric_cols.append("ATR_14")

    text_cols    = ["sentiment_fingpt", "text_conf"]

    df_ann = df[df["has_ques"] == 0].dropna(subset=numeric_cols + text_cols + ["price_dir_int"])

    # Sort by date to ensure temporal order
    df_ann = df_ann.sort_values("date").reset_index(drop=True)

    split_idx = int(0.8 * len(df_ann))

    # --- Add train/test split column ---
    df_ann["set_type"] = ["train"] * split_idx + ["test"] * (len(df_ann) - split_idx)

    # Save new CSV with 'set_type' column
    df_ann.to_csv("nasdaq_labeled_news_with_split.csv", index=False)
    print("✅ Saved CSV with train/test split column 'set_type' to psei_labeled_news_with_split.csv")

    # Fit scaler only on training data to prevent leakage
    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(df_ann.loc[:split_idx-1, numeric_cols].values.astype(np.float32))
    X_num_test = scaler.transform(df_ann.loc[split_idx:, numeric_cols].values.astype(np.float32))

    X_txt_train = df_ann.loc[:split_idx-1, text_cols].values.astype(np.float32)
    X_txt_test = df_ann.loc[split_idx:, text_cols].values.astype(np.float32)

    y_train = df_ann.loc[:split_idx-1, "price_dir_int"].values.astype(np.int64)
    y_test = df_ann.loc[split_idx:, "price_dir_int"].values.astype(np.int64)

    # Convert to tensors
    Xn_train, Xt_train, y_train = map(torch.tensor, (X_num_train, X_txt_train, y_train))
    Xn_test, Xt_test, y_test = map(torch.tensor, (X_num_test, X_txt_test, y_test))

    model = DualInputANN(num_numeric_features=Xn_train.shape[1], num_text_features=Xt_train.shape[1])
    weights = torch.tensor([1.0 / Counter(y_train.numpy())[i] for i in range(3)], dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimzr = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(30):
        pred = model(Xn_train, Xt_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimzr.step()
        optimzr.zero_grad()
        if (epoch+1)%5 == 0:
            print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(Xn_test, Xt_test), dim=1).numpy()

    print("\n🧠 ANN Classification Report")
    print(classification_report(y_test, preds, target_names=["Bullish","Neutral","Bearish"], zero_division=0))

    print("\n📊 Confusion Matrix — Fingpt")
    print(pd.crosstab(df['text_dir'], df['price_dir'], margins=True))

    print("\n📊 Confusion Matrix — ANN")
    print(pd.crosstab(
        pd.Series(preds).map({0:"Bullish",1:"Neutral",2:"Bearish"}),
        pd.Series(y_test).map({0:"Bullish",1:"Neutral",2:"Bearish"}),
        margins=True
    ))

if __name__ == "__main__":
    run_training()
