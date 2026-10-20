"""models/models.py — CNN-LSTM 与 CNN-GRU（唯一结构差异在时序单元）与训练循环。

CNNLSTM、_WindowDataset、train_torch、predict_torch 迁移自
dachuang_cnn_lstm/src/models/cnn_lstm.py，除 train_torch 额外返回
history/best_epoch（计划书步骤1允许的最小接口扩展）外逐行一致。
CNNGRU 仅把 nn.LSTM 换成 nn.GRU，CNN 前端、64 维输出与回归头完全相同；
参数量差异属被研究的结构差异，不通过调整隐藏宽度抵消。
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class CNNLSTM(nn.Module):
    def __init__(self, n_vars: int, window: int = 60,
                 conv_channels=(64, 32), lstm_hidden: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        in_ch = n_vars
        for out_ch in conv_channels:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool1d(2),
            ])
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.lstm = nn.LSTM(in_ch, lstm_hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(lstm_hidden, 32), nn.ReLU(),
                                  nn.Linear(32, 1))
        self.n_vars = n_vars

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, window, n_vars] → Conv1d 需要 [B, n_vars, window]
        z = self.conv(x.permute(0, 2, 1))
        out, _ = self.lstm(z.permute(0, 2, 1))
        return self.head(out[:, -1, :]).squeeze(-1)


class CNNGRU(nn.Module):
    def __init__(self, n_vars: int, window: int = 60,
                 conv_channels=(64, 32), gru_hidden: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        in_ch = n_vars
        for out_ch in conv_channels:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool1d(2),
            ])
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.gru = nn.GRU(in_ch, gru_hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(gru_hidden, 32), nn.ReLU(),
                                  nn.Linear(32, 1))
        self.n_vars = n_vars

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x.permute(0, 2, 1))
        out, _ = self.gru(z.permute(0, 2, 1))
        return self.head(out[:, -1, :]).squeeze(-1)


class _WindowDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def train_torch(model: nn.Module, X_tr: np.ndarray, y_tr: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray, epochs: int = 200,
                batch_size: int = 64, lr: float = 1e-3, patience: int = 10,
                seed: int = 42, verbose: bool = False,
                loss: str = "mse", scheduler: str | None = None) -> dict:
    """通用训练循环（CNN-LSTM 与 CNN-GRU 共用，与旧仓一致）。

    loss: "mse" | "huber" | "smooth_l1"
    scheduler: None | "cosine"
    返回额外含 history（每 epoch 验证损失）与 best_epoch（1 基）。
    """
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if loss == "huber":
        loss_fn = nn.HuberLoss(delta=1.0)
    elif loss == "smooth_l1":
        loss_fn = nn.SmoothL1Loss()
    else:
        loss_fn = nn.MSELoss()
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1)) if scheduler == "cosine" else None
    tr_loader = torch.utils.data.DataLoader(_WindowDataset(X_tr, y_tr),
                                            batch_size=batch_size, shuffle=True)
    best_val, best_state, wait, best_epoch = np.inf, None, 0, -1
    history = []
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        if sch is not None:
            sch.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(torch.tensor(X_val, dtype=torch.float32,
                                                      device=device)),
                                     torch.tensor(y_val, dtype=torch.float32,
                                                  device=device)))
        history.append(val_loss)
        if val_loss < best_val - 1e-6:
            best_val, wait, best_epoch = val_loss, 0, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
        if verbose and ep % 10 == 0:
            print(f"  epoch {ep}: val_loss={val_loss:.5f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu")
    return {"best_val_loss": best_val, "epochs_run": len(history),
            "history": history, "best_epoch": best_epoch + 1}


def predict_torch(model: nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs)
