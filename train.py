import os
import torch
import numpy as np
from torch import nn
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from fsrcnn import FSRCNN
from data.dataset import TrainDIV2K, EvalDIV2K
from data.helpers import calc_psnr


if __name__ == '__main__':
    SCALE = 2
    EPOCHS = 10
    BATCH_SIZE = 1
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    augmented_train = Path('data/augmented/train')
    augmented_valid = Path('data/augmented/valid')

    train_data = TrainDIV2K(dir_path=augmented_train, scale=SCALE)
    valid_data = EvalDIV2K(dir_path=augmented_valid, scale=SCALE)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

    model = FSRCNN(scale=SCALE).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    results = {'train_loss': [], 'val_loss': []}

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            loss = loss_fn(pred, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)
        results['train_loss'].append(train_loss)

        model.eval()
        val_loss, psnr_scores = 0, []
        with torch.inference_mode():
            for X, y in valid_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                pred = model(X)
                val_loss += loss_fn(pred, y).item()
                psnr_scores.append(calc_psnr(pred.cpu().numpy(), y.cpu().numpy()))
        val_loss /= len(valid_loader)
        results['val_loss'].append(val_loss)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | PSNR: {np.mean(psnr_scores):.2f} dB")

    models_dir = Path('app/api/models')
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / f'FSRCNN_{SCALE}x_{EPOCHS}e.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")
