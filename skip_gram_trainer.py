import torch
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Any

# ===================================================
# 5) SkipGram Trainer
# ===================================================
class SkipGramTrainer(object):
    """
    Skip-Gram (Negative Sampling) 모델의 학습/검증/테스트 Trainer
    - BCEWithLogitsLoss 기반으로 학습 (정확도 로깅 없음)
    - 에폭별 손실 기록 및 시각화
    - Early Stopping 및 Best Model (.pth) 저장 기능 포함
    """
    def __init__(
        self,
        learning_rate: float = 2e-3,
        weight_decay:  float = 1e-5,
        num_epochs:    int   = 50,
        patience:      int   = 5, # Early Stopping patience
        save_path:     str   = 'best_skipgram_model.pth', # 모델 저장 경로
        device:        str   = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.learning_rate = learning_rate
        self.weight_decay  = weight_decay
        self.num_epochs    = num_epochs
        self.patience      = patience
        self.save_path     = save_path
        self.device        = device

        self.logs = {
            'train_loss':  [],
            'valid_loss':  [],
        }
        
        dir_name = os.path.dirname(save_path)
        if dir_name: # dir_name이 빈 문자열('')이 아닐 때만 실행
            os.makedirs(dir_name, exist_ok=True)

    @staticmethod
    def _to_device_batch(pt_batch: Dict[str, Any], device: str) -> Dict[str, Any]:
        """dict(batch)를 device로 이동"""
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in pt_batch.items()}

    def train_step(self, model: nn.Module,
                   loader: torch.utils.data.DataLoader,
                   optimizer: optim.Optimizer):
        """1 에폭의 학습(train) 스텝"""
        model.train()
        running_loss, total = 0.0, 0

        for pt_batch in tqdm(loader, desc="   Train", leave=False):
            pt_batch = self._to_device_batch(pt_batch, self.device)

            optimizer.zero_grad(set_to_none=True)
            out = model(pt_batch)
            loss = out['loss']
            loss.backward()
            optimizer.step()

            B = pt_batch['center'].size(0) # SkipGram은 'center' 기준
            running_loss += loss.item() * B
            total        += B

        return running_loss / max(1, total)

    @torch.no_grad()
    def valid_step(self, model: nn.Module,
                   loader: torch.utils.data.DataLoader):
        """1 에폭의 검증(valid) 또는 테스트(test) 스텝"""
        model.eval()
        running_loss, total = 0.0, 0

        for pt_batch in tqdm(loader, desc="   Valid/Test", leave=False):
            pt_batch = self._to_device_batch(pt_batch, self.device)
            out = model(pt_batch)
            
            B = pt_batch['center'].size(0) # SkipGram은 'center' 기준
            running_loss += out['loss'].item() * B
            total        += B

        return running_loss / max(1, total)

    def train(self, model: nn.Module,
              train_loader: torch.utils.data.DataLoader,
              valid_loader: torch.utils.data.DataLoader):
        """
        모델 학습(train) 및 검증(validation)을 수행합니다.
        Early stopping 및 Best model 저장을 포함합니다.
        """
        model.to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        best_valid_loss = float('inf')
        epochs_no_improve = 0
        
        print(f"\nStart training on {self.device}...")
        print(f"Best model will be saved to: {self.save_path}")

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            
            train_loss = self.train_step(model, train_loader, optimizer)
            valid_loss = self.valid_step(model, valid_loader)

            print(f"   ▶ Train Loss: {train_loss:.4f}")
            print(f"   ▶ Valid Loss: {valid_loss:.4f}")

            self.logs['train_loss'].append(train_loss)
            self.logs['valid_loss'].append(valid_loss)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), self.save_path)
                print(f"New best model saved to {self.save_path} (Valid Loss: {valid_loss:.4f})")
            else:
                epochs_no_improve += 1
                print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= self.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break
        
        print("\nTraining complete.")
            
    @torch.no_grad()
    def test(self, model: nn.Module,
             test_loader: torch.utils.data.DataLoader):
        """
        학습된 Best Model을 로드하여 테스트(test)를 수행합니다.
        """
        try:
            model.load_state_dict(torch.load(self.save_path, map_location=self.device))
            print(f"\nLoaded best model from {self.save_path} for testing.")
        except FileNotFoundError:
            print(f"\n[Warning] Best model file not found at {self.save_path}. Testing with current model state.")
        except Exception as e:
            print(f"\n[Error] Failed to load best model: {e}. Testing with current model state.")
            
        model.to(self.device)
        
        test_loss = self.valid_step(model, test_loader)
        print(f"\nTest Result ▶ Loss: {test_loss:.4f}")
        return test_loss

    def plot_logs(self):
        """학습/검증 Loss를 시각화합니다."""
        
        num_epochs_run = len(self.logs['train_loss'])
        if num_epochs_run == 0:
            print("No logs to plot. Run train() first.")
            return
            
        epochs = range(1, num_epochs_run + 1)
        
        plt.figure(figsize=(8, 5)) # 1개 플롯이므로 (CBOW와 달리) 너비 축소

        plt.plot(epochs, self.logs['train_loss'], label='Train Loss')
        plt.plot(epochs, self.logs['valid_loss'], label='Valid Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curve'); plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()