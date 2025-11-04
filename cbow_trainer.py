import torch
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os # (save_path 관련)
from typing import Dict, List, Any

class Word2VecTrainer(object):
    """
    Word2Vec 학습/검증/테스트 Trainer
    - CrossEntropyLoss 기반 missing word를 예측합니다 (classification)
    - 에폭별 손실/정확도 기록 및 시각화
    - Early Stopping 및 Best Model (.pth) 저장 기능 포함
    """
    def __init__(
        self,
        learning_rate: float = 2e-3,
        weight_decay:  float = 1e-5,
        num_epochs:    int   = 50,
        patience:      int   = 10,  # [FIX] Early Stopping patience
        save_path:     str   = 'best_cbow_model.pth', # [FIX] 모델 저장 경로
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
            'train_acc':   [],
            'valid_loss':  [],
            'valid_acc':   []
        }

    @staticmethod
    def _to_device_batch(pt_batch: Dict[str, Any], device: str) -> Dict[str, Any]:
        """dict(batch)를 device로 이동"""
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in pt_batch.items()}

    @torch.no_grad()
    def _accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """top-1 정확도"""
        preds = logits.argmax(dim=1)
        return (preds == targets).float().mean().item()

    def train_step(self, model: nn.Module,
                   loader: torch.utils.data.DataLoader,
                   optimizer: optim.Optimizer):
        """1 에폭의 학습(train) 스텝"""
        model.train()
        running_loss, running_acc, total = 0.0, 0.0, 0

        for pt_batch in tqdm(loader, desc="   Train", leave=False):
            # 1. 배치 데이터 device 이동
            pt_batch = self._to_device_batch(pt_batch, self.device)

            # 2. Forward & Backward
            optimizer.zero_grad(set_to_none=True)
            out = model(pt_batch)  # 모델 forward (loss, logits 포함)
            loss = out['loss']
            loss.backward()
            optimizer.step()

            # 3. 로깅
            B = pt_batch['X'].size(0)
            running_loss += loss.item() * B
            running_acc  += self._accuracy(out['logits'], pt_batch['y']) * B
            total        += B

        return running_loss / max(1, total), running_acc / max(1, total)

    @torch.no_grad()
    def valid_step(self, model: nn.Module,
                   loader: torch.utils.data.DataLoader):
        """1 에폭의 검증(valid) 또는 테스트(test) 스텝"""
        model.eval()
        running_loss, running_acc, total = 0.0, 0.0, 0

        for pt_batch in tqdm(loader, desc="  Valid/Test", leave=False):
            # 1. 배치 데이터 device 이동
            pt_batch = self._to_device_batch(pt_batch, self.device)
            
            # 2. Forward (no backward)
            out = model(pt_batch)  # 모델 forward (loss, logits 포함)
            
            # 3. 로깅
            B = pt_batch['X'].size(0)
            running_loss += out['loss'].item() * B
            running_acc  += self._accuracy(out['logits'], pt_batch['y']) * B
            total        += B

        return running_loss / max(1, total), running_acc / max(1, total)

    def train(self, model: nn.Module,
              train_loader: torch.utils.data.DataLoader,
              valid_loader: torch.utils.data.DataLoader):
        """
        모델 학습(train) 및 검증(validation)을 수행합니다.
        Early stopping 및 Best model 저장을 포함합니다.
        """
        
        # 1. 모델 및 옵티마이저 초기화
        model.to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # 2. [FIX] Early Stopping 관련 변수 초기화
        best_valid_loss = float('inf')
        epochs_no_improve = 0
        
        print(f"Start training on {self.device}...")
        print(f"Best model will be saved to: {self.save_path}")

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            
            # 3. Train & Valid
            train_loss, train_acc = self.train_step(model, train_loader, optimizer)
            valid_loss, valid_acc = self.valid_step(model, valid_loader)

            print(f"  ▶ Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  ▶ Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f}")

            # 4. 로그 저장
            self.logs['train_loss'].append(train_loss)
            self.logs['train_acc'].append(train_acc)
            self.logs['valid_loss'].append(valid_loss)
            self.logs['valid_acc'].append(valid_acc)

            # 5. [FIX] Early Stopping 및 Best Model 저장 로직
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), self.save_path)
                print(f"  ▶ Valid loss improved. Saving best model to {self.save_path}")
            else:
                epochs_no_improve += 1
                print(f"  ▶ Early stopping counter: {epochs_no_improve}/{self.patience}")

            if epochs_no_improve >= self.patience:
                print(f"Early stopping triggered after {self.patience} epochs without improvement.")
                break
        
        print("\nTraining complete.")
            
    @torch.no_grad()
    def test(self, model: nn.Module,
             test_loader: torch.utils.data.DataLoader):
        """
        학습된 Best Model을 로드하여 테스트(test)를 수행합니다.
        """
        # 1. [FIX] 저장된 Best Model 로드
        try:
            model.load_state_dict(torch.load(self.save_path, map_location=self.device))
            print(f"\nLoaded best model from {self.save_path} for testing.")
        except FileNotFoundError:
            print(f"\n[Warning] Best model file not found at {self.save_path}. Testing with current model state.")
        except Exception as e:
            print(f"\n[Error] Failed to load best model: {e}. Testing with current model state.")
            
        model.to(self.device)
        
        # 2. Test
        test_loss, test_acc = self.valid_step(model, test_loader)
        print(f"\nTest Result ▶ Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")
        return test_loss, test_acc

    def plot_logs(self):
        """학습/검증 Loss/Accuracy를 시각화합니다."""
        
        # [FIX] Early Stopping으로 중단된 에폭까지만 플로팅
        num_epochs_run = len(self.logs['train_loss'])
        if num_epochs_run == 0:
            print("No logs to plot. Run train() first.")
            return
            
        epochs = range(1, num_epochs_run + 1)
        
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.logs['train_loss'], label='Train Loss')
        plt.plot(epochs, self.logs['valid_loss'], label='Valid Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curve'); plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.logs['train_acc'], label='Train Acc')
        plt.plot(epochs, self.logs['valid_acc'], label='Valid Acc')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy Curve'); plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()