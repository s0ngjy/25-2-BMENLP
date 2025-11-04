import torch
import torch.nn as nn
from typing import Dict

# =========================
# 5) CBOW 모델 (EmbeddingBag 사용)
# =========================
class CBOW(nn.Module):
    """
    CBOW 모델 (nn.EmbeddingBag 사용)
    EmbeddingBag은 여러 인덱스의 임베딩을 'mean' 또는 'sum'으로 
    효율적으로 합쳐주므로 CBOW 모델에 적합합니다.
    """
    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int = 128,
        padding_idx: int = 0,
        dropout_p:   float = 0.2,
        ):
        """
        Args:
            vocab_size (int): 어휘 사전의 크기
            embed_dim (int): 임베딩 차원
            padding_idx (int): 패딩 토큰의 인덱스 (기본값 0: '<pad>')
            dropout_p (float): 드롭아웃 확률
        """
        super().__init__()
        # EmbeddingBag: 여러 인덱스 임베딩을 평균/합산 → CBOW 평균에 적합
        # mode='mean': 문맥(context) 단어 임베딩의 평균을 계산합니다.
        self.emb = nn.EmbeddingBag(
            vocab_size, 
            embed_dim, 
            mode='mean', 
            padding_idx=padding_idx
        )
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(embed_dim, vocab_size)  # (B, embed_dim) -> (B, vocab_size)
        
        # Loss 함수는 모델 외부에 정의할 수도 있지만, 
        # 이렇게 내부에 포함하면 forward에서 loss까지 계산하기 편리합니다.
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pt_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            pt_batch (Dict): 'X' (context 텐서)와 'y' (target 텐서)를 포함한 딕셔너리
        
        Returns:
            Dict: 'logits'와 'loss'가 추가된 pt_batch 딕셔너리
        """
        X = pt_batch['X'] # (B, 2*window_size)
        y = pt_batch['y'] # (B,)

        # X shape: (B, C) where C = 2 * window_size
        B, C = X.shape
        
        # EmbeddingBag은 (Indices, Offsets) 형태의 1D 입력을 받습니다.
        # (B, C) -> (B*C)
        flat = X.reshape(-1) 
        
        # 각 "bag" (즉, 각 context)의 시작 위치
        # [0, C, 2*C, 3*C, ...]
        offsets = torch.arange(0, B*C, step=C, device=X.device, dtype=torch.long)

        # h shape: (B, embed_dim)
        h = self.emb(flat, offsets) 
        h = self.dropout(h)
        
        # logits shape: (B, vocab_size)
        logits = self.fc(h) 

        loss = self.criterion(logits, y)
        
        # Update batch dict
        pt_batch['logits'] = logits
        pt_batch['loss'] = loss
        return pt_batch