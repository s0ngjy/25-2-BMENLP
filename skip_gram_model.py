import torch
from torch import nn
from typing import Dict

# =========================
# 4) Skip-Gram 모델 (Negative Sampling 사용)
# =========================
class SkipGram(nn.Module):
    """
    Skip-Gram 모델 (Negative Sampling 포함)
    - in_emb: 중심 단어(center) 임베딩
    - out_emb: 문맥 단어(context/positive, negative) 임베딩
    - Loss: Positive 샘플과 Negative 샘플에 대한 Binary Cross Entropy
    """
    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int = 128,
        padding_idx: int = 0,
        ):
        super().__init__()
        # Input embedding (중심 단어용)
        self.in_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        # Output embedding (문맥 단어용)
        self.out_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        # (참고: 성능 향상을 위해 in_emb와 out_emb의 가중치를
        #  다르게 초기화하거나 따로 관리하는 것이 일반적입니다.
        #  여기서는 nn.Embedding의 기본 초기화를 따릅니다.)

        self.criterion = nn.BCEWithLogitsLoss() # 이진 분류 손실

    def forward(self, pt_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            pt_batch (Dict):
                - 'center':    (B,)
                - 'positive':  (B,)
                - 'negatives': (B, K)  (K=num_neg_samples)
        Returns:
            Dict: 'loss' 키가 추가된 pt_batch
        """
        center = pt_batch['center']     # (B,)
        positive = pt_batch['positive'] # (B,)
        negatives = pt_batch['negatives'] # (B, K)

        # 1. Get Embeddings
        # .unsqueeze(1) -> (B, 1, D) (bmm을 위한 차원 추가)
        center_vecs = self.in_emb(center).unsqueeze(1)
        
        # (B, D)
        pos_vecs = self.out_emb(positive)
        # (B, K, D)
        neg_vecs = self.out_emb(negatives)

        # 2. Positive Scores (B, 1, D) @ (B, D, 1) -> (B, 1, 1) -> (B,)
        # (중심 단어와 실제 문맥 단어 간의 내적)
        # .unsqueeze(2) -> (B, D, 1)
        pos_score = torch.bmm(center_vecs, pos_vecs.unsqueeze(2)).squeeze()
        pos_labels = torch.ones_like(pos_score) # Positive label = 1

        # 3. Negative Scores (B, 1, D) @ (B, D, K) -> (B, 1, K) -> (B, K)
        # (중심 단어와 K개의 가짜(Negative) 단어들 간의 내적)
        # .transpose(1, 2) -> (B, D, K)
        neg_score = torch.bmm(center_vecs, neg_vecs.transpose(1, 2)).squeeze(1)
        neg_labels = torch.zeros_like(neg_score) # Negative label = 0

        # 4. Loss
        # Positive 쌍의 로짓은 1에 가까워져야 함
        pos_loss = self.criterion(pos_score, pos_labels)
        # Negative 쌍의 로짓은 0에 가까워져야 함
        neg_loss = self.criterion(neg_score, neg_labels)
        
        loss = pos_loss + neg_loss

        pt_batch['loss'] = loss
        return pt_batch