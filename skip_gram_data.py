import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Counter
import numpy as np # (ndarray 타입 체크용)

# --- 의존성 ---
# ('vocab_utils.py'가 필요합니다)
try:
    # build_vocab: (stoi, itos, token_counts) 3개 반환
    # tokens_to_indices: List[str] -> List[int]
    from vocab_utils import build_vocab, tokens_to_indices
except ImportError:
    print("="*50)
    print("경고: 'vocab_utils.py' 파일을 찾을 수 없습니다.")
    print("="*50)
    # 임시 함수 정의 (오류 방지용)
    def build_vocab(df, min_freq=1, max_vocab=None): return {}, {}, Counter()
    def tokens_to_indices(tokens, stoi): return [0] * len(tokens)


# =========================
# 2) Skip-Gram Dataset
# =========================
class SkipGramDataset(Dataset):
    """
    Skip-Gram (with Negative Sampling) 모델을 위한 Dataset 클래스.
    (center_idx, positive_context_idx) 형태의 'Positive Pair' 샘플을 생성합니다.
    Negative Sampling은 DataModule의 collate_fn에서 처리됩니다.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        stoi: Dict[str, int],
        window_size: int = 2,
        max_samples_per_doc: int = None):
        """
        Args:
            df (pd.DataFrame): 'lemmas' (List[str] 또는 np.ndarray) 컬럼.
            stoi (Dict[str, int]): token-to-index 딕셔너리.
            window_size (int): 좌/우 문맥 크기.
            max_samples_per_doc (int, optional): 문서당 샘플 최대 개수 (다운샘플링).
        """
        super().__init__()
        self.stoi = stoi
        self.window_size = window_size
        
        # (center_idx, context_idx) 쌍을 저장
        self.samples: List[Tuple[int, int]] = []
        
        print(f"Creating Skip-Gram positive pairs with window_size={window_size}...")

        for _, row in df.iterrows():
            toks = row.get("lemmas", None)
            
            # [FIX] 'lemmas'가 numpy.ndarray 타입일 경우 list로 변환
            if isinstance(toks, np.ndarray):
                toks = toks.tolist()
            
            # [FIX] 최소 2개 단어는 있어야 (center, context) 쌍이 나옴
            if toks is None or not isinstance(toks, list) or len(toks) < 2:
                continue

            ids = tokens_to_indices(toks, self.stoi) # 기존 함수 재활용
            local_samples = []

            for i in range(len(ids)):
                center_word_idx = ids[i]
                # 현재 중심 단어(i)의 좌/우 윈도우 범위 계산
                window_start = max(0, i - window_size)
                window_end = min(len(ids), i + window_size + 1)

                for j in range(window_start, window_end):
                    if i == j:  # 중심 단어 자신은 제외
                        continue
                    context_word_idx = ids[j]
                    local_samples.append((center_word_idx, context_word_idx))

            if max_samples_per_doc is not None and len(local_samples) > max_samples_per_doc:
                local_samples = random.sample(local_samples, max_samples_per_doc)

            self.samples.extend(local_samples)

        print(f"Total {len(self.samples):,} positive pairs created.")
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        [FIX] CUDA + num_workers 오류를 피하기 위해 CPU 텐서를 반환합니다.
              .to(device)는 Trainer에서 처리합니다.
        """
        center, positive = self.samples[idx]
        return dict(center=torch.tensor(center, dtype=torch.long),
                    positive=torch.tensor(positive, dtype=torch.long))
    
    # (참고: to() 메서드는 이제 DataLoader/Trainer에서
    #  .to_device_batch()를 사용하므로, 사실상 사용되지 않습니다)
    def to(self, device_type):
        pass # 더 이상 이 메서드를 사용하지 않습니다.


# =========================
# 3) Skip-Gram Data Module
# =========================
class LemmaSkipGramDataModule(object):
    """
    Skip-Gram + Negative Sampling을 위한 데이터 모듈.
    - Vocab, Split, DataLoader 생성
    - Negative Sampling을 위한 Unigram 분포(weights) 생성
    - collate_fn에서 실시간 Negative Sampling 수행
    """
    def __init__(
        self,
        df_all:              pd.DataFrame,
        batch_size:          int   = 256,
        window_size:         int   = 2,
        min_freq:            int   = 1,
        max_vocab:           int   = None,
        max_samples_per_doc: int   = None,
        test_size:           float = 0.1,
        valid_size:          float = 0.05,
        seed:                int   = 42,
        num_neg_samples:     int   = 5,    # Negative 샘플 개수
        power:               float = 0.75,  # Unigram 분포 보정 파워
        device:              str   = 'cpu'
        ):

        self.batch_size = batch_size
        self.window_size = window_size
        self.num_neg_samples = num_neg_samples
        
        print("Initializing Skip-Gram DataModule...")

        # 1. Skip-Gram용 vocab 함수 호출 (token_counts가 필요함)
        self.stoi, self.itos, cnt = build_vocab(
            df_all, min_freq=min_freq, max_vocab=max_vocab
        )
        self.vocab_size = len(self.stoi)

        # 2. Negative Sampling을 위한 가중치 계산 (Unigram 분포)
        print("Calculating weights for Negative Sampling...")
        weights = torch.zeros(self.vocab_size, dtype=torch.float)
        for i in range(self.vocab_size):
            token = self.itos[i]
            if token in cnt: # PAD, UNK 등은 cnt에 없을 수 있음
                weights[i] = cnt[token] ** power
                
        self.sampling_weights = weights / weights.sum()
        
        # [FIX] sampling_weights는 CPU에 있어야 num_workers > 0 일 때
        #       collate_fn (CPU 프로세스)에서 접근 가능합니다.
        #       .to(device)를 제거합니다.

        # 3. 문서 단위 분할 (기존과 동일)
        idx_all = df_all.index
        train_val_idx, test_idx = train_test_split(
            idx_all, test_size=test_size, random_state=seed
        )
        train_idx, valid_idx = train_test_split(
            train_val_idx, test_size=valid_size / (1.0 - test_size), random_state=seed
        )
        df_train = df_all.loc[train_idx].reset_index(drop=True)
        df_valid = df_all.loc[valid_idx].reset_index(drop=True)
        df_test  = df_all.loc[test_idx].reset_index(drop=True)

        # 4. SkipGramDataset 사용
        # [FIX] Dataset 생성 시 device 인자 제거 (CPU 텐서 반환)
        print("-" * 30)
        print("Creating Train Dataset...")
        self.train_dataset = SkipGramDataset(df_train, self.stoi, window_size, max_samples_per_doc)
        print("Creating Valid Dataset...")
        self.valid_dataset = SkipGramDataset(df_valid, self.stoi, window_size, max_samples_per_doc)
        print("Creating Test Dataset...")
        self.test_dataset  = SkipGramDataset(df_test,  self.stoi, window_size, max_samples_per_doc)
        print("-" * 30)
        
        # [FIX] dataset.to(device) 호출 제거

        print("DataModule initialized.")
        print(f"Vocab size (Skip-Gram): {self.vocab_size}")
        print(f"#Train samples (pos pairs): {len(self.train_dataset):,}")
        print(f"#Valid samples (pos pairs): {len(self.valid_dataset):,}")
        print(f"#Test  samples (pos pairs): {len(self.test_dataset):,}")

    # 5. collate_fn: Negative Sampling 추가
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        __getitem__에서 반환된 CPU 텐서(Positive Pair) 배치를 받아,
        Negative Sampling을 (CPU에서) 수행하고 CPU 텐서 배치를 반환합니다.
        """
        # batch: [ {'center': tensor(c1), 'positive': tensor(p1)}, ... ]
        centers = torch.stack([b['center'] for b in batch], dim=0)    # (B,)
        positives = torch.stack([b['positive'] for b in batch], dim=0)  # (B,)
        B = centers.size(0)

        # Negative 샘플링 (CPU에서 수행)
        num_samples_to_draw = B * self.num_neg_samples
        negatives = torch.multinomial(
            self.sampling_weights,       # (CPU 텐서)
            num_samples=num_samples_to_draw,
            replacement=True
        ).view(B, self.num_neg_samples)  # (B, num_neg)
        
        # [FIX] negatives 텐서도 CPU 텐서로 반환합니다.
        #       .to(device)는 Trainer에서 일괄 처리합니다.
        
        return dict(center=centers, positive=positives, negatives=negatives)

    # --- DataLoader 생성 함수들 ---
    
    # [FIX] **kwargs를 추가하여 num_workers, pin_memory 등을
    #       노트북에서 유연하게 설정할 수 있도록 합니다.
    
    def get_train_loader(self, **kwargs): 
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,  
            collate_fn=self.collate_fn, # (self의 메서드를 콜백으로 전달)
            **kwargs
        )
    
    def get_valid_loader(self, **kwargs): 
        return DataLoader(
            self.valid_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=self.collate_fn,
            **kwargs
        )
    
    def get_test_loader (self, **kwargs): 
        return DataLoader(
            self.test_dataset,  
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=self.collate_fn,
            **kwargs
        )