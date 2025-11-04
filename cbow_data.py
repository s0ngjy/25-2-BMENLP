import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple
import numpy as np  # [FIX] numpy.ndarray 타입 체크를 위해 import

# --- 의존성 ---
# (이전 단계에서 생성한 'vocab_utils.py'가 필요합니다)
try:
    # build_vocab: (stoi, itos, token_counts) 3개 반환
    # tokens_to_indices: List[str] -> List[int]
    from vocab_utils import build_vocab, tokens_to_indices
except ImportError:
    print("="*50)
    print("경고: 'vocab_utils.py' 파일을 찾을 수 없습니다.")
    print("이 모듈은 'vocab_utils.py'의 'build_vocab'와 'tokens_to_indices' 함수에 의존합니다.")
    print("="*50)
    # 임시 함수 정의 (오류 방지용)
    def build_vocab(df, min_freq=1, max_vocab=None): return {}, {}, Counter()
    def tokens_to_indices(tokens, stoi): return [0] * len(tokens)


# =========================
# 2) CBOW Dataset
# =========================
class CBOWDataset(Dataset):
    """
    CBOW (Continuous Bag-of-Words) 모델을 위한 Dataset 클래스.
    (context_indices, target_idx) 형태의 샘플을 생성합니다.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        stoi: Dict[str, int],
        window_size: int = 2,
        max_samples_per_doc: int = None,
        device='cpu'):
        """
        Args:
            df (pd.DataFrame): 'lemmas' (List[str] 또는 np.ndarray) 컬럼을 포함한 데이터프레임.
            stoi (Dict[str, int]): token-to-index 딕셔너리.
            window_size (int): 좌/우 문맥 크기 (총 context 길이는 2*window_size).
            max_samples_per_doc (int, optional): 
                문서당 샘플 최대 개수. (메모리 절약용 다운샘플링). Defaults to None.
            device (str): 텐서가 위치할 디바이스.
        """
        super().__init__()
        self.stoi = stoi
        self.window_size = window_size
        self.samples: List[Tuple[List[int], int]] = []  # (context_indices, target_idx)
        self.device = device
        
        full_window_size = 2 * window_size + 1
        
        print(f"Creating CBOW samples with window_size={window_size}...")

        for _, row in df.iterrows():
            toks = row.get("lemmas", None)
            
            # ----------------------------------------------------
            # [FIX] 'lemmas'가 numpy.ndarray 타입일 경우 list로 변환
            #       (isinstance(toks, list) 체크를 통과하기 위함)
            # ----------------------------------------------------
            if isinstance(toks, np.ndarray):
                toks = toks.tolist()
            # ----------------------------------------------------
            
            # 문장이 너무 짧으면 (최소한 full_window_size) 샘플 생성 불가
            if toks is None or not isinstance(toks, list) or len(toks) < full_window_size:
                continue

            # 1. 토큰 리스트 -> 인덱스 리스트
            ids = tokens_to_indices(toks, self.stoi)
            local_samples = []

            # 2. 윈도우를 슬라이딩하며 (context, target) 샘플 추출
            for i in range(window_size, len(ids) - window_size):
                target = ids[i]
                left_ctx = ids[i - window_size:i]
                right_ctx = ids[i + 1:i + 1 + window_size]
                context = left_ctx + right_ctx
                
                local_samples.append((context, target))

            # 3. (선택적) 문서당 최대 샘플링
            if max_samples_per_doc is not None and len(local_samples) > max_samples_per_doc:
                local_samples = random.sample(local_samples, max_samples_per_doc)

            self.samples.extend(local_samples)
        
        print(f"Total {len(self.samples):,} samples created.") # (출력 포맷팅 , 추가)
        
        # 전체 샘플을 한 번 섞어줌 (DataLoader에서 shuffle=True와 별개로)
        random.shuffle(self.samples)

    def to(self, device_type):
        """데이터셋이 생성하는 텐서의 디바이스를 변경합니다."""
        self.device = device_type

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        하나의 샘플을 텐서 딕셔너리 형태로 반환합니다.
        """
        context, target = self.samples[idx]
        return dict(X=torch.tensor(context, dtype=torch.long),
                    y=torch.tensor(target, dtype=torch.long))


# =========================
# 4) CBOW Data Module
# =========================
class LemmaCBOWDataModule(object):
    """
    CBOW 학습을 위한 데이터 모듈.
    Vocab 생성, Train/Valid/Test 스플릿, DataLoader 생성을 담당합니다.
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
        valid_size:          float = 0.05, # (train+valid) 중 valid 비율
        seed:                int   = 42,
        device:              str   = 'cpu'
        ):
        
        self.batch_size = batch_size
        self.window_size = window_size
        self.device = device
        
        # 1. Vocab build
        print("Initializing DataModule...")
        self.stoi, self.itos, _ = build_vocab(
            df_all, 
            min_freq=min_freq, 
            max_vocab=max_vocab
        )
        self.vocab_size = len(self.stoi)

        # 2. 문서 단위로 데이터 분할 (Train / Valid / Test)
        idx_all = df_all.index
        train_val_idx, test_idx = train_test_split(
            idx_all, 
            test_size=test_size, 
            random_state=seed
        )
        
        train_idx, valid_idx = train_test_split(
            train_val_idx, 
            test_size=valid_size / (1.0 - test_size), # 전체 대비 5% -> (1-test) 대비 X%
            random_state=seed
        )

        df_train = df_all.loc[train_idx].reset_index(drop=True)
        df_valid = df_all.loc[valid_idx].reset_index(drop=True)
        df_test  = df_all.loc[test_idx].reset_index(drop=True)

        # (참고: df_train, df_valid, df_test는 여전히 'lemmas'가 ndarray인 상태지만,
        #  CBOWDataset 내부에서 list로 변환되므로 여기서 추가 작업 불필요)
        
        print("-" * 30)
        print("Creating Train Dataset...")
        self.train_dataset = CBOWDataset(df_train, self.stoi, window_size, max_samples_per_doc)
        print("Creating Valid Dataset...")
        self.valid_dataset = CBOWDataset(df_valid, self.stoi, window_size, max_samples_per_doc)
        print("Creating Test Dataset...")
        self.test_dataset  = CBOWDataset(df_test,  self.stoi, window_size, max_samples_per_doc)
        print("-" * 30)

        # 3. Dataset에 디바이스 설정
        self.train_dataset.to(device)
        self.valid_dataset.to(device)
        self.test_dataset.to(device)

        print("DataModule initialized.")
        print(f"Vocab size: {self.vocab_size}")
        print(f"#Train samples: {len(self.train_dataset):,}")
        print(f"#Valid samples: {len(self.valid_dataset):,}")
        print(f"#Test  samples: {len(self.test_dataset):,}")

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        CBOW는 문맥 길이가 항상 2 * window_size로 고정되어 있으므로
        패딩(padding)이 필요 없습니다. 
        단순히 텐서들을 stack(쌓기)만 하면 됩니다.
        """
        X = torch.stack([b['X'] for b in batch], dim=0)
        y = torch.stack([b['y'] for b in batch], dim=0) 
        return dict(X=X, y=y)

    # --- DataLoader 생성 함수들 ---

    def get_train_loader(self, **kwargs): 
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,  
            collate_fn=self.collate_fn,
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
    
    def get_test_loader(self, **kwargs): 
        return DataLoader(
            self.test_dataset,  
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=self.collate_fn,
            **kwargs
        )