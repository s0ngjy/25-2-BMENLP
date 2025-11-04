import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple, Counter

# =========================
# 1) 상수 정의
# =========================

PAD, UNK = "<pad>", "<unk>"
"""
특수 토큰 정의:
PAD: 패딩(Padding) 토큰. 문장의 길이를 맞추기 위해 사용됩니다.
UNK: Unseen/Unknown 토큰. Vocab에 없는 단어를 처리하기 위해 사용됩니다.
"""
SPECIALS = [PAD, UNK]


# =========================
# 2) Vocab 생성 함수
# =========================

def build_vocab(df: pd.DataFrame, min_freq: int = 1, max_vocab: int = None) -> Tuple[Dict[str, int], Dict[int, str], Counter[str]]:
    """
    Pandas DataFrame의 'lemmas' 컬럼(List[str] 형태)으로부터
    stoi, itos, 그리고 단어 빈도수(Counter)를 생성합니다.

    (기존의 build_vocab_from_lemmas_with_counts 함수입니다)
    
    이 함수 하나로 CBOW와 Skip-Gram 모델 모두에서 사용 가능합니다.
    Skip-Gram은 Negative Sampling을 위해 반환되는 Counter가 필요하고,
    CBOW는 필요 없으면 무시하고 사용하면 됩니다.

    Args:
        df (pd.DataFrame): 'lemmas' 컬럼을 포함하는 데이터프레임
        min_freq (int, optional): 어휘에 포함될 최소 빈도수. Defaults to 1.
        max_vocab (int, optional): 최대 어휘 크기 (SPECIALS 포함). Defaults to None.

    Returns:
        Tuple[Dict[str, int], Dict[int, str], Counter[str]]:
            - stoi (Dict[str, int]): token to index 딕셔너리
            - itos (Dict[int, str]): index to token 딕셔너리
            - final_cnt (Counter[str]): 최종 어휘에 포함된 단어들의 빈도수
    """
    print("Building vocabulary...")
    cnt = Counter()
    
    # 1. 모든 'lemmas' 리스트를 순회하며 빈도수 계산
    for toks in df["lemmas"]:
        if toks is None:  # 혹시 모를 None 값 처리
            continue
        for t in toks:
            cnt[t] += 1

    # 2. 최소 빈도수(min_freq) 필터링
    items = [(tok, f) for tok, f in cnt.items() if f >= min_freq]
    
    # 3. 빈도순 정렬 (1순위: 빈도 내림차순, 2순위: 단어 오름차순)
    items.sort(key=lambda x: (-x[1], x[0]))

    # 4. 최대 어휘 크기(max_vocab) 제한
    if max_vocab is not None:
        # SPECIALS 토큰 2개를 위한 자리 확보
        items = items[:max_vocab - len(SPECIALS)]

    # 5. itos, stoi 생성
    itos = SPECIALS + [tok for tok, _ in items]
    stoi = {tok: idx for idx, tok in enumerate(itos)}

    # 6. 최종 vocab에 포함된 단어들의 빈도수만 필터링하여 반환
    # (Skip-Gram의 Negative Sampling 등에서 사용됨)
    final_cnt = Counter({tok: f for tok, f in items})
    
    print(f"Vocabulary built. Total size: {len(itos)}")
    return stoi, {i: t for i, t in enumerate(itos)}, final_cnt


# =========================
# 3) 토큰 변환 유틸
# =========================

def tokens_to_indices(tokens: List[str], stoi: Dict[str, int]) -> List[int]:
    """
    토큰 리스트(List[str])를 인덱스 리스트(List[int])로 변환합니다.
    Vocab에 없는 단어는 UNK 인덱스로 처리합니다.
    """
    unk_idx = stoi[UNK]
    return [stoi.get(t, unk_idx) for t in tokens]