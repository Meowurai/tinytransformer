# src/tinytransformer/dataset.py

def count_examples(ids: list[int], block_size: int = 4) -> int:
    return max(len(ids) - block_size, 0)

def build_example(ids: list[int], start_index: int = 0, block_size: int = 4):
    if (start_index + block_size + 1) > len(ids):
        raise ValueError(f"Block size window is too large for start index {start_index}")
    
    x = ids[start_index : start_index + block_size]
    y = ids[start_index + 1 : start_index + block_size + 1]

    return x, y