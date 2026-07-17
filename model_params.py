from dataclasses import dataclass

@dataclass
class ModelParams:
    random_state: int = 0
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 6
    min_samples_split: int = 2
    test_size: float = 0.2
    cv: int = 5

@dataclass
class EncoderParams:
    handle_unknown: str = 'ignore'
    sparse_output: bool = False