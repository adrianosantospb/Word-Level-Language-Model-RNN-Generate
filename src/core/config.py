from dataclasses import dataclass

@dataclass()
class HyperParameters:
    n_epochs: int = 20
    batch_size: int = 128
    lr: float = 0.001
    num_workers: int = 8
    dir_base:str = "./weights"