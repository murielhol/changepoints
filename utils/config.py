from dataclasses import dataclass


@dataclass()
class Config:
    sequence_len: int
    model_name: str