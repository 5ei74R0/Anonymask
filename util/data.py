from dataclasses import dataclass


@dataclass
class Checkpoints:
    yolo: str
    mae: str
    swinir: str
