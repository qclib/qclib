from typing import List, Tuple, Optional

import numpy as np
from numpy import ndarray


class Split:

    fidelity_loss: float
    subsystem: Optional[Tuple[int, ...]]

    def __init__(self, subsystem: Optional[Tuple[int, ...]], fidelity_loss: float):
        self.subsystem = subsystem
        self.fidelity_loss = fidelity_loss

    def __str__(self):
        return f'{type(self).__name__}|{self.subsystem}|'

    def __repr__(self):
        return str(self)


class Node:
    split_program: Tuple[Split, ...]
    vectors: List[ndarray]
    cnot_saving: int
    fidelity_loss: float

    def __init__(self, split_program: Tuple[Split, ...], vectors: List[np.ndarray], fidelity_loss: float, cnot_saving: int):
        self.fidelity_loss = fidelity_loss
        self.cnot_saving = cnot_saving
        self.vectors = vectors
        self.split_program = split_program

    def __getitem__(self, item):
        data = [self.split_program, self.vectors, self.fidelity_loss, self.cnot_saving]
        return data[item]

    def __iter__(self):
        data = [self.split_program, self.vectors, self.fidelity_loss, self.cnot_saving]
        return iter(data)

    def __str__(self):
        return f'Node{(self.split_program, self.fidelity_loss, self.cnot_saving, self.vectors)}'

    def __repr__(self):
        return str(self)
