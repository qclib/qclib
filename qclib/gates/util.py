# Copyright 2021 qclib project.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from cmath import isclose, phase

def apply_ctrl_state(self):
    if self.ctrl_state is not None:
        for i, ctrl in enumerate(self.ctrl_state[::-1]):
            if ctrl == '0':
                self.definition.x(self.control_qubits[i])

def u2_to_su2(u_2):
    phase_factor = np.conj(np.linalg.det(u_2) ** (-1 / u_2.shape[0]))
    su_2 = u_2 / phase_factor
    return su_2, phase(phase_factor)


def check_u2(matrix):
    if matrix.shape != (2, 2):
        raise ValueError(
            "The shape of a U(2) matrix must be (2, 2)."
        )
    if not np.allclose(matrix @ np.conj(matrix.T), [[1.0, 0.0],[0.0, 1.0]]):
        raise ValueError(
            "The columns of a U(2) matrix must be orthonormal."
        )

def check_su2(matrix):
    return isclose(np.linalg.det(matrix), 1.0)
