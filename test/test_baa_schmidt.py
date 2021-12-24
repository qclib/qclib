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

"""
Tests for the baa_schmidt.py module.
"""

from unittest import TestCase
import time
import numpy as np
from qiskit import ClassicalRegister, execute
from qiskit.providers.aer.backends import AerSimulator
from qclib.util import get_state
from qclib.state_preparation.baa_schmidt import initialize

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring

class TestBaaSchmidt(TestCase):
    @staticmethod
    def fidelity(state1, state2):
        bra = np.conj(state1)
        ket = state2

        return np.power(np.abs(bra.dot(ket)), 2)

    @staticmethod
    def get_counts(circuit):
        n_qubits = circuit.num_qubits
        classical_reg = ClassicalRegister(n_qubits)
        circuit.add_register(classical_reg)
        circuit.measure(list(range(n_qubits)), classical_reg)

        backend = AerSimulator()
        counts = execute(circuit, backend, shots=8192).result().get_counts()

        counts_with_zeros = {}
        for i in range(2**n_qubits):
            pattern = '{:0{}b}'.format(i, n_qubits)
            if pattern in counts:
                counts_with_zeros[pattern] = counts[pattern]
            else:
                counts_with_zeros[pattern] = 0.0

        sum_values = sum(counts.values())
        return [ value/sum_values for (key, value) in counts_with_zeros.items() ]

    def _test_initialize_loss(self, fidelity_loss, state_vector=None,
                                    n_qubits=5, strategy='brute_force', use_low_rank=False):
        if state_vector is None:
            state_vector = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
            state_vector = state_vector / np.linalg.norm(state_vector)

        circuit = initialize(state_vector, max_fidelity_loss=fidelity_loss, strategy=strategy,
                                                                    use_low_rank=use_low_rank)

        state = get_state(circuit)

        fidelity = TestBaaSchmidt.fidelity(state_vector, state)

        self.assertTrue(round(fidelity,2)>=round(1-fidelity_loss,2))

    def test_initialize_loss_brute_force(self):
        for loss in range(10, 20):
            self._test_initialize_loss(loss/100, n_qubits=5, strategy='brute_force')

    def test_initialize_loss_brute_force_low_rank(self):
        for loss in range(10, 20):
            self._test_initialize_loss(loss/100, n_qubits=5, strategy='brute_force',
                                                                    use_low_rank=True)

    def test_initialize_loss_greedy(self):
        for loss in range(10, 20):
            self._test_initialize_loss(loss/100, n_qubits=5, strategy='greedy')

    def test_initialize_loss_greedy_low_rank(self):
        for loss in range(10, 20):
            self._test_initialize_loss(loss/100, n_qubits=5, strategy='greedy',
                                                                    use_low_rank=True)

    def test_initialize_loss_fixed_n3(self):
        state_vector = [-0.33*1j,0,-0.44-0.44*1j,0.24+0.23*1j,0,0,0,0.62-0.01*1j]
        state_vector = state_vector/np.linalg.norm(state_vector)
        for loss in [0.1, 0.28, 0.9]:
            self._test_initialize_loss(loss, state_vector=state_vector, strategy='brute_force')
            self._test_initialize_loss(loss, state_vector=state_vector, strategy='greedy')

    def test_initialize_loss_fixed_n4(self):
        state_vector = [0.04214906+0.25870366j, 0.18263984+0.05596082j, 0.17202687+0.1843925j ,
                        0.24972444+0.04666321j, 0.03311006+0.28233458j, 0.26680588+0.22211721j,
                        0.07205056+0.04556719j, 0.27982261+0.01626855j, 0.22908475+0.25461504j,
                        0.14290823+0.2425394j , 0.14213592+0.08282699j, 0.0068727 +0.03378424j,
                        0.2016483 +0.298073j  , 0.07520782+0.0639856j , 0.01026576+0.07669651j,
                        0.31755857+0.09279232j]
        state_vector = state_vector/np.linalg.norm(state_vector)
        for loss in [0.1, 0.15, 0.18, 0.2]:
            self._test_initialize_loss(loss, state_vector=state_vector, strategy='brute_force')
            self._test_initialize_loss(loss, state_vector=state_vector, strategy='greedy')

    def test_initialize_loss_fixed_n5(self):
        state_vector = [0.17777766+0.10171662j, 0.19896424+0.10670792j, 0.07982054+0.19653055j,
                        0.18155708+0.05746777j, 0.04259147+0.17093567j, 0.21551328+0.08246133j,
                        0.09549255+0.1117806j , 0.20562749+0.12218064j, 0.16191832+0.01653411j,
                        0.12255337+0.14109365j, 0.20090638+0.11119666j, 0.19851901+0.04543331j,
                        0.06842539+0.16671467j, 0.03209685+0.16839388j, 0.01707365+0.20060943j,
                        0.03853768+0.08183117j, 0.00073591+0.10084589j, 0.09524694+0.18785593j,
                        0.06005853+0.06977443j, 0.01553849+0.05363906j, 0.10294799+0.12558734j,
                        0.20142903+0.06801796j, 0.05282011+0.20879126j, 0.11257846+0.20746226j,
                        0.17737416+0.03461382j, 0.01689154+0.06600272j, 0.06428148+0.06199636j,
                        0.1163249 +0.160533j  , 0.14177201+0.10456823j, 0.03156739+0.04567818j,
                        0.02078566+0.02023752j, 0.18967059+0.03469463j]
        state_vector = state_vector/np.linalg.norm(state_vector)
        for loss in [0.1, 0.12, 0.14, 0.16, 0.2]:
            self._test_initialize_loss(loss, state_vector=state_vector, strategy='brute_force')
            self._test_initialize_loss(loss, state_vector=state_vector, strategy='greedy')

    def test_initialize_no_loss(self):
        state_vector = np.random.rand(32) + np.random.rand(32) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit = initialize(state_vector)

        state = get_state(circuit)

        self.assertTrue(np.allclose(state_vector, state))

    def test_initialize_ame(self):
        """ Test initialization of a absolutely maximally entangled state"""
        state_vector = [1, 1, 1, 1,1,-1,-1, 1, 1,-1,-1, 1, 1, 1,1,1,
                        1, 1,-1,-1,1,-1, 1,-1,-1, 1,-1, 1,-1,-1,1,1]
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit = initialize(state_vector)

        state = get_state(circuit)

        self.assertTrue(np.allclose(state_vector, state))

    def test_measurement_no_loss(self):
        state_vector = np.random.rand(32) + np.random.rand(32) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit = initialize(state_vector)

        state = TestBaaSchmidt.get_counts(circuit)

        self.assertTrue(np.allclose( np.power(np.abs(state_vector),2), state,
                        rtol=1e-01, atol=0.005))

    def test_compare_strategies(self):
        fidelities1 = []
        fidelities2 = []
        for n_qubits in range(3,7):
            state_vector = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
            state_vector = state_vector / np.linalg.norm(state_vector)
            for loss in range(10, 20):
                circuit = initialize(state_vector, max_fidelity_loss=loss/100,
                                                        strategy='brute_force')
                state = get_state(circuit)
                fidelity1 = TestBaaSchmidt.fidelity(state_vector, state)

                circuit = initialize(state_vector, max_fidelity_loss=loss/100,
                                                        strategy='greedy')
                state = get_state(circuit)
                fidelity2 = TestBaaSchmidt.fidelity(state_vector, state)

                fidelities1.append(fidelity1)
                fidelities2.append(fidelity2)

        self.assertTrue(np.allclose(fidelities1, fidelities2, rtol=0.125, atol=0.0))
