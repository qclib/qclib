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
Tests for the SynQ integration (qclib.synq, SynqInitialize and
unitary(decomposition="synq")). Skipped when SynQ is not installed.
"""

from unittest import TestCase, skipUnless
from unittest import mock
import numpy as np
from scipy.stats import unitary_group
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator, Statevector
from qclib import synq
from qclib.state_preparation import SynqInitialize
from qclib.unitary import unitary
from qclib.util import get_state

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring


@skipUnless(synq.is_available(), "SynQ is not installed")
class TestSynq(TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def _random_state(self, n_qubits, sparsity=0.0, real=False):
        state = self.rng.normal(size=2**n_qubits)
        if not real:
            state = state + 1j * self.rng.normal(size=2**n_qubits)
        state[self.rng.random(2**n_qubits) < sparsity] = 0
        if np.linalg.norm(state) == 0:
            state[0] = 1
        return state / np.linalg.norm(state)

    # state preparation

    def test_state_preparation(self):
        for n_qubits in range(1, 8):
            for sparsity, real in [(0.0, False), (0.0, True), (0.7, False)]:
                state = self._random_state(n_qubits, sparsity, real)
                circuit = synq.state_preparation(state)
                self.assertTrue(np.allclose(Statevector(circuit).data, state))

    def test_initialize(self):
        for n_qubits in range(1, 7):
            state = self._random_state(n_qubits)
            circuit = QuantumCircuit(n_qubits)
            SynqInitialize.initialize(circuit, state)
            self.assertTrue(np.allclose(get_state(circuit), state))

    def test_initialize_subset_of_qubits(self):
        state = self._random_state(3)
        circuit = QuantumCircuit(5)
        SynqInitialize.initialize(circuit, state, [1, 2, 4])
        # qubits [1, 2, 4] hold the state; build the expected vector explicitly
        expected = np.zeros(32, dtype=complex)
        for index, amplitude in enumerate(state):
            bits = [(index >> k) & 1 for k in range(3)]
            expected[(bits[0] << 1) | (bits[1] << 2) | (bits[2] << 4)] = amplitude
        self.assertTrue(np.allclose(get_state(circuit), expected))

    def test_basis_states(self):
        for index in range(8):
            state = np.zeros(8, dtype=complex)
            state[index] = np.exp(0.3j * index)
            self.assertTrue(np.allclose(Statevector(synq.state_preparation(state)).data, state))

    def test_initialize_inverse(self):
        state = self._random_state(4)
        circuit = QuantumCircuit(4)
        SynqInitialize.initialize(circuit, state)
        circuit = circuit.compose(circuit.inverse())
        zero = np.zeros(16)
        zero[0] = 1
        self.assertTrue(np.allclose(get_state(circuit), zero))

    def test_invalid_state(self):
        with self.assertRaises(ValueError):
            SynqInitialize([0.5, 0.5, 0.5])
        with self.assertRaises(ValueError):
            synq.state_preparation([1.0, 1.0])

    # unitary

    def _assert_unitary(self, gate):
        circuit = unitary(gate, decomposition="synq")
        # exact, including the global phase
        self.assertTrue(np.allclose(Operator(circuit).data, gate))

    def test_unitary_random(self):
        for n_qubits in range(1, 6):
            self._assert_unitary(unitary_group.rvs(2**n_qubits, random_state=n_qubits))

    def test_unitary_structured(self):
        for n_qubits in range(1, 5):
            dim = 2**n_qubits
            self._assert_unitary(np.eye(dim))
            self._assert_unitary(-np.eye(dim))
            self._assert_unitary(np.eye(dim)[self.rng.permutation(dim)])
            self._assert_unitary(np.diag(np.exp(1j * self.rng.uniform(0, 2 * np.pi, dim))))
            if n_qubits > 1:
                self._assert_unitary(Operator(QFTGate(n_qubits)).data)
                self._assert_unitary(
                    np.kron(unitary_group.rvs(2, random_state=1),
                            unitary_group.rvs(dim // 2, random_state=2))
                )

    def test_unitary_toffoli(self):
        toffoli = np.eye(8)
        toffoli[[6, 7]] = toffoli[[7, 6]]
        self._assert_unitary(toffoli)

    def test_unitary_basis_gates(self):
        circuit = unitary(unitary_group.rvs(8, random_state=3), decomposition="synq")
        self.assertTrue(set(circuit.count_ops()) <= {"rz", "ry", "cx", "cz"})
        two_qubit = circuit.count_ops().get("cx", 0) + circuit.count_ops().get("cz", 0)
        transpiled = transpile(circuit, basis_gates=["u", "cx"], optimization_level=0)
        self.assertEqual(transpiled.count_ops()["cx"], two_qubit)

    def test_unitary_cnot_count(self):
        # optimizations A.1 and A.2 of Shende, Bullock and Markov
        for n_qubits, expected in [(3, 20), (4, 100), (5, 444)]:
            circuit = unitary(unitary_group.rvs(2**n_qubits, random_state=7), decomposition="synq")
            ops = circuit.count_ops()
            self.assertEqual(ops.get("cx", 0) + ops.get("cz", 0), expected)

    def test_qasm_matches_gates(self):
        # the OpenQASM output and the gate list describe the same circuit
        import synq as synq_module  # pylint: disable=import-outside-toplevel
        gate = unitary_group.rvs(8, random_state=5)
        from_qasm = synq.qasm_to_circuit(synq_module.unitary(gate))
        self.assertTrue(np.allclose(Operator(from_qasm).data, gate))

    def test_invalid_unitary(self):
        with self.assertRaises(ValueError):
            unitary(np.ones((4, 4)), decomposition="synq")
        with self.assertRaises(ValueError):
            unitary(np.eye(3), decomposition="synq")

    # diagonal

    def test_diagonal(self):
        for n_qubits in range(1, 6):
            diag = np.exp(1j * self.rng.uniform(-np.pi, np.pi, 2**n_qubits))
            circuit = synq.diagonal(diag)
            self.assertTrue(np.allclose(Operator(circuit).data, np.diag(diag)))

    # parser

    def test_qasm_to_circuit_rejects_unknown_gates(self):
        qasm = 'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[1] q;\nx q[0];\n'
        with self.assertRaises(ValueError):
            synq.qasm_to_circuit(qasm)


class TestSynqConversion(TestCase):
    """Conversion of SynQ output into Qiskit circuits (does not need SynQ)."""

    # every gate type: rz, ry, p, cx, cz, h on 3 qubits, plus a global phase
    _GATES = (
        np.array([0, 1, 2, 3, 4, 5, 5, 3], dtype=np.uint8),
        np.array([[0, -1], [1, -1], [2, -1], [0, 1], [2, 0], [1, -1], [2, -1], [1, 2]], dtype=np.int32),
        np.array([0.3, -1.2, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
        0.7,
    )
    _QASM = (
        'OPENQASM 3.0;\ninclude "stdgates.inc";\n\nqubit[3] q; \n'
        "rz(0.3) q[0];\nry(-1.2) q[1];\np(2.5) q[2];\ncx q[0], q[1];\ncz q[2], q[0];\n"
        "h q[1];\nh q[2];\ncx q[1], q[2];\ngphase(0.7);\n"
    )

    @staticmethod
    def _expected():
        circuit = QuantumCircuit(3, global_phase=0.7)
        circuit.rz(0.3, 0)
        circuit.ry(-1.2, 1)
        circuit.p(2.5, 2)
        circuit.cx(0, 1)
        circuit.cz(2, 0)
        circuit.h(1)
        circuit.h(2)
        circuit.cx(1, 2)
        return circuit

    def _assert_expected(self, circuit):
        expected = self._expected()
        self.assertEqual(circuit.count_ops(), expected.count_ops())
        self.assertTrue(np.allclose(Operator(circuit).data, Operator(expected).data))

    def test_gates_to_circuit(self):
        self._assert_expected(synq.gates_to_circuit(3, self._GATES))

    def test_gates_to_circuit_fallbacks(self):
        # the slower conversions used when Qiskit's StandardGate API is missing
        with mock.patch.object(synq, "_FROM_STANDARD", None):
            self._assert_expected(synq.gates_to_circuit(3, self._GATES))
        types, qubits, angles, phase = self._GATES
        for append in (synq._append_fast, synq._append_public):  # pylint: disable=protected-access
            circuit = QuantumCircuit(3, global_phase=phase)
            append(circuit, types.tolist(), qubits.tolist(), angles.tolist())
            self._assert_expected(circuit)

    def test_qasm_to_circuit(self):
        self._assert_expected(synq.qasm_to_circuit(self._QASM))
