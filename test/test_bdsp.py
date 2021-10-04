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
from unittest import TestCase
from qiskit import QuantumCircuit, ClassicalRegister, execute, Aer
from qclib.state_preparation.bdsp import initialize

backend = Aer.get_backend('qasm_simulator') 
shots   = 8192

class TestBdsp(TestCase):

	@staticmethod
	def measurement(circuit, n, c):
		circuit.measure(list(range(n)), c)

		job = execute(circuit, backend, shots=shots, optimization_level=3)
		
		counts = job.result().get_counts(circuit)
		v = sum(counts.values())
		
		counts2 = {}
		for m in range(2**n):
			pattern = '{:0{}b}'.format(m, n)
			if pattern in counts:
				counts2[pattern] = counts[pattern]
			else:
				counts2[pattern] = 0.0

		return [ value/v for (key, value) in counts2.items() ]

	@staticmethod
	def bdsp_experiment(state, s=None):
		circuit = initialize(state, s)

		n = int(np.log2(len(state)))
		c = ClassicalRegister(n)
		circuit.add_register(c)

		return TestBdsp.measurement(circuit, n, c)
		
	def test_bottom_up(self):
		a = np.random.rand(16) + np.random.rand(16) * 1j
		a = a / np.linalg.norm(a)

		state = TestBdsp.bdsp_experiment(a, 1)

		self.assertTrue(np.allclose( np.power(np.abs(a),2), state, rtol=1e-01, atol=0.005))

	def test_top_down(self):
		a = np.random.rand(16) + np.random.rand(16) * 1j
		a = a / np.linalg.norm(a)

		state = TestBdsp.bdsp_experiment(a, int(np.log2(len(a))))

		self.assertTrue(np.allclose( np.power(np.abs(a),2), state, rtol=1e-01, atol=0.005))

	def test_sublinear(self):
		a = np.random.rand(16) + np.random.rand(16) * 1j
		a = a / np.linalg.norm(a)

		state = TestBdsp.bdsp_experiment(a)

		self.assertTrue(np.allclose( np.power(np.abs(a),2), state, rtol=1e-01, atol=0.005))
