from unittest import TestCase
from qclib.state_preparation.bidirectional.state_tree_preparation import Amplitude
from qiskit import QuantumCircuit, ClassicalRegister, execute, Aer

import numpy as np
import qclib.state_preparation.bdsp as bd

backend = Aer.get_backend('qasm_simulator') 
shots   = 8192

class TestBidirectional(TestCase):

	@staticmethod
	def measurement(circuit, q, c):
		circuit.measure(q, c)

		job = execute(circuit, backend, shots=shots, optimization_level=3)
		
		counts = job.result().get_counts(circuit)
		v = sum(counts.values())
		
		n = len(q)
		counts2 = {}
		for m in range(2**n):
			pattern = '{:0{}b}'.format(m, n)
			if pattern in counts:
				counts2[pattern] = counts[pattern]
			else:
				counts2[pattern] = 0.0

		return [ value/v for (key, value) in counts2.items() ]

	@staticmethod
	def bidirectional_experiment(circuit, input_state, s=None):
		state = [Amplitude(i, a) for i, a in enumerate(input_state)]

		q_output, state_tree, angle_tree = bd.initialize(circuit, state, s)

		n = int(np.log2(len(input_state)))
		c = ClassicalRegister(n)
		circuit.add_register(c)

		return TestBidirectional.measurement(circuit, q_output, c)
		
	def test_bottom_up(self):
		a = np.random.rand(16) + np.random.rand(16) * 1j
		a = a / np.linalg.norm(a)

		circuit = QuantumCircuit()
		state = TestBidirectional.bidirectional_experiment(circuit, a, 1)

		self.assertTrue(np.allclose( np.power(np.abs(a),2), state, rtol=1e-02, atol=1e-03))

	def test_top_down(self):
		a = np.random.rand(16) + np.random.rand(16) * 1j
		a = a / np.linalg.norm(a)

		circuit = QuantumCircuit()
		state = TestBidirectional.bidirectional_experiment(circuit, a, int(np.log2(len(a))))

		self.assertTrue(np.allclose( np.power(np.abs(a),2), state, rtol=1e-02, atol=1e-03))

	def test_sublinear(self):
		a = np.random.rand(16) + np.random.rand(16) * 1j
		a = a / np.linalg.norm(a)

		circuit = QuantumCircuit()
		state = TestBidirectional.bidirectional_experiment(circuit, a)

		self.assertTrue(np.allclose( np.power(np.abs(a),2), state, rtol=1e-02, atol=1e-03))

