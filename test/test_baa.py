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
Tests for the baa.py module.
"""
import datetime
from multiprocessing import Pool
from unittest import TestCase

import numpy as np
import pandas as pd
import qiskit
from qiskit import QiskitError
from qiskit.circuit.random import random_circuit
from qiskit.providers import aer

from qclib.state_preparation.baa_schmidt import initialize
from qclib.state_preparation.util.baa import adaptive_approximation, geometric_entanglement, _cnots
from qclib.util import get_state
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
from test.test_baa_schmidt import TestBaaSchmidt


use_parallel = True


def get_iota(j: int, n: int, b: int, basis_state: int):
    assert b in [0, 1]
    full_mask = 2**n - 1

    mask_j = 1 << j
    value = (mask_j & basis_state) >> j

    low_mask = full_mask >> (n - j)
    high_mask = full_mask & (full_mask << (j + 1))
    new_basis_state = ((basis_state & high_mask) >> 1) + (basis_state & low_mask)

    return value == b, new_basis_state


def generalized_cross_product(u: np.ndarray, v: np.ndarray):
    entries = []
    for j in range(u.shape[0]):
        for i in range(j):
            entry = np.abs(u[i] * v[j] - u[j] * v[i])**2
            entries.append(entry)
    return np.sum(entries)


def calculate_entropy_meyer_wallach(vector: np.ndarray):
    num_qb = int(np.ceil(np.log2(vector.shape[0])))
    meyer_wallach_entry = np.zeros(shape=(num_qb, 1))
    for j in range(num_qb):
        psi_0 = np.zeros(shape=(vector.shape[0]//2, 1), dtype=complex)  # np.zeros(shape=())
        psi_1 = np.zeros(shape=(vector.shape[0]//2, 1), dtype=complex)  # np.zeros(shape=())
        for basis_state, entry in enumerate(vector):
            delta_0, new_basis_state_0 = get_iota(j, num_qb, 0, basis_state)
            delta_1, new_basis_state_1 = get_iota(j, num_qb, 1, basis_state)

            if delta_0:
                psi_0[new_basis_state_0] = entry
            if delta_1:
                psi_1[new_basis_state_1] = entry

        entry = generalized_cross_product(psi_0, psi_1)
        meyer_wallach_entry[j] = entry

    return np.sum(meyer_wallach_entry) * (4/num_qb)


class TestBaa(TestCase):

    @staticmethod
    def get_vector(e_lower: float, e_upper: float, num_qubits: int, start_depth_multiplier=1, measure='meyer_wallach'):
        entanglement = -1.0
        multiplier = start_depth_multiplier
        iteration = 0
        entanglements = []
        vector = np.ndarray(shape=(0,))
        while e_lower > entanglement or entanglement > e_upper:
            qc: qiskit.QuantumCircuit = random_circuit(num_qubits, multiplier * num_qubits)
            qc.save_statevector('label')
            job: aer.AerJob = qiskit.execute(qc, backend=aer.AerSimulator(method="statevector"))
            vector = job.result().data()['label']
            if measure == 'geometric':
                entanglement = geometric_entanglement(vector)
            elif measure == 'meyer_wallach':
                entanglement = calculate_entropy_meyer_wallach(vector)
            else:
                raise ValueError(f'Entanglement Measure {measure} unknown.')
            iteration += 1
            if iteration > 100:
                multiplier += 1
                iteration = 0
                if not use_parallel:
                    print(f'{multiplier} ({np.min(entanglements):.4f}-{np.max(entanglements):.4f})', end='\n', flush=True)
                entanglements = []
            else:
                entanglements.append(entanglement)
                if not use_parallel:
                    print('.', end='', flush=True)
        if not use_parallel:
            print(f'Final {multiplier} ({np.min(entanglements):.4f}-{np.max(entanglements):.4f})', end='\n', flush=True)
        return vector, entanglement, multiplier * num_qubits

    @staticmethod
    def initialize_loss(fidelity_loss, state_vector=None, n_qubits=5, strategy='brute_force', use_low_rank=False):

        if state_vector is None:
            state_vector = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
            state_vector = state_vector / np.linalg.norm(state_vector)

        circuit = initialize(state_vector, max_fidelity_loss=fidelity_loss, strategy=strategy, use_low_rank=use_low_rank)
        state = get_state(circuit)
        overlap = TestBaaSchmidt.fidelity(state_vector, state)
        assert f'Overlap must be 1 ({overlap})', round(overlap, 2) >= 1-fidelity_loss

        try:
            basis_circuit = qiskit.transpile(circuit, basis_gates=['rx', 'ry', 'rz', 'cx'], optimization_level=3)
            cnots = len([d[0] for d in basis_circuit.data if d[0].name == 'cx'])
            depth = basis_circuit.depth()
        except QiskitError as ex:
            print(ex)
            return -1, -1

        return cnots, depth

    @staticmethod
    def execute_experiment(exp_idx,  num_qubits, entanglement_bounds, max_fidelity_loss):

        # State Generation
        state_vector, entganglement, depth = TestBaa.get_vector(*entanglement_bounds, num_qubits, 1)
        mw = calculate_entropy_meyer_wallach(state_vector)
        ge = geometric_entanglement(state_vector)
        cnots = _cnots(num_qubits)
        if not use_parallel:
            print(f"Found state for entanglement bounds {entganglement} in {entanglement_bounds}. State preparation needs {cnots}.")

        # Benchmark against real Algorithm
        real_cnots_benchmark, real_depth_benchmark = TestBaa.initialize_loss(
            state_vector=state_vector, fidelity_loss=0.0, use_low_rank=False
        )
        data_result = []
        for use_low_rank in [False, True]:
            for strategy in ['brute_force', 'greedy']:
                if not use_parallel:
                    print(f"{strategy.upper()} {'With' if use_low_rank else 'No'} Low Rank Processing....")
                node = adaptive_approximation(state_vector, max_fidelity_loss, use_low_rank=use_low_rank, strategy=strategy)
                # Result
                data = list(
                    zip(node.k_approximation, [list(v.shape) for v in node.vectors])
                )
                start_time = datetime.datetime.now()
                real_cnots, real_depth = TestBaa.initialize_loss(
                    state_vector=state_vector, fidelity_loss=max_fidelity_loss, use_low_rank=use_low_rank, strategy=strategy
                )
                end_time = datetime.datetime.now()
                duration = (end_time - start_time).total_seconds()
                data = [
                    exp_idx, use_low_rank, strategy, num_qubits, depth, cnots, ge, mw,
                    max_fidelity_loss, node.total_saved_cnots, node.total_fidelity_loss, data,
                    real_cnots, real_cnots_benchmark, real_depth, real_depth_benchmark, duration
                ]
                data_result.append(data)

        # Experiment transcription
        df = pd.DataFrame(data=data_result, columns=[
            'id', 'with_low_rank', 'strategy', 'num_qubits', 'depth', 'cnots', 'entganglement', 'entganglement (MW)',
            'max_fidelity_loss', 'total_saved_cnots', 'total_fidelity_loss', 'data', 'real_cnots',
            'real_cnots_no_approx', 'real_depth', 'real_depth_no_approx', 'duration'
        ])
        if use_parallel:
            print(f"Done {exp_idx,  num_qubits, entanglement_bounds, max_fidelity_loss}")
        return df

    def test(self):
        num_qubits = 7
        entanglement_bounds = (0.7, 1.0)

        data = [(i, num_qubits, entanglement_bounds, max_fidelity_loss)
                for max_fidelity_loss in np.linspace(0.1, 1.0, 10)
                for i in range(100)]
        if use_parallel:
            with Pool() as pool:
                result = pool.starmap(TestBaa.execute_experiment, data)
        else:
            result = [TestBaa.execute_experiment(*d) for d in data]

        df = pd.concat(result, ignore_index=True)
        print(df.to_string(), flush=True)
        timestamp_sec = int(datetime.datetime.now().timestamp())
        df.to_pickle(f'./{timestamp_sec}.test_baa.pickle')
        df.to_csv(f'./{timestamp_sec}.test_baa.csv')
