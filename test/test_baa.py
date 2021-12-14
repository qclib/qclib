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
import os
from multiprocessing import Pool
from unittest import TestCase

import numpy as np
import pandas as pd
import qiskit
from qiskit import QiskitError
from qiskit.circuit.random import random_circuit

from qclib.state_preparation.baa_schmidt import initialize
from qclib.state_preparation.schmidt import cnot_count as schmidt_cnots
from qclib.state_preparation.util.baa import adaptive_approximation, geometric_entanglement
from qclib.util import get_state
from test.test_baa_schmidt import TestBaaSchmidt

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring


use_parallel = bool(os.getenv('QLIB_TEST_PARALLEL', 'False'))


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


def get_vector(e_lower: float, e_upper: float, num_qubits: int, start_depth_multiplier=1, measure='meyer_wallach'):
    entanglement = -1.0

    if isinstance(start_depth_multiplier, int):
        multiplier = start_depth_multiplier
    elif isinstance(start_depth_multiplier, (tuple, list, np.ndarray)):
        assert len(start_depth_multiplier) > 1
        multiplier = np.random.randint(start_depth_multiplier[0], start_depth_multiplier[1])
    else:
        raise ValueError("start_depth_multiplier must be either an int or an array-like of int.")

    iteration = 0
    entanglements = []
    vector = np.ndarray(shape=(0,))
    while e_lower > entanglement or entanglement > e_upper:
        qc: qiskit.QuantumCircuit = random_circuit(num_qubits, multiplier * num_qubits)
        vector = get_state(qc)
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


def initialize_loss(fidelity_loss, state_vector=None, n_qubits=5, strategy='brute_force', use_low_rank=False):

    if state_vector is None:
        state_vector = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

    circuit = initialize(state_vector, max_fidelity_loss=fidelity_loss, strategy=strategy, use_low_rank=use_low_rank)
    state = get_state(circuit)
    fidelity = TestBaaSchmidt.fidelity(state_vector, state)

    try:
        basis_circuit = qiskit.transpile(circuit, basis_gates=['rx', 'ry', 'rz', 'cx'], optimization_level=3)
        cnots = len([d[0] for d in basis_circuit.data if d[0].name == 'cx'])
        depth = basis_circuit.depth()
    except QiskitError as ex:
        print(ex)
        return -1, -1, -1

    return cnots, depth, round(1 - fidelity, 4)


def execute_experiment(exp_idx,  num_qubits, entanglement_bounds, max_fidelity_losses, return_state=False,
                       start_depth_multiplier=1):
        print(f"Starting {exp_idx,  num_qubits, entanglement_bounds, max_fidelity_losses}")

        # State Generation
        state_vector, entganglement, depth = get_vector(
            *entanglement_bounds, num_qubits, start_depth_multiplier=start_depth_multiplier, measure='geometric'
        )
        mw = calculate_entropy_meyer_wallach(state_vector)
        ge = geometric_entanglement(state_vector)
        cnots = schmidt_cnots(state_vector)
        if not use_parallel:
            print(f"Found state for entanglement bounds {entganglement} in {entanglement_bounds}. State preparation needs {cnots}.")

        # Benchmark against real Algorithm
        real_cnots_benchmark, real_depth_benchmark, real_fidelity_loss_benchmark = initialize_loss(
            state_vector=state_vector, fidelity_loss=0.0, use_low_rank=False
        )
        data_result = []
        for max_fidelity_loss in max_fidelity_losses:
            for use_low_rank in [False, True]:
                for strategy in ['brute_force', 'greedy']:
                    if not use_parallel:
                        print(f"[{max_fidelity_loss}] {strategy.upper()} {'With' if use_low_rank else 'No'} Low Rank Processing....", end='')
                    node = adaptive_approximation(state_vector, max_fidelity_loss, use_low_rank=use_low_rank, strategy=strategy)
                    # Result
                    data = list(
                        zip(node.k_approximation, [list(v.shape) for v in node.vectors])
                    )
                    start_time = datetime.datetime.now()
                    real_cnots, real_depth, real_fidelity_loss = initialize_loss(
                        state_vector=state_vector, fidelity_loss=max_fidelity_loss, use_low_rank=use_low_rank, strategy=strategy
                    )
                    end_time = datetime.datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    data = [
                        exp_idx, use_low_rank, strategy, num_qubits, depth, cnots, ge, mw,
                        max_fidelity_loss, node.total_saved_cnots, node.total_fidelity_loss, data,
                        real_cnots, real_cnots_benchmark, real_depth, real_depth_benchmark, real_fidelity_loss,
                        real_fidelity_loss_benchmark, duration
                    ]
                    if not use_parallel:
                        print(f"in {duration} secs")
                    data_result.append(data)

        # Experiment transcription
        df = pd.DataFrame(data=data_result, columns=[
            'id', 'with_low_rank', 'strategy', 'num_qubits', 'depth', 'cnots', 'entganglement', 'entganglement (MW)',
            'max_fidelity_loss', 'total_saved_cnots', 'total_fidelity_loss', 'data', 'real_cnots',
            'real_cnots_no_approx', 'real_depth', 'real_depth_no_approx', 'real_fidelity_loss',
            'real_fidelity_loss_benchmark', 'duration'
        ])
        print(f"Done {exp_idx,  num_qubits, entanglement_bounds, max_fidelity_losses}")
        if return_state:
            return df, state_vector
        else:
            return df


class TestBaa(TestCase):

    def test(self):
        num_qubits = 7
        entanglement_bounds = (0.4, 1.0)
        max_fidelity_loss = np.linspace(0.1, 1.0, 4)

        data = [(i, num_qubits, entanglement_bounds, max_fidelity_loss) for i in range(5)]
        if use_parallel:
            with Pool() as pool:
                result = pool.starmap(execute_experiment, data)
        else:
            result = [execute_experiment(*d) for d in data]

        df = pd.concat(result, ignore_index=True)

        # Tests:
        # Calculation Tests
        # The benchmark will not change the state at all, so it must be essentially zero
        df['benchmark_fidelity_loss_pass'] = np.abs(df['real_fidelity_loss_benchmark']) < 1e-6
        # The expected / predicted fidelity loss must be less or equal to the max fidelity loss
        df['approximation_calculation_pass'] = df['max_fidelity_loss'] >= df['total_fidelity_loss']

        # The real Tests
        # The real measured fidelity measure must be less or equal to the configured mx fidelity loss
        df['real_approximation_calculation_pass'] = df['real_fidelity_loss'] < df['max_fidelity_loss']
        # The predicted maximum CNOT gates and the no-approximation count should be within 10%
        df['cnot_prediction_calculation_pass'] = np.abs(df['real_cnots_no_approx'] - df['cnots']) < 0.1 * df['cnots']
        # The predicted CNOT gates should be within an error margin of 10%
        df['saved_cnots_calculation_pass'] = np.abs(df['cnots'] - df['total_saved_cnots'] - df['real_cnots']) <= 0.1 * (df['cnots'] - df['total_saved_cnots'])

        test_passed = True
        if df.shape[0] != df[df['benchmark_fidelity_loss_pass']].shape[0]:
            print("[FAIL] All benchmark_fidelity_loss_pass must be true")
            test_passed = False
        if df.shape[0] != df[df['approximation_calculation_pass']].shape[0]:
            print("[FAIL] All approximation_calculation_pass must be true")
            test_passed = False
        if df.shape[0] != df[df['real_approximation_calculation_pass']].shape[0]:
            print("[FAIL] All real_approximation_calculation_pass must be true")
            test_passed = False
        if df.shape[0] != df[df['cnot_prediction_calculation_pass']].shape[0]:
            print("[FAIL] All cnot_prediction_calculation_pass must be true")
            test_passed = False
        if df.shape[0] != df[df['saved_cnots_calculation_pass']].shape[0]:
            print("[FAIL] All saved_cnots_calculation_pass must be true")
            test_passed = False

        if not test_passed:
            print(df.to_string(), flush=True)

        self.assertTrue(test_passed, 'The tests should all pass.')

    def test_no_ops(self):
        # The Test is based on randomly generated states. They all should create the correct fidelity (1.0)
        # After 10 attempts we may well find one that fails. If it doesn't it is probably okay.
        for _ in range(10):
            state_vector, entganglement, depth = get_vector(0.7, 1.0, 8, 1, measure='geometric')
            cnots, depth, fidelity_loss = initialize_loss(0.0, state_vector)
            if cnots == depth == fidelity_loss == -1:
                continue
            self.assertAlmostEqual(0.0, fidelity_loss, 4)
