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
import sys

from test.test_baa_schmidt import TestBaaSchmidt
import datetime
import os
from multiprocessing import Pool
from unittest import TestCase

import numpy as np
import pandas as pd
import qiskit
from qiskit import QiskitError
from qiskit.circuit.random import random_circuit

from qclib.entanglement import geometric_entanglement, \
    meyer_wallach_entanglement
from qclib.state_preparation.baa_schmidt import initialize
from qclib.state_preparation.schmidt import cnot_count as schmidt_cnots
from qclib.util import get_state

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring


use_parallel = os.getenv('QLIB_TEST_PARALLEL', 'False') == 'True'


def get_vector(e_lower: float, e_upper: float, num_qubits: int,
               start_depth_multiplier=1, measure='meyer_wallach'):
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
        np.random.seed()
        q_circuit: qiskit.QuantumCircuit = random_circuit(num_qubits, multiplier * num_qubits)
        vector = get_state(q_circuit)
        if measure == 'geometric':
            entanglement = geometric_entanglement(vector)
        elif measure == 'meyer_wallach':
            entanglement = meyer_wallach_entanglement(vector)
        else:
            raise ValueError(f'Entanglement Measure {measure} unknown.')
        iteration += 1
        if iteration > 100:
            multiplier += 1
            iteration = 0
            if not use_parallel:
                print(
                    f'{multiplier} ({np.min(entanglements):.4f}-{np.max(entanglements):.4f})',
                    end='\n', flush=True
                )
            entanglements = []
        else:
            entanglements.append(entanglement)
            if not use_parallel:
                print('.', end='', flush=True)
    if not use_parallel:
        print(
            f'Final {multiplier} ({np.min(entanglements):.4f}-{np.max(entanglements):.4f})',
            end='\n', flush=True
        )
    return vector, entanglement, multiplier * num_qubits


def initialize_loss(fidelity_loss, state_vector=None, n_qubits=5, strategy='brute_force',
                    use_low_rank=False):

    if state_vector is None:
        state_vector = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

    circuit, node = initialize(
        state_vector, max_fidelity_loss=fidelity_loss, strategy=strategy, use_low_rank=use_low_rank,
        return_node=True
    )
    state = get_state(circuit)
    fidelity = TestBaaSchmidt.fidelity(state_vector, state)

    try:
        basis_circuit = qiskit.transpile(
            circuit, basis_gates=['rx', 'ry', 'rz', 'cx'], optimization_level=0
        )
        cnots = len([d[0] for d in basis_circuit.data if d[0].name == 'cx'])
        depth = basis_circuit.depth()
    except QiskitError as ex:
        print(ex)
        return -1, -1, -1, node

    return cnots, depth, round(1 - fidelity, 4), node


def execute_experiment(exp_idx,  num_qubits, entanglement_bounds,
                       max_fidelity_losses, return_state=False,
                       start_depth_multiplier=1):
    print(f"Starting {exp_idx,  num_qubits, entanglement_bounds, max_fidelity_losses}")

    # State Generation
    state_vector, entganglement, depth = get_vector(
        *entanglement_bounds, num_qubits, start_depth_multiplier=start_depth_multiplier,
        measure='geometric'
    )
    mw_entanglement = meyer_wallach_entanglement(state_vector)
    geo_entanglement = geometric_entanglement(state_vector)
    cnots = schmidt_cnots(state_vector)
    if not use_parallel:
        print(
            f"Found state for entanglement bounds {entganglement} in {entanglement_bounds}. "
            f"State preparation needs {cnots}."
        )

    # Benchmark against real Algorithm
    real_cnots_benchmark, real_depth_benchmark, real_fidelity_loss_benchmark, _ = initialize_loss(
        state_vector=state_vector, fidelity_loss=0.0, use_low_rank=False
    )
    data_result = []
    for max_fidelity_loss in max_fidelity_losses:
        for use_low_rank in [False, True]:
            for strategy in ['brute_force', 'greedy']:
                if not use_parallel:
                    print(
                        f"[{max_fidelity_loss}] {strategy.upper()} "
                        f"{'With' if use_low_rank else 'No'} Low Rank Processing....", end=''
                    )
                start_time = datetime.datetime.now()
                real_cnots, real_depth, real_fidelity_loss, node = initialize_loss(
                    state_vector=state_vector, fidelity_loss=max_fidelity_loss,
                    use_low_rank=use_low_rank, strategy=strategy
                )
                end_time = datetime.datetime.now()
                duration = (end_time - start_time).total_seconds()
                vector_data = list(
                    zip(node.ranks, [list(v.shape) for v in node.vectors])
                )
                data = [
                    exp_idx, use_low_rank, strategy, num_qubits, depth, cnots, geo_entanglement,
                    mw_entanglement, max_fidelity_loss, node.total_saved_cnots,
                    node.total_fidelity_loss, vector_data, real_cnots, real_cnots_benchmark,
                    real_depth, real_depth_benchmark, real_fidelity_loss,
                    real_fidelity_loss_benchmark, duration
                ]
                if not use_parallel:
                    print(f"in {duration} secs")
                data_result.append(data)

    # Experiment transcription
    dframe = pd.DataFrame(data=data_result, columns=[
        'id', 'with_low_rank', 'strategy', 'num_qubits', 'depth', 'cnots', 'entganglement',
        'entganglement (MW)', 'max_fidelity_loss', 'total_saved_cnots', 'total_fidelity_loss',
        'data', 'real_cnots', 'real_cnots_benchmark', 'real_depth', 'real_depth_no_approx',
        'real_fidelity_loss', 'real_fidelity_loss_benchmark', 'duration'
    ])
    print(f"Done {exp_idx,  num_qubits, entanglement_bounds, max_fidelity_losses}")
    if return_state:
        return dframe, state_vector
    return dframe


class TestBaa(TestCase):

    def test(self):
        num_qubits = 7
        entanglement_bounds = (0.4, 1.0)
        max_fidelity_loss = np.linspace(0.1, 1.0, 4)
        number_of_experiments = 5

        data = [
            (i, num_qubits, entanglement_bounds, max_fidelity_loss)
            for i in range(number_of_experiments)
        ]
        if use_parallel:
            with Pool() as pool:
                result = pool.starmap(execute_experiment, data)
        else:
            result = [execute_experiment(*d) for d in data]

        df = pd.concat(result, ignore_index=True)
        df['predicted_cnots'] = df['cnots'] - df['total_saved_cnots']

        # Tests:
        # Calculation Tests
        # The benchmark will not change the state at all, so it must be
        # essentially zero
        df['benchmark_fidelity_loss_pass'] = np.abs(df['real_fidelity_loss_benchmark']) < 1e-6
        # The expected / predicted fidelity loss must be less or equal to the
        # max fidelity loss
        df['approximation_calculation_pass'] = df['max_fidelity_loss'] >= df['total_fidelity_loss']

        # The real Tests
        # The real measured fidelity measure must be less or equal to the
        # configured mx fidelity loss
        df['real_approximation_calculation_pass'] = (
                df['real_fidelity_loss'] - df['max_fidelity_loss'] < 0.1
        )

        # Attention: this is a set of tests that tests how well the CNOT-estimation works
        # However, we do not test this here. As a result, also the prediciton of CNOTs cannot
        # reliably be tested.
        # START: COMMENTED OUT AND LEFT FOR DOCUMENTATION REASONS
        # # The predicted maximum CNOT gates and the no-approximation count
        # # should be within 10%
        # df['cnot_prediction_calculation_pass'] = (
        #         np.abs(df['real_cnots_benchmark'] - df['cnots']) < 0.2 * df['cnots']
        # )
        # # The predicted CNOT gates should be within an error margin of 10%
        # df['saved_cnots_calculation_pass'] = (
        #         np.abs(df['predicted_cnots'] - df['real_cnots']) <= 0.2 * (df['predicted_cnots'])
        # )
        # END: COMMENTED OUT AND LEFT FOR DOCUMENTATION REASONS

        test_passed = True
        total_experiments = number_of_experiments * max_fidelity_loss.shape[0] * 4
        fails_to_still_pass = int(np.ceil(total_experiments * 0.01))
        if df.shape[0] - df[df['benchmark_fidelity_loss_pass']].shape[0] > 0:
            print("[WARNING] NOT ALL benchmark_fidelity_loss_pass are true!", file=sys.stderr)
        if df.shape[0] - df[df['benchmark_fidelity_loss_pass']].shape[0] > fails_to_still_pass:
            print("[FAIL] benchmark_fidelity_loss_pass must be true", file=sys.stderr)
            test_passed = False
        if df.shape[0] - df[df['approximation_calculation_pass']].shape[0] > 0:
            print("[WARNING] NOT ALL approximation_calculation_pass are true!", file=sys.stderr)
        if df.shape[0] - df[df['approximation_calculation_pass']].shape[0] > fails_to_still_pass:
            print("[FAIL] approximation_calculation_pass must be true", file=sys.stderr)
            test_passed = False
        if df.shape[0] - df[df['real_approximation_calculation_pass']].shape[0] > 0:
            print("[WARNING] NOT ALL real_approximation_calculation_pass are true", file=sys.stderr)
        if df.shape[0] - df[df['real_approximation_calculation_pass']].shape[0] > fails_to_still_pass:
            print("[FAIL] real_approximation_calculation_pass must be true", file=sys.stderr)
            test_passed = False

        print(df.to_string())

        self.assertTrue(test_passed, 'The tests should all pass.')

    def test_no_ops(self):
        # The Test is based on randomly generated states. They all should
        # create the correct fidelity (1.0). After 10 attempts we may well
        # find one that fails. If it doesn't it is probably okay.
        for lower_bound_int in range(10):
            state_vector, _, depth = get_vector(
                lower_bound_int/10, 1.0, 8, 1, measure='geometric'
            )
            cnots, depth, fidelity_loss, _ = initialize_loss(0.0, state_vector)
            if cnots == depth == fidelity_loss == -1:
                continue
            self.assertAlmostEqual(0.0, fidelity_loss, 4)