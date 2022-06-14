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

from test.test_baa_lowrank import TestBaaLowRank
import datetime
import os
from multiprocessing import Pool
from unittest import TestCase

import numpy as np
import qiskit
from qiskit import QiskitError
from qiskit.circuit.random import random_circuit

from qclib.entanglement import geometric_entanglement, \
    meyer_wallach_entanglement
from qclib.state_preparation import BaaLowRankInitialize
from qclib.state_preparation.lowrank import cnot_count as schmidt_cnots
from qclib.state_preparation.util.baa import adaptive_approximation
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

    opt_params = {'max_fidelity_loss': fidelity_loss,
                  'strategy': strategy, 'use_low_rank': use_low_rank}
    circuit = BaaLowRankInitialize(state_vector, opt_params=opt_params).definition

    state = get_state(circuit)
    fidelity = TestBaaLowRank.fidelity(state_vector, state)

    node = adaptive_approximation(state_vector, max_fidelity_loss=fidelity_loss,
                                  strategy=strategy, use_low_rank=use_low_rank)

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
    data_result = np.asarray(data_result, dtype=object)
    print(f"Done {exp_idx,  num_qubits, entanglement_bounds, max_fidelity_losses}")
    if return_state:
        return data_result, state_vector
    return data_result


class TestBaa(TestCase):
    columns = [
        'id', 'with_low_rank', 'strategy', 'num_qubits', 'depth', 'cnots', 'entganglement',
        'entganglement (MW)', 'max_fidelity_loss', 'total_saved_cnots', 'total_fidelity_loss',
        'data', 'real_cnots', 'real_cnots_benchmark', 'real_depth', 'real_depth_no_approx',
        'real_fidelity_loss', 'real_fidelity_loss_benchmark', 'duration',
        'predicted_cnots', 'benchmark_fidelity_loss_pass', 'approximation_calculation_pass',
        'real_approximation_calculation_pass'
    ]

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

        test_data = np.concatenate(result)
        # Predicted CNOTs
        test_data = np.hstack([
            test_data,
            (test_data[:, self.columns.index('cnots')]
             - test_data[:, self.columns.index('total_saved_cnots')]).reshape(-1, 1)
        ])

        # Tests:
        # Calculation Tests
        # The benchmark will not change the state at all, so it must be
        # essentially zero
        test_data = np.hstack([
            test_data,
            (np.abs(test_data[:, self.columns.index('real_fidelity_loss_benchmark')]) < 1e-6
             ).reshape(-1, 1)
        ])
        # The expected / predicted fidelity loss must be less or equal to the
        # max fidelity loss
        test_data = np.hstack([
            test_data,
            (test_data[:, self.columns.index('max_fidelity_loss')]
             >= test_data[:, self.columns.index('total_fidelity_loss')]
             ).reshape(-1, 1)
        ])

        # The real Tests
        # The real measured fidelity measure must be less or equal to the
        # configured mx fidelity loss
        test_data = np.hstack([
            test_data,
            (test_data[:, self.columns.index('real_fidelity_loss')]
             - test_data[:, self.columns.index('max_fidelity_loss')] < 0.1
             ).reshape(-1, 1)
        ])

        # Attention: this is a set of tests that tests how well the CNOT-estimation works
        # However, we do not test this here. As a result, also the prediciton of CNOTs cannot
        # reliably be tested.
        # START: COMMENTED OUT AND LEFT FOR DOCUMENTATION REASONS
        # # The predicted maximum CNOT gates and the no-approximation count
        # # should be within 10%
        # test_data['cnot_prediction_calculation_pass'] = (
        #         np.abs(test_data['real_cnots_benchmark']
        #         - test_data['cnots']) < 0.2 * test_data['cnots']
        # )
        # # The predicted CNOT gates should be within an error margin of 10%
        # test_data['saved_cnots_calculation_pass'] = (
        #         np.abs(test_data['predicted_cnots']
        #         - test_data['real_cnots']) <= 0.2 * (test_data['predicted_cnots'])
        # )
        # END: COMMENTED OUT AND LEFT FOR DOCUMENTATION REASONS

        test_passed = True
        total_experiments = number_of_experiments * max_fidelity_loss.shape[0] * 4
        fails_to_still_pass = int(np.ceil(total_experiments * 0.01))
        benchmark_fidelity_loss_fail_count = (
                test_data.shape[0]
                - test_data[
                    test_data[:, self.columns.index('benchmark_fidelity_loss_pass')].astype(bool)
                ].shape[0]
        )
        approximation_calculation_fail_count = (
                test_data.shape[0]
                - test_data[
                    test_data[:, self.columns.index('approximation_calculation_pass')].astype(bool)
                ].shape[0]
        )
        real_approximation_calculation_fail_count = (
                test_data.shape[0] - test_data[
                test_data[:, self.columns.index('real_approximation_calculation_pass')].astype(bool)
            ].shape[0]
        )
        if benchmark_fidelity_loss_fail_count > 0:
            print(f"[WARNING] NOT ALL benchmark_fidelity_loss_pass are true ({benchmark_fidelity_loss_fail_count})",
                  file=sys.stderr)
        if benchmark_fidelity_loss_fail_count > fails_to_still_pass:
            print("[FAIL] benchmark_fidelity_loss_pass must be true", file=sys.stderr)
            test_passed = False
        if approximation_calculation_fail_count > 0:
            print(f"[WARNING] NOT ALL approximation_calculation_pass are true ({approximation_calculation_fail_count})",
                  file=sys.stderr)
        if approximation_calculation_fail_count > fails_to_still_pass:
            print("[FAIL] approximation_calculation_pass must be true", file=sys.stderr)
            test_passed = False
        if real_approximation_calculation_fail_count > 0:
            print(f"[WARNING] NOT ALL real_approximation_calculation_pass are true "
                  f"({real_approximation_calculation_fail_count})", file=sys.stderr)
        if real_approximation_calculation_fail_count > fails_to_still_pass:
            print("[FAIL] real_approximation_calculation_pass must be true", file=sys.stderr)
            test_passed = False

        import importlib.util
        pandas_loader = importlib.util.find_spec('pandas')
        if pandas_loader is None:
            with np.printoptions(precision=2, linewidth=1000, suppress=True, threshold=sys.maxsize, floatmode='fixed'):
                print(test_data)
        else:
            import pandas as pd
            print(pd.DataFrame(test_data, columns=self.columns).to_string())

        self.assertTrue(test_passed, 'The tests should all pass.')

    # def test_no_ops(self):
    #     # The Test is based on randomly generated states. They all should
    #     # create the correct fidelity (1.0). After 10 attempts we may well
    #     # find one that fails. If it doesn't it is probably okay.
    #     for lower_bound_int in range(10):
    #         state_vector, _, depth = get_vector(
    #             lower_bound_int/10, 1.0, 8, 1, measure='geometric'
    #         )
    #         cnots, depth, fidelity_loss, _ = initialize_loss(0.0, state_vector)
    #         if cnots == depth == fidelity_loss == -1:
    #             continue
    #         self.assertAlmostEqual(0.0, fidelity_loss, 4)

    def test_node_state_vector(self):
        """
        The method Node.state_vector() is an important function for analytics, but it was wrong. The ordering
        of the qubits must be taken into account. One example, visible to the human eye, was a probability
        distribution, the log-normal. For a max-fidelity-loss of 0.09, the ordering is computed with qubits
        [(1,), (3,), (0, 2, 4, 5, 6)] and partitions [None, None, (1, 4)].
        The resulting state has an actual fidelity loss of about 0.0859261062108.
        The fix is to take the ordering into account. Here we test that for this example, the
        fidelity is actually correct.

        """

        from qclib.state_preparation.util import baa

        # geometric measure = 0.11600417225836746
        state = [0.07790067, 0.12411293, 0.10890448, 0.09848761, 0.05027826, 0.05027438,
                 0.03067742, 0.11846638, 0.09868947, 0.10711022, 0.01826258, 0.12535963,
                 0.11613662, 0.05865527, 0.05427737, 0.05451262, 0.07021045, 0.09220849,
                 0.08365776, 0.06869252, 0.09956702, 0.04754114, 0.0688004,  0.07704547,
                 0.08596225, 0.11279128, 0.05687908, 0.09127936, 0.09797266, 0.02743385,
                 0.09921588, 0.05256358, 0.03246542, 0.1239935,  0.12508287, 0.11444701,
                 0.07025331, 0.03978115, 0.10529168, 0.08444882, 0.04446722, 0.08957199,
                 0.02360472, 0.12138094, 0.06475262, 0.10360775, 0.07106703, 0.09179565,
                 0.09411756, 0.05472767, 0.12533861, 0.1120676,  0.12337869, 0.12040975,
                 0.0984252,  0.12221594, 0.03786564, 0.05635093, 0.02707025, 0.07260295,
                 0.07935725, 0.06630651, 0.11587787, 0.07602843, 0.06746749, 0.09377139,
                 0.04778426, 0.11400727, 0.03475503, 0.12645201, 0.11185863, 0.05674245,
                 0.00945899, 0.11494597, 0.10701826, 0.10868207, 0.11178804, 0.03463689,
                 0.07621068, 0.04332871, 0.11825607, 0.10049395, 0.07322158, 0.03209064,
                 0.0709839,  0.07258655, 0.10872672, 0.10163697, 0.11989633, 0.08747055,
                 0.04401971, 0.10750071, 0.11102557, 0.09536318, 0.11176607, 0.08944697,
                 0.09203053, 0.0832302,  0.02029422, 0.04181051, 0.02256621, 0.10154549,
                 0.07136789, 0.0907753,  0.12126382, 0.06355451, 0.08154299, 0.110643,
                 0.06088612, 0.03531675, 0.06851802, 0.05110969, 0.12273343, 0.11442741,
                 0.10130534, 0.11882721, 0.11411204, 0.05498105, 0.12025703, 0.09348119,
                 0.11437924, 0.12049476, 0.07178074, 0.04222706, 0.06077118, 0.08318802,
                 0.11512578, 0.1180934]

        node = baa.adaptive_approximation(state, use_low_rank=True, max_fidelity_loss=0.09, strategy='brute_force')
        fidelity = np.vdot(state, node.state_vector())**2

        self.assertAlmostEqual(node.total_fidelity_loss, 1 - fidelity, places=4)
