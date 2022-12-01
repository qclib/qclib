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

""" Test for entanglement.py module"""

from unittest import TestCase
import time
import numpy as np
from qclib.entanglement import (
    geometric_entanglement,
    schmidt_decomposition,
    schmidt_composition,
    randomized_svd,
    _separation_matrix,
    qb_approximation,
    _undo_separation_matrix
)

class TestEntanglement(TestCase):

    """ Tests for entanglement.py"""
    def test_geometric_measure_entanglement(self):
        """ Test geometric measure of entanglement """
        ghz4 = np.zeros(16)
        ghz4[0] = 1 / np.sqrt(2)
        ghz4[15] = 1 / np.sqrt(2)
        gme = geometric_entanglement(ghz4)
        self.assertTrue(np.isclose(gme, 0.5))

        nqbits = 5
        w6_state = np.zeros(2**nqbits)
        for k in range(nqbits):
            w6_state[2**k] = 1 / np.sqrt(nqbits)

        gme = geometric_entanglement(w6_state)
        self.assertTrue(np.abs(gme - 0.59) < 1e-3)


    def test_schmidt_composition(self):
        state = np.random.rand(2**8) + np.random.rand(2**8) * 1.0j
        state = state / np.linalg.norm(state)
        _, svd_u, svd_s, svd_v = schmidt_decomposition(state, [0, 1, 2])
        svd_s = svd_s / np.linalg.norm(svd_s)

        state_rebuilt = schmidt_composition(svd_u, svd_v, svd_s, [0, 1, 2])

        self.assertTrue(np.allclose(state_rebuilt, state))


    def test_geometric_positive(self):
        """Test if geometric entanglemen is positive"""
        vector = np.array(
            [
                0.00186779 + 0.01306012j,
                -0.01104266 - 0.07721341j,
                -0.01704493 + 0.00199258j,
                0.10077226 - 0.01178041j,
                -0.00188656 - 0.01597529j,
                0.01115364 + 0.09444834j,
                0.01555059 + 0.00254013j,
                -0.09193749 - 0.01501764j,
                -0.00979644 - 0.06849949j,
                0.05791804 + 0.40497947j,
                0.0893996 - 0.01045093j,
                -0.52854411 + 0.06178752j,
                0.00989489 + 0.08378937j,
                -0.05850013 - 0.49537556j,
                -0.08156188 - 0.01332283j,
                0.48220629 + 0.07876658j,
            ]
        )
        gme = geometric_entanglement(vector.tolist())
        self.assertTrue(gme > 0)


    def test_randomized_svd(self):
        print_output = False

        n_qubits = 20

        iters = 3
        rank = 1

        partition_size = round(n_qubits/2.5)+1
        partition = list(range(partition_size))

        random = []
        regular = []
        rnd_qb = []
        for i in range(iters):
            state = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1.0j
            state = state / np.linalg.norm(state)

            sep_matrix = _separation_matrix(n_qubits, state, partition)

            # regular numpy SVD
            start_time = time.time()
            reg_u, reg_s, reg_v = np.linalg.svd(sep_matrix, full_matrices=False)
            reg_u, reg_s, reg_v = reg_u[:, :rank], reg_s[:rank], reg_v[:rank, :]
            reg_approx = schmidt_composition(reg_u, reg_v, reg_s, partition)
            reg_time = (time.time() - start_time)+10**-8

            # randomized truncated SVD
            start_time = time.time()
            rnd_u, rnd_s, rnd_v = randomized_svd(sep_matrix, rank=rank, n_iter=2, over_sampling=12)
            rnd_approx = schmidt_composition(rnd_u, rnd_v, rnd_s, partition)
            rnd_time = (time.time() - start_time)+10**-8

            # randomized QB approximation
            start_time = time.time()
            qb_approx = qb_approximation(sep_matrix, rank=rank, n_iter=3, over_sampling=0)
            qb_approx = _undo_separation_matrix(n_qubits, qb_approx, partition)
            qb_time = (time.time() - start_time)+10**-8

            random.append(rnd_time)
            regular.append(reg_time)
            rnd_qb.append(qb_time)

            if print_output:
                print(
                    f'{i}\t'
                    f'regular={reg_time:.6f}\t'
                    f'random={rnd_time:.6f}\t'
                    f'qb={qb_time:.6f}\t'
                    f'ratio rnd={reg_time/rnd_time:.6f}\t'
                    f'ratio qb={reg_time/qb_time:.6f}'
                )

            self.assertTrue(reg_time > rnd_time)
            self.assertTrue(reg_time > qb_time)
            self.assertTrue(np.allclose(reg_approx, rnd_approx, rtol=1e-04, atol=0.0))
            self.assertTrue(np.allclose(reg_approx, qb_approx, rtol=1e-04, atol=0.0))

        if print_output:
            avg_regular = sum(regular)/iters
            avg_random = sum(random)/iters
            avg_qb = sum(rnd_qb)/iters
            avg_rnd_ratio = sum(np.array(regular)/random)/iters
            avg_qb_ratio = sum(np.array(regular)/rnd_qb)/iters

            print(
                f'avg.\t'
                f'regular={avg_regular:.6f}\t'
                f'random={avg_random:.6f}\t'
                f'qb={avg_qb:.6f}\t'
                f'ratio rnd={avg_rnd_ratio:.6f}\t'
                f'ratio qb={avg_qb_ratio:.6f}'
            )


    def test_randomized_svd_singular_values(self):
        n_qubits = 20

        iters = 3
        rank = 1

        partition_size = round(n_qubits/2.5)+1
        partition = list(range(partition_size))

        for _ in range(iters):
            state = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1.0j
            state = state / np.linalg.norm(state)

            sep_matrix = _separation_matrix(n_qubits, state, partition)

            # regular numpy SVD
            _, regular_s, _ = np.linalg.svd(sep_matrix, full_matrices=False)
            regular_s = regular_s[:rank]

            # randomized truncated SVD
            _, rnd_s, _ = randomized_svd(sep_matrix, rank=rank, n_iter=2*rank**7, over_sampling=12)

            self.assertTrue(np.allclose(rnd_s, regular_s))
