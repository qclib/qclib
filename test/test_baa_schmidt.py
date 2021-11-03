import datetime
import logging
from typing import List

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
from qiskit.providers import aer

from qclib.state_preparation import schmidt
from qclib.state_preparation.baa_schmidt import adaptive_approximation, initialize
from qclib.state_preparation.util.entanglement_measure import calculate_entropy_meyer_wallach, compute_Q_ptrace

logging.basicConfig(format='%(asctime)s::' + logging.BASIC_FORMAT, level='ERROR')
LOG = logging.getLogger(__name__)


def calculate_state(vectors: List[np.ndarray]):
    state = np.ones(1)
    for p in vectors:
        state = np.kron(p, state)
    return state


# TODO: do tests!!!
if __name__ == "__main__":
    exp_time_start = datetime.datetime.now()
    LOG.setLevel('INFO')
    num_qubits = 4
    mw_limit_lower = 0.1
    mw_limit_upper = 0.3
    for _ in range(10):
        mw = -1.0
        while mw < mw_limit_lower or mw > mw_limit_upper:
            qc: qiskit.QuantumCircuit = random_circuit(num_qubits, 2*num_qubits)
            job: aer.AerJob = qiskit.execute(qc, backend=aer.StatevectorSimulator())
            vector = job.result().get_statevector()
            mw = compute_Q_ptrace(vector)
            assert abs(mw - calculate_entropy_meyer_wallach(vector)) < 1e-3

        LOG.debug(f"The Circuit\n{qc.draw(fold=-1)}")
        LOG.debug(f"Vector: {np.linalg.norm(vector)}\n {vector}")
        LOG.debug(f"Meyer-Wallach: {mw}.")

        start = datetime.datetime.now()
        max_fidelity_loss = 0.1
        node = adaptive_approximation(vector, max_fidelity_loss)
        end = datetime.datetime.now()

        expected_state = calculate_state(node.vectors)
        expected_overlap = np.abs(np.vdot(vector, expected_state)) ** 2

        qc = initialize(vector, max_fidelity_loss)
        qc_benchmark = schmidt.initialize(vector)
        qc_basis = qiskit.transpile(qc, basis_gates=['u', 'cx'])
        qc_benchmark = qiskit.transpile(qc_benchmark, basis_gates=['u', 'cx'])

        needed_cnots = qc_basis.num_nonlocal_gates()
        needed_cnots_benchmark = qc_benchmark.num_nonlocal_gates()

        LOG.info(f'State Preparation (MW: {mw}) has now {needed_cnots} from {needed_cnots_benchmark} CNOT-gates '
                 f'with max fidelity loss {max_fidelity_loss}. ({end - start})')
        pass
        # if node is None:
        #     LOG.info(f'[{max_fidelity_loss}] No approximation could be found (MW: {mw}). ({end - start})')
        # else:
        #     moettoenen_sp = sum([2**n for n in range(1, num_qubits)])
        #     LOG.info(f'[{max_fidelity_loss}] With fidelity loss {node.fidelity_loss} (MW: {mw}) we can '
        #              f'save {node.cnot_saving} of {sp_cnots(num_qubits)} (Moettonen:{moettoenen_sp}) CNOT-gates. ({end - start})')
