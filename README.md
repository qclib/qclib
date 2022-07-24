# Quantum computing library (qclib)
Qclib is a quantum computing library implemented using qiskit.

The focus of qclib is on preparing quantum states, but it is not limited to that.

## Instalation
The easiest way of installing qclib is by using pip:

```python
pip install qclib
``` 

## Initializing your first quantum state with qclib
Now that qclib is installed, you can start building quantum circuits to prepare quantum states. Here is a basic example:

```
$ python
```

```python
import numpy as np
from qiskit import transpile
from qiskit.providers.aer.backends import AerSimulator
from qclib.state_preparation import LowRankInitialize

# Generate 3-qubit random input state vector
n = 3
rnd = np.random.RandomState(42)
input_vector = rnd.rand(2 ** n) + rnd.rand(2 ** n) * 1j
input_vector = input_vector / np.linalg.norm(input_vector)

# Build a quantum circuit to initialize the input vector
circuit = LowRankInitialize(input_vector).definition

# Construct an ideal simulator
backend = AerSimulator()

# Tests whether the produced state vector is equal to the input vector.
t_circuit = transpile(circuit, backend)
t_circuit.save_statevector()
state_vector = backend.run(t_circuit).result().get_statevector()
print('Equal:', np.allclose(state_vector, input_vector))
# Equal: True
```

## Comparing algorithms for quantum state preparation
The following table shows the depth, number of qubits, and CNOTs of circuits produced by different state preparation methods for the same random 15-qubit state.

| method | lib | qubits | cnots | depth |
| --- | --- |:---:|:---:|:---:|
| low-rank | qclib | 15 | 30999 | 53645 |
| svd | qclib | 15 | 38813 | 71581 |
| isometry | qiskit | 15 | 32752 | 131011 |
| multiplexor | qiskit | 15 | 65504 | 131025 |
| bdsp | qclib | 1151 | 72320 | 1603 |
| dcsp | qclib | 32767 | 262016 | 899 |

You can reproduce the results [here](example/comparison.ipynb).

## Authors
The first version of qclib was developed at [Centro de Informática](https://portal.cin.ufpe.br) - UFPE.
Qclib is an active project, and [other people](https://github.com/qclib/qclib/graphs/contributors) have contributed.

If you are doing research using qclib, please cite our project.
We use a [CITATION.cff](https://citation-file-format.github.io/) file, so you can easily copy the citation information from the repository landing page.

## License
qclib is **free** and **open source**, released under the Apache License, Version 2.0.
