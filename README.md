# Quantum computing library (qclib)

Qclib is a quantum computing library implemented using qiskit.

## Instalation
The easiest way of installing Qclib is by using pip:

```python
pip install qclib
``` 

## Initializing your first quantum state with Qclib
Now that Qclib is installed, you can start building quantum circuits to prepare quantum states. Here is a basic example:

```
$ python
```

```python
import numpy as np
from qiskit import transpile
from qiskit.providers.aer.backends import AerSimulator
from qclib.state_preparation.schmidt import initialize

# Generate 3-qubit random input state vector
n = 3
rnd = np.random.RandomState(42)
input_vector = rnd.rand(2**n) + rnd.rand(2**n) * 1j
input_vector = input_vector/np.linalg.norm(input_vector)

# Build a quantum circuit to initialize the input vector
circuit = initialize(input_vector)

# Construct an ideal simulator
backend = AerSimulator()

# Tests whether the produced state vector is equal to the input vector.
t_circuit = transpile(circuit, backend)
t_circuit.save_statevector()
state_vector = backend.run(t_circuit).result().get_statevector()
print('Equal:', np.allclose(state_vector, input_vector))
#Equal: True
```