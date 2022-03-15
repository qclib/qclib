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
from qiskit import execute
from qiskit.providers.aer.backends import AerSimulator
from qclib.state_preparation.schmidt import initialize

# Generate 3-qubit random input state vector
n = 3
rnd = np.random.RandomState(42)
input_state = rnd.rand(2**n) + rnd.rand(2**n) * 1j
input_state = input_state/np.linalg.norm(input_state)

# Build a quantum circuit to initialize the input state
circuit = initialize(input_state)
circuit.measure_all()

# Construct an ideal simulator
backend = AerSimulator()

# Perform an ideal simulation
result = execute(circuit, backend).result()
counts = result.get_counts(0)
print('Counts:', counts)
#Counts: {'101': 15, '010': 114, '110': 5, '100': 128, '001': 270, '000': 91, '111': 147, '011': 254}
```