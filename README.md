# Mapping Phase to Eigenstate Through Quantum Entanglement

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.quantum_info import Statevector
import numpy as np
import matplotlib as mpl
```

We are given a set of states $|x> \in {|0>,|1>,...|2^n - 1>}$, where each state is an eigenstate of the operator $U$, with the 
eigenvalue being a unique phase $e^{i 2\pi \theta(x)} = e^{i 2\pi t(x)}$, where $t(x) \in [0,1)$ and $2^d \theta(x)$ is an
integer (for a particular positive integer $d$) for any x. The goal is to find the state $\ket{x} = \ket{x_0}$, given $t(x_0)$.

We use $n = 4$ (corresponding to 16 possible basis states $\ket{x}$).
