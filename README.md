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

For our example, we use $n = 4$ (corresponding to 16 possible basis states $\ket{x}$).

```python
n = 4
```

The core mechanism is to ultimately entangle each $x$ with the corresponding $t(x)$. To that end, we need to add the number of 
ancillary bits necessary for representing all possible values of $t$. Since $2^d \theta(x)$ is an integer for each value of x, 
and since $\theta(x)$ and $t(x)$ differ by an integer, we deduce that $2^d t(x)$ is an integer for every $x$. Furthermore, the
fact that $t \in [0,1)$ implies that $2^d t \in [0,2^d)$, meaning that we can represent every value of $t(x)$ with $d$ bits.

To determine the lower bound for $d$, we consider the fact that there are $2^n$ unique values of t and $2^d$ available slots for
those values. Consequently, $2^d >= 2^n$, implying that $d >= n$. We choose $d = 6$.
