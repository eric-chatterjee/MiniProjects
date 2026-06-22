# Mapping Phase to Eigenstate Through Quantum Entanglement

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.quantum_info import Statevector
import numpy as np
import matplotlib as mpl
```

## Goal Statement

We are given a set of states $|x'> \in {|0>,|1>,...|2^n - 1>}$, where each state is an eigenstate of the operator $U$, with the eigenvalue being a unique phase $e^{i 2\pi \theta(x')} = e^{i 2\pi t(x')}$, where $t(x') \in [0,1)$ and $2^d \theta(x')$ is an integer (for a particular positive integer $d$) for any x'. The goal is to find the state $\ket{x'} = \ket{x}$, given $t(x)$.

## Establishing Inputs

For our example, we use $n = 4$ (corresponding to 16 possible basis states $\ket{x'}$).

```python
n = 4
```

The core mechanism is to ultimately entangle each $x'$ with the corresponding $t(x')$. To that end, we need to add the number of ancillary bits necessary for representing all possible values of $t$. Since $2^d \theta(x')$ is an integer for each value of x',  and since $\theta(x')$ and $t(x')$ differ by an integer, we deduce that $2^d t(x')$ is an integer for every $x'$. Furthermore, the fact that $t \in [0,1)$ implies that $2^d t \in [0,2^d)$, meaning that we can represent every value of $t(x')$ with $d$ bits.

To determine the lower bound for $d$, we consider the fact that there are $2^n$ unique values of $t$ and $2^d$ available slots for those values. Consequently, $2^d >= 2^n$, implying that $d >= n$. We choose $d = 6$.

```python
d = 6
```

We thus randomly choose $t(x')$ values for all $x'$ from 0 to 15, with the condition being that each $t(x')$ is an integer multiple of $1/2^d$ with the integer being in the range 0 through $2^d-1$ (i.e., from 0 through 63 given $d = 6$).

```python
tarray = 1/2**d*np.array([10, 5, 7, 47, 62, 2, 55, 34, 20, 38, 61, 25, 17, 44, 36, 52])
```

We convert this array to a phase array and then to the diagonal gate $U$ (where the basis states are $(\ket{0},...,\ket{2^n - 1}))$.

```python
phasearray = np.exp(1j*2*np.pi*tarray)
from qiskit.circuit.library import DiagonalGate
ugate = DiagonalGate(phasearray)
ugate.name = "U"
ugateinv = ugate.inverse()
ugateinv.name = "Uinv"
```

Note that we have defined $U^{-1}$ using the name "Uinv". We will use this later when establishing the diffusion operator for Grover's Algorithm.

## Quantum Phase Estimation: Preparing the Entangled State

We now consider the quantum circuit itself. We initialize $n+d$ bits (where we aim to use the first $n$ bits and the next $d$ bits to represent $x'$ and $2^d t(x')$, respectively).

```python
qc = QuantumCircuit(n+d)
```

<img width="144" height="689" alt="InitializedCircuit" src="https://github.com/user-attachments/assets/ce9f3a5f-8d62-436e-95d3-c303f483c375" />

<img width="105" height="27" alt="StateAfterInitializedCircuit" src="https://github.com/user-attachments/assets/bad5b488-1cc4-4045-afa1-2d7810495e62" />

We apply the Hadamard gate on all bits, yielding the superposition $\frac{1}{\sqrt{2^{n+d}}} \sum_{t',x'} |2^d*t'> |x'>$, where the $t'$ values represent all possible "phase slots" (i.e., all values in the range $2^{-d} \times \textrm{(}0,1,2,...,2^d-1$)).
