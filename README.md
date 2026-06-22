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

$\ket{0000000000}$

We apply the Hadamard gate on all bits, yielding the superposition $\frac{1}{\sqrt{2^{n+d}}} \sum_{t',x'} |2^d*t'> |x'>$, where the $t'$ values represent all possible "phase slots" (i.e., all values in the range $2^{-d} \times \textrm{(}0,1,2,...,2^d-1$)).

```python
qc.h(range(n+d))
```

<img width="176" height="689" alt="AllHadamardCircuit" src="https://github.com/user-attachments/assets/01535f7f-4b7e-41bb-96a6-97b8b2706603" />

$\frac{1}{32}\ket{000000}\ket{0000} + \frac{1}{32}\ket{000000}\ket{0001} + \frac{1}{32}\ket{000000}\ket{0010} + \frac{1}{32}\ket{000000}\ket{0011} + \frac{1}{32}\ket{000000}\ket{0100} + \frac{1}{32}\ket{000000}\ket{0101} + ... + \frac{1}{32}\ket{111111}\ket{1011} + \frac{1}{32}\ket{111111}\ket{1100} + \frac{1}{32}\ket{111111}\ket{1101} + \frac{1}{32}\ket{111111}\ket{1110} + \frac{1}{32}\ket{111111}\ket{1111}$

Therefore, for each $\ket{x'}$, this state features a superposition of all numerically possible phases. However, the goal is to create an entangled state where each $\ket{x'}$ is only associated with the corresponding phase $\ket{t(x')}$. To achieve this, we apply the quantum phase estimation method. Specifically, we want to map $\ket{2^d t'}\ket{x'} \rightarrow e^{i 2\pi t(x') (2^d t')} \ket{2^d t'}\ket{x'}$, which is achieved by applying the operator $U^{2^d t'} = \prod_{l=0}^{d-1} C_{n+l} U^{2^l}$, where we apply the operator on the first $n$ bits while using the next $d$ bits individually as control qubits.

```python
for dindex in range(d):
    ugateraised = ugate**(2**dindex)
    cugateraised = ugateraised.control(1)
    qc.append(cugateraised,[n+dindex] + list(range(n)))
```

<img width="443" height="563" alt="AfterControlUCircuit" src="https://github.com/user-attachments/assets/427f8815-30a9-4b92-a144-623a7eb253e4" />

$\frac{1}{32}\ket{000000}\ket{0000} + \frac{1}{32}\ket{000000}\ket{0001} + \frac{1}{32}\ket{000000}\ket{0010} + \frac{1}{32}\ket{000000}\ket{0011} + \frac{1}{32}\ket{000000}\ket{0100} + \frac{1}{32}\ket{000000}\ket{0101} + ... + \frac{1}{32}e^{i 2\pi (63/64) 25}\ket{111111}\ket{1011} + \frac{1}{32}e^{i 2\pi (63/64) 17}\ket{111111}\ket{1100} + \frac{1}{32}e^{i 2\pi (63/64) 44}\ket{111111}\ket{1101} + \frac{1}{32}e^{i 2\pi (63/64) 36}\ket{111111}\ket{1110} + \frac{1}{32}e^{i 2\pi (63/64) 52}\ket{111111}\ket{1111}$

The key here is that for each $\ket{x'}$, there is no longer a symmetric superposition of all $\ket{2^d t'}$, but rather a phase-weighted superposition, corresponding to the composite state $\sum_{t',x'} e^{i (2\pi/2^d) (2^d t(x')) (2^d t')} \ket{2^d t'} \ket{x'}$. Conceptually, the fact that consecutive integer states $\ket{2^d t'} \in (\ket{0},...,\ket{2^d - 1})$ are separated by a well-defined phase (invariant in the particular integer pair), while the amplitudes for the integer states are uniform, clearly demonstrates that this composite state is the QFT of $\sum_{x'} \ket{2^d t(x')} \ket{x'}$, where every $x'$ is entangled with the corresponding phase $t(x')$, as desired. We thus apply the inverse QFT on the $d$-bit system to achieve this entangled state.

```python
from qiskit.circuit.library import QFT
iqft = QFT(num_qubits=d, inverse=True)
qc.append(iqft,range(n,n+d))
```

<img width="79" height="563" alt="AfterIQFTCircuit" src="https://github.com/user-attachments/assets/6f15f600-f7e9-41fb-b94d-581b25dec80d" />

$\frac{1}{4}\ket{000010}\ket{0101} + \frac{1}{4}\ket{000101}\ket{0001} + \frac{1}{4}\ket{000111}\ket{0010} + \frac{1}{4}\ket{001010}\ket{0000} + \frac{1}{4}\ket{010001}\ket{1100} + \frac{1}{4}\ket{010100}\ket{1000} + ... + \frac{1}{4}\ket{101111}\ket{0011} + \frac{1}{4}\ket{110100}\ket{1111} + \frac{1}{4}\ket{110111}\ket{0110} + \frac{1}{4}\ket{111101}\ket{1010} + \frac{1}{4}\ket{111110}\ket{0100}$

Note the coefficient amplitude has changed from $1/\sqrt{2^{n+d}}$ to $1/\sqrt{2^n}$. This corresponds to the fact that for each $\ket{x'}$, the $\ket{t'}$ superposition collapsed to the well-defined state $\ket{t(x')}$.

## Grover's Algorithm: Distilling the Value of x
