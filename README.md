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

Next, we seek to use Grover's algorithm to distill this superposition down to the desired value of $x'$ (i.e., $x' = x$). Conveniently, the $x$ is fully marked by the corresponding given phase $t(x)$ in the current entangled state. The key is thus to define an oracle operator that targets the $d$-bit state $\ket{2^d t(x)}$. This can be accomplished by using a series of $X$ gates to turn all $d$ dits in that state into a string of 1s and then applying a multi-controlled $Z$ gate using $d-1$ bits as controls and the leftover bit as the target. This will flip the phase of $\ket{2^d t(x)}$, while leaving all other $d$-bit states \ket{2^d t(x')} in the entangled superposition untouched. 

The $X$-gate series can be practically implemented for a given value of $2^d t(x)$, which we label as "tvalueexpanded" as follows. For our example, we will pick $2^d t(x) = 25$, corresponding to $x = 11$ (see "tarray" above).

```python
tvalueexpanded = 25 # represents 2^d*t(x)
```

We define a recursive parameter "numhold" which we initialize at the value of tvalueexpanded. We start by subtracting $2^d$ from numhold. If the result is non-negative, that means the $d^\mathrm{th}$ bit corresponding to the $2^d t(x)$ value is already 1, and we continue on to the next bit with the new value of numhold. On the other hand, if the result is negative, that means that the $d^\mathrm{th}$ bit is 0. In this case, we apply the $X$ gate to flip it to 1, along with reverting the numhold value to the original. We apply the same protocol to the $(d-1)^\mathrm{st}$ bit, $(d-2)^\mathrm{st}$ bit, all the way to the $0^\mathrm{th}$ bit.

```python
numhold = tvalueexpanded

for dindexreversed in range(d):
    dindex = d - dindexreversed - 1
    twoexpdindex = 2**dindex
    numhold = numhold - twoexpdindex
    if numhold < 0:
        qc.x(n+dindex)
        numhold = numhold + twoexpdindex
```

<img width="70" height="563" alt="AfterXGatesCircuit" src="https://github.com/user-attachments/assets/d047b2cf-95eb-4142-ba67-819bc73585e4" />

$\frac{1}{4}\ket{000000}\ket{1001} + \frac{1}{4}\ket{000010}\ket{1110} + \frac{1}{4}\ket{000100}\ket{0111} + \frac{1}{4}\ket{001001}\ket{0011} + \frac{1}{4}\ket{001010}\ket{1101} + \frac{1}{4}\ket{010001}\ket{0110} + ... + \frac{1}{4}\ket{100100}\ket{0101} + \frac{1}{4}\ket{101100}\ket{0000} + \frac{1}{4}\ket{110010}\ket{1000} + \frac{1}{4}\ket{110111}\ket{1100} + \frac{1}{4}\ket{111111}\ket{1011}$

As the diagram shows, the $X$ gates were applied to the 1st, 2nd, and 5th bits of the $d$-bit string (where we define the string indices as ranging from 0 through 5). This comports with the positions of the 0s in the binary string for $t(x) = 25$ (i.e., $\ket{011001}$), which we have now converted to $\ket{111111}$. Consequently, we can see from the last state in the new superposition that the $d$-bit state consisting of all 1s is now entangled with the $x = 11$ (i.e., $\ket{1011}$). 

We now construct the oracle operator using the multi-controlled $Z$ gate on the $d$-bit string, as mentioned above. To do so, we use the general MCPhase gate with the phase set to $\pi$.

```python
from qiskit.circuit.library import PhaseGate
zgate = PhaseGate(np.pi)
czgate = zgate.control(d-1)
qc.append(czgate,range(n,n+d))
```

<img width="72" height="563" alt="AfterMCZCircuit" src="https://github.com/user-attachments/assets/297ec372-6e92-494e-8083-5186f51c25aa" />

$\frac{1}{4}\ket{000000}\ket{1001} + \frac{1}{4}\ket{000010}\ket{1110} + \frac{1}{4}\ket{000100}\ket{0111} + \frac{1}{4}\ket{001001}\ket{0011} + \frac{1}{4}\ket{001010}\ket{1101} + \frac{1}{4}\ket{010001}\ket{0110} + ... + \frac{1}{4}\ket{100100}\ket{0101} + \frac{1}{4}\ket{101100}\ket{0000} + \frac{1}{4}\ket{110010}\ket{1000} + \frac{1}{4}\ket{110111}\ket{1100} - \frac{1}{4}\ket{111111}\ket{1011}$

As desired, the phase associated with the last state in this superposition (i.e., $x = 11$) has flipped, while all other phases are unchanged. 

We now construct the diffusion operator. To do so, we apply the composite operator $A (2\ket{0...0}\bra{0...0} - I) A^†$, where $A$ is the entire set of operations that we used to go from the $(n+d)$-bit vacuum state $\ket{0...0}$ to the entangled superposition upon which we applied the oracle operator.

The operator $2\ket{0...0}\bra{0...0} - I$ is in turn equivalent (up to a global sign flip) to applying the $X$ gate on all bits, then applying an $MCZ$ gate using $n+d-1$ bits as controls and the remaining bit as target (thus flipping the phase corresponding to the state $\ket{1...1}$), and finally reverting the basis states back by applying the $X$ gate on all bits again. The overall effect is to flip the phase of the $\ket{0...0}$ basis state in a given superposition while leaving the coefficients for all other basis states unchanged. For this, we define the necessary $MCZ$ gate as follows:

```python
czgate2 = zgate.control(n+d-1)
```

We thus break the diffusion operator into individual steps, starting with $A^†$:

```python
numhold = tvalueexpanded
for dindexreversed in range(d):
    dindex = d - dindexreversed - 1
    twoexpdindex = 2**dindex
    numhold = numhold - twoexpdindex
    if numhold < 0:
        qc.x(n+dindex)
        numhold = numhold + twoexpdindex
qc.append(iqft.inverse(),range(n,n+d))
for dindexreversed in range(d):
    dindex = d - dindexreversed - 1
    ugateinvraised = ugateinv**(2**dindex)
    cugateinvraised = ugateinvraised.control(1)
    qc.append(cugateinvraised,[n+dindex] + list(range(n)))
qc.h(range(n+d))
```

Next, we apply $I - 2\ket{0...0}\bra{0...0}$ using the protocol laid out above:

```python
qc.x(range(n+d))
qc.append(czgate2,range(n+d))
qc.x(range(n+d))
```

Finally, we apply $A$:

```python
qc.h(range(n+d))
for dindex in range(d):
    ugateraised = ugate**(2**dindex)
    cugateraised = ugateraised.control(1)
    qc.append(cugateraised,[n+dindex] + list(range(n)))
qc.append(iqft,range(n,n+d))
numhold = tvalueexpanded
for dindexreversed in range(d):
    dindex = d - dindexreversed - 1
    twoexpdindex = 2**dindex
    numhold = numhold - twoexpdindex
    if numhold < 0:
        qc.x(n+dindex)
        numhold = numhold + twoexpdindex
```

<img width="3522" height="1329" alt="GroverCycle" src="https://github.com/user-attachments/assets/4981d876-12f0-440f-b9d5-c34bc2000e7a" />

$-\frac{3}{16}\ket{000000}\ket{1001} - \frac{3}{16}\ket{000010}\ket{1110} - \frac{3}{16}\ket{000100}\ket{0111} - \frac{3}{16}\ket{001001}\ket{0011} - \frac{3}{16}\ket{001010}\ket{1101} - \frac{3}{16}\ket{010001}\ket{0110} - ... - \frac{3}{16}\ket{100100}\ket{0101} - \frac{3}{16}\ket{101100}\ket{0000} - \frac{3}{16}\ket{110010}\ket{1000} - \frac{3}{16}\ket{110111}\ket{1100} - \frac{11}{16}\ket{111111}\ket{1011}$

In the above diagram, Uinv represents the inverse of U. As the state coefficients show, the state corresponding to $x = 11$ has been dramatically amplified in the composite superposition. We run the oracle-diffusion cycles for the total number of $2\phi$ rotations (where $\phi = \textrm{sin}^{-1}(1/\sqrt{2^n}) \sim 1/\sqrt{2^n}$) required to get the phase angle as close to $\pi/2$ as possible. This is defined by "numgrovercycles" below:

```python
numgrovercycles = round(np.pi/(4*np.arcsin(1/2**(n/2))) - 1/2)
```

For our case, this amounts to 3 total cycles. Since we have already completed 1 cycle above, we run another numgrovercycles-1 cycles (2 in our example) using a for loop:

```python
for cycle in range(1,numgrovercycles):
    
    qc.append(czgate,range(n,n+d))
    
    numhold = tvalueexpanded
    for dindexreversed in range(d):
        dindex = d - dindexreversed - 1
        twoexpdindex = 2**dindex
        numhold = numhold - twoexpdindex
        if numhold < 0:
            qc.x(n+dindex)
            numhold = numhold + twoexpdindex
    qc.append(iqft.inverse(),range(n,n+d))
    for dindexreversed in range(d):
        dindex = d - dindexreversed - 1
        ugateinvraised = ugateinv**(2**dindex)
        cugateinvraised = ugateinvraised.control(1)
        qc.append(cugateinvraised,[n+dindex] + list(range(n)))
    qc.h(range(n+d))
    
    qc.x(range(n+d))
    qc.append(czgate2,range(n+d))
    qc.x(range(n+d))
    
    qc.h(range(n+d))
    for dindex in range(d):
        ugateraised = ugate**(2**dindex)
        cugateraised = ugateraised.control(1)
        qc.append(cugateraised,[n+dindex] + list(range(n)))
    qc.append(iqft,range(n,n+d))
    numhold = tvalueexpanded
    for dindexreversed in range(d):
        dindex = d - dindexreversed - 1
        twoexpdindex = 2**dindex
        numhold = numhold - twoexpdindex
        if numhold < 0:
            qc.x(n+dindex)
            numhold = numhold + twoexpdindex
```

$0.05078125\ket{000000}\ket{1001} + 0.05078125\ket{000010}\ket{1110} + 0.05078125\ket{000100}\ket{0111} + 0.05078125\ket{001001}\ket{0011} + 0.05078125\ket{001010}\ket{1101} + 0.05078125\ket{010001}\ket{0110} + ... + 0.05078125\ket{100100}\ket{0101} + 0.05078125\ket{101100}\ket{0000} + 0.05078125\ket{110010}\ket{1000} + 0.05078125\ket{110111}\ket{1100} - 0.98046875\ket{111111}\ket{1011}$

We finish by measuring the rightmost $n$ bits (in our example, 4 bits). There is a 96% probability that we will get the correct answer $\ket{1011}$, i.e., $x = 11$.

More generally, we note that since each Grover cycle rotates the composite state by a $2\phi$ step, the optimal number of cycles will yield a state with a maximum separation of $\phi$ from the ideal angle $\pi/2$. Since $\phi = \textrm{sin}^{-1}(1/\sqrt{2^n})$, this means that in the final superposition, the overall amplitude of the "wrong" states is no more than $1/\sqrt{2^n}$, resulting in a measurement error probability of no more than $1/2^n$ (i.e., the measurement error is $O(1/2^n)$, as desired).
