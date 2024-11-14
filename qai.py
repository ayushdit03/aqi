
"""#### New way"""

from qiskit import QuantumCircuit, transpile
from qiskit import QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

q = QuantumRegister(16, 'q')
c = ClassicalRegister(16, 'c')
circuit = QuantumCircuit(q, c)

circuit.h(q)
circuit.draw()
circuit.measure(q, c)
circuit.draw()

simulator = AerSimulator()

transpiled_circuit = transpile(circuit, simulator)
job = simulator.run(transpiled_circuit, shots=1)

print("Executing Job.....")
result = job.result()

counts = result.get_counts(circuit)
print("Result: ", counts)

"""## Prac-2"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit_aer.noise import NoiseModel,depolarizing_error

qc=QuantumCircuit(3,3)

qc.h(0)
qc.cx(0,1)
qc.cx(0,2)
qc.measure([0,1,2],[0,1,2])

noise_model = NoiseModel()

# Add depolarizing noise for single-qubit gates and two-qubit gates
depol_1q = depolarizing_error(0.01, 1)
depol_2q = depolarizing_error(0.02, 2)
noise_model.add_all_qubit_quantum_error(depol_1q, ['u3', 'x', 'h'])
noise_model.add_all_qubit_quantum_error(depol_2q, ['cx'])

# Use AerSimulator and noise model for simulation
backend = AerSimulator()

# Execute the quantum circuit with noise model
result_with_noise = backend.run(qc,noise_model=noise_model, shots=1024).result()

# Get the raw counts with noise
noisy_counts = result_with_noise.get_counts(qc)

# Simple mitigation technique: scale counts based on expected noise
mitigated_counts = {key: noisy_counts[key] * (1 - 0.02) for key in noisy_counts}

# Plot the mitigated results
plot_histogram(mitigated_counts)

print("Original counts:")
print(noisy_counts)

print("Mitigated counts:")
print(mitigated_counts)

plot_histogram([noisy_counts, mitigated_counts], legend=['Original', 'Mitigated'])



"""## Prac-3"""

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister,transpile
from qiskit.circuit.library import GroverOperator
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import seaborn as sns

def create_3_puzzle_circuit():
    """Create the quantum circuit for solving the 3-puzzle problem using Grover's algorithm."""
    # Initialize quantum and classical registers
    qr = QuantumRegister(3, 'q')
    cr = ClassicalRegister(3, 'c')
    qc = QuantumCircuit(qr, cr)

    # Initial state |000>
    # CNOT gates to exchange positions of qubits
    qc.cx(qr[0], qr[1])
    qc.cx(qr[1], qr[2])

    # Apply Hadamard gates to create superposition
    qc.h(qr[0])
    qc.h(qr[1])
    qc.h(qr[2])

    # Placeholder for Grover's algorithm
    # This requires defining an oracle and the Grover operator
    # Here we use a simple placeholder to illustrate the concept
    oracle = QuantumCircuit(3)
    oracle.z(0)  # Example oracle condition (adjust based on the puzzle's rules)
    oracle_gate = oracle.to_gate(label="Oracle")

    # Grover operator
    grover_operator = GroverOperator(oracle)
    qc.append(grover_operator, qr)

    # Measurement
    qc.measure(qr, cr)
    return qc

# Create the 3-puzzle quantum circuit
qc = create_3_puzzle_circuit()

# Execute the circuit on a quantum simulator
simulator = AerSimulator()
transpiled_qc = transpile(qc, simulator)

job = simulator.run(transpiled_qc, shots=1024)
result = job.result()
counts = result.get_counts()

# Output the result
print("Measurement results from the 3-puzzle quantum circuit:")
print(counts)

sns.barplot(x=list(counts.keys()), y=list(counts.values()))
plt.xlabel('Measurement Outcomes')
plt.ylabel('Counts')
plt.title('3-Puzzle Quantum Circuit Measurement Results')
plt.show()



"""## Prac-4"""

from qiskit.circuit import QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit import transpile

circuit = QuantumCircuit(3, 3)

circuit.x(0)
circuit.barrier(range(3))
circuit.h(1)
circuit.cx(1, 2)
circuit.cx(0, 1)
circuit.h(0)
circuit.barrier(range(3))
circuit.measure(range(2), range(2))
circuit.barrier(range(3))
circuit.cx(1, 2)
circuit.cz(0, 2)
circuit.draw()

backend = AerSimulator()
qc_compiled = transpile(circuit, backend)

job_sim = backend.run(qc_compiled, shots=1024)
result_sim = job_sim.result()

counts = result_sim.get_counts(qc_compiled)
print(counts)

plot_histogram(counts)



"""## Prac-5"""

import numpy as np
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit,transpile
from qiskit_aer import AerSimulator

from qiskit.circuit.library import QFT

# Set up the AerSimulator backend
simulator = AerSimulator()

q = QuantumRegister(5,'q')
c = ClassicalRegister(5,'c')
circuit = QuantumCircuit(q,c)

circuit = QuantumCircuit(q, c)
circuit.h(q)

# Apply X gates to specific qubits (as per your original code)
circuit.x(q[4])
circuit.x(q[2])
circuit.x(q[0])

# Apply the Quantum Fourier Transform (QFT)
qft_circuit = QFT(num_qubits=5, approximation_degree=0, do_swaps=True, inverse=False, insert_barriers=False, name="qft")
circuit.append(qft_circuit, q)

circuit = circuit.compose(qft_circuit)
circuit.measure(q,c) # Measure the qubits and store the result in classical register
circuit.draw()

transpiled_circuit = transpile(circuit, simulator)
job = simulator.run(transpiled_circuit,shots=1000)

print("Job is running...")
print(f"Final job status: {job.status()}")

job_result = job.result()
counts = job_result.get_counts()
print("\n QFT Output")
print("-------------")
print(counts)

