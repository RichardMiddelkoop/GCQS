from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.circuit import CircuitInstruction
from qiskit import QuantumRegister
import numpy as np

def gate_encoding(gene, nr_of_qubits):
    # first six bits of string define gate time, rest define which qubits to apply to
    complexity = 0
    gate_string = gene[:6]
    # based on the qubit string a permutation of the qubits is made, which is
    # used in the gate(a single bit gate only takes the first, a two bit gate)
    # takes the first two etc.
    qubit_string = gene[6:]
    qubit_seed = int(qubit_string, 2)
    qubits = np.random.RandomState(seed=qubit_seed).permutation(nr_of_qubits)
    temp_circuit = QuantumCircuit(nr_of_qubits)

    if gate_string == "000000":
        temp_circuit.x(qubit=qubits[0])
        print(qubits)
        return temp_circuit.to_instruction(), [qubits], complexity
    else:
        temp_circuit = None
        return temp_circuit
    
def genome_to_circuit(genome, nr_of_qubits, nr_of_gates):
    qr = QuantumRegister(nr_of_qubits)
    circuit = QuantumCircuit(qr)
    total_complexity = 0
    gene_length = len(genome)//nr_of_gates
    for i in range(nr_of_gates):
        gene = genome[i * gene_length : (i+1) * gene_length]
        gate, qubits, complexity = gate_encoding(gene, nr_of_qubits)
        if gate is not None:
            circuit.append(gate, [qr[bits] for bits in qubits])
            total_complexity += complexity
    circuit.draw()
    return

print(gate_encoding("00000000000",5))
print(genome_to_circuit("00000000000",5,1))