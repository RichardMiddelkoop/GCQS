from qiskit import QuantumCircuit
import qiskit.circuit as qc
from qiskit import QuantumRegister
from qiskit_ibm_provider import IBMProvider
from qiskit import transpile
import numpy as np
import math

# TODO: possibly include more gates (like full layer gates)
def gate_encoding(circuit, gene, nr_of_qubits, complexity):
    max_params = 3
    # first six bits of string define gate time, rest define which qubits to apply to
    gate_string = gene[:6]
    # based on the qubit string a permutation of the qubits is made, which is
    # used in the gate(a single bit gate only takes the first, a two bit gate)
    # takes the first two etc.
    qubit_string = gene[6:]
    qubit_seed = int(qubit_string, 2)
    qubits = np.random.RandomState(seed=qubit_seed).permutation(nr_of_qubits)
    # TODO: decide on the parameter initialisation
    # sample max_params times a parameter between [0,2pi) based on the seed value
    parameters = 2*math.pi*np.random.RandomState(seed=qubit_seed).random_sample((max_params,))

    ## Single Qubit gates
    if gate_string == "000000":
        # Pauli-X
        circuit.x(qubit=qubits[0])
        return circuit, complexity+1
    if gate_string == "000001":
        # originally u1 gate
        circuit.p(parameters[0],qubit=qubits[0])
        return circuit, complexity+2
    if gate_string == "000010":
        # originally u2 gate
        circuit.u(math.pi/2,parameters[0],parameters[1],qubit=qubits[0])
        return circuit, complexity+2
    if gate_string == "000011":
        # originally u3 gate
        circuit.u(parameters[0],parameters[1],parameters[2],qubit=qubits[0])
        return circuit, complexity+2
    if gate_string == "000100":
        # Pauli-Y
        circuit.y(qubit=qubits[0])
        return circuit, complexity+1
    if gate_string == "000101":
        # Pauli-Z
        circuit.z(qubit=qubits[0])
        return circuit, complexity+1
    if gate_string == "000110":
        # Hadamard
        circuit.h(qubit=qubits[0])
        return circuit, complexity+1
    if gate_string == "000111":
        # S gate
        circuit.s(qubit=qubits[0])
        return circuit, complexity+1
    if gate_string == "001000":
        # S conjugate gate
        circuit.sdg(qubit=qubits[0])
        return circuit, complexity+1
    if gate_string == "001001":
        # T gate
        circuit.t(qubit=qubits[0])
        return circuit, complexity+1
    if gate_string == "001010":
        # T conjugate gate
        circuit.tdg(qubit=qubits[0])
        return circuit, complexity+1
    if gate_string == "001011":
        # rx gate
        circuit.rx(parameters[0],qubit=qubits[0])
        return circuit, complexity+2
    if gate_string == "001100":
        # ry gate
        circuit.ry(parameters[0],qubit=qubits[0])
        return circuit, complexity+2
    if gate_string == "001101":
        # rz gate
        circuit.rz(parameters[0],qubit=qubits[0])
        return circuit, complexity+2
    ## Multi Qubit gates
    if gate_string == "100000":
        # Controlled NOT gate
        circuit.cx(qubits[0],qubits[1])
        return circuit, complexity+2
    if gate_string == "100001":
        # Controlled Y gate
        circuit.cy(qubits[0],qubits[1])
        return circuit, complexity+2
    if gate_string == "100011":
        # Controlled Z gate
        circuit.cz(qubits[0],qubits[1])
        return circuit, complexity+2
    if gate_string == "100100":
        # Controlled H gate
        circuit.ch(qubits[0],qubits[1])
        return circuit, complexity+2
    if gate_string == "100101":
        # Controlled rotation Z gate
        circuit.crz(parameters[0],qubits[0],qubits[1])
        return circuit, complexity+3
    if gate_string == "100110":
        # Controlled phase rotation gate
        circuit.cp(parameters[0],qubits[0],qubits[1])
        return circuit, complexity+3
    if gate_string == "100111":
        # SWAP gate
        circuit.swap(qubits[0],qubits[1])
        return circuit, complexity+2
    if gate_string == "101000":
        # Toffoli gate
        circuit.ccx(qubits[0],qubits[1],qubits[2])
        return circuit, complexity+4
    if gate_string == "101001":
        # controlled swap gate
        circuit.cswap(qubits[0],qubits[1],qubits[2])
        return circuit, complexity+4       
    else:
        return circuit, complexity
    
def genome_to_circuit(genome, nr_of_qubits, nr_of_gates):
    qr = QuantumRegister(nr_of_qubits)
    circuit = QuantumCircuit(qr)
    total_complexity = 0
    gene_length = len(genome)//nr_of_gates
    for i in range(nr_of_gates):
        gene = genome[i * gene_length : (i+1) * gene_length]
        circuit, total_complexity = gate_encoding(circuit, gene, nr_of_qubits, total_complexity)
    return circuit, total_complexity

def configure_circuit_to_backend(circuit, backend):

    provider = IBMProvider()
    backend = provider.get_backend(backend)
    circuit_basis = transpile(circuit, backend=backend)
    return circuit_basis

def get_circuit_properties(circuit, backend):
    complexity = 0
    circuit_error = 0
    provider = IBMProvider()
    backend = provider.get_backend(backend)
    for gate in circuit.data:
        if gate.operation.name == "cx":
            cx_bits = [int(gate.qubits[0]._index), int(gate.qubits[1]._index)]
            circuit_error += backend.properties().gate_error('cx',cx_bits)
            complexity += 2



    return complexity, circuit_error

circuit, _ = genome_to_circuit("100011101111010010110000011010001101111101011000011101101100001011",5,6)

a, b = get_circuit_properties(configure_circuit_to_backend(circuit, 'ibm_perth'), 'ibm_perth')
print("complexity = ",a," circuit_error = ", b)