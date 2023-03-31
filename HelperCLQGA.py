from qiskit import QuantumCircuit
import qiskit.circuit as qc
from qiskit.circuit import Parameter
from qiskit import QuantumRegister
from qiskit_ibm_provider import IBMProvider
from qiskit import transpile
import numpy as np
import math
import random

# TODO: possibly include more gates (like full layer gates)
def gate_encoding(circuit, gene, nr_of_qubits, number_of_parameters):
    # first six bits of string define gate time, rest define which qubits to apply to
    gate_string = gene[:6]
    # based on the qubit string a permutation of the qubits is made, which is
    # used in the gate(a single bit gate only takes the first, a two bit gate)
    # takes the first two etc.
    qubit_string = gene[6:]
    qubit_seed = int(qubit_string, 2)
    qubits = np.random.RandomState(seed=qubit_seed).permutation(nr_of_qubits)
    # TODO: decide on the parameter initialisation

    ## Single Qubit gates
    if gate_string == "000000":
        # Pauli-X
        circuit.x(qubit=qubits[0])
        return circuit, number_of_parameters
    if gate_string == "000001":
        # originally u1 gate
        circuit.p(Parameter(str(number_of_parameters)),qubit=qubits[0])
        number_of_parameters += 1
        return circuit, number_of_parameters
    if gate_string == "000010":
        # originally u2 gate
        circuit.u(math.pi/2,Parameter(str(number_of_parameters)),Parameter(str(number_of_parameters+1)),qubit=qubits[0])
        number_of_parameters += 2
        return circuit, number_of_parameters
    if gate_string == "000011":
        # originally u3 gate
        circuit.u(Parameter(str(number_of_parameters)),Parameter(str(number_of_parameters+1)),Parameter(str(number_of_parameters+2)),qubit=qubits[0])
        number_of_parameters += 3
        return circuit, number_of_parameters
    if gate_string == "000100":
        # Pauli-Y
        circuit.y(qubit=qubits[0])
        return circuit, number_of_parameters
    if gate_string == "000101":
        # Pauli-Z
        circuit.z(qubit=qubits[0])
        return circuit, number_of_parameters
    if gate_string == "000110":
        # Hadamard
        circuit.h(qubit=qubits[0])
        return circuit, number_of_parameters
    if gate_string == "000111":
        # S gate
        circuit.s(qubit=qubits[0])
        return circuit, number_of_parameters
    if gate_string == "001000":
        # S conjugate gate
        circuit.sdg(qubit=qubits[0])
        return circuit, number_of_parameters
    if gate_string == "001001":
        # T gate
        circuit.t(qubit=qubits[0])
        return circuit, number_of_parameters
    if gate_string == "001010":
        # T conjugate gate
        circuit.tdg(qubit=qubits[0])
        return circuit, number_of_parameters
    if gate_string == "001011":
        # rx gate
        circuit.rx(Parameter(str(number_of_parameters)),qubit=qubits[0])
        number_of_parameters += 1
        return circuit, number_of_parameters
    if gate_string == "001100":
        # ry gate
        circuit.ry(Parameter(str(number_of_parameters)),qubit=qubits[0])
        number_of_parameters += 1
        return circuit, number_of_parameters
    if gate_string == "001101":
        # rz gate
        circuit.rz(Parameter(str(number_of_parameters)),qubit=qubits[0])
        number_of_parameters += 1
        return circuit, number_of_parameters
    ## Multi Qubit gates
    if gate_string == "100000":
        # Controlled NOT gate
        circuit.cx(qubits[0],qubits[1])
        return circuit, number_of_parameters
    if gate_string == "100001":
        # Controlled Y gate
        circuit.cy(qubits[0],qubits[1])
        return circuit, number_of_parameters
    if gate_string == "100011":
        # Controlled Z gate
        circuit.cz(qubits[0],qubits[1])
        return circuit, number_of_parameters
    if gate_string == "100100":
        # Controlled H gate
        circuit.ch(qubits[0],qubits[1])
        return circuit, number_of_parameters
    if gate_string == "100101":
        # Controlled rotation Z gate
        circuit.crz(Parameter(str(number_of_parameters)),qubits[0],qubits[1])
        number_of_parameters += 1
        return circuit, number_of_parameters
    if gate_string == "100110":
        # Controlled phase rotation gate
        circuit.cp(Parameter(str(number_of_parameters)),qubits[0],qubits[1])
        number_of_parameters += 1
        return circuit, number_of_parameters
    if gate_string == "100111":
        # SWAP gate
        circuit.swap(qubits[0],qubits[1])
        return circuit, number_of_parameters
    if gate_string == "101000":
        # Toffoli gate
        circuit.ccx(qubits[0],qubits[1],qubits[2])
        return circuit, number_of_parameters
    if gate_string == "101001":
        # controlled swap gate
        circuit.cswap(qubits[0],qubits[1],qubits[2])
        return circuit, number_of_parameters
    else:
        return circuit, number_of_parameters
    
def genome_to_circuit(genome, nr_of_qubits, nr_of_gates):
    qr = QuantumRegister(nr_of_qubits)
    circuit = QuantumCircuit(qr)
    gene_length = len(genome)//nr_of_gates
    nr_of_parameters = 0
    for i in range(nr_of_gates):
        gene = genome[i * gene_length : (i+1) * gene_length]
        circuit,nr_of_parameters = gate_encoding(circuit, gene, nr_of_qubits, nr_of_parameters)
    return circuit

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
        if "c" in gate.operation.name:
            cx_bits = [int(gate.qubits[0]._index), int(gate.qubits[1]._index)]
            circuit_error += backend.properties().gate_error(gate.operation.name,cx_bits)
            complexity += 2

    return complexity, circuit_error

def compute_expected_energy(counts,h,j):
    '''
    returns the expected energy of a circuit given the counts, the 1d-ising parameters h and j
    '''
    def bool_to_state(integer):
        # Convert the 1/0 of a bit to +1/-1
        return 2*int(integer)-1

    # Get total energy of each count
    r1=list(counts.keys())
    r2=list(counts.values())
    total_energy = 0

    for k in range(0,len(r1)):
        # r2[k] is the number of shots that have this result
        # r1[k] is the result as qubits (like 0001)
        
        # Energy of h
        for bit_h in range(0,len(r1[k])):
            total_energy += bool_to_state(r1[k][bit_h]) * h[bit_h]*r2[k]
        # Energy of j
        for bit_value in range(0,len(r1[k])-1):
            total_energy += bool_to_state(r1[k][bit_value])*bool_to_state(r1[k][bit_value+1])*j[bit_value]*r2[k]
    # Divide over the total count(shots)
    expectation_value = total_energy/sum(r2)
    return expectation_value

#TODO: add option to fix random seed!
def ising_1d_instance(qubits):
    def rand1d(qubits):
        return [random.choice([+1, -1]) for _ in range(qubits)]

    # transverse field terms
    h = rand1d(qubits)
    # links between lines
    j = rand1d(qubits-1)
    return (h, j)

def add_measurement(circuit, qubits):
    # Create a Quantum Circuit
    meas = QuantumCircuit(qubits, qubits)
    meas.barrier(range(qubits))
    # map the quantum measurement to the classical bits
    meas.measure(range(qubits), range(qubits))

    # The Qiskit circuit object supports composition using
    # the compose method.
    circuit.add_register(meas.cregs[0])
    qc = circuit.compose(meas)
    return qc

#TODO
def energy_from_circuit(circuit, qubits):
    meas_circuit =  add_measurement(circuit, qubits)
    return

def compute_gradient(circuit, parameter_length, h, j, shots=1024):
    #TODO: use the same epsilon as used by IC
    epsilon = 10**-3
    gradient = 0
    #TODO: add possible fixed random seed for experiment
    parameter_selection = np.random.random()* 2*math.pi
    parameters = [parameter_selection for _ in range(parameter_length)]

    for i,_ in enumerate(parameters):
        temp_parameters = parameters
        # Alpha-component of the gradient
        temp_parameters[i] += epsilon/2
        grad_param = energy_from_circuit(circuit.bind_parameters(parameters))
        temp_parameters[i] -= epsilon
        grad_param -= energy_from_circuit(circuit.bind_parameters(parameters))
        grad_param /= epsilon
        gradient += grad_param

    # compute_expected_energy(counts,h,j)

    return gradient