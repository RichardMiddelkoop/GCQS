import cirq
import numpy as np
import random
import sympy
#!https://quantumai.google/cirq/experiments/variational_algorithm
# Set this for experiment, use None otherwise!!
RANDOM_SEED = 10

def rot_x_layer(length, half_turns):
    """cirq sub-circuit: Yields X rotations by half_turns on a square grid of given length."""

    # Define the gate once and then re-use it for each Operation.
    rot = cirq.XPowGate(exponent=half_turns)

    # Create an X rotation Operation for each qubit in the grid.
    for i in range(length):
        for j in range(length):
            yield rot(cirq.GridQubit(i, j))

def rand2d(rows, cols):
    global RANDOM_SEED
    random.seed(RANDOM_SEED)
    return [[random.choice([+1, -1]) for _ in range(cols)] for _ in range(rows)]
    
def random_instance(length):
    """Generates a random instance with the parameters h and j, returns (h: the field terms, jr: links in the row and jc: links in the column)"""
    global RANDOM_SEED
    random.seed(RANDOM_SEED)
    # transverse field terms
    h = rand2d(length, length)
    # links within a row
    jr = rand2d(length - 1, length)
    # links within a column
    jc = rand2d(length, length - 1)
    return (h, jr, jc)

def prepare_plus_layer(length):
    for i in range(length):
        for j in range(length):
            yield cirq.H(cirq.GridQubit(i, j))

def rot_z_layer(h, half_turns):
    """cirq sub-circuit: Yields Z rotations by half_turns conditioned on the field h."""
    gate = cirq.ZPowGate(exponent=half_turns)
    for i, h_row in enumerate(h):
        for j, h_ij in enumerate(h_row):
            if h_ij == 1:
                yield gate(cirq.GridQubit(i, j))

def rot_11_layer(jr, jc, half_turns):
    """cirq sub-circuit: Yields rotations about |11> conditioned on the jr and jc fields."""
    cz_gate = cirq.CZPowGate(exponent=half_turns)    
    for i, jr_row in enumerate(jr):
        for j, jr_ij in enumerate(jr_row):
            q = cirq.GridQubit(i, j)
            q_1 = cirq.GridQubit(i + 1, j)
            if jr_ij == -1:
                yield cirq.X(q)
                yield cirq.X(q_1)
            yield cz_gate(q, q_1)
            if jr_ij == -1:
                yield cirq.X(q)
                yield cirq.X(q_1)

    for i, jc_row in enumerate(jc):
        for j, jc_ij in enumerate(jc_row):
            q = cirq.GridQubit(i, j)
            q_1 = cirq.GridQubit(i, j + 1)
            if jc_ij == -1:
                yield cirq.X(q)
                yield cirq.X(q_1)
            yield cz_gate(q, q_1)
            if jc_ij == -1:
                yield cirq.X(q)
                yield cirq.X(q_1)

def initial_step(length):
    yield prepare_plus_layer(length)

def one_step(h, jr, jc, x_half_turns, h_half_turns, j_half_turns):
    length = len(h)
    yield rot_z_layer(h, h_half_turns)
    yield rot_11_layer(jr, jc, j_half_turns)
    yield rot_x_layer(length, x_half_turns)

# Using all function create an ansatz instance 
def create_instance(length=3, p1=0.1, p2=0.2, p3=0.3):
    """Return a problem instance"""
    h, jr, jc = random_instance(length)
    circuit = cirq.Circuit()  
    circuit.append(initial_step(len(h)))
    circuit.append(one_step(h, jr, jc, p1, p2, p3))
    # print(circuit)
    return h,jr,jc,circuit


def energy_func(length, h, jr, jc):
    def energy(measurements):
        # Reshape measurement into array that matches grid shape.
        meas_list_of_lists = [measurements[i * length:(i + 1) * length]
                              for i in range(length)]
        # Convert true/false to +1/-1.
        pm_meas = 1 - 2 * np.array(meas_list_of_lists).astype(np.int32)

        tot_energy = np.sum(pm_meas * h)
        for i, jr_row in enumerate(jr):
            for j, jr_ij in enumerate(jr_row):
                tot_energy += jr_ij * pm_meas[i, j] * pm_meas[i + 1, j]
        for i, jc_row in enumerate(jc):
            for j, jc_ij in enumerate(jc_row):
                tot_energy += jc_ij * pm_meas[i, j] * pm_meas[i, j + 1]
        return tot_energy
    return energy

def create_instance_and_calculate_expected_value(length=3,p1=0.1, p2=0.2, p3=0.3, repetitions=100):
    simulator = cirq.Simulator()
    qubits = cirq.GridQubit.square(length)
    h,jr,jc,circuit = create_instance(length,p1,p2,p3)
    circuit.append(cirq.measure(*qubits, key='x'))
    result = simulator.run(circuit, repetitions=repetitions)
    energy_hist = result.histogram(key='x', fold_func=energy_func(3, h, jr, jc))
    return np.sum([k * v for k,v in energy_hist.items()]) / result.repetitions