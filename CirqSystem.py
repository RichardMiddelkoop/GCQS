import cirq
import numpy as np
# Pick a qubit.
# qubit = cirq.GridQubit(0, 0)

# # Create a circuit
# circuit = cirq.Circuit(
#     cirq.X(qubit)**0.5,  # Square root of NOT.
#     cirq.measure(qubit, key='m')  # Measurement.
# )
# print("Circuit:")
# print(circuit)

# # Simulate the circuit several times.
# simulator = cirq.Simulator()
# result = simulator.simulate(circuit)
# print("Results:")
# print(result)
# print("Final state vector:")
# print(np.around(result.final_state_vector, 3))


## Attempt at creating an fitness function
### Idea is very simple 
# real_score = (1-abs(target_value_real-individual_value_real))^2
# imaginary_score = (1-abs(target_value_imaginary-individual_value_imaginary))^2
# fitness_value = real_score + imaginary_score

def mutated_genes(i):
    return cirq.GridQubit(i, 0)


TARGET = [0,0.5,0,1]
GENES_1 = [cirq.X,cirq.Y,cirq.Z,cirq.H,cirq.S,cirq.T]
GENES_2 = [cirq.CZ,cirq.CNOT,cirq.SWAP,cirq.XX,cirq.YY,cirq.ZZ]
GENES_3 = [cirq.CCNOT,cirq.CCZ,cirq.CSWAP]
gnome_qubit_len = int(np.log2(np.shape(TARGET)[0]))
print(gnome_qubit_len)

# print(cirq.Circuit(GENES_1[0](qubit) for qubit in cirq.LineQubit.range(gnome_qubit_len)))
import random
print(GENES_1+GENES_2)
print(random.choice(GENES_1+GENES_2))
print(random.randrange(3))
