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


TARGET = [0,0]
gnome_qubit_len = int(np.shape(TARGET)[0]/2)
print(gnome_qubit_len)
# print([mutated_genes(i) for i in range(gnome_len)])
