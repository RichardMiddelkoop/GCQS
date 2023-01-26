import cirq
import numpy as np
# Pick a qubit.
q0, q1 = cirq.LineQubit.range(2)

# Create a circuit
circuit = cirq.Circuit(
    cirq.YY(q0,q1)
    # cirq.measure(qubit, key='m')  # Measurement.
)
print("Circuit:")
print(circuit)
result = cirq.Simulator().simulate(circuit)
print("Results:")
print(np.around(result.final_state_vector, 3))
# # Simulate the circuit several times.
# simulator = cirq.Simulator()
# result = simulator.simulate(circuit)
# print("Results:")
# print(result)
# print("Final state vector:")
# print(np.around(result.final_state_vector, 3))

