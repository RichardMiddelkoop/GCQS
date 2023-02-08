import cirq
import numpy as np
# Pick a qubit.
q0, q1 = cirq.LineQubit.range(2)
ps0 = cirq.Z(q0)
qubit_map={q0: 0}
# Create a circuit
circuit = cirq.Circuit(
    cirq.X(q0)
    # cirq.measure(qubit, key='m')  # Measurement.
)
print("Circuit:")
print(circuit)
print(cirq.unitary(circuit))
result = cirq.final_state_vector(circuit)

print("Results:")
print(ps0.expectation_from_state_vector(result, qubit_map).real)
# # Simulate the circuit several times.
# simulator = cirq.Simulator()
# result = simulator.simulate(circuit)
# print("Results:")
# print(result)
# print("Final state vector:")
# print(np.around(result.final_state_vector, 3))

