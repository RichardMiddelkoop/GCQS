import cirq
import numpy as np
import sympy
import random
from HelperQGA import create_instance, energy_func
# TODO: make the gradients variable for the number of parameters
def generate_parameter_circuit(length=3):
    alpha = sympy.Symbol('alpha')
    beta = sympy.Symbol('beta')
    gamma = sympy.Symbol('gamma')
    return create_instance(length=length, p1=alpha, p2=beta, p3=gamma)

# number of repitions in the sampling
REPETITIONS = 100
# The parameters used for the problem instance
PARAMETERS = ["alpha","beta","gamma"]
LENGTH = 3
H,JR,JC,TEST_INSTANCE = generate_parameter_circuit(LENGTH)

def energy_from_params(alpha, beta, gamma):
    """Returns the energy given values of the parameters."""
    global REPETITIONS, H, JR, JC, TEST_INSTANCE
    simulator = cirq.Simulator()
    qubits = cirq.GridQubit.square(LENGTH)
    circuit = cirq.resolve_parameters(TEST_INSTANCE, {"alpha": alpha, "beta": beta, "gamma": gamma})
    circuit.append(cirq.measure(*qubits, key='x'))
    result = simulator.run(circuit, repetitions=REPETITIONS)
    energy_hist = result.histogram(key='x', fold_func=energy_func(3, H, JR, JC))
    return np.sum([k * v for k,v in energy_hist.items()]) / result.repetitions

# TODO: Test different variations of gradient estimations
def gradient_energy(alpha, beta, gamma):
    """Uses a symmetric difference to calulate the gradient."""
    epsilon = 10**-3  # Try different values of the discretization parameter

    # Alpha-component of the gradient
    grad_a = energy_from_params(alpha + epsilon, beta, gamma)
    grad_a -= energy_from_params(alpha - epsilon, beta, gamma)
    grad_a /= 2 * epsilon

    # Beta-compoonent of the gradient
    grad_b = energy_from_params(alpha, beta + epsilon, gamma)
    grad_b -= energy_from_params(alpha, beta - epsilon, gamma)
    grad_b /= 2 * epsilon

    # Gamma-component of the gradient
    grad_g = energy_from_params(alpha, beta, gamma + epsilon)
    grad_g -= energy_from_params(alpha, beta, gamma - epsilon)
    grad_g /= 2 * epsilon

    return grad_a, grad_g, grad_b

def gradient_descent(iterations=10000, learning_rate=0.001):
    # initialise the parameters
    alpha = random.random()
    beta = random.random()
    gamma = random.random()

    for i in range(iterations + 1):
        # Compute the gradient.
        grad_a, grad_g, grad_b = gradient_energy(alpha, beta, gamma)

        # Update the parameters.
        alpha -= learning_rate * grad_a
        gamma -= learning_rate * grad_g
        beta -= learning_rate * grad_b

        # Status updates.
        if not i % 25:
            print("Step: {} Energy: {}, Params: {},{},{}".format(i, energy_from_params(alpha, beta, gamma),alpha, beta, gamma))

gradient_descent()