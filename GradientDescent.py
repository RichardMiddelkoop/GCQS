import sympy
import random
from HelperQGA import create_instance, calculate_expected_value
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
H,JR,JC,TEST_INSTANCE = generate_parameter_circuit(len(PARAMETERS))

def energy_from_params(params):
    """Returns the energy given values of the parameters."""
    global REPETITIONS, H, JR, JC, TEST_INSTANCE
    return calculate_expected_value(H,JR,JC,TEST_INSTANCE,params,REPETITIONS)

# TODO: Test different variations of gradient estimations
def symmetric_gradient_energy(alpha, beta, gamma):
    """Uses a symmetric difference to calulate the gradient."""
    epsilon = 10**-3  # Try different values of the discretization parameter

    # Alpha-component of the gradient
    grad_a = energy_from_params({"alpha":alpha + epsilon, "beta":beta, "gamma":gamma})
    grad_a -= energy_from_params({"alpha":alpha - epsilon, "beta":beta, "gamma":gamma})
    grad_a /= 2 * epsilon

    # Beta-compoonent of the gradient
    grad_b = energy_from_params({"alpha":alpha, "beta":beta + epsilon, "gamma":gamma})
    grad_b -= energy_from_params({"alpha":alpha, "beta":beta - epsilon, "gamma":gamma})
    grad_b /= 2 * epsilon

    # Gamma-component of the gradient
    grad_g = energy_from_params({"alpha":alpha, "beta":beta, "gamma":gamma + epsilon})
    grad_g -= energy_from_params({"alpha":alpha, "beta":beta, "gamma":gamma - epsilon})
    grad_g /= 2 * epsilon

    return grad_a, grad_g, grad_b

def gradient_descent(version="symmetric", iterations=10000, learning_rate=0.001):
    # initialise the parameters
    alpha = random.random()
    beta = random.random()
    gamma = random.random()

    for i in range(iterations + 1):
        # Compute the gradient.
        if version=="symmetric":
            grad_a, grad_g, grad_b = symmetric_gradient_energy(alpha, beta, gamma)

        # Update the parameters.
        alpha -= learning_rate * grad_a
        gamma -= learning_rate * grad_g
        beta -= learning_rate * grad_b

        # Status updates.
        if not i % 25:
            print("Step: {} Energy: {}, Params: {},{},{}".format(i, energy_from_params({"alpha":alpha, "beta":beta, "gamma":gamma}),alpha, beta, gamma))

gradient_descent()