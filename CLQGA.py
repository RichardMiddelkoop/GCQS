import random
import time
import argparse
from HelperCLQGA import genome_to_circuit, configure_circuit_to_backend, get_circuit_properties, ising_1d_instance, compute_gradient

## parameters for the algorithm ##
# number of individuals in each generation
POPULATION_SIZE = 20
# maximum number of generation the algorithm can run
MAX_GENERATIONS = 20
# mutation rate of a gene in the mutation phase
# 0 < MUTATION_RATE < 1
MUTATION_RATE = 0.20
# the selection of best performing individuals that will be saved in the selection phase
# 0 < ELITISM_RATE < 1
ELITISM_RATE = 0.10
# number of layers in the ansatz
CIRCUIT_DEPTH = 6
# the encoding of the circuit takes 11 bits for each circuit layer
# do not change unless the encoding of gates has changed!
GATE_ENCODING_LENGTH = 6 
# nr of qubits in the system
QUBIT_SECTIONING_LENGTH = 5
CHROMOSOME_LENGTH = CIRCUIT_DEPTH * (GATE_ENCODING_LENGTH + QUBIT_SECTIONING_LENGTH)

# the path to IBM chip
CHIP_BACKEND = "ibm_perth"
CHIP_BACKEND_SIMULATOR = "local_qasm_simulator"

## subtract the required information from the given path
NR_OF_QUBITS = 5
NR_OF_ISING_QUBITS = 4
NR_OF_GATES = CIRCUIT_DEPTH
NR_OF_SHOTS = 1024

def read_arg_string_from_file(parameter_file):
    arg_file = open(parameter_file, 'r')
    arg_lines = arg_file.readlines()
    arg_dict = {}
    for arg_line in arg_lines:
        param = arg_line[:arg_line.find("=")].strip()
        value = arg_line[arg_line.find("=")+1:].strip()
        arg_dict[param] = value
    return arg_dict

def arg_string_to_dict(arg_string, dict):
    arg_lines = arg_string.split(",")
    arg_dict = dict
    for arg_line in arg_lines:
        param = arg_line[:arg_line.find("=")].strip()
        value = arg_line[arg_line.find("=")+1:].strip()
        arg_dict[param] = value
    return arg_dict

def assign_parameters(parameter_file, arg_string):
    global POPULATION_SIZE, MAX_GENERATIONS, MUTATION_RATE, ELITISM_RATE, CIRCUIT_DEPTH, GATE_ENCODING_LENGTH, QUBIT_SECTIONING_LENGTH, CHROMOSOME_LENGTH, CHIP_BACKEND, CHIP_BACKEND_SIMULATOR, NR_OF_QUBITS, NR_OF_ISING_QUBITS, NR_OF_SHOTS
    arg_dict = {}
    if not(parameter_file == None):
        arg_dict = read_arg_string_from_file(parameter_file)
    else:
        arg_dict = arg_string_to_dict(arg_string, arg_dict)
    for argument in arg_dict:
        if argument == "POPULATION_SIZE":
            POPULATION_SIZE = int(arg_dict[argument])
        elif argument == "MAX_GENERATIONS":
            MAX_GENERATIONS = int(arg_dict[argument])
        elif argument == "MUTATION_RATE":
            MUTATION_RATE = float(arg_dict[argument])
        elif argument == "ELITISM_RATE":
            ELITISM_RATE = float(arg_dict[argument])
        elif argument == "CIRCUIT_DEPTH":
            CIRCUIT_DEPTH = int(arg_dict[argument])
            CHROMOSOME_LENGTH = CIRCUIT_DEPTH * (GATE_ENCODING_LENGTH + QUBIT_SECTIONING_LENGTH)
        elif argument == "GATE_ENCODING_LENGTH":
            GATE_ENCODING_LENGTH = int(arg_dict[argument])
            CHROMOSOME_LENGTH = CIRCUIT_DEPTH * (GATE_ENCODING_LENGTH + QUBIT_SECTIONING_LENGTH)
        elif argument == "QUBIT_SECTIONING_LENGTH":
            QUBIT_SECTIONING_LENGTH = int(arg_dict[argument])
            CHROMOSOME_LENGTH = CIRCUIT_DEPTH * (GATE_ENCODING_LENGTH + QUBIT_SECTIONING_LENGTH)
        elif argument == "CHIP_BACKEND":
            CHIP_BACKEND = arg_dict[argument]
        elif argument == "CHIP_BACKEND_SIMULATOR":
            CHIP_BACKEND_SIMULATOR = arg_dict[argument]
        elif argument == "NR_OF_QUBITS":
            NR_OF_QUBITS = int(arg_dict[argument])
        elif argument == "NR_OF_ISING_QUBITS":
            NR_OF_ISING_QUBITS = int(arg_dict[argument])
        elif argument == "NR_OF_SHOTS":
            NR_OF_SHOTS = int(arg_dict[argument])
    return



class Individual(object):
    '''
    class representing individual in population
    '''
    def __init__(self, chromosome):
        self.chromosome = chromosome 
        self.fitness = -1

    @classmethod
    def create_gnome(self):
        '''
        create chromosome representing quantum circuit
        '''
        global CHROMOSOME_LENGTH
        return ''.join(str(random.randint(0,1)) for _ in range(CHROMOSOME_LENGTH))

def mutation(population):
    '''
    performs the mutation phase for a single generation phase, returns the mutated population
    '''
    global MUTATION_RATE, CHROMOSOME_LENGTH

    for individual in population:
        for sliceIndex, _ in enumerate(individual.chromosome):
            if random.uniform(0.0,1.0) <= MUTATION_RATE:
                individual.chromosome = individual.chromosome[0:sliceIndex] + str(random.randint(0,1)) + individual.chromosome[sliceIndex+1:CHROMOSOME_LENGTH]

    return population

def combination(children_population, parent_population):
    '''
    perform mating and produce new offspring
    '''
    global CHROMOSOME_LENGTH

    for _ in range( int((len(parent_population) - len(children_population))/2)):
        parents = random.sample(parent_population, 2)
        slice = random.randint(0,CHROMOSOME_LENGTH)
        children_population.append(Individual(parents[0].chromosome[:slice]+parents[1].chromosome[slice:]))
        children_population.append(Individual(parents[1].chromosome[:slice]+parents[0].chromosome[slice:]))

    return children_population

def selection(parent_population):
    '''
    perform selection using elitism, returns initial children_population
    '''
    global ELITISM_RATE

    parent_population = sorted(parent_population, key=lambda individual: individual.fitness, reverse=True)
    print("population sorted!!")
    for kid in parent_population:
        print("Child, ",kid.fitness)
    cut_off = int(ELITISM_RATE*len(parent_population))
    children_population = parent_population[:cut_off]

    return children_population
    
# ### The current fitness calculation is very simple 
def fitness(population, observable_h, observable_j):
    '''
    calculate fitness score
    '''
    global CHIP_BACKEND, NR_OF_QUBITS, NR_OF_GATES, NR_OF_ISING_QUBITS, NR_OF_SHOTS
    # This stops repeated queries to IBM for the circuit properties
    backend = CHIP_BACKEND
    #TODO: calculate the complexity value of both the circuit and added complexity due to the required changes of the circuit given the chip layout.
    #TODO: decide upon a maximum CNOT value allow within a circuit
    #TODO: calculate the energy/gradient of the circuit using the H and the initial state
    for individual in population:
        genome = individual.chromosome
        circuit, nr_of_parameters = genome_to_circuit(genome, NR_OF_QUBITS, NR_OF_GATES)
        gradient, circuit = compute_gradient(circuit, nr_of_parameters, NR_OF_ISING_QUBITS, observable_h, observable_j, NR_OF_SHOTS)
        configured_circuit = configure_circuit_to_backend(circuit, CHIP_BACKEND)
        complexity, circuit_error, backend = get_circuit_properties(configured_circuit, backend)
        individual.fitness = 1/(1+complexity) * 1/(1+circuit_error) * gradient
    
    return population

def main():
    global POPULATION_SIZE, MAX_GENERATIONS, NR_OF_ISING_QUBITS

    # initialise parameters of the observable (1d ising model)
    observable_h, observable_j = ising_1d_instance(NR_OF_ISING_QUBITS)

    # initialisation of variables
    generation = 1
    found = False
    population = []
    start = time.perf_counter()
    # initial population
    for _ in range(POPULATION_SIZE):
        gnome = Individual.create_gnome()
        population.append(Individual(gnome))
    population = fitness(population, observable_h, observable_j)
    #TODO: Include stopping criteria based on improvement
    while not found:

        new_population = selection(population)
        new_population = combination(new_population, population)
        population = mutation(new_population)
        population = fitness(population, observable_h, observable_j)
        if generation == 1:
            # print expected runtime 
            print("Expected runtime: {}".format(time.strftime("%H:%M:%S", time.gmtime((time.perf_counter() - start)*MAX_GENERATIONS))))
        if generation % 5 == 0:
            print("Generation: {}\nCircuit: \n{}\nFitness: {}".format(generation,population[0].chromosome,population[0].fitness))
        if generation == MAX_GENERATIONS: 
            print("max gen reached!!")
            found = True
        generation += 1
    
    # # if wanted, uncomment to see the final gate
    global NR_OF_QUBITS, NR_OF_GATES
    print(genome_to_circuit(population[0].chromosome, NR_OF_QUBITS, NR_OF_GATES)[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arguments', required=False, help="Input the parameters to use for the algorithm, either input the name of a textfile(\".txt\"!) located in the same textfolder as the algorithm OR input a \",\" string listing the parameters that you want changed, like \"param1=5,param2=3,...\"")
    args = parser.parse_args()
    parameter_file = "default_parameters.txt"
    if args.arguments:
        if args.arguments[len(args.arguments)-4:] == ".txt": 
            parameter_file = args.arguments
            arg_string = None
        else: 
            arg_string = args.arguments
    else: 
        arg_string = None
    assign_parameters(parameter_file, arg_string)
    main()