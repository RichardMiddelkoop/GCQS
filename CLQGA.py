#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import time
import argparse
from HelperCLQGA import genome_to_circuit, configure_circuit_to_backend, get_circuit_properties, ising_1d_instance, compute_gradient, calculate_crowd_distance
from Experiments import saveLoad

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
NR_OF_GATES = 6
# the encoding of the circuit takes 11 bits for each circuit layer
# do not change unless the encoding of gates has changed!
GATE_ENCODING_LENGTH = 5
# nr of qubits in the system
QUBIT_SECTIONING_LENGTH = 5
CHROMOSOME_LENGTH = NR_OF_GATES * (GATE_ENCODING_LENGTH + QUBIT_SECTIONING_LENGTH)
# seed used for the randomiser
RANDOM_SEED = None
MODIFIED_UNIFORM_CROSSOVER = False

# the path to IBM chip
CHIP_BACKEND = "ibm_perth"
CHIP_BACKEND_SIMULATOR = "local_qasm_simulator"

## subtract the required information from the given path
NR_OF_QUBITS = 5
NR_OF_ISING_QUBITS = 4
NR_OF_SHOTS = 1024
IMPROVEMENT_CRITERIA = 0.02

# For saving data to file
OUTPUT_NAME = None

def write_output_to_file(output):
    global OUTPUT_NAME
    saveLoad("save",OUTPUT_NAME, output)
    return

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

def assign_parameters(parameter_file, arg_string, output_file):
    global POPULATION_SIZE, MAX_GENERATIONS, MUTATION_RATE, ELITISM_RATE, IMPROVEMENT_CRITERIA, GATE_ENCODING_LENGTH, QUBIT_SECTIONING_LENGTH, CHROMOSOME_LENGTH, CHIP_BACKEND, CHIP_BACKEND_SIMULATOR, NR_OF_QUBITS, NR_OF_ISING_QUBITS, NR_OF_SHOTS, RANDOM_SEED, MODIFIED_UNIFORM_CROSSOVER, OUTPUT_NAME, NR_OF_GATES
    arg_dict = {}
    if not(parameter_file == None):
        arg_dict = read_arg_string_from_file(parameter_file)
    if not(arg_string == None):
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
        elif argument == "IMPROVEMENT_CRITERIA":
            IMPROVEMENT_CRITERIA = float(arg_dict[argument])
        elif argument == "NR_OF_GATES":
            NR_OF_GATES = int(arg_dict[argument])
            CHROMOSOME_LENGTH = NR_OF_GATES * (GATE_ENCODING_LENGTH + QUBIT_SECTIONING_LENGTH)
        elif argument == "QUBIT_SECTIONING_LENGTH":
            QUBIT_SECTIONING_LENGTH = int(arg_dict[argument])
            CHROMOSOME_LENGTH = NR_OF_GATES * (GATE_ENCODING_LENGTH + QUBIT_SECTIONING_LENGTH)
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
        elif argument == "RANDOM_SEED":
            if not arg_dict[argument] == "None":
                RANDOM_SEED = int(arg_dict[argument])
        elif argument == "MODIFIED_UNIFORM_CROSSOVER":
            MODIFIED_UNIFORM_CROSSOVER = (arg_dict[argument]=="True")
        if not output_file == None:
            OUTPUT_NAME = output_file
    return



class Individual(object):
    '''
    class representing individual in population
    '''
    def __init__(self, chromosome):
        self.chromosome = chromosome 
        self.fitness = -1
        # Only for experimental insight
        self.error = -1
        self.ops = -1

    @classmethod
    def create_gnome(self):
        '''
        create chromosome representing quantum circuit
        '''
        global CHROMOSOME_LENGTH, RANDOM_SEED
        random.seed(RANDOM_SEED)
        return ''.join(str(random.randint(0,1)) for _ in range(CHROMOSOME_LENGTH))

def mutation(population):
    '''
    performs the mutation phase for a single generation phase, returns the mutated population
    '''
    global MUTATION_RATE, CHROMOSOME_LENGTH, RANDOM_SEED
    
    for individual in population:
        for sliceIndex, _ in enumerate(individual.chromosome):
            if random.uniform(0.0,1.0) <= MUTATION_RATE:
                individual.chromosome = individual.chromosome[0:sliceIndex] + str(random.randint(0,1)) + individual.chromosome[sliceIndex+1:CHROMOSOME_LENGTH]

    return population

def combination(children_population, parent_population):
    '''
    perform mating and produce new offspring
    '''
    global CHROMOSOME_LENGTH, MODIFIED_UNIFORM_CROSSOVER
    elitism_population = children_population
    
    for _ in range( int((len(parent_population) - len(children_population))/2)):
        parents = random.sample(parent_population, 2)
        chromosome_child_0 = ""
        chromosome_child_1 = ""
        
        if MODIFIED_UNIFORM_CROSSOVER:
            swap_chance = (calculate_crowd_distance(elitism_population, parents[0]) + calculate_crowd_distance(elitism_population, parents[1]))/2
            for i in range(0,CHROMOSOME_LENGTH):
                # modified uniform random, higher crowd distance results in less swapping
                if random.random() < swap_chance:
                    # If True swap gene
                    chromosome_child_0 += parents[1].chromosome[i]
                    chromosome_child_1 += parents[0].chromosome[i]
                else:
                    chromosome_child_0 += parents[0].chromosome[i]
                    chromosome_child_1 += parents[1].chromosome[i]
        else:
            for i in range(0,CHROMOSOME_LENGTH):
                if random.randint(0,1):
                    # If True swap gene
                    chromosome_child_0 += parents[1].chromosome[i]
                    chromosome_child_1 += parents[0].chromosome[i]
                else:
                    chromosome_child_0 += parents[0].chromosome[i]
                    chromosome_child_1 += parents[1].chromosome[i]
        children_population.append(Individual(chromosome_child_0))
        children_population.append(Individual(chromosome_child_1))

        # FOR RANDOM BENCHMARKING ONLY!!
        # children_population.append(Individual(Individual.create_gnome()))
        # children_population.append(Individual(Individual.create_gnome()))

    return children_population

def selection(parent_population):
    '''
    perform selection using elitism, returns initial children_population
    '''
    global ELITISM_RATE
    cut_off = int(ELITISM_RATE*len(parent_population))
    children_population = parent_population[:cut_off]

    return children_population
    
# ### The current fitness calculation is very simple 
def fitness(population, observable_h, observable_j):
    '''
    calculate fitness score
    '''
    global CHIP_BACKEND, NR_OF_QUBITS, NR_OF_GATES, NR_OF_ISING_QUBITS, NR_OF_SHOTS, CHIP_BACKEND_SIMULATOR, RANDOM_SEED
    for individual in population:
        genome = individual.chromosome
        circuit, nr_of_parameters = genome_to_circuit(genome, NR_OF_QUBITS, NR_OF_GATES)
        gradient, energy, circuit = compute_gradient(circuit, nr_of_parameters, NR_OF_ISING_QUBITS, observable_h, observable_j, NR_OF_SHOTS, CHIP_BACKEND_SIMULATOR, RANDOM_SEED)
        
        configured_circuit, backend = configure_circuit_to_backend(circuit, CHIP_BACKEND)
        if not type(backend) == str:
            CHIP_BACKEND = backend
        complexity, circuit_error = get_circuit_properties(configured_circuit, CHIP_BACKEND)
        individual.fitness = 1/(1+complexity) * 1/(1+circuit_error) * gradient * -energy
        # For experiments
        individual.error = circuit_error
        individual.ops = configured_circuit.count_ops()

    return sorted(population, key=lambda individual: individual.fitness, reverse=True)

def main():
    global POPULATION_SIZE, MAX_GENERATIONS, NR_OF_ISING_QUBITS, IMPROVEMENT_CRITERIA, RANDOM_SEED, OUTPUT_NAME, ELITISM_RATE, CHIP_BACKEND, NR_OF_QUBITS, NR_OF_GATES
    # initialise parameters of the observable (1d ising model)
    observable_h, observable_j = ising_1d_instance(NR_OF_ISING_QUBITS, RANDOM_SEED)
    # initialisation of variables
    generation = 1
    found = False
    population = []
    # For experimental data saving
    start_total_run_time = time.perf_counter()
    data_average_fitness = []
    data_average_error = []
    data_average_crowd_score = []
    data_best_individual = []
    data_best_family = []
    data_circuit_evolution = []
    # initial population
    for _ in range(POPULATION_SIZE):
        gnome = Individual.create_gnome()
        population.append(Individual(gnome))
    population = fitness(population, observable_h, observable_j)
    population = sorted(population, key=lambda individual: individual.fitness, reverse=True)
    average_fitness = []
    average_error = []
    # first generation so its always the best individual and best family
    # for the individual the structure is [fitness, error, chromosome, current generation, configured circuit]
    # for the family the structure is [average elitism fitness, average elitism error, generation, family set of chromosomes]
    data_best_individual.append([population[0].fitness,population[0].error,population[0].chromosome,generation-1,genome_to_circuit(population[0].chromosome, NR_OF_QUBITS, NR_OF_GATES)[0],configure_circuit_to_backend(genome_to_circuit(population[0].chromosome, NR_OF_QUBITS, NR_OF_GATES)[0], CHIP_BACKEND)])
    family_pool = selection(population)
    data_best_family.append([sum([i.fitness for i in family_pool])/len(family_pool),sum([i.error for i in family_pool])/len(family_pool),generation-1,[i.chromosome for i in family_pool]])
    while not found:
        start = time.perf_counter()
        new_population = selection(population)
        new_population = combination(new_population, population)
        population = mutation(new_population)
        population = fitness(population, observable_h, observable_j)
        average_fitness.append(population[0].fitness)
        average_error.append(population[0].error)
        if len(average_fitness) > 40:
            average_fitness.pop(0)
            average_error.pop(0)
        # print statements during processing
        print("generation {}: {}".format(generation, population[0].fitness))
        if generation == 1:
            # print expected runtime 
            print("Expected runtime: {}".format(time.strftime("%H:%M:%S", time.gmtime((time.perf_counter() - start)*MAX_GENERATIONS))))
        if generation % 50 == 0:
            data_average_fitness.append(sum(average_fitness[int(len(average_fitness)/2):])/int(len(average_fitness)/2))
            data_average_error.append(sum(average_error[int(len(average_error)/2):])/int(len(average_error)/2))
            total_crowd_distances = [calculate_crowd_distance(population[:int(ELITISM_RATE*len(population))], population[i]) for i in range(0,len(population))]
            data_average_crowd_score.append(sum(total_crowd_distances)/len(total_crowd_distances))
            print("Generation: {}\nCircuit: \n{}\nFitness: {}".format(generation,population[0].chromosome,sum(average_fitness[int(len(average_fitness)/2):])/int(len(average_fitness)/2)))
        if population[0].fitness > data_best_individual[-1][0]:
            data_best_individual.append([population[0].fitness,population[0].error,population[0].chromosome,generation,genome_to_circuit(population[0].chromosome, NR_OF_QUBITS, NR_OF_GATES)[0],configure_circuit_to_backend(genome_to_circuit(population[0].chromosome, NR_OF_QUBITS, NR_OF_GATES)[0], CHIP_BACKEND)])
        family_pool = selection(population)
        data_circuit_evolution.append([[sum(population[0].ops.values()),sum([population[0].ops[i] for i in population[0].ops.keys() if 'c' in i])],[sum([sum(x.ops.values()) for x in family_pool])/len(family_pool),sum([sum([x.ops[i] for i in x.ops.keys() if 'c' in i]) for x in family_pool])/len(family_pool)]])
        if sum([i.fitness for i in family_pool])/len(family_pool) > data_best_family[-1][0]:
            data_best_family.append([sum([i.fitness for i in family_pool])/len(family_pool),sum([i.error for i in family_pool])/len(family_pool),generation,[i.chromosome for i in family_pool]])
        
        # stopping criteria
        if generation == MAX_GENERATIONS: 
            print("max gen reached!!")
            found = True
        ## (New-Old)/Old to check if there are still improvements in the fitness of the population
        if abs((sum(average_fitness[int(len(average_fitness)/2):]) - sum(average_fitness[:int(len(average_fitness)/2)]))/(sum(average_fitness[int(len(average_fitness)/2):])+1))<IMPROVEMENT_CRITERIA:
            print("improvement threshold breached!")
            found = True
        generation += 1
    
    # # if wanted, uncomment to see the final gate
    print(genome_to_circuit(population[0].chromosome, NR_OF_QUBITS, NR_OF_GATES)[0])
    if not OUTPUT_NAME == None:
        write_output_to_file([population,time.gmtime((time.perf_counter() - start_total_run_time)),data_average_fitness, data_average_crowd_score, data_average_error, data_best_individual, data_best_family, data_circuit_evolution,[observable_h,observable_j]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arguments', required=False, help="Input the parameters to use for the algorithm, either input the name of a textfile(\".txt\"!) located in the same textfolder as the algorithm OR input a \",\" string listing the parameters that you want changed, like \"param1=5,param2=3,...\"")
    parser.add_argument('--write', required=False, help="Input name of file to pickle the experimental data to")
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

    if args.write:
        assign_parameters(parameter_file, arg_string, args.write)
    else:
        assign_parameters(parameter_file, arg_string, None)
    main()