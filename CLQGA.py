import random
import time

## parameters for the algorithm ##
# number of individuals in each generation
POPULATION_SIZE = 100
# maximum number of generation the algorithm can run
MAX_GENERATIONS = 100
# mutation rate of a gene in the mutation phase
# 0 < MUTATION_RATE < 1
MUTATION_RATE = 0.20
# the selection of best performing individuals that will be saved in the selection phase
# 0 < ELITISM_RATE < 1
ELITISM_RATE = 0.10
# number of layers in the ansatz
CIRCUIT_DEPTH = 6
# the encoding of the circuit takes 11 bits for each circuit layer
CHROMOSOME_LENGTH = CIRCUIT_DEPTH * 11

#TODO: implement these steps
# the hamiltonion used as observable
H = None
# the initial quantum state used
INITIAL_STATE = None
# the path to IBM chip csv
CHIP_LAYOUT_PATH = None

## subtract the required information from the given path
NR_OF_QUBITS = None

class Individual(object):
    '''
    class representing individual in population
    '''
    def __init__(self, chromosome):
        self.chromosome = chromosome 
        self.fitness = 0

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
    cut_off = int(ELITISM_RATE*len(parent_population))
    children_population = parent_population[:cut_off]
    return children_population
    
# ### The current fitness calculation is very simple 
def fitness(population):
    '''
    calculate fitness score
    '''
    global H,INITIAL_STATE, CHIP_LAYOUT
    for individual in population:
        genome = individual.chromosome
        individual.fitness = 0
    return population

def main():
    global POPULATION_SIZE, MAX_GENERATIONS

    # initialisation of variables
    generation = 1
    found = False
    population = []

    # initial population
    for _ in range(POPULATION_SIZE):
        gnome = Individual.create_gnome()
        population.append(Individual(gnome))

    while not found:

        if generation == 1:
            # used to calculate expected runtime
            start = time.process_time()
        population = fitness(population)
        new_population = selection(population)
        new_population = combination(new_population, population)
        population = mutation(new_population)
        if generation == 1:
            # print expected runtime 
            print("Expected runtime: {}".format(time.strftime("%H:%M:%S", time.gmtime((time.process_time() - start)*MAX_GENERATIONS))))
        if generation % 50 == 0:
            print("Generation: {}\nCircuit: \n{}\nFitness: {}".format(generation,population[0].chromosome,population[0].fitness))
        if generation == MAX_GENERATIONS: 
            print("max gen reached!!")
            found = True
        generation += 1


if __name__ == '__main__':
    main()