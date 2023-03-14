import random

## parameters for the algorithm ##
# number of individuals in each generation
POPULATION_SIZE = 100
# mutation rate of a gene in the mutation phase
MUTATION_RATE = 0.20
# number of layers in the ansatz
CIRCUIT_DEPTH = 6
# the encoding of the circuit takes 11 bits for each circuit layer
CHROMOSOME_LENGTH = CIRCUIT_DEPTH * 11

class Individual(object):
    '''
    class representing individual in population
    '''
    def __init__(self, chromosome):
        self.chromosome = chromosome 
        self.fitness = self.cal_fitness()

    @classmethod
    def create_gnome(self):
        '''
        create chromosome of parameters initialised between 0 and 1
        '''
        params = ["alpha","beta","gamma"]
        return {param: random.random() for param in params}

    def mutation(population):
        '''
        performs the mutation phase for a single generation phase, returns the mutated population
        '''
        global MUTATION_RATE, CHROMOSOME_LENGTH
        for individual in population:
            for sliceIndex, _ in enumerate(individual.string):
                if random.uniform(0.0,1.0) <= MUTATION_RATE:
                    individual.chromosome = individual.chromosome[0:sliceIndex] + str(random.randint(0,1)) + individual.string[sliceIndex+1:CHROMOSOME_LENGTH]

        return population
    
    def combination(children_population, parent_population):
        '''
        perform mating and produce new offspring
        '''
        global CHROMOSOME_LENGTH

        for _ in range( int((len(parent_population) - len(children_population))/2)):
            parents = random.sample(parent_population, 2)
            slice = random.randint(0,CHROMOSOME_LENGTH)
            children_population.append(Individual(parents[0][:slice]+parents[1][slice:]))
            children_population.append(Individual(parents[1][:slice]+parents[0][slice:]))
        return children_population
    
    def selection(parent_population):
        global ELITISM_RATE
        parent_population = sorted(parent_population, key=lambda individual: individual.fitness, reverse=True)
        cut_off = int((10*POPULATION_SIZE)/100)
        children_population = parent_population[:]
        return children_population
    
    ### The current fitness calculation is very simple 
    def cal_fitness(self):
        '''
        TODO: Allow for more flexibility in instances
        calculate fitness score
        '''
        global H,JR,JC,TEST_INSTANCE, REPETITIONS
        return calculate_expected_value(H,JR,JC,TEST_INSTANCE,self.chromosome,REPETITIONS)
