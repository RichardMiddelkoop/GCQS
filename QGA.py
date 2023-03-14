# Quantum Genetic Algorithm
import random
import sympy
from HelperQGA import create_instance, calculate_expected_value
import time

def generate_parameter_circuit(length=3):
    alpha = sympy.Symbol('alpha')
    beta = sympy.Symbol('beta')
    gamma = sympy.Symbol('gamma')
    return create_instance(length=length, p1=alpha, p2=beta, p3=gamma)

## parameters for the algorithm ##
# number of individuals in each generation
POPULATION_SIZE = 100
# number of repitions in the sampling
REPETITIONS = 100
# length of the VQA
LENGTH = 3
# stopping criteria
MAX_GENERATIONS = 1000
## created variables of the problem instance ##
H,JR,JC,TEST_INSTANCE = generate_parameter_circuit(LENGTH)
##################################

class Individual(object):
    '''
    class representing individual in population
    '''
    def __init__(self, chromosome):
        self.chromosome = chromosome 
        self.fitness = self.cal_fitness()

    @classmethod
    def mutated_genes(self):
        '''
        create random number for a gene to be mutated by
        '''
        # [-2,-1) negate increase
        # (-1,0) negate decrease
        # (0,1) decrease
        # (1,2] increase
        mutation_odds = random.random()
        if mutation_odds < 0.9:
            return random.uniform(-2,2)
        # 10% chance to have a more sizable mutation
        return 100*random.uniform(-2,2)

    @classmethod
    def create_gnome(self):
        '''
        create chromosome of parameters initialised between 0 and 1
        '''
        params = ["alpha","beta","gamma"]
        return {param: random.random() for param in params}
    
    def mate(self, parent2):
        '''
        perform mating and produce new offspring
        '''

        # chromosome for offspring
        child_chromosome = {}

        for param in self.chromosome:    
            # random probability  
            prob = random.random()

            # from parent 1
            # if prob is less than 0.45, insert gene
            # if larger than 0.45 mutate the gene
            if prob <= 0.50:
                child_chromosome[param] = self.chromosome[param]
                if prob > 0.45:
                    child_chromosome[param] *= self.mutated_genes()
            else:
                # from parent 2
                # if prob is less than 0.95, insert gene
                # if larger than 0.95 mutate the gene
                child_chromosome[param] = parent2.chromosome[param]
                if prob > 0.95:
                    child_chromosome[param] *= self.mutated_genes()
        return Individual(child_chromosome)
    
    ### The current fitness calculation is very simple 
    def cal_fitness(self):
        '''
        TODO: Allow for more flexibility in instances
        calculate fitness score
        '''
        global H,JR,JC,TEST_INSTANCE, REPETITIONS
        return calculate_expected_value(H,JR,JC,TEST_INSTANCE,self.chromosome,REPETITIONS)

def main():
    global POPULATION_SIZE, MAX_GENERATIONS

    # initialisation of variables
    generation = 1
    found = False
    population = []

    # used to calculate expected runtime
    start = time.process_time()

    # initial population
    for i in range(POPULATION_SIZE):
                gnome = Individual.create_gnome()
                population.append(Individual(gnome))
    
    # print expected runtime 
    print("Expected runtime: {}".format(time.strftime("%H:%M:%S", time.gmtime((time.process_time() - start)*MAX_GENERATIONS))))
    while not found:

        # sort the population in increasing order of fitness score
        population = sorted(population, key = lambda x:x.fitness, reverse=False)

        # TODO: decide on possible succes stopping criteria
        # If population is contains optimal fitness
        # if population[0].fitness >= 0.95:
        #     found = True
        #     break

        new_generation = []
        # Perform Elitism, that means 10% of fittest population
        # goes to the next generation
        s = int((10*POPULATION_SIZE)/100)
        new_generation.extend(population[:s])

        # From 50% of fittest population, Individuals 
        # will mate to produce offspring
        s = int((90*POPULATION_SIZE)/100)
        for _ in range(s):
            parent1 = random.choice(population[:50])
            parent2 = random.choice(population[:50])
            child = parent1.mate(parent2)
            new_generation.append(child)

        population = new_generation
        if generation % 50 == 0:
            print("Generation: {}\nCircuit: \n{}\nFitness: {}".format(generation,population[0].chromosome,population[0].fitness))
        
        if generation == MAX_GENERATIONS: 
            print("max gen reached!!")
            found = True
        generation += 1


    print("Generation: {}\nCircuit: \n{}\nFitness: {}".format(generation,population[0].chromosome,population[0].fitness))


if __name__ == '__main__':
    main()