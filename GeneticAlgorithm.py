# shell Genetic Algorithm
# Based on https://www.geeksforgeeks.org/genetic-algorithms/

import random
  
# number of individuals in each generation
POPULATION_SIZE = 100
  
# valid genes of the chromosome
GENES = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP
QRSTUVWXYZ 1234567890, .-;:_!"#%&/()=?@${[]}'''

# target string to be generated
TARGET = "GeneticAlgorithm"
  
class Individual(object):
    '''
    Class representing individual in population
    '''
    def __init__(self, chromosome):
        self.chromosome = chromosome 
        self.fitness = self.cal_fitness()
  
    @classmethod
    def mutated_genes(self):
        '''
        TODO: Implement an mutation for Cirq system
        create random genes for mutation
        '''
        global GENES
        gene = random.choice(GENES)
        return gene
  
    @classmethod
    def create_gnome(self):
        '''
        TODO: Implement an initial Cirq system as a choromosome
        create chromosome or string of genes
        '''
        global TARGET
        gnome_len = len(TARGET)
        return [self.mutated_genes() for _ in range(gnome_len)]
    
    def mate(self, par2):
        '''
        TODO: Decide if Elitism is also a solid choice for the Thesis
        Perform mating and produce new offspring
        '''
  
        # chromosome for offspring
        child_chromosome = []
        for gp1, gp2 in zip(self.chromosome, par2.chromosome):    
  
            # random probability  
            prob = random.random()
  
            # if prob is less than 0.45, insert gene
            # from parent 1 
            if prob < 0.45:
                child_chromosome.append(gp1)
  
            # if prob is between 0.45 and 0.90, insert
            # gene from parent 2
            elif prob < 0.90:
                child_chromosome.append(gp2)
  
            # otherwise insert random gene(mutate), 
            # for maintaining diversity
            else:
                child_chromosome.append(self.mutated_genes())
  
        # create new Individual(offspring) using 
        # generated chromosome for offspring
        return Individual(child_chromosome)
  
    def cal_fitness(self):
        '''
        TODO: Estime energy level of the given quantum circuit
        Calculate fitness score, it is the number of
        characters in string which differ from target
        string.
        '''
        global TARGET
        fitness = 0
        for gs, gt in zip(self.chromosome, TARGET):
            if gs != gt: fitness+= 1
        return fitness
  
def main():
    global POPULATION_SIZE
  
    # initialisation of variables
    generation = 1
    found = False
    population = []

    # initial population
    for _ in range(POPULATION_SIZE):
                gnome = Individual.create_gnome()
                population.append(Individual(gnome))
  
    while not found:

        # sort the population in increasing order of fitness score
        population = sorted(population, key = lambda x:x.fitness)

        # If population is contains optimal fitness
        if population[0].fitness <= 0:
            found = True
            break

        new_generation = []
  
        # Perform Elitism, that mean 10% of fittest population
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
  
        print("Generation: {}\tString: {}\tFitness: {}".format(generation,"".join(population[0].chromosome),population[0].fitness))
  
        generation += 1
  
      
    print("Generation: {}\tString: {}\tFitness: {}".format(generation,"".join(population[0].chromosome),population[0].fitness))
  
if __name__ == '__main__':
    main()