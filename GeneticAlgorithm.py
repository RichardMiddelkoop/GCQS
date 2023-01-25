######
# TODO: add possibility for ancillairy bits
######

# shell Genetic Algorithm
# Based on the structure of https://www.geeksforgeeks.org/genetic-algorithms/

import random
import cirq
import numpy as np
  
# number of individuals in each generation
POPULATION_SIZE = 10
  
# valid genes of the chromosome
## TODO: replace with all tested and allowed circuit items.
GENES_1 = [cirq.X,cirq.Y,cirq.Z,cirq.H,cirq.S,cirq.T]
GENES_2 = [cirq.CZ,cirq.CNOT,cirq.SWAP,cirq.XX,cirq.YY,cirq.ZZ]
GENES_3 = [cirq.CCNOT,cirq.CCZ,cirq.CSWAP]

# target state vector to be generated
TARGET = [0,0]
  
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
        global GENES_1,GENES_2,GENES_3
        gnome_qubit_len = int(np.log2(np.shape(TARGET)[0]))
        if gnome_qubit_len > 2:
            GENES = GENES_1 + GENES_2 + GENES_3
        elif gnome_qubit_len > 1:
            GENES = GENES_1 + GENES_2
        else:
            GENES = GENES_1
        
        gene = random.choice(GENES)
        if gene in GENES_3:
            return gene(random.sample(range(0, gnome_qubit_len), 3))
        elif gene in GENES_2:
            return gene(random.sample(range(0, gnome_qubit_len), 2))
        else:
            return gene(random.sample(range(0, gnome_qubit_len), 1))
  
    @classmethod
    def create_gnome(self):
        '''
        create chromosome as a string of qubitgenes in superpostion
        '''
        global TARGET
        gnome_qubit_len = int(np.log2(np.shape(TARGET)[0]))
        return cirq.Circuit(cirq.H(qubit) for qubit in cirq.LineQubit.range(gnome_qubit_len))
    
    def mate(self, parent2):
        '''
        TODO: Decide if Elitism is also a solid choice for the Thesis
        Perform mating and produce new offspring
        '''
  
        # chromosome for offspring
        child_chromosome = cirq.Circuit()

        # Seperate both parents in moments
        ## Moments are a collection of Operations taht all act during the same abstract time slice.
        ## Note that if the systems aren't of identical length only the matching section are considered.
        for gene_parent_1, gene_parent_2 in zip(self.chromosome, parent2.chromosome):    
  
            # random probability  
            prob = random.random()
  
            # if prob is less than 0.45, insert gene
            # from parent 1 
            if prob < 0.45:
                child_chromosome.append(gene_parent_1)
  
            # if prob is between 0.45 and 0.90, insert
            # gene from parent 2
            elif prob < 0.90:
                child_chromosome.append(gene_parent_2)
  
            # otherwise insert random gene(mutate), 
            # for maintaining diversity
            else:
                child_chromosome.append(self.mutated_genes())
  
        # create new Individual(offspring) using 
        # generated chromosome for offspring
        return Individual(child_chromosome)
    
    ### The current fitness calculation is very simple 
    # real_score = (1-abs(target_value_real-individual_value_real))^2
    # imaginary_score = (1-abs(target_value_imaginary-individual_value_imaginary))^2
    # fitness_value = real_score + imaginary_score
    def cal_fitness(self):
        '''
        TODO: Estime energy level of the given quantum circuit
        Calculate fitness score, it is the penalites difference between the target state vector 
        and the actual state vector.
        '''
        global TARGET
        fitness = 0
        chromosome_state_vector = cirq.Simulator.simulate(self.chromosome)
        for gene_self, gene_target in zip(chromosome_state_vector, TARGET):
            fitness += (1-abs(gene_self.real-gene_target.real))^2+(1-abs(gene_self.imag-gene_target.imag))^2
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