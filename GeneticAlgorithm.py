######
# TODO: add possibility for ancillairy bits
# TODO: implement PowGates and variable rotation gates
######

# shell Genetic Algorithm
# Based on the structure of https://www.geeksforgeeks.org/genetic-algorithms/

import random
import cirq
import numpy as np
  
# number of individuals in each generation
POPULATION_SIZE = 100
  
# valid genes of the chromosome
## TODO: replace with all tested and allowed circuit items.
GENES_1 = [cirq.I,cirq.X,cirq.Y,cirq.Z,cirq.H,cirq.S,cirq.T]
GENES_2 = [cirq.CZ,cirq.CNOT,cirq.SWAP,cirq.XX,cirq.YY,cirq.ZZ]
GENES_3 = [cirq.CCNOT,cirq.CCZ,cirq.CSWAP]

# target state vector to be generated
TARGET = [0,0,0,0,0.707,0.707,0,0]
# calculation of the number of qbits
QBITLEN = int(np.log2(np.shape(TARGET)[0]))

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
        global GENES_1,GENES_2,GENES_3,QBITLEN
        QBITLEN = int(np.log2(np.shape(TARGET)[0]))
        if QBITLEN > 2:
            GENES = GENES_1 + GENES_2 + GENES_3
        elif QBITLEN > 1:
            GENES = GENES_1 + GENES_2
        else:
            GENES = GENES_1
        
        gene = random.choice(GENES)
        if gene in GENES_3:
            q0, q1, q2 = random.sample(range(0, QBITLEN), 3)
            return gene(cirq.LineQubit(q0),cirq.LineQubit(q1),cirq.LineQubit(q2))
        elif gene in GENES_2:
            q0, q1 = random.sample(range(0, QBITLEN), 2)
            return gene(cirq.LineQubit(q0),cirq.LineQubit(q1))
        else:
            q0 = random.sample(range(0, QBITLEN), 1)
            return gene(cirq.LineQubit(q0[0]))
  
    @classmethod
    def create_gnome(self):
        '''
        create chromosome as a string of qubitgenes in superpostion
        '''
        global TARGET, QBITLEN
        return cirq.Circuit(cirq.I(qubit) for qubit in cirq.LineQubit.range(QBITLEN))
    
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
                if prob > 0.96: # RARE additional mutation to create an additional moment
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
        global TARGET,QBITLEN
        fitness = 0
        maxfitness = 8*(2**QBITLEN)
        chromosome_results = cirq.Simulator().simulate(self.chromosome)
        chromosome_state_vector = np.around(chromosome_results.final_state_vector, 3)
        for gene_self, gene_target in zip(chromosome_state_vector, TARGET):
            fitness += (2-abs(gene_self.real-gene_target.real))**2+(2-abs(gene_self.imag-gene_target.imag))**2
            # Possible heuristic: decreasing the importance of the zero states to try to increase diversity
            zero_state_penalty = 0.85
            if gene_self.real == 0 and gene_self.imag == 0 and gene_target.real == 0 and gene_self.imag == 0:
                fitness -= 8*zero_state_penalty
                maxfitness -= 8*zero_state_penalty
        return fitness/maxfitness
  
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

        # sort the population in decreasing order of fitness score
        population = sorted(population, key = lambda x:x.fitness, reverse=True)

        # If population is contains optimal fitness
        if population[0].fitness >= 0.95:
            found = True
            break

        new_generation = []
  
        # Perform Elitism, that mean 10% of fittest population
        # goes to the next generation
        s = int((10*POPULATION_SIZE)/100)
        new_generation.extend(population[:s])
  
        # From 75% of fittest population, Individuals 
        # will mate to produce offspring
        s = int((90*POPULATION_SIZE)/100)
        for _ in range(s):
            parent1 = random.choice(population[:75])
            parent2 = random.choice(population[:75])
            child = parent1.mate(parent2)
            new_generation.append(child)
  
        population = new_generation
  
        if generation % 50 == 0:
            print("Generation: {}\tCircuit: \n{}\tFitness: {}".format(generation,population[0].chromosome,population[0].fitness))
        
        if generation == 2000: 
            print("max gen reached!!")
            found = True
        generation += 1
  
      
    print("Generation: {}\nCircuit: \n{}\nFitness: {}".format(generation,population[0].chromosome,population[0].fitness))
  
if __name__ == '__main__':
    main()