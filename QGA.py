# Quantum Genetic Algorithm

import random
import cirq
import numpy as np
import sympy
from HelperQGA import create_instance, energy_func

def generate_parameter_circuit():
    alpha = sympy.Symbol('alpha')
    beta = sympy.Symbol('beta')
    gamma = sympy.Symbol('gamma')
    return create_instance(length=3, p1=alpha, p2=beta, p3=gamma)

# number of individuals in each generation
POPULATION_SIZE = 100

# The choice of instance used for the individuals
## TODO: Allow for more choices to choose from
PARAMETERS = ["alpha","beta","gamma"]
H,JR,JC,TEST_INSTANCE = generate_parameter_circuit()


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
        create random number for a gene to be mutated by
        '''
        return random.randint(-100,100)*random.random()
  
    @classmethod
    def create_gnome(self):
        '''
        create chromosome of parameters initialised between 0 and 1
        '''
        global PARAMETERS
        return {param: random.random() for param in PARAMETERS}
    
    def mate(self, parent2):
        '''
        Perform mating and produce new offspring
        '''
  
        # chromosome for offspring
        child_chromosome = {}

        for param in self.chromosome:    
            # random probability  
            prob = random.random()
  
            # if prob is less than 0.45, insert gene
            # from parent 1
            if prob <= 0.50:
                child_chromosome[param] = self.chromosome[param]
                if prob > 0.45:
                    child_chromosome[param] *= self.mutated_genes()
            else:
                child_chromosome[param] = parent2.chromosome[param]
                if prob > 0.95:
                    child_chromosome[param] *= self.mutated_genes()
        return Individual(child_chromosome)
    
    ### The current fitness calculation is very simple 
    # real_score = (1-abs(target_value_real-individual_value_real))^2
    # imaginary_score = (1-abs(target_value_imaginary-individual_value_imaginary))^2
    # fitness_value = real_score + imaginary_score
    def cal_fitness(self):
        '''
        TODO: Allow for more flexibility in instances, possibly more a large portion back to the helper file
        Calculate fitness score
        '''
        global TEST_INSTANCE
        simulator = cirq.Simulator()
        qubits = cirq.GridQubit.square(3)
        circuit = cirq.resolve_parameters(TEST_INSTANCE, self.chromosome)
        circuit.append(cirq.measure(*qubits, key='x'))
        result = simulator.run(circuit, repetitions=100)
        energy_hist = result.histogram(key='x', fold_func=energy_func(3, H, JR, JC))
        return np.sum([k * v for k,v in energy_hist.items()]) / result.repetitions
  
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
        population = sorted(population, key = lambda x:x.fitness, reverse=False)

        # TODO: decide on possible succes stopping criteria
        # If population is contains optimal fitness
        # if population[0].fitness >= 0.95:
        #     found = True
        #     break

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
  
        if generation % 50 == 0:
            print("Generation: {}\nCircuit: \n{}\nFitness: {}".format(generation,population[0].chromosome,population[0].fitness))
        
        if generation == 2000: 
            print("max gen reached!!")
            found = True
        generation += 1
  
      
    print("Generation: {}\nCircuit: \n{}\nFitness: {}".format(generation,population[0].chromosome,population[0].fitness))
  
if __name__ == '__main__':
    main()