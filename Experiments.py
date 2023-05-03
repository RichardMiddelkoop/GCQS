#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import subprocess

def saveLoad(opt,pickleName, object):
    '''opt "save" or "load"'''
    save = None
    if opt == "save":
        f = open(pickleName, 'wb')
        pickle.dump(object, f)
        f.close()
        print('data saved')
    elif opt == "load":
        f = open(pickleName, 'rb')
        save = pickle.load(f)
        f.close()
    else:
        print('Invalid saveLoad option')
    return save

def experiment():
    ####### Settings
    # Experiment    
    parameters = ["NR_OF_QUBITS=4,NR_OF_ISING_QUBITS=4,NR_OF_GATES=20","NR_OF_QUBITS=6,NR_OF_ISING_QUBITS=6,NR_OF_GATES=30","NR_OF_QUBITS=8,NR_OF_ISING_QUBITS=8,NR_OF_GATES=40","NR_OF_QUBITS=10,NR_OF_ISING_QUBITS=10,NR_OF_GATES=50"]
    input = "python3 ./CLQGA.py --arguments "
    pickle_name = "experiment_3_may_gateScaling_evolution_2000"
    ####### Experiments
    for i in range(0,len(parameters)):        
        exp_pickle_name = str(pickle_name) + "_" + str(i) 
        exp_input = str(input) + str(parameters[i]) + " --write " + str(exp_pickle_name)
        out = subprocess.call(exp_input, shell=True)
    
if __name__ == '__main__':
    experiment()