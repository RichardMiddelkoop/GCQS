#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

from experiments.HelperExperiments import LearningCurvePlot, smooth

def average_over_repetitions(backup, n_repetitions, n_timesteps, policy='egreedy', smoothing_window=51):

    reward_results = np.empty([n_repetitions,n_timesteps]) # Result array
    now = time.time()


    for rep in range(n_repetitions): # Loop over repetitions
        if backup == 'DQN':
            rewards = dqn(False, False, (policy != 'egreedy'), n_timesteps)
        elif backup == 'DQN-TN':
            rewards = dqn(False, True, (policy != 'egreedy'), n_timesteps)
        elif backup == 'DQN-EP-TN':
            rewards = dqn(True, True, (policy != 'egreedy'), n_timesteps)

        reward_results[rep] = rewards
        
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))    
    learning_curve = np.mean(reward_results,axis=0) # average over repetitions
    learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve  

def experiment():
    ####### Settings
    # Experiment    
    n_repetitions = 2
    smoothing_window = 1001 
    n_timesteps = 25000
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    backup = 'DQN' # 'DQN' or 'DQN-ER' or 'DQN-TN' or 'DQN-ER-TN'
        
    # Plotting parameters
    plot = False
    
    ####### Experiments
    #### Ablation Study
    Plot = LearningCurvePlot(title = 'Deep Q-learning: effect of experience replay and a target network')    
    policy = 'egreedy'
    backups = ['DQN','DQN-TN','DQN-EP-TN']
    for backup in backups:        
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, policy, smoothing_window)
        Plot.add_curve(learning_curve,label=r'Model {} using policy {}'.format(backup,policy))    
    policy = 'softmax'
    for backup in backups:
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, policy, smoothing_window)
        Plot.add_curve(learning_curve,label=r'Model {} using policy {}'.format(backup,policy))
    Plot.save('ablation.png')

if __name__ == '__main__':
    experiment()