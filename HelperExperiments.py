import numpy as np
import matplotlib.pyplot as plt
import argparse
from Experiments import saveLoad
from scipy.signal import savgol_filter
from CLQGA import Individual
from os import walk


class LearningCurvePlot:

    def __init__(self,xlabel, ylabel, title=None):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)      
        if title is not None:
            self.ax.set_title(title)
        
    def add_curve(self,y,label=None):
        ''' y: the curve
        label: string to appear as label in plot legend '''
        if label is not None:
            x = np.arange(len(y))
            self.ax.plot(x*50,y,label=label)
        else:
            self.ax.plot(y)
    
    def add_point(self,x,y, label=None):
        '''y: the point
        label: string to appear as label in plot legend'''
        
        if label is not None:
            self.ax.scatter(x, y,label=label)
        else:
            self.ax.scatter(x, y)
    def set_ylim(self,lower,upper):
        self.ax.set_ylim([lower,upper])

    def add_hline(self,height,label):
        self.ax.axhline(height,ls='--',c='k',label=label)

    def save(self,name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(name,dpi=300)

def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly)

if __name__ == '__main__':
    # Test Learning curve plot
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=False, help="Input name pickle file with the experimental data to generate graphs, to add multiple curve use a comma separator eg. \"result1,result2\"")
    parser.add_argument('--x', required=False, help="x label name")
    parser.add_argument('--y', required=False, help="y label name")
    parser.add_argument('--out', required=False, help="Input name of Plot title and filename")
    args = parser.parse_args()
    if args.input == None:
        filenames = next(walk('./'), (None, None, []))[2]  # [] if no file
        outputs = []
        for filename in filenames:
            if "experiment" in filename: outputs.append(filename)
    else:
        outputs = (args.input).split(",")
    
    if args.out == None:
        filename = "test"
    else:
        filename = str(args.out).strip()
    if args.x == None:
        x = "x"
    else:
        x = args.x
    if args.y == None:
        y = "y"
    else:
        y = args.y

    graph = LearningCurvePlot(xlabel=x,ylabel=y,title=filename)
    for i in outputs:
        output = saveLoad("load",i, None)
        experiment_population = output[0]
        experiment_duration = output[1]
        experiment_average_fitness_50_increment = output[2]
        experiment_average_crowd_score_50_increment = output[3]
        experiment_average_error_rate_50_increment = output[4]
        add_choice = experiment_average_fitness_50_increment
        if len(add_choice) == 1:
            graph.add_point(experiment_average_error_rate_50_increment,add_choice,label='method {}'.format(i))
        else:
            graph.add_curve(smooth(add_choice, window=2),label='method {}'.format(i))

    graph.save(name=(filename + ".png"))