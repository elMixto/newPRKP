from pandas import *
import logging
import numpy as np
import json
import pandas as pd
import os
import random
from copy import deepcopy
import itertools
import time as t
import GAHeuristic as ga
import sys
from Instance import Instance
from solver.collection import SolverCollection

if __name__ =='__main__':
	log_name = "logs/polynomial_knapsack.log"
	logging.basicConfig(filename=log_name,format='%(asctime)s %(levelname)s: %(message)s',level=logging.INFO, datefmt="%H:%M:%S",filemode='w')
	PATH_CONFIG_FOLDER="Data"
	NAME_OUTPUT_FILE="results_GAHeu_modified.txt"
	OUTPUT_FOLDER="Results" + NAME_OUTPUT_FILE
	
	#BEST PARAMS FROM TUNING -> n_chromosomes=70; penalization=0.03; weight=0.6
	n_chromosomes = 70
	penalization = 0.03
	weight = 0.60

	from pathlib import Path
	import os

	a = Path("/home/mixto/repositories/PolynomialKnapsackProblem/PolynomialKnapsackProblem/GAHeu/Data")

	data_path = a
	for name_file in data_path.iterdir():
		print("Doing file {}".format(name_file))
		with open(name_file, 'r') as fp:
			dict_data = json.load(fp)
			instance = Instance.from_dict(dict_data)
		var_type = 'continuous'
		heuristic = False
		indexes = []
		timeStart = t.time()
		#CONTINUOUS SOLUTION
		of, sol, comp_time = SolverCollection.gurobi_remote(dict_data, var_type, heuristic, indexes)
		#START OF THE GENETIC ALGORITHM
		g = ga.GAHeuristic(sol, dict_data, n_chromosomes, penalization, weight)
		solGA, objfun = g.run()
		timeStop = t.time()

		with open(OUTPUT_FOLDER, 'a+') as f:
			f.write('{},{},{}\n'.format(name_file,objfun,round(timeStop-timeStart,3)))