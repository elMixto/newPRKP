import logging
import random
from pathlib import Path
from dataclasses import dataclass, field
import math
import json
from hashlib import sha1
import numpy as np
from dataclasses_json import dataclass_json
from functools import lru_cache
from time import time
from numpy.typing import ArrayLike,NDArray


@dataclass
class Instance:
    n_items: int
    gamma: int
    budget: float
    profits: ArrayLike  #Array of floats
    costs: NDArray[np.float64]      #Array of tuples with lower_cost and upper_cost (n,2)
    polynomial_gains: dict[set[int],int]
    optimal_solution: None | ArrayLike = field(default=None)
    optimal_objective: None | np.float64         = field(default=None)
    
    @property
    @lru_cache
    def gains(self):
        return len(self.polynomial_gains)
    
    def evaluate(self,sol):
        """ calculate the score of each possible solution
		Args: 
			chromosome: a possible solution
		Return: 
			of: value of the objective function of this chromosome
		"""
        syn_work=[key.replace("(","").replace(")","").replace("'","").split(",") for key in self.polynomial_gains.keys()]
        synSet = [set(map(int,k)) for k in syn_work]
        of = 0
        investments = [i for i in range(0,len(sol)) if sol[i] == 1]	
        investments.sort(key = lambda x: self.costs[x][1] - self.costs[x][0], reverse = True)
        # CHECK FOR FEASIBILITY
        upperCosts = np.sum([self.costs[x][1] for x in investments[:self.gamma]])
        nominalCosts = np.sum([self.costs[x][0] for x in investments[self.gamma:]])
        # IF FEASIBLE, CALCULATE THE OBJECTIVE FUNCTION
        if upperCosts + nominalCosts <= self.budget:
            of += np.sum([self.profits[x] for x in investments])
            of -= upperCosts
            of -= nominalCosts
            investments=set(investments)
            for it in range(len(synSet)):
                syn=synSet[it]
                if syn.issubset(investments):
                    of += self.polynomial_gains[list(self.polynomial_gains.keys())[it]]
        # IF INFEASIBLE, RETURN -1
        else:
            of = -1
        return of

    @staticmethod
    def key_to_set(k0):
        input_string = k0
        cleaned_string = input_string.replace("(", "").replace(")", "").replace(" ", "")
        number_strings = cleaned_string.split(",")
        return set(int(num) for num in number_strings)
    
    @lru_cache
    def precalcs(self):
        syns = [list([0,0]) for i in range(self.n_items)]
        for pol_gain, value in self.polynomial_gains.items():
            if value < 0:
                reference = 0
            else:
                reference = 1
            for item in self.key_to_set(pol_gain):
                syns[item][reference] += 1
        return syns

    #Loader classes
    @classmethod
    def from_file(cls,json_file):
        with open(json_file,"r",encoding="utf8") as f:
            json_file = json.load(f)
        output = cls.from_dict(json_file)

        output.created_time = time()
        return output

    @classmethod
    def from_dict(cls,json_file: dict):
        """Carga la instancia desde un archivo .json"""
        logging.info("Loading instance")
        n_items = json_file['n_items']
        gamma = json_file['gamma']
        budget = json_file['budget']
        profits = np.array(json_file['profits'])
        costs = np.array(json_file['costs'])
        polynomial_gains = json_file['polynomial_gains']
        instance = cls(n_items,gamma,budget,profits,costs,polynomial_gains)
        instance.optimal_objective = json_file['optimal_objective']
        instance.optimal_solution = np.array(json_file['optimal_solution'])
        return instance

    def save(self, folder_path: str | Path) -> str:
        """Guarda la instancia en una ruta 
        (Para guardar en el directorio de trabajo usar \"/\")"""
        json_output = self.to_json_string()
        file_path = Path(folder_path) / f"In{self.n_items}g{self.gamma}#{abs(hash(self))}.json"
        with open(file_path, "w", encoding="utf-8") as json_file:
            json_file.write(json_output)
        return file_path

    def to_json_string(self) -> str:
        """
            Esta funcion retorna un json de la instancia, en un formato,
            que pueda cargarse en el futuro, y además incluye la solucion optima.
        """
        output = {}
        output['n_items'] = self.n_items
        output['gamma'] = self.gamma
        
        
        output['optimal_solution'] = None
        if self.optimal_solution is not None:
            output['optimal_solution'] = self.optimal_solution.tolist()
        output['optimal_objective'] = self.optimal_objective
        output['budget'] = self.budget
        output['profits'] = self.profits.tolist()
        output['costs'] = self.costs.tolist()
        output['polynomial_gains'] = self.polynomial_gains
        return json.dumps(output)

    @classmethod
    def generate(cls,n_items: int,gamma: int, seed=None)-> 'Instance':
        """Gamma is generally int(random.uniform(0.2, 0.6) * el)"""
        if seed is None:
            random.seed(43)
        else:
            random.seed(seed)
        instance = Instance(None,None,None,None,None,None)
        instance.created_time = time()
        instance.n_items = n_items
        instance.gamma = gamma
        matrix_costs = np.zeros((n_items, 2), dtype=float)
        d = [0.3, 0.6, 0.9]

        for i in range(n_items):
            matrix_costs[i, 0] = random.uniform(1, 50)
            matrix_costs[i, 1] = (1 + random.choice(d)) * matrix_costs[i, 0]
        array_profits = np.zeros((n_items), dtype=float)
        
        
        for i in range(n_items):
            array_profits[i] = random.uniform(0.8 * np.max(matrix_costs[:, 0]), 100)

        m = [2, 3, 4]
        instance.budget = np.sum(matrix_costs[:, 0]) / random.choice(m)
        items = list(range(n_items))
        polynomial_gains = {}
        n_it = 0
        for i in range(2, n_items):
            if n_items > 1000:
                for j in range(int(n_items / 2 ** ((i - 1)))):
                    n_it += 1
                    elem = str(tuple(np.random.choice(items, i, replace=False)))
                    polynomial_gains[elem] = random.uniform(1, 100 / i)
            elif n_items <= 1000 and n_items > 300:
                for j in range(int(n_items / 2 ** (math.sqrt(i - 1)))):
                    n_it += 1
                    elem = str(tuple(np.random.choice(items, i, replace=False)))
                    polynomial_gains[elem] = random.uniform(1, 100 / i)
            else:
                for j in range(int(n_items / (i - 1))):
                    n_it += 1
                    elem = str(tuple(np.random.choice(items, i, replace=False)))
                    polynomial_gains[elem] = random.uniform(1, 100 / i)

        array_profits = array_profits
        matrix_costs = matrix_costs.reshape(n_items, 2)
        instance.profits = array_profits
        instance.costs = matrix_costs
        instance.polynomial_gains = polynomial_gains
        return instance

    def _id(self):
        return str(sha1(self.to_json_string().encode()).hexdigest())

    def __hash__(self) -> int:
        """El hash se obtiene del string que representa todos los parametros, sin tomar en cuenta
            los valores de la solucion optima
        """
        output = {}
        output['n_items'] = self.n_items
        output['gamma'] = self.gamma
        output['budget'] = self.budget
        output['profits'] = self.profits.tolist()
        output['costs'] = self.costs.tolist()
        output['polynomial_gains'] = self.polynomial_gains
        
        return hash(str(sha1(json.dumps(output).encode()).hexdigest()))

    def __str__(self) -> str:
        return f"Instance({self.n_items},{self.gamma})#{hash(self)})"
