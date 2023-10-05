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

class Instance:
    def __init__(self,n_items: int,gamma: int,budget: float,profits: ArrayLike,
                    costs: NDArray[np.float64],polynomial_gains: dict[set[int],int],
                    optimal_solution: ArrayLike = None,optimal_objective: np.float64 = None, ):
        self.n_items = n_items
        self.budget = budget
        self.gamma = gamma
        self.profits = profits
        self.costs = costs
        self.polynomial_gains = polynomial_gains
        self.syn_work = [key.replace("(","").replace(")","").replace("'","").split(",") for key in polynomial_gains.keys()]
        self.syn_dict = dict()
        for key,value in self.polynomial_gains.items():
            key: list = key.replace("(","").replace(")","").replace("'","").split(",")
            if "" in key:
                key.remove("")       
            key = tuple(map(int,key))
            self.syn_dict[key] = value
        self.synSet = [set(map(int,k)) for k in self.syn_work]
        self.item_vect = np.linspace(1,self.n_items,self.n_items)
        self.nominal_costs = np.array(self.costs[:,0])
        self.total_nominal_costs = np.sum(self.nominal_costs)
        self.upper_costs = np.array(self.costs[:,1])
        self.costs_deltas = np.add(self.upper_costs,-1 * self.nominal_costs)
        self.pol_keys = list(self.polynomial_gains.keys())
        self.optimal_solution = optimal_solution
        self.optimal_objective = optimal_objective
    
    def evaluate(self,sol):
        """Faster eval function to train a model :D"""
        synSet = self.synSet
        sol = np.array(sol,dtype=np.uint32)
        investments = np.multiply(sol,self.item_vect)
        investments = np.array([ int(i) for i in investments if i > 0 ]) - 1
        investments = np.sort(investments)
        investments = investments.tolist()
        investments.sort(key = lambda x: self.costs_deltas[x], reverse = True)
        investments = np.array(investments)
        upperCosts = np.sum([self.upper_costs[x] for x in investments[:self.gamma]])
        nominalCosts = np.sum([self.nominal_costs[x] for x in investments[self.gamma:]])
        total_costs = upperCosts + nominalCosts
        if total_costs <= self.budget:
            of = np.sum([self.profits[x] for x in investments]) - total_costs
            investments=set(investments)
            for i,syn in enumerate(synSet): #Por cada llave en polsyns
                if syn.issubset(investments): #Checkeo si la llave es subconjunto de los investments
                    of += self.polynomial_gains[self.pol_keys[i]] #Y si es subconjunto, retorno la cosa
            return of
        return -1

    def baldo_evaluate(self,sol):
        """ 
        Baldos implementation
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
    
    @staticmethod
    def key_to_tuple(k0):
        input_string = k0
        cleaned_string = input_string.replace("(", "").replace(")", "").replace(" ", "")
        number_strings = cleaned_string.split(",")
        output = [int(num) for num in number_strings]
        output.sort()
        return tuple(output)
    
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
        if 'optimal_objective' in json_file.keys():
            instance.optimal_objective = json_file['optimal_objective']
        if 'optimal_solution' in json_file.keys():
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
    def generate(cls,n_items: int, gamma: int, seed = None)-> 'Instance':
        """Gamma is generally int(random.uniform(0.2, 0.6) * el)"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        matrix_costs = np.zeros((n_items, 2), dtype=float)
        d = [0.3, 0.6, 0.9]
        for i in range(n_items):
            matrix_costs[i, 0] = random.uniform(1, 50)
            matrix_costs[i, 1] = (1 + random.choice(d)) * matrix_costs[i, 0]

        array_profits = np.zeros((n_items), dtype=float)
        for i in range(n_items):
            array_profits[i] = random.uniform(0.8 * np.max(matrix_costs[:, 0]), 100)

        m = [2, 3, 4]
        budget = np.sum(matrix_costs[:, 0]) / random.choice(m)
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
        matrix_costs = matrix_costs.reshape(n_items, 2)
        return Instance(n_items,gamma,budget,array_profits,matrix_costs,polynomial_gains)
    

    @classmethod
    def generate_quadratic(cls,n_items: int, gamma: int, seed = None)-> 'Instance':
        """Gamma is generally int(random.uniform(0.2, 0.6) * el)"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        matrix_costs = np.zeros((n_items, 2), dtype=float)
        d = [0.3, 0.6, 0.9]
        for i in range(n_items):
            matrix_costs[i, 0] = random.uniform(1, 50)
            matrix_costs[i, 1] = (1 + random.choice(d)) * matrix_costs[i, 0]

        array_profits = np.zeros((n_items), dtype=float)
        for i in range(n_items):
            array_profits[i] = random.uniform(0.8 * np.max(matrix_costs[:, 0]), 100)

        m = [2, 3, 4]
        budget = np.sum(matrix_costs[:, 0]) / random.choice(m)
        polynomial_gains = {}
        for i in range(n_items):
            for j in range(n_items):
                if i == j:
                    continue
                tupla = [i,j]
                tupla.sort()
                tupla = tuple(tupla)
                if np.random.random() >= 0.9:
                    polynomial_gains[str(tupla)] = np.random.random()
        matrix_costs = matrix_costs.reshape(n_items, 2)
        return Instance(n_items,gamma,budget,array_profits,matrix_costs,polynomial_gains)
    
    def _id(self):
        return str(sha1(self.to_json_string().encode()).hexdigest())

    def __hash__(self) -> int:
        """
            El hash se obtiene del string que representa todos los parametros, sin tomar en cuenta
            los valores de la solucion optima.
            Idealmente cachear solo las cosas que cuestan mas en computar que este hash
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
        return f"Instance({self.n_items},{self.gamma},#{hash(self)})"
