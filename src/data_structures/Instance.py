import logging
import random
from pathlib import Path
from dataclasses import dataclass
import math
import json
from hashlib import sha1
import numpy as np
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class Solution:
    objective: float
    sol: list[bool]
    comp_time: float

@dataclass
class Instance:
    n_items: int
    gamma: int
    budget: float
    profits: list[int]
    costs: list[list[int]]
    polynomial_gains: dict[set[int],int]

    #Estas cosas son para resolver de forma optima

    def evaluate(self):
        pass

    @classmethod
    def from_file(cls,json_file):
        with open(json_file,"r",encoding="utf8") as f:
            json_file = json.load(f)
        return cls.from_dict(json_file)

    @classmethod
    def from_dict(cls,json_file: str):
        """Carga la instancia desde un archivo .json"""
        logging.info("Loading instance")
        gamma = json_file['gamma']
        budget = json_file['budget']
        profits = json_file['profits']
        costs = json_file['costs']
        polynomial_gains = json_file['polynomial_gains']
        n_items = json_file['n_items']
        logging.info("simulation end")
        return cls(n_items,gamma,budget,profits,costs,polynomial_gains)

    def save(self,folder_path: str | Path)-> None:
        """Guarda la instancia en una ruta (Para guardar en el directorio de trabajo usar \"/.\")"""
        with open(folder_path + str(self) + ".json",'w',encoding="utf8") as file:
            json.dump(self.__dict__,file)

    def to_json_string(self)->str:
        return json.dumps(self.__dict__)

    @classmethod
    def generate(cls,n_items: int,gamma: int, seed=None)-> 'Instance':
        """Gamma is generally int(random.uniform(0.2, 0.6) * el)"""
        if seed is None:
            random.seed(43)
        else:
            random.seed(seed)
        instance = Instance(None,None,None,None,None,None)
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

        array_profits = list(array_profits)
        matrix_costs = matrix_costs.reshape(n_items, 2)
        matrix_costs = matrix_costs.tolist()
        instance.profits = array_profits
        instance.costs = matrix_costs
        instance.polynomial_gains = polynomial_gains
        return instance

    def _id(self):
        return str(sha1(self.to_json_string().encode()).hexdigest())

    def __hash__(self) -> int:
        return hash(self._id())

    def __str__(self) -> str:
        return f"Instance_{self.n_items}_{self.gamma}_{round(self.budget,3)}"
    