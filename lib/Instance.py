import logging
import numpy as np
import random
from pathlib import Path
from dataclasses import dataclass
import math
import json

class Instance:
    def __init__(self) -> None:
        pass
    
    def evaluate(self):
        pass

    @classmethod
    def from_file(cls,json_file: dict):
        """Carga la instancia desde un archivo .json"""
        logging.info("Loading instance")
        self = cls()
        self.gamma = json_file['gamma']
        self.budget = json_file['budget']
        self.sizes = np.around(np.random.uniform(json_file['n_items']))
        self.profits = json_file['profits']
        self.costs = json_file['costs']
        self.polynomial_gains = json_file['polynomial_gains']
        self.n_items = json_file['n_items']
        logging.info("simulation end")
        return self

    def save(self,folder_path: str | Path)-> None:
        """Guarda la instancia en una ruta (Para guardar en el directorio de trabajo usar \"/.\")"""
        file = open(folder_path + str(self) + ".json",'w')
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
        instance = cls()
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
        from hashlib import sha1
        return str(sha1(self.to_json_string().encode()).hexdigest())

    def __hash__(self) -> int:
        hash(self._id())

    def __str__(self) -> str:
        return f"Instance_{self.n_items}_{self.gamma}_{round(self.budget,3)}"

if __name__ == "__main__":
    """Esta cosa para manejar tests"""
    test_file = "/home/mixto/repositories/PRKP/data/S_900_324_5669.147.json"
    file = json.load(open(test_file,'r'))
    instance = Instance.from_file(file)
    instance.save("")
    random_instance = Instance.generate(n_items = 1000,gamma=10,seed=10)
    random_instance.save("")
    print(instance._id)

    
