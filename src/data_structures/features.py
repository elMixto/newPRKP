from abc import ABC,abstractmethod
from src.data_structures import Instance
import numpy as np
from numpy.typing import ArrayLike
from functools import lru_cache
#Si agrego las Features a data structures, voy a terminar creando dependencia circular

class ItemSingleFeature(ABC):
    """Item feature es una feature que se puede obtener para aun item individual de una instancia"""
    @property
    @abstractmethod
    def name(self)-> str:
        pass

    @staticmethod
    @abstractmethod
    def evaluate(instance: Instance, item: int) -> float:
        """Esta funcion debe tomar una instancia y un item, y retornar un flotante"""
        pass


class ItemBatchFeature(ABC):
    """Item Batch Feature, es una feature que se puede obtener para todos los items de una instancia,
        en una sola operacion, como por ejemplo resolver la versión relajada del problema."""
    @property
    @abstractmethod
    def name(self)-> str:
        pass

    @staticmethod
    @abstractmethod
    def batch_evaluate(instance: Instance) -> ArrayLike:
        """Esta funcion debe tomar una instancia y generar las features para toda la instancia"""
        pass

### Features

class NItems(ItemBatchFeature):
    name = "NItems"

    @staticmethod
    def batch_evaluate(instance: Instance) -> ArrayLike:
        return np.array([instance.n_items/10000 for i in range(instance.n_items)])


class Budget(ItemBatchFeature):
    name = "Budget"
    @staticmethod
    def batch_evaluate(instance: Instance) -> float:
        """Normalized Budget?"""
        return np.array([ instance.budget/instance.n_items  for i in range(instance.n_items)])


class Noise(ItemBatchFeature):
    """Para medir si otras features son tan utiles como el ruido"""
    name = "Noise"
    @staticmethod
    def batch_evaluate(instance: Instance) -> ArrayLike:
        """Average of """
        salida = np.random.random(instance.n_items)
        return salida
    
    
class IsInContSol(ItemBatchFeature):
    """
    Resuelve la relajacion continua del problema de optimizacion y retorna la solucion para cada uno de los items, si se encuentra o no en la solucion
    (Usa el solver cacheado para poder reconstruir la trainingdata de forma rapida)
    """
    name = "IsInContSol"
    @staticmethod
    def batch_evaluate(instance: Instance) -> ArrayLike:
        from src.solvers.collection import SolverCollection,SolverConfig
        return SolverCollection.gurobi_continous(instance).sol
    
class IsInOptSol(ItemBatchFeature):
    """Resuelve la relajacion continua del problema de optimizacion y retorna la solucion para cada uno de los items, si se encuentra o no en la solucion"""
    name = "IsInOptSol"
    @staticmethod
    def batch_evaluate(instance: Instance) -> ArrayLike:
        from src.solvers.collection import SolverCollection,SolverConfig
        return SolverCollection.gurobi_optimal(instance).sol


class ProfitOverBudget(ItemSingleFeature,ItemBatchFeature):
    name = "ProfitOverBudget"

    @staticmethod
    def batch_evaluate(instance: Instance) -> ArrayLike:
        return np.array(instance.profits) / instance.budget
    
    @staticmethod
    def evaluate(instance: Instance, item: int) -> float:
        return instance.profits[item]/instance.budget
    
class LowerCostOverBudget(ItemSingleFeature,ItemBatchFeature):
    name = "LowerCostOverBudget"
    @staticmethod
    def batch_evaluate(instance: Instance) -> ArrayLike:
        return np.array(instance.costs[:,0]) / instance.budget
    
    @staticmethod
    def evaluate(instance: Instance, item: int) -> float:
        return instance.costs[item,0]/instance.budget
    
class UpperCostOverBudget(ItemSingleFeature,ItemBatchFeature):
    name = "UpperCostOverBudget"
    @staticmethod
    def batch_evaluate(instance: Instance) -> ArrayLike:
        return np.array(instance.costs[:,1]) / instance.budget
    
    @staticmethod
    def evaluate(instance: Instance, item: int) -> float:
        return instance.costs[item,1]/instance.budget


class CountPSynergiesOverNItems(ItemBatchFeature):
    """Cuenta las sinergias postivas asociadas a un item especifico, y guarda en memoria los resultados para cada
    instancia que se pase."""

    name = "CountPSynergiesOverNItems"
    @lru_cache(maxsize=None)
    @staticmethod
    def syns(instance: Instance):
        syns = [list([0,0]) for i in range(instance.n_items)]
        for pol_gain, value in instance.polynomial_gains.items():
            if value < 0:
                reference = 0
            else:
                reference = 1
            for item in instance.key_to_set(pol_gain):
                syns[item][reference] += 1
        syns = np.array(syns)
        return syns

    @staticmethod
    def batch_evaluate(instance: Instance) -> ArrayLike:
        return CountPSynergiesOverNItems.syns(instance)[:,1]/instance.n_items
    
    @staticmethod
    def evaluate(instance: Instance,item: int) -> ArrayLike:
        return CountPSynergiesOverNItems.syns(instance)[item,1]/instance.n_items
    

class SynergieReduction(ItemBatchFeature):
    pass