from abc import ABC,abstractmethod
from src.data_structures import Instance
import numpy as np
from numpy.typing import ArrayLike
#Si agrego las Features a data structures, voy a terminar creando dependencia circular

class ItemSingleFeature(ABC):
    """Item feature es una feature que se puede obtener para aun item individual de una instancia"""
    @staticmethod
    @abstractmethod
    def evaluate(instance: Instance, item: int) -> float:
        """Esta funcion debe tomar una instancia y un item, y retornar un flotante"""
        pass
    

class ItemBatchFeature(ABC):
    """
        Item Batch Feature, es una feature que se puede obtener para todos los items de una instancia,
        en una sola operacion, como por ejemplo resolver la versión relajada del problema.
    """
    
    @staticmethod
    @abstractmethod
    def batch_evaluate(instance: Instance) -> ArrayLike:
        """Esta funcion debe tomar una instancia y generar las features para toda la instancia"""
        pass



### Features


class IsInContSol(ItemBatchFeature):
    """Resuelve la relajacion continua del problema de optimizacion y retorna la solucion para cada uno de los items, si se encuentra o no en la solucion"""
    
    @staticmethod
    def batch_evaluate(instance: Instance) -> ArrayLike:
        from src.solvers.collection import SolverCollection,SolverConfig
        return SolverCollection.gurobi(instance,SolverConfig.continous).sol
    
class ProfitOverBudget(ItemSingleFeature,ItemBatchFeature):
    @staticmethod
    def batch_evaluate(instance: Instance) -> ArrayLike:
        return np.array(instance.profits) / instance.budget
    @staticmethod
    def evaluate(instance: Instance, item: int) -> float:
        return instance.profits[item]/instance.budget
    
class LowerCostOverBudget(ItemSingleFeature,ItemBatchFeature):
    
    @staticmethod
    def batch_evaluate(instance: Instance) -> ArrayLike:
        return np.array(instance.costs[0]) / instance.budget
    
    @staticmethod
    def evaluate(instance: Instance, item: int) -> float:
        return instance.costs[item]/instance.budget
    