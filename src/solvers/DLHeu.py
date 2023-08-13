from src.data_structures.Instance import Instance
from fastai.tabular.all import RandomSplitter,TabularPandas,CategoryBlock,accuracy,slide,valley,tabular_learner
from fastai.tabular.all import Learner, DataLoader
import pandas as pd
import numpy as np
from functools import lru_cache
from src.data_structures.features import ItemBatchFeature,ItemSingleFeature

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance


class DLHeu:
    def __init__(self,
                 data_features:      list[ItemSingleFeature | ItemBatchFeature],
                 objective_features: list[ItemSingleFeature | ItemBatchFeature]
                 ) -> None:
    
        self.data_features = data_features
        self.objective_features = objective_features
        self.leaner: Learner | None = None
        self.training_dl: DataLoader | None = None

    def create_training_data(self,instances: list[Instance])->pd.DataFrame:
        """Toma las instancias, calcula todas las features y las empaqueta en un dataframe"""
        features: list[ItemBatchFeature | ItemSingleFeature] = self.data_features + self.objective_features
        features_df = None
        for instance in instances:
            data = []
            for feature in features:
                if issubclass(feature,ItemBatchFeature):
                    data.append(feature.batch_evaluate(instance))
                    continue
                elif issubclass(feature,ItemSingleFeature):
                    data.append([feature.evaluate(instance,x) for x in range(instance.n_items)])
                else:
                    raise Exception("Feature is not subclass of ItemBatchFeature or ItemSingleFeature")
                
            data = np.vstack(data).T
            data = pd.DataFrame(data,columns=[f.name for f in features])
            if features_df is None:
                features_df = data
            else:
                features_df = pd.concat([features_df,data])

            return features_df

    def create_model(self,training_data: pd.DataFrame) -> Learner:
        pass




    @staticmethod
    @lru_cache
    def training_features(self: Instance):
        from src.solvers.collection import SolverCollection,SolverConfig,VAR_TYPE
        optimal_solution = SolverCollection.gurobi(self,SolverConfig(VAR_TYPE.BINARY,False,[]))
        #continuous_solution = SolverCollection.gurobi(self,SolverConfig(VAR_TYPE.CONTINOUS,False,[]))
        items = []
        synergies = self.precalcs()
        for i in range(self.n_items):
            feature_array = np.array([
                                        synergies[i][1]/self.n_items,
         #                               self.profits[i]/continuous_solution.o,
          #                              continuous_solution.sol[i],
                                        self.profits[i]/self.budget,
                                        self.costs[i][0]/self.budget,
                                        self.costs[i][1]/self.budget,
                                        optimal_solution.sol[i],
                                            ])
            items.append(feature_array)
        return np.array(items)
    
    @staticmethod
    def features(self: Instance):
        from src.solvers.collection import SolverCollection,SolverConfig,VAR_TYPE
        #continuous_sol = SolverCollection.gurobi(self,SolverConfig(VAR_TYPE.CONTINOUS,False,[]))
        items = []
        synergies = self.precalcs()
        for i in range(self.n_items):
            feature_array = np.array([
                                        synergies[i][1]/self.n_items,
                                        #self.profits[i]/self.budget,
                                        #self.profits[i]/continuous_sol,
                                        #continuous_sol.sol[i],
                                        self.profits[i]/self.budget,
                                        self.costs[i][0]/self.budget,
                                        self.costs[i][1]/self.budget,
                                            ])
            items.append(feature_array)
        return np.array(items)

        