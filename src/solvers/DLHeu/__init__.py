from src.data_structures.Instance import Instance
from fastai.tabular.all import RandomSplitter,TabularPandas,CategoryBlock,accuracy,slide,valley,tabular_learner
from fastai.tabular.all import Learner
import pandas as pd
import numpy as np
from functools import lru_cache

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class DLHeu:
    def __init__(self) -> None:
        self.training_data = None
        self.learner = None

    def create_training_data(self)-> pd.DataFrame:
        features_df = None
        for i in range(100):
            print(f"Solving {i}")
            instance = Instance.generate(100,10,seed=i*2)
            features =	self.training_features(instance)
            data = pd.DataFrame(features,columns=["p_syn","profit","l_cost","u_cost","is_in_opt_sol"])
            if features_df is None:
                features_df = data
            else:
                features_df = pd.concat([features_df,data])
        self.training_data = features_df
        return self.training_data

    def train(self,training_data: pd.DataFrame)->Learner :
        splits = RandomSplitter(seed=42)(training_data)
        dls = TabularPandas(
                training_data,
                splits = splits,
                procs = [],
                cat_names = [],
                cont_names = ["p_syn","profit","l_cost","u_cost"],
                y_names =["is_in_opt_sol"],
                y_block = CategoryBlock()
                ).dataloaders(path=".",bs=64)
        learn = tabular_learner(dls,metrics=accuracy,layers=[100,100,100])
        a = learn.lr_find(suggest_funcs=(slide, valley),stop_div=False)
        learn.fit(20,lr=(a[1]+a[0])/2)
        self.learner = learn
        return learn
    
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

        