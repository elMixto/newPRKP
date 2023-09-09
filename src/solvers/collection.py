from ..data_structures.Instance import Instance
import requests
import json
from src.config import REMOTE_SOLVER_HOST
from src.solvers.gurobi import SolverConfig,solve_polynomial_knapsack,VAR_TYPE
from src.solvers.MLHeu.functions_ml import prepare_set,fix_variables
from src.config import MLH_MODEL
from src.solvers.ga.GAHeuristic import GAHeuristic
from src.data_structures.Instance import Instance
from time import time
import numpy as np
import torch
from pathlib import Path
import pickle
from numpy.typing import ArrayLike
from dataclasses import dataclass
from src.solvers.DLHeu import DLHeu
import pandas as pd
from functools import lru_cache

@dataclass
class Solution:
    o: float
    sol: ArrayLike
    time: float #Seconds

    def __repr__(self) -> str:
        return f"Sol(of:{self.o},time:{self.time})"

class SolverCollection:

    @staticmethod
    def gurobi_local(instance: Instance, solver_config: SolverConfig = SolverConfig.optimal()):
        return solve_polynomial_knapsack(instance,solver_config)

    @staticmethod
    def gurobi_remote(instance: Instance,solver_config: SolverConfig = SolverConfig.optimal() ):
        host = REMOTE_SOLVER_HOST
        data = {"instance": instance.to_json_string(),
                "solver_config": solver_config.to_json()
                }
        response = requests.post(host,json=data)
        return json.loads(response.content.decode())
    
    @staticmethod
    def gurobi(instance: Instance, solver_config: SolverConfig)->Solution:
        """Esta funcioncilla intenta resolver la instancia con las 2 funciones"""
        try:
            o,sol,t =  SolverCollection.gurobi_remote(instance,solver_config)
            instance.optimal_solution = np.array(sol)
            instance.optimal_objective = o
            return Solution(o,sol,t)
        except Exception as e:
            """
            I know this is bad, but gurobi exceptions dont appear to work with my LSP :c
            so.
            """
            pass
        
        try:
            o,sol,t = SolverCollection.gurobi_local(instance,solver_config)
            instance.optimal_solution = np.array(sol)
            instance.optimal_objective = o
            return Solution(o,sol,t)
        except Exception as e:
            pass
    

    @staticmethod
    def gurobi_optimal(instance: Instance)-> Solution:
        """This is a wrapper to have some solutions cached"""
        if instance.optimal_objective is not None:
            return Solution(instance.optimal_objective,np.array(list(map(abs,instance.optimal_solution))),1)
        return SolverCollection.gurobi(instance,SolverConfig.optimal())
    
    @staticmethod
    def gurobi_continous(instance: Instance)-> Solution:
        """This is a wrapper to have some solutions cached"""
        return SolverCollection.gurobi(instance,SolverConfig.continous())
    

    @staticmethod
    def baldo_GA(instance: Instance, n_chromosomes: int = 70, penalization: float = 0.03, weight: float = 0.6) -> Solution:
        start = time()
        solution = SolverCollection.gurobi(instance,SolverConfig(VAR_TYPE.CONTINOUS,False,[]))
        ga_solver = GAHeuristic(list(solution.sol),instance,n_chromosomes,penalization,weight)
        solGA, objfun = ga_solver.run()
        solGA = list(map(lambda x: int(x), list(solGA)))
        solGA = list(solGA)
        return Solution(objfun,solGA, time() - start)
    
    @staticmethod
    def baldo_ML(instance: Instance)-> Solution:
        start = time()
        model_file = MLH_MODEL
        n_features = 6
        fixed_percentage = 0.85
        clf = pickle.load(open(model_file, 'rb'))
        sol_cont = SolverCollection.gurobi(instance, SolverConfig(VAR_TYPE.CONTINOUS,False,[]))
        X = prepare_set(n_features,instance, sol_cont.sol)
        y_mlProba = clf.predict_proba(X)
        y_ml = fix_variables(instance.n_items, y_mlProba, fixed_percentage)
        discrete_config = SolverConfig(VAR_TYPE.BINARY,True,y_ml)
        final_gurobi = SolverCollection.gurobi(instance,discrete_config)
        return Solution(final_gurobi.o,final_gurobi.sol,time()-start)


    

    @staticmethod
    def DL(instance: Instance)-> Solution:
        """
        Falta manejar que el modelo que sea carga tenga las mismas
        features que la instancia generada
        """
        """This is still experimental :D"""
        from src.data_structures.features import IsInContSol,ProfitOverBudget,LowerCostOverBudget
        from src.data_structures.features import UpperCostOverBudget
        from src.data_structures.features import CountPSynergiesOverNItems
        from src.data_structures.features import IsInOptSol,Budget,Noise
        from fastai.learner import load_learner
        deepl = DLHeu(
                [
                    
                    #ExperimentalFeature,
                    Noise,
                    Budget,
                    UpperCostOverBudget,
                    LowerCostOverBudget,
                    ProfitOverBudget,
                    #CountPSynergiesOverNItems,
                    #StdOfItemProfits,
                    #StdOfItemSyns,
                    #IsInContSol,
                 ],
                [IsInOptSol],
                [10,10]
            )
        learner = deepl.load_model()
        return deepl.solve(instance,learner)

        

