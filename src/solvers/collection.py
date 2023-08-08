from ..data_structures.Instance import Instance
import requests
import json
from config import REMOTE_SOLVER_HOST
from src.solvers.gurobi import SolverConfig,solve_polynomial_knapsack,VAR_TYPE
from src.solvers.ga.GAHeuristic import GAHeuristic
from time import time
from pathlib import Path
import pickle
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
    def gurobi(instance: Instance, solver_config: SolverConfig):
        try:
            return SolverCollection.gurobi_local(instance,solver_config)
        except:
            pass
        try:
            return SolverCollection.gurobi_remote(instance,solver_config)
        except:
            pass

    @staticmethod
    def baldo_GA(instance: Instance, n_chromosomes: int = 70, penalization: float = 0.03, weight: float = 0.6):
        start = time()
        _, solution, _ = SolverCollection.gurobi(instance,SolverConfig(VAR_TYPE.CONTINOUS,False,[]))
        ga_solver = GAHeuristic(solution,instance,n_chromosomes,penalization,weight)
        solGA, objfun = ga_solver.run()
        solGA = list(map(lambda x: int(x), list(solGA)))
        return objfun,solGA, time() - start
    
    @staticmethod
    def baldo_ML(instance: Instance):
        model_file = Path("/home/mixto/PRKP/src/solvers/MLHeu/model_data/finalized_model_rTrees.sav")
        n_features = 6
        fixed_percentage = 0.85
        clf = pickle.load(open(model_file, 'rb'))
        of, sol_cont, comp_time = solve_polynomial_knapsack(instance, SolverConfig(VAR_TYPE.CONTINOUS,False,[]))
        from src.solvers.MLHeu.functions_ml import prepare_set,fix_variables
        X = prepare_set(n_features,instance, sol_cont)
        y_mlProba = clf.predict_proba(X)		
        y_ml = fix_variables(instance.n_items, y_mlProba, fixed_percentage)
        print(y_ml)
        discrete_config = SolverConfig(VAR_TYPE.BINARY,True,y_ml)
        return solve_polynomial_knapsack(instance,discrete_config)
        #Agregar un sistema iterativo
        #Crear caracteristicas automaticas
    
