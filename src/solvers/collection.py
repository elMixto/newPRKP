from ..data_structures.Instance import Instance
import requests
import json
from config import REMOTE_SOLVER_HOST
from src.solvers.gurobi import SolverConfig,solve_polynomial_knapsack,VAR_TYPE
from src.solvers.ga.GAHeuristic import GAHeuristic
from time import time

class SolverCollection:
    
    @staticmethod
    def gurobi_local(instance: Instance, solver_config: SolverConfig):
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
    