from ..data_structures.Instance import Instance
import requests
import json
from config import REMOTE_SOLVER_HOST
from src.solvers.gurobi import SolverConfig,solve_polynomial_knapsack,VAR_TYPE
class SolverCollection:
    
    @staticmethod
    def gurobi_local(instance: Instance, solver_config: SolverConfig):
        return solve_polynomial_knapsack(instance,solver_config)

    @staticmethod
    def gurobi_remote(instance: Instance,solver_config: SolverConfig ):
        host = REMOTE_SOLVER_HOST
        data = {"instance": instance.to_json_string(),
                "solver_config": solver_config.to_json()
                }
        response = requests.post(host,json=instance.__dict__)
        return json.loads(response.content.decode())
    