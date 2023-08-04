from ..Instance import Instance
import requests
import json
from config import REMOTE_SOLVER_HOST

class SolverCollection:
    
    @staticmethod
    def gurobi_local(instance: Instance):
        from lib.gurobi_solver import solve_polynomial_knapsack,VAR_TYPE
        return solve_polynomial_knapsack(instance, VAR_TYPE.BINARY, True, [], gap=None, time_limit=None, verbose=False)

    @staticmethod
    def gurobi_remote(instance: Instance):
        host = REMOTE_SOLVER_HOST
        response = requests.post(host,json=instance.__dict__)
        return json.loads(response.content.decode())
    