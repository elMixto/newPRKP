from flask import Flask,request
from asgiref.wsgi import WsgiToAsgi
from flask import jsonify
from lib.solvers.gurobi.gurobi_solver import solve_polynomial_knapsack,VAR_TYPE,SolverConfig
from lib.data_structures.Instance import Instance

app = Flask(__name__)

@app.route("/",methods = ["POST"])
def solve():
    data = request.json
    instance = Instance.from_dict(data['instance'])
    solver_config = SolverConfig.from_dict(instance['solver_config'])
    return jsonify(solve_polynomial_knapsack(instance,solver_config))

asgiapp = WsgiToAsgi(app)
