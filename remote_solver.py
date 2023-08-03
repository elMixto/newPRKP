from flask import Flask,request
from asgiref.wsgi import WsgiToAsgi
from flask import jsonify
from lib.gurobi_solver import solve_polynomial_knapsack,VAR_TYPE
from lib.Instance import Instance

app = Flask(__name__)

@app.route("/",methods = ["POST"])
def solve():
    data = request.json
    instance = Instance.from_dict(data)    
    return jsonify(solve_polynomial_knapsack(instance,VAR_TYPE.BINARY,False,[]))

asgiapp = WsgiToAsgi(app)
