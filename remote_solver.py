from flask import Flask
from asgiref.wsgi import WsgiToAsgi
from flask import jsonify
from lib.gurobi_solver import solve_polynomial_knapsack

app = Flask(__name__)

@app.route("/",methods = ["POST"])
def solve():
    return jsonify({"data":"Hola"})


asgiapp = WsgiToAsgi(app)
