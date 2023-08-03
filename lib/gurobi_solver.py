import time
import logging
from gurobipy import GRB,Model,quicksum
from lib.Instance import Instance
from enum import Enum,auto

class VAR_TYPE(Enum):
    CONTINOUS = auto()
    BINARY = auto()

def solve_polynomial_knapsack(
        instance: Instance, 
        var_type: VAR_TYPE, 
        heuristic: bool, 
        indexes, 
        gap=None, 
        time_limit=None, 
        verbose=False):

    n_items = instance.n_items
    items = range(instance.n_items)
    n_hog = len(instance.polynomial_gains)
    hogs = range(n_hog)
    
    if var_type == VAR_TYPE.CONTINOUS:
        var_type = GRB.CONTINUOUS
    else:
        var_type = GRB.BINARY

    problem_name = "polynomial_knapsack"
    logging.info("{}".format(problem_name))

    model = Model(problem_name)
    X = model.addVars(n_items,lb=0,ub=1,vtype=var_type,name='X')  
    Z = model.addVars(n_hog,lb=0,ub=1,vtype=var_type,name='Z')
    Pi = model.addVars(n_items,lb=0,vtype=GRB.CONTINUOUS,name='Pi')
    Rho = model.addVar(lb=0,vtype=GRB.CONTINUOUS,name='Rho')

    obj_funct = quicksum(instance.profits[i] * X[i] for i in items)
    for h, key in enumerate(instance.polynomial_gains):
        obj_funct += instance.polynomial_gains[key] * Z[h]
    obj_funct -= quicksum(instance.costs[i][0] * X[i] for i in items)
    obj_funct -= (instance.gamma * Rho + quicksum(Pi[i] for i in items))
    model.setObjective(obj_funct, GRB.MAXIMIZE)

    #CONSTRAINS
    model.addConstr(
         quicksum(instance.costs[i][0] * X[i] for i in items) 
         + instance.gamma*Rho 
         + quicksum(Pi[i] for i in items) <= instance.budget,
        "budget_limit"
    )

    for i in items:
        model.addConstr(
            Rho + Pi[i] >= (instance.costs[i][1]-instance.costs[i][0]) * X[i],
            "duality_{}".format(i)
        )

    for h, k in enumerate(instance.polynomial_gains):
        k=k.replace("(","").replace(")","").replace("'","").split(",")
        key=[]
        for i in k:
            key.append(int(i))
        key=tuple(key)
        #print("",dict_data['polynomial_gains'][str(key)],"\n",key,"\n")
        if instance.polynomial_gains[str(key)] > 0:
            model.addConstr(
                quicksum(X[i] for i in key) >= len(key) * Z[h],
                "hog {}".format(h)
            )
        else:
            model.addConstr(
                quicksum(X[i] for i in key) <= len(key) - 1 + Z[h],
                "hog {}".format(h)
            )
    if heuristic:
        for i in indexes:
            model.addConstr(
                X[i] >= 1, "mathheur_constr{}".format(i)
            )


    model.update()
    if gap:
        model.setParam('MIPgap', gap)
    if time_limit:
        model.setParam(GRB.Param.TimeLimit, time_limit)
    if verbose:
        model.setParam('OutputFlag', 1)
    else:
        model.setParam('OutputFlag', 0)
    model.setParam('LogFile', './logs/gurobi.log')
    start = time.time()
    model.optimize()
    end = time.time()
    comp_time = end - start
    if model.status == GRB.Status.OPTIMAL:
        sol = [0] * n_items
        for i in items:
            grb_var = model.getVarByName(
                "X[{}]".format(i)
            )
            sol[i] = grb_var.X
        return model.getObjective().getValue(), sol, comp_time
    else:
        return -1, [], comp_time