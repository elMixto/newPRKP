from lib.heuristic import Heuristic
from pathlib import Path

HEURISTICS_TO_RUN = [Heuristic]
from lib.Instance import Instance

###TODO: Crear un registro lazy con todos los resultados de cada instancia en cada heuristica.
from lib.gurobi_solver import solve_polynomial_knapsack,VAR_TYPE

def main():
    data_path = Path("./data")
    instance_files = [i for i in data_path.glob("*.json")]
    instance0 = Instance.from_file(instance_files[0])
    solution = solve_polynomial_knapsack(instance0,VAR_TYPE.BINARY,False,[])
    print(solution)

if __name__ == "__main__":
    main()