from lib.heuristic import Heuristic
from pathlib import Path

HEURISTICS_TO_RUN = [Heuristic]


###TODO: Crear un registro lazy con todos los resultados de cada instancia en cada heuristica.

def main():
    data_path = Path("./data")
    instance_files = [i for i in data_path.glob("*.json")]
    for instance in instance_files:
        pass
    








if __name__ == "__main__":
    main()