import unittest
from src.data_structures.Instance import Instance
from src.solver import SolverCollection

class TestStringMethods(unittest.TestCase):

    def test_generator(self):
        """Test intance generator"""
        Instance.generate(n_items = 10, gamma = 0)


    def test_gurobi_solver(self):
        """Test local or remote gurobi solver"""
        instance = Instance.generate(n_items = 10, gamma = 0)
        failed = 0
        try:
            SolverCollection.gurobi_local(instance)
        except Exception:
            failed += 1
        try:
            SolverCollection.gurobi_remote(instance)
        except Exception:
            failed += 1
        if failed >= 2:
            raise Exception()
                

if __name__ == '__main__':
    unittest.main()