import os
import unittest

import numpy as np

from pymoo.algorithms.nsga2 import nsga2
from pymoo.configuration import get_pymoo
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from tests.test_usage import test_usage


class AlgorithmTest(unittest.TestCase):

    def test_algorithms(self):
        folder = os.path.join(get_pymoo(), "pymoo", "usage", "algorithms")
        test_usage([os.path.join(folder, fname) for fname in os.listdir(folder)])

    def test_same_seed_same_result(self):
        problem = get_problem("zdt3")
        algorithm = nsga2(pop_size=100, elimate_duplicates=True)

        res1 = minimize(problem, algorithm, ('n_gen', 20), seed=1)
        res2 = minimize(problem, algorithm, ('n_gen', 20), seed=1)

        self.assertEqual(res1.X.shape, res2.X.shape)
        self.assertTrue(np.all(np.allclose(res1.X, res2.X)))


if __name__ == '__main__':
    unittest.main()
