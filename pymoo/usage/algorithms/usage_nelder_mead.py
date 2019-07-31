from pymoo.factory import get_problem, get_algorithm
from pymoo.optimize import minimize

problem = get_problem("go-xinsheyang04")

res = minimize(problem,
               get_algorithm("nelder-mead", n_max_restarts=100),
               seed=1,
               verbose=False)

print(res.X)
print(res.F)
