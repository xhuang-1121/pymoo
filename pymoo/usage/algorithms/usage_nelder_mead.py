from pymoo.factory import get_problem, get_algorithm
from pymoo.optimize import minimize

problem = get_problem("go-damavandi")

res = minimize(problem,
               get_algorithm("nelder-mead"),
               seed=None,
               verbose=True)

print(problem.success(res.X))
print(res.X)
print(res.F)
