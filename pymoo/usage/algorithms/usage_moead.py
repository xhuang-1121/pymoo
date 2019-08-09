from pymoo.algorithms.moead import moead
from pymoo.factory import get_problem, get_visualization, get_reference_directions
from pymoo.optimize import minimize

ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

algorithm = moead(
    ref_dirs=ref_dirs,
    n_neighbors=15,
    decomposition="pbi",
    prob_neighbor_mating=0.7
)

res = minimize(get_problem("dtlz2"),
               algorithm,
               termination=('n_gen', 200)
               )

get_visualization("scatter", angle=(45, 45)).add(res.F).show()
