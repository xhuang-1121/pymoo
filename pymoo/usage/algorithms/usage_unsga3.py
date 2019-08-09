# START unsga3
import numpy as np

from pymoo.factory import get_algorithm, get_problem
from pymoo.optimize import minimize

# create the reference directions to be used for the optimization - just a single one here
ref_dirs = np.array([[1.0]])

# create the algorithm object
algorithm = get_algorithm("unsga3",
                          pop_size=100,
                          ref_dirs=ref_dirs)

# execute the optimization
res = minimize(get_problem("ackley", n_var=30),
               algorithm,
               termination=('n_gen', 150),
               save_history=True,
               seed=1)

print("UNSGA3: Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
# END unsga3


# START no_unsga3
_res = minimize(get_problem("ackley", n_var=30),
                get_algorithm("nsga3", pop_size=100, ref_dirs=ref_dirs),
                termination=('n_gen', 150),
                save_history=True,
                seed=1)
print("NSGA3: Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
# END no_unsga3


# START unsga3_comp

import numpy as np
import matplotlib.pyplot as plt

ret = [np.min(e.pop.get("F")) for e in res.history]
_ret = [np.min(e.pop.get("F")) for e in _res.history]

plt.plot(np.arange(len(ret)), ret, label="unsga3")
plt.plot(np.arange(len(_ret)), _ret, label="nsga3")
plt.title("Convergence")
plt.xlabel("Generation")
plt.ylabel("F")
plt.legend()
plt.show()
# END unsga3_comp
