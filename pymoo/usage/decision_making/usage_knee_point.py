# START knee_point_2d
import os

import numpy as np

from pymoo.configuration import get_pymoo
from pymoo.factory import get_decision_making
from pymoo.visualization.scatter import scatter

pf = np.loadtxt(os.path.join(get_pymoo(), "pymoo", "usage", "decision_making", "knee-2d-all.out"))
dm = get_decision_making("knee_point", no_extreme_points=False)

I = dm.do(pf)

plot = scatter()
plot.add(pf, alpha=0.2)
plot.add(pf[I], color="red", s=100)
plot.show()

# END knee_point_2d



# START knee_point_3d

pf = np.loadtxt(os.path.join(get_pymoo(), "pymoo", "usage", "decision_making", "knee-3d.out"))
dm = get_decision_making("knee_point", no_extreme_points=True)

I = dm.do(pf)

plot = scatter(angle=(10, 140))
plot.add(pf, alpha=0.2)
plot.add(pf[I], color="red", s=100)
plot.show()
# END knee_point_3d

