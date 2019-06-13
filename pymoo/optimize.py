import copy

from pymoo.factory import get_termination
from pymoo.model.termination import Termination


def minimize(problem,
             method,
             termination=None,
             copy_method=True,
             **kwargs):
    """

    Minimization of function of one or more variables, objectives and constraints.

    This is used as a convenience function to execute several algorithms with default settings which turned
    out to work for a test single. However, evolutionary computations utilizes the idea of customizing a
    meta-algorithm. Customizing the algorithm using the object oriented interface is recommended to improve the
    convergence.

    Parameters
    ----------

    problem : pymop.problem
        A problem object defined using the pymop framework. Either existing test single or custom single
        can be provided. please have a look at the documentation.

    method : :class:`~pymoo.model.algorithm.Algorithm`
        The algorithm object that should be used for the optimization.

    termination : tuple
        The termination criterium that is used to stop the algorithm when the result is satisfying.

    Returns
    -------
    res : :class:`~pymoo.model.result.Result`
        The optimization result represented as an object.

    """

    # create an evaluator defined by the termination criterion
    if termination is None:
        pass
    elif not isinstance(termination, Termination):
        if isinstance(termination, str):
            termination = get_termination(termination)
        else:
            termination = get_termination(*termination)

    # create a copy of the algorithm object
    if copy_method:
        method = copy.deepcopy(method)

    # use it to solve the problem
    res = method.solve(problem, termination, **kwargs)

    # store the method object in the result
    res.method = method

    return res
