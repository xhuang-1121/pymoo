import numpy as np

from pymoo.model.algorithm import Algorithm
from pymoo.model.individual import Individual
from pymoo.model.population import Population
from pymoo.model.termination import Termination
from pymoo.operators.repair.out_of_bounds_repair import repair_out_of_bounds
from pymoo.util.display import disp_single_objective
from pymoo.util.misc import at_least_2d_array
# =========================================================================================================
# Implementation
# =========================================================================================================
from pymoo.util.normalization import denormalize


def default_params(*args):
    alpha = 1
    beta = 2.0
    gamma = 0.5
    delta = 0.05
    return alpha, beta, gamma, delta


def adaptive_params(problem):
    n = problem.n_var

    alpha = 1
    beta = 1 + 2 / n
    gamma = 0.75 - 1 / 2 * n
    delta = 1 - 1 / n
    return alpha, beta, gamma, delta


class NelderAndMeadTermination(Termination):

    def __init__(self, xtol=1e-6, ftol=1e-6, n_max_iter=1e6, n_max_evals=1e6):
        super().__init__()
        self.xtol = xtol
        self.ftol = ftol
        self.n_max_iter = n_max_iter
        self.n_max_evals = n_max_evals

    def _do_continue(self, algorithm):
        pop, problem = algorithm.pop, algorithm.problem

        X, F = pop.get("X", "F")

        ftol = np.abs(F[1:] - F[0]).max() <= self.ftol

        # if the problem has bounds we can normalize the x space to to be more accurate
        if problem.has_bounds():
            xtol = np.abs((X[1:] - X[0]) / (problem.xu - problem.xl)).max() <= self.xtol
        else:
            xtol = np.abs(X[1:] - X[0]).max() <= self.xtol

        max_iter = algorithm.n_gen > self.n_max_iter
        max_evals = algorithm.evaluator.n_eval > self.n_max_evals

        return not (ftol or xtol or max_iter or max_evals)

    def do_restart(self, algorithm):
        return self.has_finished(algorithm)


def bring_back_to_bounds_if_necessary(x, direction, problem):
    n, xl, xu = problem.n_var, problem.xl, problem.xu

    _x = x

    def is_in_bounds(x):
        return np.all(x >= xl) and np.all(x <= xu)

    # no repair is necessary
    if is_in_bounds(x):
        return x

    # otherwise bring back to bounds
    else:

        # for each axis
        for ax in range(n):

            if x[ax] < xl[ax]:
                val = (xl[ax] - x[ax]) / direction[ax]
                x = x + direction * val

            elif x[ax] > xu[ax]:
                val = (xu[ax] - x[ax]) / direction[ax]
                x = x + direction * val

    return repair_out_of_bounds(problem, x)


class NelderMead(Algorithm):

    def __init__(self,
                 x0=None,
                 func_params=adaptive_params,
                 termination=NelderAndMeadTermination(xtol=1e-6, ftol=1e-6, n_max_iter=1e6, n_max_evals=1e6),
                 restart_criterion=NelderAndMeadTermination(xtol=1e-2, ftol=1e-2),
                 n_max_restarts=10,
                 **kwargs):

        super().__init__(**kwargs)

        # the function to return the parameter
        self.func_params = func_params

        # the attributes for the simplex operations
        self.alpha, self.beta, self.gamma, self.delta = None, None, None, None

        # the scaling used for the initial simplex
        self.simplex_scaling = None

        # the specified termination criterion
        self.termination = termination

        # restart attributes
        self.n_max_restarts = n_max_restarts
        self.restarts_disabled = False
        self.restart_criterion = restart_criterion
        self.restart_history = []

        # the initial point to be used to build the simplex
        self.x0 = x0

        self.func_display_attrs = disp_single_objective

    def _next(self):

        # perform  step of nelder and mead algorithm and sort by F
        pop = self._step()
        self.pop = pop[np.argsort(pop.get("F")[:, 0])]

        # if a restart should be considered
        if self.n_max_restarts is not None and not self.restarts_disabled:

            # if restarts happened before check if has improved
            no_restart_or_has_improved = len(self.restart_history) == 0 \
                                         or (self.restart_history[-1].get("F").min() - self.pop.get("F").min() > 1e-8)

            # if there should be a restart
            if self.restart_criterion.do_restart(self):

                # if restarts are still left and there was an improvement
                if no_restart_or_has_improved and len(self.restart_history) < self.n_max_restarts:
                    self.pop = self._restart()

                # otherwise just focus on the best result found so far until converging
                else:

                    if len(self.restart_history) > 1:
                        self.pop = self.restart_history[-1]
                    self.restarts_disabled = True

        return self.pop

    def _restart(self):

        # store the current population found in the history
        self.restart_history.append(self.pop)

        # sort the restart history by best values
        I = np.array([h.get("F").min() for h in self.restart_history]).argsort()
        best = self.restart_history[I[0]].copy()

        # initialize new simplex around best point and sort again
        best[1:].set("X", self.initialize_simplex(best[0].X))
        best[1:] = self.evaluator.eval(self.problem, best[1:])
        return best[np.argsort(best.get("F")[:, 0])]

    def _initialize(self):

        # initialize the function parameters
        self.alpha, self.beta, self.gamma, self.delta = self.func_params(self.problem)

        # reset the restart history
        self.restart_history = []

        pop = self._parse_pop(self.x0)

        # the corresponding x values
        x0 = pop[0].X

        # if lower and upper bounds are given take 5% of the range and add
        if self.problem.has_bounds():
            self.simplex_scaling = 0.05 * (self.problem.xu - self.problem.xl)

        # no bounds are given do it based on x0 - MATLAB procedure
        else:
            self.simplex_scaling = 0.05 * x0
            # some value needs to be added if x0 is zero
            self.simplex_scaling[self.simplex_scaling == 0] = 0.00025

        # initialize the simplex
        X = self.initialize_simplex(x0)

        # create a population object
        pop = pop.merge(Population().new("X", X))

        # evaluate the values that are not already evaluated
        self._evaluate_if_not_done_yet(pop)

        # sort by its corresponding function values
        pop = pop[np.argsort(pop.get("F")[:, 0])]

        return pop

    def _step(self):

        # number of variables increased by one - matches equations in the paper
        pop, n = self.pop, self.problem.n_var - 1

        # calculate the centroid
        centroid = pop[:n + 1].get("X").mean(axis=0)

        # reflect and evaluate
        x_reflect = centroid + self.alpha * (centroid - pop[n + 1].X)
        x_reflect = bring_back_to_bounds_if_necessary(x_reflect, (centroid - pop[n + 1].X), self.problem)
        reflect = self.evaluator.eval(self.problem, Individual(X=x_reflect), algorithm=self)

        # whether a shrink is necessary or not
        shrink = False

        # if the new point is not better than the best, but better than second worst - just take it
        if pop[0].F <= reflect.F < pop[n].F:
            pop[n + 1] = reflect

        # if even better than the best - check for expansion
        elif reflect.F < pop[0].F:

            # reflect and evaluate
            x_expand = centroid + self.beta * (x_reflect - centroid)
            x_expand = bring_back_to_bounds_if_necessary(x_expand, (x_reflect - centroid), self.problem)
            expand = self.evaluator.eval(self.problem, Individual(X=x_expand), algorithm=self)

            # if the expansion further improved the function value
            if expand.F < reflect.F:
                pop[n + 1] = expand
            else:
                pop[n + 1] = reflect

        # if not worst than the worst - outside contraction
        elif pop[n].F <= reflect.F < pop[n + 1].F:

            x_contract_outside = centroid + self.gamma * (x_reflect - centroid)
            x_contract_outside = bring_back_to_bounds_if_necessary(x_contract_outside, (x_reflect - centroid),
                                                                   self.problem)
            contract_outside = self.evaluator.eval(self.problem, Individual(X=x_contract_outside), algorithm=self)

            if contract_outside.F <= reflect.F:
                pop[n + 1] = contract_outside
            else:
                shrink = True

        # if the reflection was worse than the worst - inside contraction
        else:

            x_contract_inside = centroid - self.gamma * (x_reflect - centroid)
            contract_inside = self.evaluator.eval(self.problem, Individual(X=x_contract_inside), algorithm=self)

            if contract_inside.F < pop[n + 1].F:
                pop[n + 1] = contract_inside
            else:
                shrink = True

        if shrink:
            for i in range(1, len(pop)):
                pop[i].X = pop[0].X + self.delta * (pop[i].X - pop[0].X)

            pop[1:] = self.evaluator.eval(self.problem, pop[1:], algorithm=self)

        return pop

    def initialize_simplex(self, x0):

        n, xl, xu = self.problem.n_var, self.problem.xl, self.problem.xu

        # repeat the x0 already and add the values
        X = x0[None, :].repeat(n, axis=0)
        X = X - self.simplex_scaling * np.eye(n)

        return X

    # makes sure that a population object is initialized
    def _parse_pop(self, pop):

        # if no initial point is provided at all
        if pop is None:
            if not self.problem.has_bounds():
                raise Exception("Either provide an x0 or a problem with variable bounds!")

            X = np.random.random((1, self.problem.n_var))
            pop = Population().new("X", denormalize(X, self.problem.xl, self.problem.xu * 0.95))

        # if vector provided but not already a population object
        elif not isinstance(pop, Population):
            pop = Population().new("X", at_least_2d_array(pop))

        # if already a population object - assume this is the simplex
        else:

            # if the simplex has not the correct number of points
            if len(pop) != self.problem.n_var + 1:
                raise Exception("If  population object is provided it must be length of n+1")

        return pop


# =========================================================================================================
# Interface
# =========================================================================================================

def nelder_mead(**kwargs):
    return NelderMead(**kwargs)
