from pymoo.model.decomposition import Decomposition


class ASF(Decomposition):

    def _do(self, F, weights, **kwargs):
        _weights = weights.astype(float)
        _weights[weights == 0] = 1e-10
        return ((F - self.utopian_point) / _weights).max(axis=1)
