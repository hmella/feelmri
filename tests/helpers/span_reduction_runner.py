"""Does the hold-out guard's field span survive a rank that owns no elements?

`_holdout_residual` reduces the field's range with MIN/MAX across ranks. A rank
with no elements has no opinion about that range, so it has to contribute the
reduction IDENTITIES; contributing 0.0 makes the span straddle zero, and on a
field that does not -- a shim written as a large uniform offset plus a small
spatial term, which is how `B0Field` documents them -- the span then picks up
the whole offset and the interpolation threshold can no longer fire.

Rank 0 owns every element and rank 1 owns none, which is the shape a graded
mesh produces on its own at enough ranks.
"""
import numpy as np

from feelmri.MPIUtilities import MPI_rank
from feelmri.MRObjects import B0Field


class _Mesh:
  def __init__(self, nodes, cells):
    self.local_nodes, self.local_elements = nodes, cells


def main():
  lat = np.stack(np.meshgrid(*[np.linspace(-0.1, 0.1, 4)] * 3,
                             indexing='ij'), -1).reshape(-1, 3)
  elems = (np.random.default_rng(0).permutation(len(lat))[:60].reshape(-1, 4)
           if MPI_rank == 0 else np.zeros((0, 4), dtype=int))
  shim = lambda q: 5.0 + 1.0e-3 * q[:, 2]
  fitted = B0Field.fit(shim, lat, collective=False)
  _gap, span = B0Field._holdout_residual(shim, _Mesh(lat, elems), lat,
                                         fitted, True)
  print(f'RANK {MPI_rank} SPAN {span:.6e}', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
