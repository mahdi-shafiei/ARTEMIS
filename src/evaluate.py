import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import jax
import jax.numpy as jnp
import ott

## mmd distance
def mmd_distance(x, y, gamma):
    xx = rbf_kernel(x, x, gamma)
    xy = rbf_kernel(x, y, gamma)
    yy = rbf_kernel(y, y, gamma)

    return xx.mean() + yy.mean() - 2 * xy.mean()

def compute_scalar_mmd(target, transport, gammas=None):
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]

    def safe_mmd(*args):
        try:
            mmd = mmd_distance(*args)
        except ValueError:
            mmd = np.nan
        return mmd

    return np.mean(list(map(lambda x: safe_mmd(target, transport, x), gammas)))

## wasserstein-2 loss
def make_geometry(t0_points, t1_points):
    """ Set up inital/final cloud points living in space endowed with squared Eucliden distance
    """
    point_cloud = ott.geometry.pointcloud.PointCloud(t0_points, t1_points, ott.geometry.costs.SqEuclidean())
    return point_cloud
def compute_ot(t0_points, t1_points):
    """ Solve OT problem
    """
    point_cloud = make_geometry(t0_points, t1_points)
    sinkhorn = ott.solvers.linear.sinkhorn.Sinkhorn()(ott.problems.linear.linear_problem.LinearProblem(point_cloud))
    return sinkhorn
def transport(ot, init_points):
    return ot.to_dual_potentials().transport(init_points)

def compute_wasserstein_2(preds, true):
    ot = compute_ot(preds, true)
    return jnp.sqrt(ot.transport_cost_at_geom(make_geometry(preds, true))).item()

def compute_metrics_subset(target, predicted):
    return {
        'w2': compute_wasserstein_2(target, predicted),
            }
