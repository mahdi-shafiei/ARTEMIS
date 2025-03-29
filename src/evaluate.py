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


########## run evaluation on multiple GPUs ############

def compute_wasserstein_2_single(preds, true):
    ot_result = compute_ot(preds, true)
    geom = make_geometry(preds, true)
    cost = ot_result.transport_cost_at_geom(geom)
    return jnp.sqrt(cost)


p_compute_w2 = jax.pmap(compute_wasserstein_2_single)

def compute_wasserstein_2_multi(preds, true):

    num_devices = jax.local_device_count()

    if preds.shape[0] != true.shape[0]:
        n = preds.shape[0]
        n_trim = n - (n % num_devices)  # largest multiple of num_devices
        preds = preds[:n_trim]
        true = true[:n_trim]

    num_devices = jax.local_device_count()
    preds_shards = jnp.array_split(preds, num_devices)
    true_shards = jnp.array_split(true, num_devices)

    preds_sharded = jax.device_put_sharded(preds_shards, jax.local_devices())
    true_sharded = jax.device_put_sharded(true_shards, jax.local_devices())

    w2_shards= p_compute_w2(preds_sharded, true_sharded)

    return jnp.mean(w2_shards).item()

#########################################################


def compute_metrics_subset(target, predicted):
    return {
        'w2': compute_wasserstein_2(target, predicted), 
        # 'w2_multi': compute_wasserstein_2_multi(target, predicted)
            }