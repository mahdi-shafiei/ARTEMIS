# code adapted from https://github.com/matteopariset/unbalanced_sb/tree/main/udsb_f

import jax
from jax import vmap # type: ignore
import jax.random as random
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from functools import partial
from typing import List

from ott.geometry import pointcloud
# from ott.tools.sinkhorn_divergence import sinkhorn_divergence
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn,  acceleration
# from ott.solvers.linear.acceleration import AndersonAcceleration

from utils import *
from datasets import *
from models import *
from sde import *



def init_ipf_loss(dataset, training_setup, sde, model, direction, objective: str):
    """ Initialize the IPF (MM) loss
    """
    if is_forward(direction):
        sign = +1
    else:
        sign = -1

    ipf_mask_dead = training_setup.ipf_mask_dead

    times = jnp.array(dataset.times)
    mass = jnp.array(dataset.mass)

    def _mean_matching_objective(params_train, params_eval, key, k, pos_k, pos_k_plus_1, statuses, steps_num):
        """ Mean-matching objective _(De Bortoli 2021)_

        `loss(Z) = || (X_{k+1} - X_k) - (f ± gZ) Δt ||`
        """

        t = k/steps_num

        # Because of how integrals are discretized, `pos` corresponds to:
        #  - X[k]: for the FORWARD direction
        #  - X[k+1]: for the BACKWARD direction
        if is_forward(direction):
            pos = pos_k
        else:
            pos = pos_k_plus_1

        key, key_model = random.split(key)

        preds = (sde.f(t, pos) + sign * sde.g(t, pos) * training_setup.score(model[direction])(params_train, key_model, t, pos)) / steps_num
        vals = pos_k_plus_1 - pos_k

        mse_loss_vec = jnp.sqrt(jnp.sum(jnp.square(preds - vals), axis=-1))

        if ipf_mask_dead:
            interval_indicator = (times[:-1] <= k) * (k < times[1:])
            mass_delta = jnp.abs((interval_indicator * mass[1:]).sum() - (interval_indicator * mass[:-1]).sum()) / mass.max()
            # alive_mask = jnp.logical_or(statuses.at[k].get(), random.uniform(key, statuses.shape[1:]) < (1. - mass_delta))
            alive_mask = jnp.logical_or(statuses.at[k].get(), statuses[:-1].at[jnp.mod(k-sign, steps_num+1)].get() * random.uniform(key, statuses.shape[1:]) < (1. - mass_delta))
        else:
            alive_mask = jnp.ones_like(statuses[k]).astype(bool)

        mse_loss = jnp.sum(alive_mask * mse_loss_vec) / jnp.clip(alive_mask.sum(), 1)

        return mse_loss

    def _divergence_objective(params_train, params_eval, key, k, pos_k, pos_k_plus_1, statuses, steps_num):
        """ Divergence-based objective. Inspired by _(TChen 2021)_
            but including missing terms to make its magnitude comparable to other losses in _(Liu 2022)_
        
        FORWARD:
        `loss(Z) = 1/2 ||Z||^2 + ∇·(f + gZ) + <Z_hat, Z> - V...`

        BACKWARD:
        `loss(Z_hat) = 1/2 ||Z_hat||^2 + ∇·(-f + gZ_hat) + <Z, Z_hat> + V...`
        """

        t = k/steps_num

        Z_train, Z_eval = training_setup.score(model[direction]), training_setup.score(model[reverse(direction)])

        if is_forward(direction):
            pos = pos_k
            params_forward = params_train
        else:
            pos = pos_k_plus_1
            params_forward = params_eval

        key, key_z_train, key_div, key_z_eval = random.split(key, 4)

        Z_train_value = Z_train(params_train, key_z_train, t, pos)
        Z_eval_value = Z_eval(params_eval, key_z_eval, t, pos)

        def _divergence_arg(t, pos):
            
            return sign * sde.f(t, pos) + sde.g(t, pos) * Z_train(params_train, key_z_train, t, pos)
    

        vec_obj = (.5 * jnp.sum(jnp.square(Z_train_value), axis=1) + divergence(key_div, _divergence_arg, t, pos) + jnp.sum(Z_eval_value * Z_train_value, axis=1)) / steps_num
        
        if ipf_mask_dead:
            alive_mask = statuses.at[k].get()
        else:
            alive_mask = jnp.ones_like(statuses[k]).astype(bool)

        alive_num = jnp.clip(alive_mask.sum(), 1)
        
        # obj = jnp.mean(vec_obj)
        obj = jnp.sum(alive_mask * vec_obj) / alive_num

        return obj

    def _combined_objective(params_train, params_eval, key, k, pos_k, pos_k_plus_1, statuses, steps_num):
        return .8 * _mean_matching_objective(params_train, params_eval, key, k, pos_k, pos_k_plus_1, statuses, steps_num) + .2 * _divergence_objective(params_train, params_eval, key, k, pos_k, pos_k_plus_1, statuses, steps_num)

    if objective == "mean_matching":
        return _mean_matching_objective
    elif objective == "divergence":
        return _divergence_objective
    elif objective == "combined":
        return _combined_objective
    else:
        raise ValueError(f"Unknown training objective: {objective}")


def init_ferryman_loss(dataset, training_setup, sde, ferryman):
    """ Initialize the Ferryman loss
    """

    assert len(dataset.times) == len(dataset.mass), "The number of mass checkpoints should equal to the number of times"
    assert len(dataset.times) >= 2, "Must at least specify mass variation over one interval"
    assert 0 <= min(dataset.times), "Times should be >= 0"
    assert max(dataset.times) <= training_setup.steps_num, "Times should be >= 0"
    assert ((jnp.array(dataset.times)[1:] - jnp.array(dataset.times)[:-1]) > 0).all()

    steps_num = training_setup.steps_num
    intervals_num = len(dataset.times)-1

    assignment_matrix = jnp.zeros((intervals_num, steps_num+1))

    for int_num, (b, e) in enumerate(zip(dataset.times[:-1], dataset.times[1:])):
        assignment_matrix = assignment_matrix.at[int_num,b:e].set(1.)

    mass_max = max(dataset.mass)

    norm_masses = jnp.array(dataset.mass) / mass_max

    def _ferryman_loss(params_train, key, trajectory_direction, trajs, statuses, td_coeff, ferryman_coeff=1.0):
        """ Ferryman Loss (Ours)

        Used to learn the network Q
        """

        time = jnp.linspace(0, 1, steps_num+1)

        if is_forward(trajectory_direction):
            death_sign = 1
            m_init, m_end = norm_masses[:-1], norm_masses[1:]
            interval_assignment = assignment_matrix
        else:
            death_sign = -1
            # Flip the trajectory 
            trajs = trajs[::-1]
            statuses = statuses[::-1]
            time = time[::-1]
            m_init, m_end = norm_masses[::-1][:-1], norm_masses[::-1][1:]
            interval_assignment = assignment_matrix[::-1,::-1]

        statuses = jnp.concatenate([statuses[:1], statuses[:-1]], axis=0)

        interval_assignment = interval_assignment[:,:-1]

        key, key_seed = random.split(key)
        key_model = random.split(key_seed, steps_num+1)
        eval_ferryman = partial(ferryman.apply, params=params_train, direction=trajectory_direction)
        prev_trajs = jnp.concatenate([trajs[:1], trajs[:-1]], axis=0)

        death_threshold = vmap(lambda t, x: sde.killer(t, x, trajectory_direction))(time, trajs)
        infinite_barriers = jnp.isinf(death_threshold)[:-1]

        raw_death_rates = vmap(
            lambda key, t, prev_x, prev_status, x, death_rate: death_sign * sde.reweight_killing(key, trajectory_direction, death_rate, eval_ferryman, prev_x, prev_status, t, x, clip=False)[1]
        )(key_model, time, prev_trajs, statuses, trajs, jnp.nan_to_num(death_threshold, posinf=0.))[:-1]

        death_rates = jnp.logical_not(infinite_barriers) * jnp.clip(raw_death_rates, -1., 1.) + infinite_barriers * 1.


        alive = statuses[:-1]
        alive_to_dead = jnp.logical_and(alive, jnp.logical_not(statuses[1:]))

        # calculations correspond to the first term, when the particle is alive and we calc. its death rate
        all_deaths = alive * jnp.clip(death_rates, 0.)
        real_deaths = alive_to_dead * jnp.clip(death_rates, 0.)

        dead = jnp.logical_not(statuses[:-1])
        dead_to_alive = jnp.logical_and(dead, statuses[1:])

        # Account for birth by splitting (i.e., births from living particles)
        key, key_death_type = random.split(key)
        splitting_birth_selector = random.uniform(key_death_type, shape=dead.shape) < sde.splitting_births_frac
        can_give_birth = jnp.logical_or(dead * jnp.logical_not(splitting_birth_selector), alive * splitting_birth_selector)

        # calculations for first term, when particle is dead and we calc its birth rate (or neg. death rates)
        all_births = can_give_birth * jnp.clip(death_rates, None, 0.)
        real_births = dead_to_alive * jnp.clip(death_rates, None, 0.)

        transitions_num = jnp.clip((training_setup.reality_coefficient * (alive_to_dead + dead_to_alive).astype(float) + (1-training_setup.reality_coefficient) * (alive + dead).astype(float)).sum(axis=1), 1.)

        # # Compute change of mass contraints
        # change_of_mass_constraint = jnp.abs(((possible_deaths + possible_births).sum(axis=1) / transitions_num).sum(axis=0) - (m_end[0]-m_init[-1]))

        transitions = training_setup.reality_coefficient * (real_deaths + real_births) + (1-training_setup.reality_coefficient) * (all_deaths + all_births)
        predicted_mass_variations = jnp.matmul(interval_assignment, (transitions.sum(axis=1) / transitions_num))

        # jax.debug.print("{x})
        #print (predicted_mass_variations.shape, interval_assignment.shape, transitions.shape,jnp.cumsum(predicted_mass_variations).shape)

        change_of_mass_constraint = jnp.abs(m_init[0] - jnp.cumsum(predicted_mass_variations) - m_end).sum()

        # Killing rate regularization
        out_of_bounds_reg = (jnp.clip(jnp.abs(smooth_interval_indicator(raw_death_rates, -1.0, 1.0, 30) * raw_death_rates), 1.) - 1.).sum() / jnp.clip(jnp.logical_not(infinite_barriers).sum(), 1.)

        return (td_coeff > 0.) * ferryman_coeff * (change_of_mass_constraint) + out_of_bounds_reg

    return _ferryman_loss        



def sinkhorn_loss(x,y):

    a = jnp.ones(x.shape[0]) / x.shape[0]  # uniform weights for x
    b = jnp.ones(y.shape[0]) / (y.shape[0])

    geom = pointcloud.PointCloud(x, y)
    """
    # sinkhorn EMD
    prob = linear_problem.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn()
    out = solver(prob)
    loss = out.reg_ot_cost
    """
    """
    # sinkhorn divergence
    # ot = sinkhorn_divergence(geom,x=geom.x,y=geom.y)
    #sinkhorn_kwargs={"rank":10,"initializer":'random'})
    # loss = ot.divergence
    """
    #aa = AndersonAcceleration()

    prob = linear_problem.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn()#momentum=acceleration.Momentum(value=1.0, start=20))
    out = solver(prob)
    loss = out.reg_ot_cost
    return loss


def binary_cross_entropy(x: jnp.ndarray, logits: jnp.ndarray) -> jnp.ndarray:
    """Calculate binary (logistic) cross-entropy from distribution logits.
    Args:
        x: input variable tensor, must be of same shape as logits
        logits: log odds of a Bernoulli distribution, i.e. log(p/(1-p))

    Returns:
        A scalar representing binary CE for the given Bernoulli distribution.
    """
    if x.shape != logits.shape:
        raise ValueError("inputs x and logits must be of the same shape")

    x = jnp.reshape(x, (x.shape[0], -1))
    logits = jnp.reshape(logits, (logits.shape[0], -1))

    return -jnp.sum(x * logits - jnp.logaddexp(0.0, logits), axis=-1)


def kl_gaussian(mean: jnp.ndarray, var: jnp.ndarray) -> jnp.ndarray:
    r"""Calculate KL divergence between given and standard gaussian distributions.

    KL(p, q) = H(p, q) - H(p) = -\int p(x)log(q(x))dx - -\int p(x)log(p(x))dx
            = 0.5 * [log(|s2|/|s1|) - 1 + tr(s1/s2) + (m1-m2)^2/s2]
            = 0.5 * [-log(|s1|) - 1 + tr(s1) + m1^2] (if m2 = 0, s2 = 1)

    Args:
        mean: mean vector of the first distribution
        var: diagonal vector of covariance matrix of the first distribution

    Returns:
        A scalar representing KL divergence of the two Gaussian distributions.
    """
    return -0.5 * jnp.sum(1.0 + jnp.log(var) - var - jnp.square(mean), axis=-1)

def mean_squared_error(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    """Calculate mean squared error between two tensors.

    Args:
            x1: variable tensor
            x2: variable tensor, must be of same shape as x1

    Returns:
            A scalar representing mean square error for the two input tensors.
    """
    if x1.shape != x2.shape:
        raise ValueError("x1 and x2 must be of the same shape")

    x1 = jnp.reshape(x1, (x1.shape[0], -1))
    x2 = jnp.reshape(x2, (x2.shape[0], -1))

    return jnp.sum(jnp.square(x1 - x2), axis=-1)

