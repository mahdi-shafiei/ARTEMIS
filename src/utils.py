# code adapted from https://github.com/matteopariset/unbalanced_sb/tree/main/udsb_f

import jax
import jax.random as random
import jax.numpy as jnp
from jax import vmap # type: ignore


## Globals

PRNGKey = jnp.ndarray
eval_frequency = 10

directions = {"forward":"forward", "backward":"backward"}
models = directions | {"ferryman": "ferryman"}

# Representations of directions
FORWARD = directions["forward"]
BACKWARD = directions["backward"]
FERRYMAN = models["ferryman"]

## Utils

def info(*msg):
    print("[INFO]:", *msg)

def is_forward(direction):
    """
    Safe test of direction: returns True, if `direction` == FORWARD
    but raises an exception if `direction` has an invalid value
    """
    if direction == FORWARD:
        return True
    elif direction == BACKWARD:
        return False
    else:
        raise ValueError(f"Unknown direction: {direction}")

def reverse(direction):
    if is_forward(direction):
        return BACKWARD
    else:
        return FORWARD
    
def broadcast(f, *args, score_only=True):
    if score_only:
        models_list = directions
    else:
        models_list = models

    return {k: f(*[arg[k] for arg in args]) for k in models_list.values()}

def split_key(key):
    key, *rs = random.split(key, 3)
    return key, {
        FORWARD: rs[0],
        BACKWARD: rs[1]
    }

def gaussian_kernel(var_coeff=2.):
    kde_kernel = vmap(lambda pos,x: 1/jnp.sqrt(jnp.pow(2*jnp.pi*var_coeff**2, x.shape[-1])) * jnp.exp(-jnp.sum(jnp.square(jnp.expand_dims(x,0) - pos), axis=-1)/(2*var_coeff**2)),
                      in_axes=(None, 0))
    return kde_kernel

def kde(kernel,pos,x):
    return kernel(pos,x)

### Miscellaneous

def ema(old_params, new_params, decay=0.99):
    return jax.tree_util.tree_map(lambda old, new: (1-decay)*old + decay*new, old_params, new_params)


## Computational resources

def divergence(key, F, t, x):
    """ Estimate the divergence of `F` using random (+/- 1) projections.

    For `(a_i) ~ unif{-1,+1}`, the function `Q` defined by:
    
     `Q = a · ∇(a·F)`
    
    is an unbiased estimator of `∇·F`, i.e.

     `∇·F = E[Q]`
    """

    n = x.shape[0] #n,d
    # Repetitions to compute empirical expectation
    r = 20
    d = x.shape[1]
    rand_a = random.rademacher(key, (n, r, d))

    def _gaussian_proj(a, t, x):
        F_of_t_x = F(t, jnp.expand_dims(x, axis=0))[0]   # shape: (d,)
        inner_prod = jnp.inner(a, F_of_t_x)
        return jnp.inner(a, F_of_t_x)
    
    grad = jax.grad(_gaussian_proj, argnums=2) # shape: (d,)
    assert len(x.shape) == 2, x.shape
    vec_grad = jax.vmap(jax.vmap(grad, in_axes=(0, None, 0), out_axes=0), in_axes=(1, None, None), out_axes=1)(rand_a, t, x)  # shape: (n, r, d)
    Q = jnp.sum(rand_a * vec_grad, axis=2)  # shape: (n, r)
    return jnp.sum(Q, axis=1) / r  # shape: (n,)


### Reproducibility

def init_logs(epoch: int):
    return {
        'epoch': epoch,
        'ipf_loss': 0.,
        'latent_loss':0,
        'recon_loss':0,
        'td_loss': 0.,
        'ferryman_loss': 0.,
        'loss': 0.,
    }

def print_logs(logs):
    print(f"EPOCH #{logs['epoch']} \t loss={logs['loss']:.3f} \t ipf_loss={logs['ipf_loss']:.3f} \t latent_loss={logs['latent_loss']:.3f} \t recon_loss={logs['recon_loss']:.3f} \t ferryman_loss={logs['ferryman_loss']:.3f}")

def get_logs(logs):
    return {
        "loss":logs['loss'],
        "ipf_loss": logs['ipf_loss'], "latent_loss":logs['latent_loss'],
        "recon_loss":logs['recon_loss'],"ferryman_loss":logs['ferryman_loss']}


# datasets

def triangular_diffusivity(t, max_g=2., min_g=.1):
    # other diffusivities: https://github.com/vsomnath/aligned_diffusion_bridges/blob/main/sbalign/training/diffusivity.py
    return max_g - (max_g-min_g) * 2*jnp.abs(t-.5)

def constant_diffusivity(t, g_max):
    return jnp.ones_like(t) * g_max

def inverse_triangular_diffusivity(t, g_max):
    g_min = .01
    return g_min - 2 * jnp.abs(t - .5) * (g_min-g_max)

def decreasing_diffusivity(t, g_max):
    g_min = .1
    return g_max - jnp.square(t) * (g_max-g_min)

# SDE

def match_locations_with_candidates(locs_mask, cands_mask):
    """ Returns an array of indices, where the `i`-th component points to the location `j` to associate to candidate `i`

    **WARNING**: The indices associated to non-candidates are set to `0`: should therefore filter by `cands_mask` after applying the matches
    """
    locs_idxs = jnp.where(locs_mask, jnp.arange(locs_mask.shape[0]), locs_mask.shape[0])
    cands_idxs = jnp.where(cands_mask, jnp.arange(cands_mask.shape[0]), cands_mask.shape[0])

    # Must extend conditions to avoid out-of-bounds indices
    locs_idxs = jnp.concatenate([locs_idxs, jnp.array([locs_mask.shape[0]])]) # type: ignore
    cands_idxs = jnp.concatenate([cands_idxs, jnp.array([cands_mask.shape[0]])]) # type: ignore
    
    matches = jnp.sort(locs_idxs).at[jnp.argsort(jnp.argsort(cands_idxs))].get().at[:-1].get()
    #jax.debug.print("matches before:{y}", y=matches)

    # Replace placeholder indices with index 0 (leaving invalid indices could be catastrophic in Jax)

    placeholders_mask = (matches == locs_mask.shape[0])
    matches = jnp.logical_not(placeholders_mask) * matches + placeholders_mask * 0
    #jax.debug.print("matches after:{y}", y=matches)

    return matches

def choose_birth_type(key, splitting_frac, birth_mask):
    from_splitting = jnp.logical_and((random.uniform(key, birth_mask.shape) < splitting_frac), birth_mask)
    from_shadow_paths = jnp.logical_and(jnp.logical_not(from_splitting), birth_mask)

    return from_shadow_paths, from_splitting

def birth_from_shadow_paths(status, birth_mask):
    reborn = jnp.logical_and(jnp.logical_not(status), birth_mask)
    status = jnp.logical_or(status, reborn)

    return status

def birth_by_splitting( key, status, birth_mask, reference_pos):

    key, key_half_splitting = random.split(key)

    birth_locations_mask = jnp.logical_and(status, birth_mask)

    dead_trajs_count = jnp.cumsum(jnp.logical_not(status).astype(int))
    # TODO: Shuffle trajectories to resurrect (otherwise there may be bias coming from the description of the marginal sampler, e.g. rectangle_*)
    # Ensure same number of locations & birth candidates 
    candidate_particles_mask = jnp.logical_and(jnp.logical_not(status), dead_trajs_count <= birth_locations_mask.astype(int).sum())
    birth_locations_mask = jnp.logical_and(birth_locations_mask, jnp.cumsum(birth_locations_mask.astype(int)) <= candidate_particles_mask.astype(int).sum())

    matches = match_locations_with_candidates(birth_locations_mask, candidate_particles_mask)
    
    extended_candidate_parts_mask = jnp.expand_dims(candidate_particles_mask, axis=-1)
    next_pos = jnp.logical_not(extended_candidate_parts_mask) * reference_pos + extended_candidate_parts_mask * reference_pos.at[matches].get()

    curr_status = jnp.logical_or(status, candidate_particles_mask)
    
    # IMP note1: extended_candidate_parts_mask-> True for locations where new particles will arrive.
    # IMP note2: matches->stores the indices of particles that are splitting into daughter cells.
    # IMP note3: the order of new particles arriving is given by the particle indices in matches and 
    # filled into locations where extended_candidate_parts_mask is True.

    # DEBUG:
    # store birth particles and new birth locations
    #matches = matches.at[2].set(4)
    #extended_candidate_parts_mask = extended_candidate_parts_mask.at[2].set(True)
    #jax.debug.print("🤯 {y} 🤯", y=matches)
    #jax.debug.print("{y}",y=extended_candidate_parts_mask)
    
    birth_status = matches*extended_candidate_parts_mask.reshape(matches.shape[0],)
    
    # DEBUG:
    #jax.debug.print("{y}",y=birth_status)
    

    return key, curr_status, birth_status, next_pos


# Analysis utils


