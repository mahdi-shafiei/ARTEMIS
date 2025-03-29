# code adapted from https://github.com/matteopariset/unbalanced_sb/tree/main/udsb_f

import jax
import jax.random as random
import jax.numpy as jnp
from jax.lax import fori_loop

from utils import *
from models import *
from datasets import *

class SDE:

    def __init__(self,dataset: Input_Dataset, steps_num=100, batch_size=512):

        self.dataset = dataset
        self.f = dataset.f
        self.g = dataset.g

        self.killer = dataset.killing_function()

        kde_var_coeff = 2
        self.kde_kernel = gaussian_kernel(kde_var_coeff)

        self.splitting_births_frac = dataset.splitting_births_frac

        self.steps_num = steps_num
        self.batch_size = batch_size

        # Available choices: "keep", "freeze"
        self.dead_mode = "keep"

    def get_normalized_mass(self,direction):

        assert len(self.dataset.times) >= 2
        assert len(self.dataset.mass) == len(self.dataset.times)
        assert self.dataset.times[0] == 0
        assert self.dataset.times[-1] == self.steps_num

        mass_max = max(self.dataset.mass)

        if is_forward(direction):
            return self.dataset.mass[0]/mass_max
        else:
            return self.dataset.mass[-1]/mass_max
        

    def langevin_correct(self, key, model, t, pos):
        """ Applies Langevin correction, as detailed in (TChen 2021)
        """
        r = 0.005
        def _langevin_step(i, l_args):
            key, corrected_pos = l_args
            key, key_score, key_score_hat, key_epsilon = random.split(key, 4)
            epsilon_i = random.normal(key_epsilon, shape=corrected_pos.shape)

            Z_t_i = model[FORWARD](key_score, t, corrected_pos)
            Z_hat_t_i = model[BACKWARD](key_score_hat, t, corrected_pos)

            sigma_t_i = jnp.expand_dims(2 * jnp.square(r * self.g(t, corrected_pos)) * jnp.sum(jnp.square(epsilon_i), axis=1) / (jnp.sum(jnp.square(Z_t_i + Z_hat_t_i), axis=1)), axis=1)  # shape: (n,1)

            score_t_i = (Z_t_i + Z_hat_t_i) / self.g(t, corrected_pos)
            corrected_pos = corrected_pos + sigma_t_i * score_t_i + jnp.sqrt(2 * sigma_t_i) * epsilon_i

            return key, corrected_pos
        
        key, next_pos = fori_loop(0, 20, _langevin_step, (key, pos)) # type: ignore
        return key, next_pos
    
    def estimate_density(self, ref_pos, status, x):
        density_est = (status * self.kde_kernel(ref_pos, x)).sum(axis=1) / jnp.clip(status.sum(), 1)

        return density_est
    
    def apply_killer(self, key, death_threshold):
        key, key_deaths, key_births = random.split(key, 3)
        deaths = random.uniform(key_deaths, (death_threshold.shape[0],)) < death_threshold
        births = -random.uniform(key_births, (death_threshold.shape[0],)) < -1-death_threshold

        return key, deaths, births

    def reweight_killing(self, key, direction, death_threshold, ferryman, f_ref_pos, status, t, pos, clip=True):
        """ Computes the posterior death/birth probabilities starting from the prior ones, in `death_threshold`.
        """
        # Non-parametric density estimation
        f_density_est = self.estimate_density(f_ref_pos, status, pos)

        key, key_ferryman = random.split(key)

        # Add 1 to the density to avoid unreachability walls at the beginning of training
        if is_forward(direction):
            death_reweighting = ferryman(rng=key_ferryman, t=t) / (1. + self.dataset.eps + f_density_est)
        else:
            K_t = jnp.clip(1-status.mean(), 1/self.batch_size)
            death_reweighting = ferryman(rng=key_ferryman, t=t) / (1. + self.dataset.eps + f_density_est)

        death_threshold = death_threshold * death_reweighting

        if clip:
            death_threshold = jnp.clip(death_threshold, -1., 1.)
        return key, death_threshold
    

    def sample_f_trajectory(self, key, x_0, score, ferryman, corrector, t_0=0, test=False, max_size=None):
        steps_num = self.steps_num

        if test and max_size is not None:
            traj = jnp.zeros((steps_num+1, max_size, x_0.shape[-1]))
            traj = traj.at[t_0,:len(x_0)].set(x_0)
        else:
            traj = jnp.zeros((steps_num+1, *x_0.shape))
            traj = traj.at[t_0].set(x_0)

        key, key_init_status = random.split(key)

    
        statuses = jnp.zeros(traj.shape[:-1]).astype(bool)
        birth_statuses = jnp.zeros(traj.shape[:-1]).astype(int)

        # Start with right proportion of dead/alive particles
        if test and max_size is not None:
            statuses = statuses.at[t_0,:len(x_0)].set(jnp.bool((1)))
        else:
            statuses = statuses.at[t_0].set(random.uniform(key_init_status, shape=(x_0.shape[0],)) < self.get_normalized_mass(FORWARD))

        key, key_density_b_init = random.split(key)

        def _step(i, args):
            key, curr_pos, part_traj, part_statuses, part_birth_statuses = args

            key, key_score, key_score_hat, key_brownian, key_y = random.split(key, 5)
            rand = {
                'Z': key_score,
                'Z_hat': key_score_hat,
                'dB': key_brownian
            }

            t = (i-1)/steps_num
            dt = 1/steps_num
            curr_g = self.g(t, curr_pos)
            curr_score = score[FORWARD](key_score, t, curr_pos)

            if test and max_size is not None:
                next_pos = curr_pos + (self.f(t, curr_pos) + curr_g * curr_score) * dt + curr_g * jnp.sqrt(dt) * random.normal(key_brownian, traj.at[t_0].get().shape)
            else:
                next_pos = curr_pos + (self.f(t, curr_pos) + curr_g * curr_score) * dt + curr_g * jnp.sqrt(dt) * random.normal(key_brownian, x_0.shape)

            if corrector == "langevin":
                key, next_pos = self.langevin_correct(key, score, t, next_pos)

            if self.killer is not None:
                death_threshold = self.killer(t, next_pos, FORWARD)
                key, death_threshold = self.reweight_killing(key, FORWARD, death_threshold, ferryman, curr_pos, part_statuses.at[i-1].get(), t, next_pos)
                
                key, dead_mask, birth_mask = self.apply_killer(key, death_threshold)

                

                curr_status = part_statuses.at[i-1].get()

                # DEATHS ######################
                curr_status = jnp.logical_and(curr_status, jnp.logical_not(dead_mask))
                ###############################
                
                # BIRTHS ######################
                key, key_birth_type = random.split(key)
                shadow_births_mask, splitting_births_mask = choose_birth_type(key_birth_type, self.splitting_births_frac, birth_mask)

                # ...from shadow paths
                curr_status = birth_from_shadow_paths(curr_status, shadow_births_mask)
                # ...from splitting
                key, curr_status, birth_status, next_pos = birth_by_splitting(key, curr_status, splitting_births_mask, next_pos)
                ###############################

                part_statuses = part_statuses.at[i].set(curr_status)
                part_birth_statuses = part_birth_statuses.at[i].set(birth_status)

                if self.dead_mode == "keep":
                    pass
                elif self.dead_mode == "freeze":
                    raise NotImplementedError("Cannot freeze trajectories")

            part_traj = part_traj.at[i].set(next_pos)
            return key, next_pos, part_traj, part_statuses, part_birth_statuses

        if test and max_size is not None:
            key, x_1, traj, statuses, birth_statuses = fori_loop(t_0+1, steps_num+1, _step, (key, traj.at[t_0].get(), traj, statuses, birth_statuses)) # type: ignore
        else:
            key, x_1, traj, statuses, birth_statuses = fori_loop(t_0+1, steps_num+1, _step, (key, x_0, traj, statuses, birth_statuses)) # type: ignore
        return traj, None, statuses, birth_statuses


    def sample_b_trajectory(self, key, x_1, score, ferryman, corrector, test=False, max_size=None):

        steps_num = self.steps_num

        if test and max_size is not None:
            traj = jnp.zeros((steps_num+1, max_size, x_1.shape[-1]))
            traj = traj.at[-1,:len(x_1)].set(x_1)
        else:
            traj = jnp.zeros((steps_num+1, *x_1.shape))
            traj = traj.at[-1].set(x_1)

        key, key_init_status = random.split(key)

        statuses = jnp.zeros(traj.shape[:-1]).astype(bool)  # shape (k+1,n)
        birth_statuses = jnp.zeros(traj.shape[:-1]).astype(int)

        if test and max_size is not None:
            statuses = statuses.at[-1,:len(x_1)].set(jnp.bool((1)))
            birth_places = jnp.copy(traj.at[-1].get())  # shape (n,d)
        else:
            # Start with right proportion of dead/alive particles
            statuses = statuses.at[-1].set(random.uniform(key_init_status, shape=(x_1.shape[0],)) < self.get_normalized_mass(BACKWARD))
            birth_places = jnp.copy(x_1)  # shape (n,d)

        key, key_density_f_init = random.split(key)

        def _step(i, args):
            key, curr_pos, part_traj, part_statuses, part_birth_statuses, part_birth_places = args
            key, key_score, key_score_hat, key_brownian, key_y = random.split(key, 5)
            rand = {
                "Z": key_score,
                "Z_hat": key_score_hat,
                "dB": key_brownian
            }

            i = steps_num - i
            
            t = (i+1)/steps_num
  
            dt = 1/steps_num
            curr_g = self.g(t, curr_pos)
            curr_score = score[BACKWARD](key_score_hat, t, curr_pos)

            if test and max_size is not None:
                next_pos = curr_pos - ((self.f(t, curr_pos) - curr_g * curr_score) * dt + curr_g * jnp.sqrt(dt) * random.normal(key_brownian, traj.at[-1].get().shape))
            else:
                next_pos = curr_pos - ((self.f(t, curr_pos) - curr_g * curr_score) * dt + curr_g * jnp.sqrt(dt) * random.normal(key_brownian, x_1.shape))

            
            if corrector == "langevin":
                key, next_pos = self.langevin_correct(key, score, t, next_pos)

            if self.killer is not None:
                death_threshold = self.killer(t, next_pos, BACKWARD)

                key, death_threshold = self.reweight_killing(key, BACKWARD, death_threshold, ferryman, curr_pos, part_statuses.at[i+1].get(), t, next_pos)
                
                key, birth_mask, dead_mask = self.apply_killer(key, death_threshold)
                curr_status = part_statuses.at[i+1].get()

                # DEATHS ######################
                curr_status = jnp.logical_and(curr_status, jnp.logical_not(dead_mask))
                ###############################
                
                # BIRTHS ######################
                key, key_birth_type = random.split(key)
                shadow_births_mask, splitting_births_mask = choose_birth_type(key_birth_type, self.splitting_births_frac, birth_mask)

                # ...from shadow paths
                curr_status = birth_from_shadow_paths(curr_status, shadow_births_mask)
                # ...from splitting
                key, curr_status, birth_status, next_pos = birth_by_splitting(key, curr_status, splitting_births_mask, next_pos)
                ###############################

                part_statuses = part_statuses.at[i].set(curr_status)

                part_birth_statuses = part_birth_statuses.at[i].set(birth_status)

                born_now = jnp.logical_and(jnp.logical_not(part_statuses.at[i+1].get()), part_statuses.at[i].get())
                born_now_mask = jnp.expand_dims(born_now, axis=1)
                part_birth_places = born_now_mask * next_pos + jnp.logical_not(born_now_mask) * part_birth_places

            part_traj = part_traj.at[i].set(next_pos)
            return key, next_pos, part_traj, part_statuses, part_birth_statuses, part_birth_places

        if test and max_size is not None:
            key, x_0, traj, statuses, birth_statuses, _ = fori_loop(1, steps_num+1, _step, (key, traj.at[-1].get(), traj, statuses, birth_statuses, birth_places)) # type: ignore
        else:
            key, x_0, traj, statuses, birth_statuses, _ = fori_loop(1, steps_num+1, _step, (key, x_1, traj, statuses, birth_statuses, birth_places)) # type: ignore

        if self.dead_mode == "keep":
            pass
        elif self.dead_mode == "freeze":
            assert False, "Unimplemented"

        return traj, None, statuses, birth_statuses


    def sample_trajectory(self, key, direction, x_init, score, ferryman, t_0=0, corrector="", test=False, max_size=None):
        if is_forward(direction):
            return self.sample_f_trajectory(key, x_init, score, ferryman, corrector,  t_0=t_0, test=test, max_size=max_size)
        else:
            return self.sample_b_trajectory(key, x_init, score, ferryman, corrector, test=test, max_size=max_size)



