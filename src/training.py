# code adapted from https://github.com/matteopariset/unbalanced_sb/tree/main/udsb_f

import jax
from jax import grad, value_and_grad, jit, tree_map # type: ignore
import jax.random as random
import jax.numpy as jnp
from jax.lax import fori_loop

import haiku as hk
import matplotlib.pyplot as plt
import optax
from tqdm.auto import tqdm
from functools import partial
from typing import NamedTuple, List, Tuple, Callable
import ipywidgets

import copy
from scipy import linalg
import matplotlib as mpl
import wandb

from utils import *
from datasets import *
from models import *
from loss import *
from training_setup import *

class Trainer:
    def __init__(self, dataset: Dataset, ts: Training_Setup, key, lr = 1e-3, 
                 ferryman_lr=1e-3, vae_lr=1e-3, beta1=1, beta2 = 1, ferryman_coeff=1.0) -> None:

        self.training_setup = ts
        self.dataset = dataset
        
        self.beta1=beta1
        self.beta2=beta2
        self.ferryman_coeff = ferryman_coeff
        self.vae_lr=vae_lr
        self.lr=lr
        self.ferryman_lr=ferryman_lr
        

        key, keys_init = split_key(key)
        key, key_ferryman = random.split(key)

        init_params = broadcast(
            lambda key, model: model.init(key, t=jnp.zeros((ts.batch_size,1)), x=jnp.zeros((ts.batch_size, ts.vae_latent_dim))),
            keys_init,ts.model
        ) | {
            FERRYMAN: ts.ferryman.init(key_ferryman, t=jnp.zeros((ts.batch_size,1)), direction=FORWARD)
        }

        rng_seq = hk.PRNGSequence(1)
        self.vae_params = self.training_setup.vae_model.init(next(rng_seq), t= np.zeros((self.training_setup.vae_batch_size, 1)), 
                                                         x=np.zeros((self.training_setup.vae_batch_size,self.training_setup.vae_input_dim)))

        self.dec_params = self.training_setup.dec.init(next(rng_seq), 
                                                       z=np.zeros((self.training_setup.batch_size, self.training_setup.vae_latent_dim+self.training_setup.vae_t_dim)))

        self.vae_lr_schedule = optax.cosine_decay_schedule(
            init_value=vae_lr,
            decay_steps=100,
        )
        self.vae_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Clip gradients at 1
            optax.adamw(learning_rate = vae_lr),
            # optax.scale_by_schedule(self.vae_lr_schedule)
        )
        self.vae_opt_state = self.vae_optimizer.init(self.vae_params)

        ipf_loss = broadcast(
            lambda direction: value_and_grad(init_ipf_loss(dataset,ts,ts.sde, ts.model, direction, ts.objective)),
            directions
        )
            
        ferryman_loss = value_and_grad(init_ferryman_loss(dataset, ts, ts.sde, ts.ferryman))
  
        optimizer = broadcast(lambda _: optax.chain(optax.clip(1.0), optax.adamw(learning_rate=lr)), models) | {FERRYMAN: optax.chain(optax.clip(1.0), optax.adamw(learning_rate=ferryman_lr))}
        init_opt_state = broadcast(lambda opt, init_params: opt.init(init_params), optimizer, init_params, score_only=False)

        def _zero_model(t, x):
            return jnp.zeros_like(x)
        zero_model = hk.transform(_zero_model)
        _ = zero_model.init(None, t=jnp.zeros((self.training_setup.batch_size, 1)), x=jnp.zeros((self.training_setup.batch_size, self.dataset.input_dim)))

        # Save training entities
        self.ipf_loss = ipf_loss
        self.ferryman_loss = ferryman_loss

        self.zero_model = zero_model

        self.optimizer = optimizer

        self.training_setup.state = (key, init_params)
        self.opt_state = init_opt_state

    @partial(jit, static_argnames=['self', 'd', 'is_warmup'])
    def training_step(self, d, key, params, opt_state, logs, is_warmup=False, td_coeff=.001,):
        sde = self.training_setup.sde
        model = self.training_setup.model
        ferryman = self.training_setup.ferryman

        score = self.training_setup.score

        start_marginals_sampler = self.training_setup.start_marginals_sampler

        paths_reuse = self.training_setup.paths_reuse
        steps_num = self.training_setup.steps_num


        key[d], key_init_points, key_traj, key_ferryman_loss = random.split(key[d], 4)

        if is_warmup:
            sampling_score = {
                FORWARD: partial(self.zero_model.apply, None),
                BACKWARD: partial(score(model[BACKWARD]), params[BACKWARD])
            }
            td_coeff = 0.
        else:
            sampling_score = broadcast(lambda d, m: partial(score(m), params[d]), directions, model)

        sampling_ferryman = partial(ferryman.apply, params=params[FERRYMAN], direction=reverse(d))

        init_points = start_marginals_sampler[reverse(d)](key_init_points)
        trajs, _, statuses, birth_statuses = sde.sample_trajectory(key_traj, reverse(d), init_points, sampling_score, sampling_ferryman,)
      

        # Need for losses which use difference between successive samples (e.g. mean-matching)
        padded_trajs = jnp.concatenate([trajs, trajs.at[-1:].get()], axis=0)  # shape (k+2,n,d)
        padded_statuses = jnp.concatenate([statuses, statuses.at[-1:].get()], axis=0)  # shape (k+2,n)
        

        def _step(k: int, args):
            if not is_forward(d):
                k = steps_num - k

            sink_step_loss, sink_step_grad = 0.0,0.0

            key, params, opt_state, grads, logs = args

            pos_k, pos_k_plus_1 = padded_trajs.at[k].get(), padded_trajs.at[k+1].get()

            key[d], key_ipf_loss, key_sink_loss = random.split(key[d], 3)

            if is_forward(d):
                euler_m_k = k
            else:
                # When sampling backward, must use k+1 in Euler-Maruyama discretization
                euler_m_k = (k+1)

            ipf_step_loss, ipf_step_grad = self.ipf_loss[d](params[d], params[reverse(d)], key_ipf_loss, euler_m_k, pos_k, pos_k_plus_1, padded_statuses, steps_num)

            logs['ipf_loss'] = logs['ipf_loss'] + ipf_step_loss
    

            loss = ipf_step_loss + sink_step_loss
            
            logs['loss'] = logs['loss'] + loss


            # Accumulate gradients
            grads = tree_map(lambda g_acc, g_ipf: g_acc+g_ipf, grads, ipf_step_grad)

            return (
                key,
                params,
                opt_state,
                grads,
                logs
            )

        logs['ferryman_loss'] = 0.

        for _ in range(paths_reuse):
            # Reset gradients
            grads = tree_map(lambda w: jnp.zeros_like(w), params[d])

            path_state = (key, params, opt_state, grads, logs)

            key, params, opt_state, grads, logs = fori_loop(0, steps_num+1, _step, path_state)


            # Follow gradients
            updates, opt_state[d] = self.optimizer[d].update(grads, opt_state[d], params[d])        
            new_params = optax.apply_updates(params[d], updates)
            params[d] = ema(params[d], new_params)

            sampling_d = reverse(d)

            if not is_warmup and is_forward(sampling_d):
                # Learn Ferryman
                ferryman_loss, ferryman_grad = self.ferryman_loss(params[FERRYMAN], key_ferryman_loss, sampling_d, trajs, statuses, td_coeff, self.ferryman_coeff)

                logs['ferryman_loss'] = logs['ferryman_loss'] + ferryman_loss

                ferryman_updates, opt_state[FERRYMAN] = self.optimizer[FERRYMAN].update(ferryman_grad, opt_state[FERRYMAN], params[FERRYMAN])
                new_ferryman_params = optax.apply_updates(params[FERRYMAN], ferryman_updates)
                
                # TODO: Debug. Deactivate ema?
                # params[FERRYMAN] = ema(params[FERRYMAN], new_ferryman_params)
                params[FERRYMAN] = new_ferryman_params

        state = key, params, opt_state, logs

        state[-1]['ipf_loss'] /= (paths_reuse * (steps_num+1))
        state[-1]['ferryman_loss'] /= paths_reuse
        state[-1]['loss'] /= (paths_reuse * (steps_num+1))
        return state
    
    def compute_trajectory_loss(self, key, sde_params, opt_state, logs, d=BACKWARD, eval_val=False, calc_lantent_loss=True,calc_recon_loss=True):

        latent_loss=0.0
        recon_loss=0.0

        #loss1: compute sinkhorn loss between sde latent and vae latent
        if calc_lantent_loss:
            latent_loss, sde_params[d], self.vae_params, self.vae_opt_state, opt_state[d] = self.l_joint_latent_update_fn([sde_params[d],self.vae_params],sde_params, opt_state, self.vae_opt_state, key, d)

            if eval_val:
                self.l_joint_latent_loss_val([sde_params[d],self.vae_params], sde_params, key, d)

        logs["latent_loss"] += latent_loss
        #wandb.log({"vae_latent_loss":latent_loss})

        # loss 2: compute reconstruction loss between VAE recon and groundtruth (only update decoder)
        #recon_loss, sde_params, self.vae_params, self.vae_opt_state = self.l_joint_recon_dec_update_fn(self.vae_params, self.dec_params, sde_params.copy(), self.vae_opt_state, key)
        
        # Loss 3: compute reconstruction loss between VAE recon and groundtruth (update entire vae)
        if calc_recon_loss:
            recon_loss, self.vae_params, self.vae_opt_state = self.l_joint_recon_update_fn(self.vae_params, sde_params, self.vae_opt_state, key,d)

            if eval_val:
                self.l_joint_recon_loss_val(self.vae_params, sde_params, key, d)

        logs["recon_loss"] += recon_loss
        #wandb.log({"vae_recon_loss":recon_loss})

        logs['loss'] += logs['recon_loss'] + logs['latent_loss']
        return key, sde_params, opt_state, logs
    


    @partial(jit, static_argnames=['self','d'])
    def l_joint_latent_update_fn(self, params, sde_params, opt_state, vae_opt_state, key,d=BACKWARD):

        #params = self.training_setup.state[1]

        loss, grads = value_and_grad(self.l_joint_latent_loss)(params, sde_params, key, d)
        updates, new_opt_state = self.optimizer[d].update(grads[0], opt_state[d], params[0])
        new_sde_params_backward = optax.apply_updates(params[0], updates)
        vae_updates, new_vae_opt_state = self.vae_optimizer.update(grads[1], vae_opt_state, params[1])
        new_vae_params = optax.apply_updates(params[1], vae_updates)
        return loss, new_sde_params_backward, new_vae_params, new_vae_opt_state, new_opt_state

    @partial(jit, static_argnames=['self','d'])
    def l_joint_latent_loss(self, params, sde_params, key, d=BACKWARD):

        rng_seq = hk.PRNGSequence(1)
        sde = self.training_setup.sde
        model = self.training_setup.model
        ferryman = self.training_setup.ferryman
        score = self.training_setup.score
        start_marginals_sampler = self.training_setup.start_marginals_sampler
        n_samples = self.training_setup.batch_size

        key, key_init_points, key_traj = random.split(key, 3)
        
        sampling_score = {FORWARD: partial(score(model[FORWARD]), sde_params[FORWARD]), 
                          BACKWARD: partial(score(model[BACKWARD]), params[0])}        
        sampling_ferryman = partial(ferryman.apply, sde_params[FERRYMAN], direction=reverse(d))
        init_points = start_marginals_sampler[d](key_init_points)
        trajs, _, statuses, birth_statuses = sde.sample_trajectory(key_traj, d, init_points, sampling_score, sampling_ferryman,)
        # compute latent loss between SDE and VAE Latents
        loss = 0.0
        times_orig, times = self.dataset.times_orig, self.dataset.times

        for t_orig,t in zip(times_orig, times):
            import numpy as np
            tot_samples = self.dataset.x[self.dataset.x["time"]==t_orig]["time"].values.shape[0]
            sel_idxs=np.random.permutation(tot_samples)[:min(n_samples, tot_samples)]

            train_t=self.dataset.x[self.dataset.x["time"]==t_orig]["time"].values[sel_idxs]
            train_data_t = self.dataset.x[self.dataset.x["time"]==t_orig].values[sel_idxs,:-1]

            outputs: VAEOutput = self.training_setup.vae_model.apply(params[1], next(rng_seq), train_t, train_data_t )
            s_loss = sinkhorn_loss(trajs.at[t].get(), outputs.latent)
            loss = loss+ s_loss 
        return jnp.mean(loss)*self.beta1


    @partial(jit, static_argnames=['self','d'])
    def l_joint_recon_update_fn(self, params, sde_params, vae_opt_state, key,d=BACKWARD):

        loss, grads = value_and_grad(self.l_joint_recon_loss)(params, sde_params, key, d)
        vae_updates, new_vae_opt_state = self.vae_optimizer.update(grads, vae_opt_state, params)
        new_vae_params = optax.apply_updates(params, vae_updates)
        return loss, new_vae_params, new_vae_opt_state,
    
    @partial(jit, static_argnames=['self','d'])
    def l_joint_recon_loss(self, params, sde_params, key, d=BACKWARD):

        rng_seq = hk.PRNGSequence(1)
        n_samples = self.training_setup.batch_size
        key, key_init_points, key_traj = random.split(key, 3)
        loss = 0.0
        times_orig, times = self.dataset.times_orig, self.dataset.times

        for t_orig,t in zip(times_orig, times):
            tot_samples = self.dataset.x[self.dataset.x["time"]==t_orig]["time"].values.shape[0]
            sel_idxs=np.random.permutation(tot_samples)[:min(n_samples, tot_samples)]

            train_t=self.dataset.x[self.dataset.x["time"]==t_orig]["time"].values[sel_idxs]
            train_data_t = self.dataset.x[self.dataset.x["time"]==t_orig].values[sel_idxs,:-1]

            outputs: VAEOutput = self.training_setup.vae_model.apply(params, next(rng_seq), train_t, train_data_t)
        
            s_loss = sinkhorn_loss(train_data_t, outputs.logits)
            # s_loss = jnp.mean(mean_squared_error(train_data_t, outputs.logits))
            loss = loss+ s_loss 
        return jnp.mean(loss)*self.beta1


    @partial(jit, static_argnames=['self'])
    def l_joint_recon_dec_update_fn(self, params, dec_params, sde_params, vae_opt_state, key):

        for param_key in params.keys():
            if "dec" in param_key:
                dec_params[param_key.split("~/")[1]] = params[param_key]

        loss, grads = value_and_grad(self.l_joint_recon_dec_loss)(dec_params, sde_params, key)
        vae_param_key="'variational_auto_encoder/~/"
        vae_grads = copy.deepcopy(params)

        for vae_key in vae_grads:
            if vae_key.split("~/")[1] in grads.keys():
                vae_grads[vae_key] = grads[vae_key.split("~/")[1]]
            else:
                for sub_keys in vae_grads[vae_key].keys():
                    vae_grads[vae_key][sub_keys] = 0.0*vae_grads[vae_key][sub_keys]

        vae_updates, new_vae_opt_state = self.vae_optimizer.update(vae_grads, vae_opt_state, params)
        new_vae_params = optax.apply_updates(params, vae_updates)

        for new_vae_key in new_vae_params.keys():
            if "enc" in new_vae_key:
                new_vae_params[new_vae_key] = params[new_vae_key]
        return loss, sde_params, new_vae_params, new_vae_opt_state
    
    @partial(jit, static_argnames=['self','d'])
    def l_joint_recon_dec_loss(self,params, sde_params, key, d=BACKWARD):

        rng_seq = hk.PRNGSequence(1)
        sde = self.training_setup.sde
        model = self.training_setup.model
        ferryman = self.training_setup.ferryman
        score = self.training_setup.score
        start_marginals_sampler = self.training_setup.start_marginals_sampler

        key, key_init_points, key_traj = random.split(key, 3)
        sampling_score = broadcast(lambda d, m: partial(score(m), sde_params[d]), directions, model)
        
        sampling_ferryman = partial(ferryman.apply, sde_params[FERRYMAN], direction=reverse(d))
        init_points = start_marginals_sampler[d](key_init_points)
        trajs, _, statuses, birth_statuses = sde.sample_trajectory(key_traj, d, init_points, sampling_score, sampling_ferryman,)

        # loss1: compute latent loss between SDE and VAE Latents
        loss = 0.0
        times_orig, times = self.dataset.times_orig, self.dataset.times

        for t_orig,t in zip(times_orig, times):
            #train_t=self.dataset.x[self.dataset.x["time"]==t_orig]["time"].values
            trajs_data_t = trajs.at[t].get()
            trajs_t = np.array([t_orig]*trajs_data_t.shape[0])
            t_emb = get_timestep_embedding(trajs_t, self.training_setup.vae_t_dim)

            z_dec = jnp.concatenate((trajs_data_t,t_emb),-1)
            logits = self.training_setup.dec.apply(params, None, z_dec)
            train_data_t = self.dataset.x[self.dataset.x["time"]==t_orig].values[:,:-1]
            s_loss = sinkhorn_loss(train_data_t, logits)
            loss = loss+ s_loss 
        return jnp.mean(loss)*self.beta2
    
    def l_joint_recon_loss_val(self, params, sde_params, key, d=BACKWARD):

        rng_seq = hk.PRNGSequence(1)
        n_samples = self.training_setup.batch_size
        key, key_init_points, key_traj = random.split(key, 3)
        loss = 0.0
        times_orig, times = self.dataset.times_orig, self.dataset.times
        val_loss=0.0
        
        for t_orig,t in zip(times_orig, times):

            tot_samples = self.dataset.x_val[self.dataset.x_val["time"]==t_orig]["time"].values.shape[0]
            sel_idxs=np.random.permutation(tot_samples)[:min(n_samples, tot_samples)]

            val_t=self.dataset.x_val[self.dataset.x_val["time"]==t_orig]["time"].values[sel_idxs]
            val_data_t = self.dataset.x_val[self.dataset.x_val["time"]==t_orig].values[sel_idxs,:-1]

            outputs: VAEOutput = self.training_setup.vae_model.apply(params, next(rng_seq), val_t, val_data_t)
            s_loss = sinkhorn_loss(val_data_t, outputs.logits)
            val_loss = val_loss+ s_loss 
        #wandb.log({"val: vae_recon_loss": jnp.mean(val_loss)})

    
    def l_joint_latent_loss_val(self, params, sde_params, key, d=BACKWARD):

        rng_seq = hk.PRNGSequence(1)
        sde = self.training_setup.sde
        model = self.training_setup.model
        ferryman = self.training_setup.ferryman
        score = self.training_setup.score
        start_marginals_sampler = self.training_setup.start_marginals_sampler
        n_samples = self.training_setup.batch_size

        key, key_init_points, key_traj = random.split(key, 3)
        sampling_score = {FORWARD: partial(score(model[FORWARD]), sde_params[FORWARD]), 
                          BACKWARD: partial(score(model[BACKWARD]), params[0])}
        sampling_ferryman = partial(ferryman.apply, sde_params[FERRYMAN], direction=reverse(d))

        val_loss=0.0
        import numpy as np
        pi_0_source_val = self.dataset.time_x_val[self.dataset.times[0]]
        tot_samples = pi_0_source_val.shape[0]
        sel_idxs = np.random.permutation(tot_samples)[:min(n_samples, tot_samples)]
        init_points = pi_0_source_val[sel_idxs]

        times_orig, times = self.dataset.times_orig, self.dataset.times

        trajs, _, statuses, birth_statuses = sde.sample_trajectory(key_traj, d, init_points, sampling_score, sampling_ferryman,)
        for t_orig,t in zip(times_orig, times):
            
            tot_samples = self.dataset.x_val[self.dataset.x_val["time"]==t_orig]["time"].values.shape[0]
            sel_idxs=np.random.permutation(tot_samples)[:min(n_samples, tot_samples)]

            val_t=self.dataset.x_val[self.dataset.x_val["time"]==t_orig]["time"].values[sel_idxs]
            val_data_t = self.dataset.x_val[self.dataset.x_val["time"]==t_orig].values[sel_idxs,:-1]

            outputs: VAEOutput = self.training_setup.vae_model.apply(params[1], next(rng_seq), val_t, val_data_t)
            s_loss = sinkhorn_loss(trajs.at[t].get(), outputs.latent)
            val_loss = val_loss+ s_loss 
        #wandb.log({"val: vae_latent_loss": jnp.mean(val_loss)*self.beta1})

    
    @partial(jit, static_argnames=['self', 'get_indiv_loss'])
    def vae_loss(self, params: hk.Params, rng_key: PRNGKey, batch: np.ndarray, 
                time: np.ndarray, get_indiv_loss=False, beta=1e-3):
        """ELBO: E_p[log(x)] - KL(d||q), where p ~ Be(0.5) and q ~ N(0,1)."""

        outputs: VAEOutput = self.training_setup.vae_model.apply(params, rng_key, time, batch )

        log_likelihood = mean_squared_error(batch, outputs.logits)
        kl = kl_gaussian(outputs.mean, jnp.square(outputs.stddev))
        
        kl = beta*kl
        elbo = log_likelihood + kl

        if get_indiv_loss:
            return jnp.mean(elbo), jnp.mean(log_likelihood), jnp.mean(kl)
        return jnp.mean(elbo)
    
    @partial(jit, static_argnames=['self'])
    def vae_update_fn(self,  params: hk.Params, rng_key: PRNGKey, opt_state: optax.OptState, batch: np.ndarray, time: np.ndarray, beta=1e-3):
        """Single SGD update step."""
        loss, grads = value_and_grad(self.vae_loss)( params, rng_key, batch, time, False, beta)
        updates, new_opt_state = self.vae_optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_opt_state
    
    def pretrain_vae(self,):
        rng_seq = hk.PRNGSequence(1)
        train_data = self.dataset.x.values[:,:-1]
        train_time = self.dataset.x.values[:,-1]

        if self.training_setup.dataset.val_split:
            val_data = self.dataset.x_val.values[:,:-1]
            val_time = self.dataset.x_val.values[:,-1]
        
        losses = []
        batch_size= self.training_setup.vae_batch_size

        for e in tqdm(range(1,self.training_setup.vae_epochs+1)):
            epoch_loss = []
            epoch_kl_loss = []
            epoch_mse_loss = []
            for tr_batch,i in enumerate(range(0, len(train_data),batch_size)):

                batch = train_data[i:i+batch_size,:]
                batch_time = train_time[i:i+batch_size]
                mid = (self.training_setup.vae_epochs/2)

                # Implement KL anneal:
                # kl_anneal = 1 / (1 + np.exp( 5 * (mid - e) / mid))
                #kl_anneal = 1 / (1 + np.exp( mid - e))
                # kl_anneal = e/self.training_setup.vae_epochs
                # kl_anneal=0.0
                
                kl_anneal=1.0
                _, mse, kl = self.vae_loss(self.vae_params, next(rng_seq), batch, batch_time, True, kl_anneal)
                
                loss, self.vae_params, self.vae_opt_state = self.vae_update_fn(self.vae_params, next(rng_seq), self.vae_opt_state, batch, batch_time, kl_anneal)
                
                epoch_kl_loss.append(kl)
                epoch_mse_loss.append(mse)
                epoch_loss.append(loss)

            tr_loss = np.mean(epoch_loss)
            mse_loss = np.mean(epoch_mse_loss)
            kl_loss = np.mean(epoch_kl_loss)
           
            #print ("STEP: %5d; Training ELBO: %.3f MSE: %.3f KL: %.3f \n"%(e, tr_loss, mse_loss, kl_loss))
            losses.append(tr_loss)
            #wandb.log({"pretrain: vae_train_loss": tr_loss, "pretrain: vae_train_mse_loss": mse_loss, "pretrain: vae_train_kl_loss": kl_loss})

            if e % eval_frequency == 0 and self.training_setup.dataset.val_split:
                
                val_loss = self.loss_fn(self.vae_params, next(rng_seq), val_data, val_time, False)
                #print("STEP: %5d; Validation ELBO: %.3f \n"%(e, val_loss))

                #wandb.log({"pretrain: vae_val_loss": val_loss})


            train_latents: VAEOutput = self.training_setup.vae_model.apply(self.vae_params, next(rng_seq), train_time, train_data )
            latent_df = pd.DataFrame(np.array(train_latents.latent))
            latent_df["time"] = train_time

            if self.training_setup.dataset.val_split:
                val_latents: VAEOutput = self.training_setup.vae_model.apply(self.vae_params, next(rng_seq), val_time, val_data )
                val_latent_df = pd.DataFrame(np.array(val_latents.latent))
                val_latent_df["time"] = val_time
            else:
                val_latent_df=None

        return self.training_setup.vae_model,self.vae_params, latent_df, val_latent_df

    @partial(jit, static_argnames=['self', 'direction', 'corrector'])
    def fast_sample_trajectory_evo(self, key, direction, x_init, params, ferryman, corrector=""):
        score = broadcast(lambda d, m: partial(self.training_setup.score(m), params[d]), directions, self.training_setup.model)

        if is_forward(direction):
            return self.training_setup.sde.sample_f_trajectory(key, x_init, score, ferryman, corrector)
        else:
            return self.training_setup.sde.sample_b_trajectory(key, x_init, score, ferryman, corrector)


    def training_phase(self, key, params, opt_state, td_coeff=None, epchs_num=5,):
        key, key_train = split_key(key)
        num_sde = self.training_setup.num_sde

        for epch in tqdm(range(epchs_num)):

            if td_coeff is None:
                # TODO: Debug. Maybe it's too much
                epch_td_coeff = 1. * epch/epchs_num
            else:
                epch_td_coeff = td_coeff

            # TODO: Refactor this + killing
            for _ in range(num_sde):
                #print ("SAMPLE-BACKWARD")
                key_train, params, opt_state, logs = self.training_step(FORWARD, key_train, params, opt_state, init_logs(epoch=epch), td_coeff=epch_td_coeff)
                #wandb.log({"forward:ipf_loss": logs["ipf_loss"],})

                if epch%2==0 and self.dataset.val_split:
                    eval_val=True
                else:
                    eval_val=False
                key, params, opt_state, logs = self.compute_trajectory_loss(key, params, opt_state, logs, d=FORWARD, eval_val=eval_val,
                                                                             calc_lantent_loss=self.training_setup.calc_latent_loss,
                                                                             calc_recon_loss=self.training_setup.calc_recon_loss)
                #print_logs(logs)

            for _ in range(num_sde):
                #print ("SAMPLE-FORWARD")
                key_train, params, opt_state, logs = self.training_step(BACKWARD, key_train, params, opt_state, init_logs(epoch=epch), td_coeff=epch_td_coeff)
                #wandb.log({"backward: ipf_loss": logs["ipf_loss"],"backward: ferryman_loss": logs["ferryman_loss"]})
                #print_logs(logs)

        return key, params, opt_state
    

    def get_model_configs(self):

        config={"vae lr": self.vae_lr, "lr":self.lr, "ferryman_lr": self.ferryman_lr,
                "steps_num": self.training_setup.steps_num, "epochs":self.training_setup.epochs,
                "vae_epochs": self.training_setup.vae_epochs, "batch_size": self.training_setup.batch_size,
                   "hidden_dim": self.training_setup.hidden_dim, "dec_hidden_dim": self.training_setup.dec_hidden_size,
                   "ipf_mask_dead": self.training_setup.ipf_mask_dead, "objective":self.training_setup.objective,
                "reality_coefficient":self.training_setup.reality_coefficient, "paths_reuse": self.training_setup.paths_reuse,
                "num_sde": self.training_setup.num_sde, "resnet": self.training_setup.resnet, "t_dim": self.training_setup.t_dim,
                "vae_input_dim": self.training_setup.vae_input_dim, "vae_enc_hidden_dim": self.training_setup.vae_enc_hidden_dim,
                "vae_dec_hidden_dim": self.training_setup.vae_dec_hidden_dim,
                "vae_t_dim": self.training_setup.vae_t_dim, "vae_latent_dim": self.training_setup.vae_latent_dim,
                "vae_batch_size": self.training_setup.vae_batch_size, "ferryman_hidden_dim": self.training_setup.ferryman_hidden_dim,
                "death_importance_rate": self.dataset.death_importance_rate, "f":self.dataset.f_val}
        
        return config

    def train(self, td_schedule=[None], project_name="training_1"):
        """ Perform `len(td_schedule)` training phases (each consisting of several epochs).

        On each phase, if `td_schedule[i] == 0`, the Ferryman network is **not** updated. All positive values of `td_schedule[i]`
        have the same effect, i.e., SGD is perfomed on the Ferryman loss.
        """
        # <<<<<<<<< Pop training state >>>>>>>>>
        key, params = self.training_setup.state
        opt_state = self.opt_state
        # <<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>

        config=self.get_model_configs()
        #wandb.init(project=f"ARTEMIS_{project_name}", config=config)

        self.training_setup.vae_model, self.vae_params, vae_latents, val_vae_latents = self.pretrain_vae()
        self.dataset.update_data_info(vae_latents, val_vae_latents)
        self.training_setup.sde.killer = self.dataset.killing_function()

        #return self
           
        for phase_num in range(len(td_schedule)):
            td_coeff = td_schedule[phase_num]
            key, params, opt_state = self.training_phase(key, params, opt_state,  td_coeff=td_coeff, 
                                                         epchs_num=self.training_setup.epochs,)
        
        # >>>>>>>>> Push training state <<<<<<<<<
        self.training_setup.state = (key, params)
        self.opt_state = opt_state
        # >>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<

        #wandb.finish()
        return self
    
    
