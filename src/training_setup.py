# code adapted from https://github.com/matteopariset/unbalanced_sb/tree/main/udsb_f

import jax
import haiku as hk
from functools import partial

import pickle
import datetime

from utils import *
from datasets import *
from models import *
from sde import *


class Training_Setup:
    def __init__(self, dataset: Input_Dataset, dataset_name="Data_1", steps_num=100, epochs = 5, vae_epochs=100, key=None, params=None, objective="divergence",
                 batch_size=512, hidden_dim=[64], dec_hidden_size=[64], ferryman_hidden_dim=[64], ferryman_activate_final=True,
                 ipf_mask_dead=False, reality_coefficient=0.1, paths_reuse=5, num_sde=10, resnet=False,
                 feature_spatial_loss=False, t_dim=16, vae_input_dim=1000, vae_enc_hidden_dim=[512,512], 
                 vae_dec_hidden_dim =[512,512], vae_t_dim=8, calc_latent_loss=True, calc_recon_loss=True,
                 vae_latent_dim=64, vae_batch_size=64, killer_func=Input_Dataset.killing_function): 

        self.dataset_name=dataset_name
        self.dataset=dataset
        self.steps_num = dataset.steps_num
        self.epochs = epochs
        self.vae_epochs = vae_epochs
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.dec_hidden_size = dec_hidden_size
        self.ipf_mask_dead = ipf_mask_dead
        self.objective=objective
        self.reality_coefficient = reality_coefficient # Birth by splitting are not accounted for when reality_coefficient is big (i.e. alive_to_alive should contribute to births)
        self.paths_reuse = paths_reuse
        self.num_sde = num_sde
        self.resnet = resnet
        self.t_dim=t_dim
        self.vae_input_dim=vae_input_dim
        self.vae_enc_hidden_dim=vae_enc_hidden_dim
        self.vae_dec_hidden_dim=vae_dec_hidden_dim
        self.vae_t_dim=vae_t_dim
        self.vae_latent_dim=vae_latent_dim
        self.vae_batch_size=vae_batch_size
        self.ferryman_hidden_dim=ferryman_hidden_dim
        self.calc_latent_loss=calc_latent_loss
        self.calc_recon_loss=calc_recon_loss
        #self.killer_func = killer_func

        #assert self.hidden_dim > dataset.input_dim, "The network hidden size should be bigger than the dimension of the state space"        

        mass_max = max(dataset.mass)

        
        # 2. define SDE class
        self.sde = SDE(dataset=dataset, steps_num=self.steps_num, batch_size=self.batch_size)

        # 3. sample marginals (training.py, viewer.py)
        self.start_marginals_sampler = { FORWARD: partial(dataset.pi_0_sample, n_samples = self.batch_size),
                                   BACKWARD: partial(dataset.pi_1_sample, n_samples = self.batch_size)}
        
        # 4a. define models: Forward, Backward
        self.model = { FORWARD: hk.transform(init_base_model( self.hidden_dim, dec_hidden_size=self.dec_hidden_size, 
                                                             resnet=self.resnet, t_dim=self.t_dim)), 
                 BACKWARD: hk.transform(init_base_model(self.hidden_dim,  dec_hidden_size=self.dec_hidden_size, t_dim=self.t_dim))}

        # 4b. define ferryman model
        self.ferryman = hk.transform(init_ferryman_model(ferryman_hidden_dim, ferryman_activate_final))


        self.vae_model = hk.transform(lambda t,x: VariationalAutoEncoder(self.vae_input_dim, self.vae_enc_hidden_dim,
                                                                         self.vae_dec_hidden_dim, 
                                                                         self.vae_latent_dim, self.vae_t_dim)(t,x))
        
        self.dec = hk.transform(lambda z: Decoder(output_shape=self.vae_input_dim, hidden_size=self.vae_dec_hidden_dim)(z))
        
        #self.optimizer = optax.adam(self.learning_rate)


        # initialize training state
        self.state = (key,params)

    @staticmethod
    def density(model):
        return model.apply

    def score(self, model):
        #Z = lambda params, key, t, pos: self.sde.g(t, pos) * model.apply(params, key, t, pos)
        # return Z
    
        log_density = Training_Setup.density(model)
        grad_log_density = jax.grad(log_density, argnums=3)
         
        grad_log_density = jax.vmap(grad_log_density, in_axes=(None, None, None, 0), out_axes=(0))

        Z = lambda params, key, t, pos: self.sde.g(t, pos) * grad_log_density(params, key, t, pos)

        return Z

    def save(self):
        tag = self.dataset_name + "_" + datetime.datetime.today().strftime('%Y_%m_%d-%H_%M_%S')
        file = open(tag+".pkl","wb")
        file.write(pickle.dumps(self.__dict__))
        file.close()

    def load(self, dataset_name):
        file = open(dataset_name+".pkl","rb")
        obj = file.read()
        file.close()
        self.__dict__ = pickle.loads(obj)


