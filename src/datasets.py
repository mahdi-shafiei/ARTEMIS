# code adapted from https://github.com/matteopariset/unbalanced_sb/tree/main/udsb_f

import jax
import jax.random as random
import jax.numpy as jnp
from jax.tree_util import tree_map

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from utils import *

class Dataset():

    def __init__(self, x, meta, meta_celltype_column=None, splitting_births_frac=0.2, eps=1e-7,steps_num=100, 
                 val_split=False, death_importance_rate=100, f_val=None):
        
        self.val_split = val_split
        self.death_importance_rate = death_importance_rate

        if val_split:
            x,meta,x_val, meta_val = self.split_train_test(x,meta)
            self.x_val, self.meta_val = x_val, meta_val
        else:
            self.x_val, self.meta_val = None, None

        self.x = x.sort_values(by=["time"])
        self.features = x.columns[:-1]
        self.input_dim = len(self.features)
        self.eps = eps
        
        self.meta = meta
        self.meta_celltype_column = meta_celltype_column
        self.steps_num=steps_num

        self.times = self.x["time"].unique().tolist()
        self.times_orig = self.x["time"].unique().tolist()

        self.mass = self.x["time"].value_counts().sort_index().to_list()
        self.splitting_births_frac = splitting_births_frac
        self.f_val=f_val

        self.time_x_groundtruth = {int(self.cells_time(k)*self.steps_num): jnp.array(x[x["time"]==k].to_numpy().astype(float)[:,:-1]) for k in self.times}
        self.time_x = {int(self.cells_time(k)*self.steps_num): jnp.array(x[x["time"]==k].to_numpy().astype(float)[:,:-1]) for k in self.times}
        self.mean = tree_map(lambda x_temp: jnp.array(x_temp.mean(axis=0, keepdims=True)), self.time_x)
        self.std = tree_map(lambda x_temp: jnp.array(x_temp.std(axis=0, keepdims=True)), self.time_x)

        if self.val_split:
            self.time_x_val = {int(self.cells_time(k)*self.steps_num): jnp.array(x_val[x_val["time"]==k].to_numpy().astype(float)[:,:-1]) for k in self.times}
    
        self.times = list(map(lambda t: int(self.cells_time(t)*self.steps_num), self.times))
    
    def update_data_info(self,x, x_val):

        self.time_x = {int(self.cells_time(k)*self.steps_num): jnp.array(x[x["time"]==k].to_numpy().astype(float)[:,:-1]) for k in self.times_orig}
        if self.val_split:
            self.time_x_val = {int(self.cells_time(k)*self.steps_num): jnp.array(x_val[x_val["time"]==k].to_numpy().astype(float)[:,:-1]) for k in self.times_orig}

        self.time_x = {int(self.cells_time(k)*self.steps_num): jnp.array(x[x["time"]==k].to_numpy().astype(float)[:,:-1]) for k in self.times_orig}
        self.mean = tree_map(lambda x_temp: jnp.array(x_temp.mean(axis=0, keepdims=True)), self.time_x)
        self.std = tree_map(lambda x_temp: jnp.array(x_temp.std(axis=0, keepdims=True)), self.time_x)
        
    def split_train_test(self,x,meta):
        X = np.arange(x.shape[0])
        y = x.values[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y , random_state=104,test_size=0.1, shuffle=True)

        if meta is not None:
            return x.iloc[X_train],meta.iloc[X_train], x.iloc[X_test], meta.iloc[X_test]
        return x.iloc[X_train],None, x.iloc[X_test], None
 
    def f(self,t,x):
        "Returns base drift"
        if self.f_val is not None:
            return self.f_val
        else:
            return 5
    
    def g(self,t,x, type="triangular"):
        "Returns diffusion coefficient"

        if type=="triangular":
            #return triangular_diffusivity(1.5,1.)
            return triangular_diffusivity(t,1.)
        elif type=="inverse_triangular":
            return inverse_triangular_diffusivity(t, g_max=1.0)
        elif type=="decreasing":
            return decreasing_diffusivity(t, g_max=1.0)
        elif type=="constant":
            return constant_diffusivity(t, g_max=1.0)
    
    def cells_time(self,t):
        return (t-self.times_orig[0]) / (self.times_orig[-1] - self.times_orig[0])
    
    def real_time(self,t):
        return (t*(self.times_orig[-1]-self.times_orig[0]) + self.times_orig[0])
    
    def pi_0_sample(self, key, n_samples=300, get_metadata=False):
        "sample cells from initial distribution"

        pi_0_source = self.time_x[self.times[0]]
        tot_samples = pi_0_source.shape[0]

        if self.meta_celltype_column == None:
            sel_idxs = random.permutation(key, tot_samples)[:min(n_samples, tot_samples)]
            return pi_0_source[sel_idxs]
        else:
            meta_t_0 = self.meta[self.meta["time"] == self.times_orig[0]]
            celltype_proportion = ((meta_t_0[self.meta_celltype_column].value_counts()/tot_samples)*min(n_samples, tot_samples)).astype(int)
            samples_per_celltype = celltype_proportion.to_dict()
            df_measurements  = pd.DataFrame(pi_0_source, index = meta_t_0.index)

            list_sampled_indexes = []

            for name, group in meta_t_0.groupby(self.meta_celltype_column):    
                n_rows_to_sample = samples_per_celltype[name]
                sampled_group = group.sample(n_rows_to_sample)
                list_sampled_indexes.extend(sampled_group.index.to_list())

            p1_0_source_sub = jnp.array(df_measurements.loc[list_sampled_indexes].to_numpy().astype(float))

            if get_metadata:
                return p1_0_source_sub, meta_t_0.loc[list_sampled_indexes]
            else:
                return p1_0_source_sub

    def pi_1_sample(self, key, n_samples=300, get_metadata=False):
        "sample cells from terminal distribution"

        pi_1_source = self.time_x[self.times[-1]]
        tot_samples = pi_1_source.shape[0]

        if self.meta_celltype_column == None:
            sel_idxs = random.permutation(key, tot_samples)[:min(n_samples, tot_samples)]
            return pi_1_source[sel_idxs]
        else:
            meta_t_1 = self.meta[self.meta["time"] == self.times_orig[-1]]
            celltype_proportion = ((meta_t_1[self.meta_celltype_column].value_counts()/tot_samples)*min(n_samples, tot_samples)).astype(int)
            samples_per_celltype = celltype_proportion.to_dict()
            df_measurements  = pd.DataFrame(pi_1_source, index = meta_t_1.index)

            list_sampled_indexes = []

            for name, group in meta_t_1.groupby(self.meta_celltype_column):    
                n_rows_to_sample = samples_per_celltype[name]
                sampled_group = group.sample(n_rows_to_sample)
                list_sampled_indexes.extend(sampled_group.index.to_list())

            p1_1_source_sub = jnp.array(df_measurements.loc[list_sampled_indexes].to_numpy().astype(float))
            if get_metadata:
                return p1_1_source_sub, meta_t_1.loc[list_sampled_indexes]
            else:
                return p1_1_source_sub

    def killing_function(self):
        
        density_kernel = gaussian_kernel()

        def _calc_threshold_violation_score(t, t_min, t_max, x, std_threshold=2., cutoff=.2):
            effecttive_t = (t-(t_min/self.steps_num)) / ((t_max/self.steps_num) - (t_min/self.steps_num))
            effective_mean = effecttive_t * self.mean[t_max] + (1-effecttive_t) * self.mean[t_min]
            effective_std = effecttive_t * self.std[t_max] + (1-effecttive_t) * self.std[t_min]
            threshold_violation_score = jnp.mean(jnp.abs(x-effective_mean) > std_threshold * effective_std, axis=1)
            return (threshold_violation_score < cutoff) * 0 + (threshold_violation_score >= cutoff)*threshold_violation_score
     
        # define prior kill-rate factor
        delta_t = jnp.array(self.times[1:]) - jnp.array(self.times[:-1])
        self.death_importance = (self.death_importance_rate)*(-1)*(1/delta_t)*jnp.log(jnp.array(self.mass[1:])/jnp.array(self.mass[:-1]))
        
        def _killer(t,x,direction=FORWARD, mb_prior=5.0, db_prior=10.0):

            mean_based_transitions = 0.0
            density_based_transitions = 0.0

            for i in range(1,len(self.times)):
                t_min, t_max = self.times[i-1], self.times[i]
               
                mean_based_transitions += self.death_importance[i-1]*((t_min/self.steps_num) <= t)*(t< (t_max/self.steps_num)) * _calc_threshold_violation_score(t,t_min,t_max,x)
                density_based_transitions += ((t_min/self.steps_num) <= t)*(t< (t_max/self.steps_num)) * jnp.square(kde(density_kernel,self.mean[t_max],x)) * -1
                
            density_based_transitions = density_based_transitions.reshape(density_based_transitions.shape[0],)
            kill_rate = (mb_prior)*mean_based_transitions +(db_prior)*density_based_transitions

            return kill_rate
         
        return _killer
    
    


    
