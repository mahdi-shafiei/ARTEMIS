import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import haiku as hk
from functools import partial

from utils import *
from evaluate import *
from models import *


def z_score_norm(df):
    
    scaler = StandardScaler()
    scaler_obj = scaler.fit(df.values[:,:-1])
    scaled_df = scaler_obj.transform(df.values[:,:-1])
    new_df = pd.DataFrame(scaled_df, columns=df.columns[:-1], index=df.index)
    new_df["time"] = df["time"].to_list()
    df = new_df.copy()
    return df

def get_model_latents(train_data, val_data,ts,tr_model):
    train_time = train_data.iloc[:,-1]
    val_time = val_data.iloc[:,-1]

    train_data = train_data.iloc[:, :-1]
    val_data = val_data.iloc[:,:-1]


    train_pred = ts.vae_model.apply(tr_model.vae_params, next(hk.PRNGSequence(1)), train_time.values, train_data.values)
    train_latent = pd.DataFrame(train_pred.latent, index=train_data.index.to_list())
    train_latent["time"] = train_time.values
    #train_latent = train_latent.sort_values(by=["time"])

    train_recon = pd.DataFrame(train_pred.logits, index=train_data.index.to_list())
    train_recon["time"] = train_time.values
    #train_recon = train_recon.sort_values(by=["time"])

    val_pred = ts.vae_model.apply(tr_model.vae_params, next(hk.PRNGSequence(1)), val_time.values, val_data.values)
    val_latent = pd.DataFrame(val_pred.latent, index=val_data.index.to_list())
    val_latent["time"] = val_time.values
    #val_latent = val_latent.sort_values(by=["time"])

    val_recon = pd.DataFrame(val_pred.logits, index=val_data.index.to_list())
    val_recon["time"] = val_time.values
    #val_recon = val_recon.sort_values(by=["time"])

    return train_recon, train_latent, val_recon, val_latent

def get_model_latents_single_data(data,ts,tr_model):
    train_time = data.iloc[:,-1]
    train_data = data.iloc[:, :-1]

    train_pred = ts.vae_model.apply(tr_model.vae_params, next(hk.PRNGSequence(1)), train_time.values, train_data.values)
    train_latent = pd.DataFrame(train_pred.latent, index=train_data.index.to_list())
    train_latent["time"] = train_time.values
    train_latent = train_latent.sort_values(by=["time"])

    train_recon = pd.DataFrame(train_pred.logits, index=train_data.index.to_list())
    train_recon["time"] = train_time.values
    train_recon = train_recon.sort_values(by=["time"])

    return train_recon, train_latent

def sample(data, size=(100, ), replace=False):
    sub = data
    idx = np.arange(sub.shape[0])
    sampled = sub[np.random.choice(idx, size=size, replace=replace if len(idx) >= size[0] else True)]
    return sampled

def get_latent_trajectories(train_dataset,val_latent, tps, tr_model, ts, t_0_orig=0, max_size=0, test=False):

    val_data_init = val_latent[val_latent["time"]==t_0_orig].iloc[:,:-1].values
    params = tr_model.training_setup.state[1]
    eval_score = broadcast(lambda model, params: partial(tr_model.training_setup.score(model), params), tr_model.training_setup.model, params)

    params_ferryman = tr_model.training_setup.state[1][FERRYMAN]
    eval_ferryman= partial(tr_model.training_setup.ferryman.apply, params=params_ferryman, direction=FORWARD)

    if max_size!=0:
        val_data_init = sample(val_data_init, size = (max_size, ))

    t_0 = int(train_dataset.cells_time(t_0_orig)*train_dataset.steps_num)

    trajs, _, statuses, birth_statuses = ts.sde.sample_trajectory(random.PRNGKey(0), FORWARD, val_data_init, eval_score, 
                                                                    eval_ferryman, t_0=t_0, corrector="",  test=test, max_size=max_size)
    steps_num=train_dataset.steps_num
    pred_trajectories = []
    pred_trajectories_alive = []
    timepoints = []
    timepoints_alive = []
    for t in tps[tps.index(t_0_orig):]:
        
        tmp_cell_time = int(tr_model.dataset.cells_time(t)*steps_num)
        
        pred_trajectories.append(trajs[tmp_cell_time])
        pred_trajectories_alive.append(trajs[tmp_cell_time][np.where(statuses[tmp_cell_time])[0]])
        timepoints.extend([t]*len(trajs[tmp_cell_time]))
        timepoints_alive.extend([t]*len(trajs[tmp_cell_time][np.where(statuses[tmp_cell_time])[0]]))

    pred_trajectories = np.concatenate((pred_trajectories))
    pred_trajectories_alive = np.concatenate((pred_trajectories_alive))

    return pred_trajectories, pred_trajectories_alive, np.array(timepoints), np.array(timepoints_alive), statuses


def get_test_trajs(df,t):
    df_sub = df[df["time"].isin(t)]
    return df_sub

def get_reconstructed_trajectory(pred_trajectories, val_data, vae_input_dim, vae_hidden_dim, vae_latent_dim, tr_model, timepoints):
    
    dec = hk.transform(lambda z: Decoder(output_shape=vae_input_dim, hidden_size=vae_hidden_dim)(z))
    rng_seq = hk.PRNGSequence(1)
    dec_init_params = dec.init(next(rng_seq), z=np.zeros((val_data.shape[0], vae_latent_dim+8)))

    for key in tr_model.vae_params.keys():
        if "dec" in key:
            dec_init_params[key.split("~/")[1]] = tr_model.vae_params[key]
            
    t_emb = get_timestep_embedding(np.array(timepoints), 8)
    pred_traj_t = jnp.concatenate((pred_trajectories,t_emb),-1)

    recon_data = dec.apply(dec_init_params, None, pred_traj_t)
    recon_data.shape

    return recon_data

def get_metrics(simulations, gt,t):
    perf_df = pd.DataFrame()
    for i in range(len(simulations.keys())):
        pred = simulations[i][1]
        pred = pred[pred["time"]==t].values[:,:-1]
        perf_i = pd.DataFrame(compute_metrics_subset(gt,pred), index=[0])
        perf_df = pd.concat([perf_df, perf_i])
    
    return perf_df

def get_predictions(train_data, val_data, train_tps, val_tps,
                    train_dataset,train_latent,tps,tr_model,ts,
                    vae_input_dim, vae_hidden_dim,vae_latent_dim, 
                    num_simulations=5, t_0_orig=0):

    tps = sorted(list(set(tps)))
    predictions = {}
    predictions_all={}

    predictions["simulations"] = {}
    predictions_all["simulations"] = {}

    for i in range(num_simulations):

        # heldout time point
        pred_trajectories, pred_trajectories_alive, timepoints, timepoints_alive, statuses = get_latent_trajectories(train_dataset,train_latent, tps, tr_model, ts, max_size=2000, t_0_orig=t_0_orig)
        
        # trained all time point
        # pred_trajectories, pred_trajectories_alive, timepoints, timepoints_alive, statuses = get_latent_trajectories(val_latent, tps, tr_model, ts, max_size=2000)
        
        pred_trajectories_alive_w_time = pd.DataFrame(pred_trajectories_alive)
        pred_trajectories_alive_w_time["time"] = timepoints_alive
        
        recon_data = get_reconstructed_trajectory(pred_trajectories_alive, val_data, vae_input_dim, vae_hidden_dim,vae_latent_dim, tr_model, timepoints_alive)
        recon_data_w_time = pd.DataFrame(recon_data)
        recon_data_w_time["time"] = timepoints_alive
        # print (set(timepoints_alive), set(timepoints))


        pred_trajectories_w_time = pd.DataFrame(pred_trajectories)
        pred_trajectories_w_time["time"] = timepoints
        
        recon_data_all = get_reconstructed_trajectory(pred_trajectories, val_data, vae_input_dim, vae_hidden_dim,vae_latent_dim, tr_model, timepoints)
        recon_data_all_w_time = pd.DataFrame(recon_data_all)
        recon_data_all_w_time["time"] = timepoints

        predictions["simulations"][i] = (pred_trajectories_alive_w_time, recon_data_w_time)
        predictions_all["simulations"][i] = (pred_trajectories_w_time, recon_data_all_w_time)
        
    predictions["train_data"] = train_data
    predictions["val_data"] = val_data
    predictions["train_tps"] = train_tps
    predictions["val_tps"] = val_tps

    predictions_all["train_data"] = train_data
    predictions_all["val_data"] = val_data
    predictions_all["train_tps"] = train_tps
    predictions_all["val_tps"] = val_tps

    return predictions
# with open("/home/sayalialatkar/Documents/UDSB/src_vae_fbsde/results/train_zebrafish_predictions.pkl","wb") as f:
#     pickle.dump(predictions,f,protocol=pickle.HIGHEST_PROTOCOL)
