# code adapted from https://github.com/matteopariset/unbalanced_sb/tree/main/udsb_f

import jax
from jax.nn import sigmoid, silu, leaky_relu
import jax.numpy as jnp
import numpy as np

import haiku as hk
from typing import NamedTuple, List, Tuple


def get_timestep_embedding(
        timesteps: jnp.ndarray,
        embedding_dim: int,
        max_positions=10000
    ) -> jnp.ndarray:
    """ Get timesteps embedding.
    Function extracted from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py

    Args:
        timesteps (jnp.ndarray): timesteps array (Nbatch,).
        embedding_dim (int): Size of the embedding.
        max_positions (int, optional): _description_. Defaults to 10000.

    Returns:
        emb (jnp.ndarray): embedded timesteps (Nbatch, embedding_dim).
    """
    assert embedding_dim > 3, "half_dim == 0"
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = jnp.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps[:, jnp.newaxis] * emb[jnp.newaxis, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jnp.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


# VAE Implementation
class Encoder(hk.Module):
    """Encoder model."""

    def __init__(self, hidden_size: list | int = 512, latent_size: int = 10):
        super().__init__()
        self._hidden_size = hidden_size
        self._latent_size = latent_size

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:

        x = hk.Flatten()(x)

        for i in range(len(self._hidden_size)):
            x = hk.Linear(self._hidden_size[i])(x)
            x = jax.nn.relu(x)

        mean = hk.Linear(self._latent_size)(x)
        log_stddev = hk.Linear(self._latent_size)(x)
        stddev = jnp.exp(log_stddev)

        return mean, stddev


class Decoder(hk.Module):
    """Decoder model."""

    def __init__(
        self,
        output_shape,
        hidden_size: list | int = 512,
    ):
        super().__init__()
        self._hidden_size = hidden_size
        self._output_shape = output_shape

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:

        for i in range(len(self._hidden_size)-1,-1,-1):
            z = hk.Linear(self._hidden_size[i])(z)
            z = jax.nn.relu(z)

        logits = hk.Linear(np.prod(self._output_shape))(z)
        #logits = jnp.reshape(logits, (-1, self._output_shape))
 
        return logits
    
    
class VAEOutput(NamedTuple):
    mean: jnp.ndarray
    stddev: jnp.ndarray
    logits: jnp.ndarray
    latent: jnp.ndarray

class VariationalAutoEncoder(hk.Module):
    """Main VAE model class, uses Encoder & Decoder under the hood."""

    def __init__(
        self,
        output_shape,
        enc_hidden_size: int = 512,
        dec_hidden_size: int = 512,
        latent_size: int = 10,
        t_emb_size = 16,
    ):
        super().__init__()
        self._enc_hidden_size = enc_hidden_size
        self._dec_hidden_size = dec_hidden_size
        self._latent_size = latent_size
        self._output_shape = output_shape
        self.t_emb_size=t_emb_size

        self.encoder = Encoder(self._enc_hidden_size, self._latent_size)
        self.decoder = Decoder(self._output_shape, self._dec_hidden_size)

    def __call__(self, t, x: jnp.ndarray) -> VAEOutput:

        t = jnp.array(t, dtype=float).reshape(-1)
        t_emb = get_timestep_embedding(t, self.t_emb_size)
        # t_emb=t

        x = x.astype(jnp.float32)
        x = jnp.concatenate((x,t_emb),-1)
        mean, stddev = self.encoder(x)
        z = mean + stddev * jax.random.normal(hk.next_rng_key(), mean.shape)

        z_dec = jnp.concatenate((z,t_emb),-1)
        logits = self.decoder(z_dec)

        #p = jax.nn.sigmoid(logits)
        #image = jax.random.bernoulli(hk.next_rng_key(), p)

        return VAEOutput(mean, stddev, logits, z)


# Schr¨odinger bridge "Drift" Networks

def MLP(hidden_shapes, bias=True, activate_final=True):
    w_init, b_init = None, None
    return hk.nets.MLP(hidden_shapes, with_bias=bias, w_init=w_init, b_init=b_init, activation=silu, activate_final=activate_final)

class BaseModel(hk.Module):
    """The standard network used to parameterize forward/backward drifts in UDSB experiments.

    Parameters:
        - `output_shape` (int): output shape.
        - `enc_shapes` (int): The shapes of the encoder.
        - `t_dim` (int): the dimension of the time embedding.
        - `dec_shapes` (int): The shapes of the decoder 
        - `resnet` (bool): if True then the network is a resnet.
    """

    def __init__(self, enc_shapes: List[int], t_dim: int, dec_shapes: List[int], resnet: bool):
        super().__init__()
        self.temb_dim = t_dim
        #t_enc_dim = t_dim * 2

        #self.output_shape = output_shape
        self.dec_shapes= dec_shapes

        self.net = MLP(
            hidden_shapes=dec_shapes,
            #output_shape=output_shape,
            bias=True,
            activate_final=False,
        )

        self.t_encoder = MLP(
            hidden_shapes=enc_shapes,
            #output_shape=t_enc_dim,
            bias=True,
            activate_final=True,
        )

        self.x_encoder = MLP(
            hidden_shapes=enc_shapes,
            #output_shape=t_enc_dim,
            bias=True,
            activate_final=True,
        )

        self.bias = hk.Bias(bias_dims=[-1])

        self.resnet = resnet

    def __call__(self, t, x):
        t = jnp.array(t, dtype=float)

        if len(x.shape) == 1:
            x_input = jnp.expand_dims(x, axis=0)
        else:
            x_input = x

        # comment these lines to avoid time embedding
        temb = get_timestep_embedding(t.reshape(-1), self.temb_dim)

        temb = self.t_encoder(temb)
        xemb = self.x_encoder(x_input)

        temb = jnp.broadcast_to(temb, [xemb.shape[0], *temb.shape[1:]])
        h = jnp.concatenate([xemb, temb], -1)

        # comment this line to avoid only values input
        #h = x_input

        # following lines are common
        out = self.net(h)

        out = self.bias(out)

        if self.resnet:
            out = x_input + out

        # if self.output_shape == 1:
        #     out = jnp.squeeze(out, axis=1)

        if self.dec_shapes[-1] == 1:
            out = jnp.squeeze(out, axis=1)

        if len(x.shape) == 1:
            out = jnp.squeeze(out, axis=0)

        return out

def smooth_interval_indicator(x, low=0., high=1., steepness=5.):
    return sigmoid(-steepness*(x-low)) + sigmoid(steepness*(x-high))


class FerrymanModel(hk.Module):
    """The standard Ferryman network used in UDSB experiments.

    Parameters:
        - `t_dim` (int): the dimension of the time embedding.
        - `hidden_dims` (list[int]): The shape of hidden layers
        - `activate_final` (bool): if True a non-linearity is placed at the output of the network
    """
    def __init__( self, t_dim: int, hidden_dims: List[int], activate_final: bool):
        super().__init__()
        self.temb_dim = t_dim

        self.t_net = hk.nets.MLP(
            hidden_dims + [1],
            with_bias=True,
            activation=leaky_relu,
            activate_final=activate_final
        )

    def __call__(self, t, direction):
        t = jnp.array(t, dtype=float)

        # time_delay = 0 if is_forward(direction) else +1.
        time_delay = 0.

        t_emb = get_timestep_embedding(time_delay + t.reshape(-1), self.temb_dim)
        out = self.t_net(t_emb)

        return jnp.abs(out.ravel())

def init_base_model( hidden_dim_size, dec_hidden_size, resnet=False, t_dim=16):
    def _query_model(t, x):
        return BaseModel(
            #output_shape=statespace_dim,
            enc_shapes=hidden_dim_size, #[hidden_dim_size, hidden_dim_size],
            t_dim=t_dim,
            dec_shapes=dec_hidden_size, #[hidden_dim_size, hidden_dim_size], # for VAE inputs
            #dec_shapes=[hidden_dim_size, hidden_dim_size],
            resnet=resnet
            
        )(t, x)
    return _query_model

def init_ferryman_model(hidden_dims, activate_final=True):
    def _query_model(t, direction):
        return FerrymanModel(
            t_dim=32,
            hidden_dims=hidden_dims,
            activate_final=activate_final
        )(t, direction)
    return _query_model
