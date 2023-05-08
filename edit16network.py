import torch
from torch.utils.data import Dataset, DataLoader
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax

class Encoder(nn.Module):
  latents: int
  @nn.compact
  def __call__(self, x):
    y1 = nn.Conv(features=4, kernel_size=(4,), strides=4, name='1mer')(x)
    y2 = nn.Conv(features=4**2, kernel_size=(4*2,), strides=4, name='2mer')(x)
    y3 = nn.Conv(features=4**3, kernel_size=(4*3,), strides=4, name='3mer')(x)
    y4 = nn.Conv(features=4**4 // 4, kernel_size=(4*4,), strides=4, name='4mer')(x)
    y5 = nn.Conv(features=4**5 // 16, kernel_size=(4*5,), strides=4, name='5mer')(x)
    y6 = nn.Conv(features=4**6 // 64, kernel_size=(4*6,), strides=4, name='6mer')(x)

    y4 = nn.max_pool(y4, window_shape=(4,), strides=(2,))
    y5 = nn.max_pool(y5, window_shape=(5,), strides=(3,))
    y6 = nn.max_pool(y6, window_shape=(6,), strides=(3,))

    y1 = y1.reshape(y1.shape[0], -1)
    y2 = y2.reshape(y2.shape[0], -1)
    y3 = y3.reshape(y3.shape[0], -1)
    y4 = y4.reshape(y4.shape[0], -1)
    y5 = y5.reshape(y5.shape[0], -1)
    y6 = y6.reshape(y6.shape[0], -1)

    y_1_6 = jnp.concatenate([y1, y2, y3, y4, y5, y6], axis=1)
    y = nn.gelu(y_1_6)
    y = nn.gelu(y)
    y = nn.Dense(256)(y)
    y = nn.gelu(y)
    y = nn.Dense(128)(y)
    y = nn.gelu(y)
    y = nn.Dense(self.latents)(y)
    return y

class PredictDistance(nn.Module):
  @nn.compact
  def __call__(self, lx, ly):
    z = jnp.concatenate((lx,ly), axis=1)
    z = nn.Dense(128)(z)
    z = nn.gelu(z)
    z = nn.Dense(128)(z)
    z = nn.gelu(z)
    z = nn.Dense(64)(z)
    z = nn.gelu(z)
    z = nn.Dense(17)(z)
    return z

class EditEmbedding(nn.Module):
  latents: int = 2
  def setup(self):
    self.encoder = Encoder(self.latents)
    self.predictdistance = PredictDistance()
  def __call__(self, triple):
    x, y, d = triple
    lx = self.encoder(x)
    ly = self.encoder(y)
    z = self.predictdistance(lx, ly)
    return z, d
