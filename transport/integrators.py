import numpy as np
import torch as th
import torch.nn as nn
from torchdiffeq import odeint
from functools import partial
from tqdm import tqdm

class sde:
    """SDE solver class"""
    def __init__(
        self,
        drift,
        diffusion,
        *,
        t0,
        t1,
        num_steps,
        sampler_type,
    ):
        assert t0 < t1, "SDE sampler has to be in forward time"

        self.num_timesteps = num_steps
        self.t = th.linspace(t0, t1, num_steps)
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type

    def __Euler_Maruyama_step(self, x, mean_x, t, model, **model_kwargs):
        x_xy, x_yt, x_xt = x
        w_cur_xy = th.randn(x_xy.size()).to(x_xy)
        w_cur_yt = th.randn(x_yt.size()).to(x_yt)
        w_cur_xt = th.randn(x_xt.size()).to(x_xt)
        t = th.ones(x[0].size(0)).to(x_xy) * t

        dw_xy = w_cur_xy * th.sqrt(self.dt)
        dw_yt = w_cur_yt * th.sqrt(self.dt)
        dw_xt = w_cur_xt * th.sqrt(self.dt)

        drift_xy, drift_yt, drift_xt = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)

        mean_x_xy = x_xy + drift_xy * self.dt
        mean_x_yt = x_yt + drift_yt * self.dt
        mean_x_xt = x_xt + drift_xt * self.dt

        x_xy = mean_x_xy + th.sqrt(2 * diffusion) * dw_xy
        x_yt = mean_x_yt + th.sqrt(2 * diffusion) * dw_yt
        x_xt = mean_x_xt + th.sqrt(2 * diffusion) * dw_xt

        return (x_xy, x_yt, x_xt), (mean_x_xy, mean_x_yt, mean_x_xt)

    def __Heun_step(self, x, _, t, model, **model_kwargs):
        x_xy, x_yt, x_xt = x
        w_cur_xy = th.randn(x_xy.size()).to(x_xy)
        w_cur_yt = th.randn(x_yt.size()).to(x_yt)
        w_cur_xt = th.randn(x_xt.size()).to(x_xt)
        dw_xy = w_cur_xy * th.sqrt(self.dt)
        dw_yt = w_cur_yt * th.sqrt(self.dt)
        dw_xt = w_cur_xt * th.sqrt(self.dt)
        t_cur = th.ones(x[0].size(0)).to(x_xy) * t
        diffusion = self.diffusion(x, t_cur)
        xhat_xy = x_xy + th.sqrt(2 * diffusion) * dw_xy
        xhat_yt = x_yt + th.sqrt(2 * diffusion) * dw_yt
        xhat_xt = x_xt + th.sqrt(2 * diffusion) * dw_xt
        K1_xy, K1_yt, K1_xt = self.drift((xhat_xy, xhat_yt, xhat_xt), t_cur, model, **model_kwargs)
        xp_xy = xhat_xy + self.dt * K1_xy
        xp_yt = xhat_yt + self.dt * K1_yt
        xp_xt = xhat_xt + self.dt * K1_xt
        K2_xy, K2_yt, K2_xt = self.drift((xp_xy, xp_yt, xp_xt), t_cur + self.dt, model, **model_kwargs)
        return (xhat_xy + 0.5 * self.dt * (K1_xy + K2_xy),
                xhat_yt + 0.5 * self.dt * (K1_yt + K2_yt),
                xhat_xt + 0.5 * self.dt * (K1_xt + K2_xt),), (xhat_xy, xhat_yt, xhat_xt) # at last time point we do not perform the heun step

    def __forward_fn(self):
        """TODO: generalize here by adding all private functions ending with steps to it"""
        sampler_dict = {
            "Euler": self.__Euler_Maruyama_step,
            "Heun": self.__Heun_step,
        }

        try:
            sampler = sampler_dict[self.sampler_type]
        except:
            raise NotImplementedError("Smapler type not implemented.")

        return sampler

    def sample(self, init, model, **model_kwargs):
        """forward loop of sde"""
        x = init
        mean_x = init
        samples = []
        sampler = self.__forward_fn()
        for ti in self.t[:-1]:
            with th.no_grad():
                x, mean_x = sampler(x, mean_x, ti, model, **model_kwargs)
                samples.append(x)

        return samples

class ode:
    """ODE solver class"""
    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol,
        rtol,
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.drift = drift
        self.t = th.linspace(t0, t1, num_steps)
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def sample(self, x, model, **model_kwargs):

        device = x[0].device if isinstance(x, tuple) else x.device
        def _fn(t, x):
            t = th.ones(x[0].size(0)).to(device) * t if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t
            model_output = self.drift(x, t, model, **model_kwargs)
            return model_output

        t = self.t.to(device)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        samples = odeint(
            _fn,
            x,
            t,
            method=self.sampler_type,
            atol=atol,
            rtol=rtol
        )
        return samples