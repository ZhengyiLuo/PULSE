from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from learning.amp_network_builder import AMPBuilder
from phc.utils.torch_utils import project_to_norm
import torch
import torch.nn as nn
import numpy as np

DISC_LOGIT_INIT_SCALE = 1.0


class AMPZReaderBuilder(AMPBuilder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        net = AMPZReaderBuilder.Network(self.params, **kwargs)
        return net

    class Network(AMPBuilder.Network):

        def __init__(self, params, **kwargs):
            self.task_obs_size_detail = kwargs['task_obs_size_detail']
            self.self_obs_size = kwargs['self_obs_size']
            self.task_obs_size = kwargs['task_obs_size']
            
            super().__init__(params, **kwargs)
            self.running_mean = kwargs['mean_std'].running_mean
            self.running_var = kwargs['mean_std'].running_var

            self.proj_norm = self.task_obs_size_detail.get("proj_norm", True)
            self.embedding_size = self.task_obs_size_detail.get("embedding_size", 256)
            self.embedding_norm = self.task_obs_size_detail.get("embedding_norm", 5)
            self.z_readout = self.task_obs_size_detail.get("z_readout", False)
            self.z_type = self.task_obs_size_detail.get("z_type", "sphere")
            self.vae_prior_policy = self.task_obs_size_detail.get("vae_prior_policy", False)
            
            if self.vae_prior_policy:
                self.decoder = self.task_obs_size_detail.get("z_decoder", None)
                self.running_mean = self.task_obs_size_detail.get("running_mean", None)
                self.running_var = self.task_obs_size_detail.get("running_var", None)


        def eval_actor(self, obs_dict):
            
            mu_proj, sigma =  super().eval_actor(obs_dict)
            if self.vae_prior_policy:
                self_obs_orig = obs_dict['obs_orig'][:, :self.self_obs_size]
                self_obs_orig_proccessed = ((self_obs_orig - self.running_mean.float()[:self.self_obs_size]) / torch.sqrt(self.running_var.float()[:self.self_obs_size] + 1e-05))
                z_prior_out = self.decoder.z_prior(self_obs_orig_proccessed)
                # prior_mu = self.decoder.z_prior_mu(z_prior_out)
                prior_log_var = self.decoder.z_prior_logvar(z_prior_out)
                return mu_proj, prior_log_var
            else:
                return mu_proj, sigma
            
            