from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from learning.amp_network_builder import AMPBuilder
import torch
import torch.nn as nn
import numpy as np

DISC_LOGIT_INIT_SCALE = 1.0

class AMPSeptBuilder(AMPBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        net = AMPSeptBuilder.Network(self.params, **kwargs)
        return net

    class Network(AMPBuilder.Network):
        def __init__(self, params, **kwargs):
            self.self_obs_size = kwargs['self_obs_size']
            self.task_obs_size = kwargs['task_obs_size']
            self.task_obs_size_detail = kwargs['task_obs_size_detail']
            self.topk = 5

            if "people" in self.task_obs_size_detail:
                self.point_net_embedding_size = kwargs.get('point_net_embedding_size', 64)
                kwargs['input_shape'] = (kwargs['self_obs_size'] + params['task_mlp']["units"][-1] + self.point_net_embedding_size, ) # Task embedding size + self_obs
            else:
                kwargs['input_shape'] = (kwargs['self_obs_size'] + params['task_mlp']["units"][-1] , ) # Task embedding size + self_obs

            super().__init__(params, **kwargs)
            self.running_mean = kwargs['mean_std'].running_mean
            self.running_var = kwargs['mean_std'].running_var
            self._build_task_mlp()
            if self.separate:
                self._build_task_mlp()

        def load(self, params):
            super().load(params)
            self._task_units = params['task_mlp']['units']
            self._task_activation = params['task_mlp']['activation']
            self._task_initializer = params['task_mlp']['initializer']
            return

        def eval_task(self, task_obs):

            if "people" in self.task_obs_size_detail:
                B, N = task_obs.shape
                mlp_input_shape = self.task_obs_size_detail['traj'] + self.task_obs_size_detail['people']  # Traj, heightmap
                mlp_input = task_obs[:, :mlp_input_shape]
                point_net_input = task_obs[:, mlp_input_shape:]

                point_net_input_unomralize = point_net_input * torch.sqrt(self.running_var[self.self_obs_size:(self.self_obs_size + self.task_obs_size)][mlp_input_shape:]).float() \
                     + self.running_mean[self.self_obs_size:(self.self_obs_size + self.task_obs_size)][mlp_input_shape:].float()

                mlp_out = self._task_mlp(mlp_input)
                point_net_out = self._point_net(point_net_input_unomralize.view(B, self.topk, -1))
                point_feat, _ = torch.max(point_net_out, dim=1)
                return torch.cat([mlp_out, point_feat], dim=-1)
            else:
                return self._task_mlp(task_obs)


        def eval_critic(self, obs_dict):
            obs = obs_dict['obs']
            self_obs = obs[:, :self.self_obs_size]
            task_obs = obs[:, self.self_obs_size:(self.self_obs_size + self.task_obs_size)]
            assert(obs.shape[-1] == self.self_obs_size + self.task_obs_size)
            #### ZL: add CNN here

            task_out = self.eval_task(task_obs)
            c_input = torch.cat([self_obs, task_out], dim = -1)

            c_out = self.critic_mlp(c_input)
            value = self.value_act(self.value(c_out))
            return value

        def eval_actor(self, obs_dict):
            obs = obs_dict['obs']
            a_out = self.actor_cnn(obs) # This is empty
            a_out = a_out.contiguous().view(a_out.size(0), -1)

            self_obs = obs[:, :self.self_obs_size]
            task_obs = obs[:, self.self_obs_size:(self.self_obs_size + self.task_obs_size)]
            assert(obs.shape[-1] == self.self_obs_size + self.task_obs_size)
            #### ZL: add CNN here

            task_out = self.eval_task(task_obs)
            actor_input = torch.cat([self_obs, task_out], dim = -1)

            a_out = self.actor_mlp(actor_input)

            if self.is_discrete:
                logits = self.logits(a_out)
                return logits

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits

            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.space_config['fixed_sigma']:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))
                return mu, sigma
            return

        def _build_task_mlp(self):
            task_obs_size, task_obs_size_detail = self.task_obs_size, self.task_obs_size_detail
            assert ("traj" in task_obs_size_detail and "heightmap" in task_obs_size_detail)
            mlp_input_shape = task_obs_size_detail['traj'] + task_obs_size_detail['heightmap'] # Traj, heightmap

            self._task_mlp = nn.Sequential()
            mlp_args = {
                'input_size': mlp_input_shape,
                'units': self._task_units,
                'activation': self._task_activation,
                'dense_func': torch.nn.Linear
            }
            self._task_mlp = self._build_mlp(**mlp_args)

            mlp_init = self.init_factory.create(**self._task_initializer)
            for m in self._task_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            if "people" in task_obs_size_detail:
                pointnet_input_shape = (task_obs_size_detail["people"]) // self.topk
                self._build_pointnet(pointnet_input_shape, embedding_shape = self.point_net_embedding_size)

            return

        def _build_critic_task_mlp(self):
            task_obs_size, task_obs_size_detail = self.task_obs_size, self.task_obs_size_detail
            assert ("traj" in task_obs_size_detail and "heightmap" in task_obs_size_detail)
            mlp_input_shape = task_obs_size_detail['traj'] + task_obs_size_detail['heightmap'] # Traj, heightmap

            self._task_mlp = nn.Sequential()
            mlp_args = {
                'input_size': mlp_input_shape,
                'units': self._task_units,
                'activation': self._task_activation,
                'dense_func': torch.nn.Linear
            }
            self._task_mlp = self._build_mlp(**mlp_args)

            mlp_init = self.init_factory.create(**self._task_initializer)
            for m in self._task_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            if "people" in task_obs_size_detail:
                pointnet_input_shape = (task_obs_size_detail["people"]) // self.topk
                self._build_pointnet(pointnet_input_shape, embedding_shape = self.point_net_embedding_size)

            return

        def _build_pointnet(self, input_shape, embedding_shape):
            self._point_net = nn.Sequential(
                nn.Linear(input_shape, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, embedding_shape),
            )
            mlp_init = self.init_factory.create(**self._task_initializer)
            for m in self._point_net.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
            return