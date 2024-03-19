# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import time
import torch
import phc.env.tasks.humanoid_im as humanoid_im
from phc.env.tasks.humanoid_amp import remove_base_rot
from phc.utils import torch_utils
from typing import OrderedDict

from isaacgym.torch_utils import *
from phc.utils.flags import flags
from rl_games.algos_torch import torch_ext
import torch.nn as nn
from phc.learning.pnn import PNN
from collections import deque
from phc.utils.torch_utils import project_to_norm
from phc.learning.network_loader import load_z_encoder, load_z_decoder, load_pnn, load_mcp_mlp

class HumanoidImDistill(humanoid_im.HumanoidIm):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)

        
        # if True:
        if self.distill and not flags.test:
            check_points = [torch_ext.load_checkpoint(ck_path) for ck_path in self.models_path]
            self.distill_z_model = self.cfg['env'].get("distill_z_model", False)
            self.distill_model_config = self.cfg['env']['distill_model_config']
            self.fut_tracks_distill = self.distill_model_config.get("fut_tracks", False)
            self.num_traj_samples_distill = self.distill_model_config.get("numTrajSamples", -1)
            self.traj_sample_timestep_distill = self.distill_model_config.get("trajSampleTimestepInv", -1)
            self.fut_tracks_dropout_distill = self.distill_model_config.get('fut_tracks_dropout', False)
            self.z_activation = self.distill_model_config['z_activation']
            self.root_height_obs_distill = self.distill_model_config.get('root_height_obs', True)
            ### Loading Distill Model ###
            
            if self.distill_z_model:    
                self.embedding_size_distill = self.distill_model_config['embedding_size']
                self.embedding_norm_distill = self.distill_model_config['embedding_norm']
                self.z_all_distill = self.distill_model_config.get('z_all', False)
                self.distill_z_type = self.distill_model_config.get("z_type", "sphere")
                self.use_vae_prior_loss = self.cfg['env'].get("use_vae_prior_loss", False)
                self.use_vae_prior = self.cfg['env'].get("use_vae_prior", False)
                self.decoder = load_z_decoder(check_points[0], activation = self.z_activation, z_type = self.distill_z_type, device = self.device) 
                self.encoder = load_z_encoder(check_points[0], activation = self.z_activation, z_type = self.distill_z_type, device = self.device)
            else:
                self.has_pnn_distill = self.distill_model_config.get("has_pnn", False)
                self.has_lateral_distill = self.distill_model_config.get("has_lateral", False)
                self.num_prim_distill = self.distill_model_config.get("num_prim", 3)
                self.discrete_moe_distill = self.distill_model_config.get("discrete_moe", False)
                if self.has_pnn_distill:
                    assert (len(self.models_path) == 2)
                    self.pnn = load_pnn(check_points[0], num_prim = self.num_prim_distill, has_lateral = self.has_lateral_distill, activation = self.z_activation, device = self.device)
                    self.running_mean, self.running_var = check_points[0]['running_mean_std']['running_mean'], check_points[0]['running_mean_std']['running_var']
                    self.composer = load_mcp_mlp(check_points[1], activation = self.z_activation, device = self.device, mlp_name = "composer")
                else:
                    self.encoder = load_mcp_mlp(check_points[0], activation = self.z_activation, device = self.device)
                # else:
                #     self.actors = [self.load_moe_actor(ck) for ck in check_points]
                #     composer_cp = torch_ext.load_checkpoint("output/klab/smpl_im_comp_10/Humanoid_00282500.pth")
                #     self.composer = self.load_moe_composer(composer_cp)

            self.running_mean, self.running_var = check_points[-1]['running_mean_std']['running_mean'], check_points[-1]['running_mean_std']['running_var']
        

        if self.save_kin_info:
            self.kin_dict = OrderedDict()
            self.kin_dict.update({
                "gt_action": torch.zeros([self.num_envs, self._num_actions]),
                "progress_buf": self.progress_buf.clone(),
                }) # current root pos + root for future aggergration
        return

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        return

    def load_pnn(self, pnn_ck):
        mlp_args = {'input_size': pnn_ck['model']['a2c_network.pnn.actors.0.0.weight'].shape[1], 'units': pnn_ck['model']['a2c_network.pnn.actors.0.2.weight'].shape[::-1], 'activation': "relu", 'dense_func': torch.nn.Linear}
        pnn = PNN(mlp_args, output_size=69, numCols=self.num_prim_distill, has_lateral=self.has_lateral_distill)
        state_dict = pnn.state_dict()
        for k in pnn_ck['model'].keys():
            if "pnn" in k:
                pnn_dict_key = k.split("pnn.")[1]
                state_dict[pnn_dict_key].copy_(pnn_ck['model'][k])
        pnn.freeze_pnn(self.num_prim_distill)
        pnn.to(self.device)
        return pnn
    
    def load_moe_actor(self, checkpoint):
        actvation_func = torch_utils.activation_facotry(self.z_activation)
        key_name = "a2c_network.actor_mlp"
        
        loading_keys = [k for k in checkpoint['model'].keys() if k.startswith(key_name)] + ["a2c_network.mu.weight", 'a2c_network.mu.bias']
        loading_keys_linear = [k for k in loading_keys if k.endswith('weight')]
        
        nn_modules = []
        for idx, key in enumerate(loading_keys_linear):
            layer = nn.Linear(*checkpoint['model'][key].shape[::-1])
            nn_modules.append(layer)
            if idx < len(loading_keys_linear) - 1:
                nn_modules.append(actvation_func())
        actor = nn.Sequential(*nn_modules)
        
        state_dict = actor.state_dict()
        
        for idx, key_affix in enumerate(state_dict.keys()):
            state_dict[key_affix].copy_(checkpoint['model'][loading_keys[idx]])
        
        for param in actor.parameters():
            param.requires_grad = False
        actor.to(self.device)
        return actor


    def load_moe_composer(self, checkpoint):
        actvation_func = torch_utils.activation_facotry(self.z_activation)
        composer = nn.Sequential(nn.Linear(*checkpoint['model']['a2c_network.composer.0.weight'].shape[::-1]), actvation_func(), 
                              nn.Linear(*checkpoint['model']['a2c_network.composer.2.weight'].shape[::-1]), actvation_func(), 
                              nn.Linear(*checkpoint['model']['a2c_network.composer.4.weight'].shape[::-1]), 
                              actvation_func()) ###### This final activation function.............. if silu, does not make any sense. 
        
        state_dict = composer.state_dict()
        state_dict['0.weight'].copy_(checkpoint['model']['a2c_network.composer.0.weight'])
        state_dict['0.bias'].copy_(checkpoint['model']['a2c_network.composer.0.bias'])
        state_dict['2.weight'].copy_(checkpoint['model']['a2c_network.composer.2.weight'])
        state_dict['2.bias'].copy_(checkpoint['model']['a2c_network.composer.2.bias'])
        state_dict['4.weight'].copy_(checkpoint['model']['a2c_network.composer.4.weight'])
        state_dict['4.bias'].copy_(checkpoint['model']['a2c_network.composer.4.bias'])

        for param in composer.parameters():
            param.requires_grad = False
        composer.to(self.device)
        return composer
    

    def step(self, actions):


        # if self.dr_randomizations.get('actions', None):
        #     actions = self.dr_randomizations['actions']['noise_lambda'](actions)
        # if flags.server_mode:
            # t_s = time.time()
        # t_s = time.time()
        # if True:
        if not flags.test and self.save_kin_info:
            with torch.no_grad():
            # Apply trained Model.

            ################ GT-Action ################
                temp_tracks = self._track_bodies_id
                self._track_bodies_id = self._full_track_bodies_id
                temp_fut, temp_fut_drop, temp_timestep, temp_num_steps, temp_root_height_obs = self._fut_tracks, self._fut_tracks_dropout, self._traj_sample_timestep, self._num_traj_samples, self._root_height_obs
                self._fut_tracks, self._fut_tracks_dropout, self._traj_sample_timestep, self._num_traj_samples, self._root_height_obs = self.fut_tracks_distill, self.fut_tracks_dropout_distill, 1/self.traj_sample_timestep_distill, self.num_traj_samples_distill, self.root_height_obs_distill
                
                if self.root_height_obs_distill != temp_root_height_obs:
                    self_obs = self.obs_buf[:, :self.get_self_obs_size()]
                    self_obs = torch.cat([self._rigid_body_pos[:, 0, 2:3], self_obs], dim = -1)
                    # self_obs = self._compute_humanoid_obs() # torch.cat([self._rigid_body_pos[:, 0, 2:3], self_obs], dim = -1) - self._compute_humanoid_obs()
                    self_obs_size = self_obs.shape[-1]
                    self_obs = ((self_obs - self.running_mean.float()[:self_obs_size]) / torch.sqrt(self.running_var.float()[:self_obs_size] + 1e-05))
                else:
                    self_obs_size = self.get_self_obs_size()
                    self_obs = ((self.obs_buf[:, :self_obs_size] - self.running_mean.float()[:self_obs_size]) / torch.sqrt(self.running_var.float()[:self_obs_size] + 1e-05))
                    
                if temp_fut == self.fut_tracks_distill and temp_fut_drop == self.fut_tracks_dropout_distill and temp_timestep == 1/self.traj_sample_timestep_distill and temp_num_steps == self.num_traj_samples_distill\
                    and temp_root_height_obs == self.root_height_obs_distill:
                    task_obs = self.obs_buf[:, self.get_self_obs_size():]
                else:
                    task_obs = self._compute_task_obs(save_buffer = False)
                    
                self._track_bodies_id = temp_tracks
                self._fut_tracks, self._fut_tracks_dropout, self._traj_sample_timestep, self._num_traj_samples, self._root_height_obs = temp_fut, temp_fut_drop, temp_timestep, temp_num_steps, temp_root_height_obs

                
                task_obs = ((task_obs - self.running_mean.float()[self_obs_size:]) / torch.sqrt(self.running_var.float()[self_obs_size:] + 1e-05))
                full_obs = torch.cat([self_obs, task_obs], dim = -1)
                full_obs = torch.clamp(full_obs, min=-5.0, max=5.0)
                
                if self.distill_z_model:
                    gt_z = self.encoder.encoder(full_obs)
                    gt_z = project_to_norm(gt_z, self.embedding_norm_distill)
                    if self.z_all_distill:
                        gt_action = self.decoder.decoder(gt_z)
                    else:
                        gt_action = self.decoder.decoder(torch.cat([self_obs, gt_z], dim = -1))
                else:
                    if self.has_pnn_distill:
                        _, pnn_actions = self.pnn(full_obs)
                        x_all = torch.stack(pnn_actions, dim=1)
                        weights = self.composer(full_obs)
                        gt_action = torch.sum(weights[:, :, None] * x_all, dim=1)
                    else:
                        gt_action = self.encoder(full_obs)
                        # x_all = torch.stack([net(full_obs) for net in self.actors], dim=1)
                
                if self.save_kin_info:
                    self.kin_dict['gt_action'] = gt_action.squeeze()
                    self.kin_dict['progress_buf'] = self.progress_buf.clone()
                    
            ################ GT-Action ################
            # actions = gt_action; print("using gt action") # Debugging 
        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame
        self._physics_step()

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()
        if flags.server_mode:
            dt = time.time() - t_s
            print(f'\r {1/dt:.2f} fps', end='')
            
        # dt = time.time() - t_s
        # self.fps.append(1/dt)
        # print(f'\r {np.mean(self.fps):.2f} fps', end='')
        

        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

