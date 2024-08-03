import time
import torch
import phc.env.tasks.humanoid as humanoid
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

from phc.learning.network_loader import load_z_encoder, load_z_decoder

from easydict import EasyDict

HACK_MOTION_SYNC = False

class HumanoidZ(humanoid.Humanoid):

    def initialize_z_models(self):
        check_points = [torch_ext.load_checkpoint(ck_path) for ck_path in self.models_path]
        ### Loading Distill Model ###
        self.distill_model_config = self.cfg['env']['distill_model_config']
        self.embedding_size_distill = self.distill_model_config['embedding_size']
        self.embedding_norm_distill = self.distill_model_config['embedding_norm']
        self.fut_tracks_distill = self.distill_model_config['fut_tracks']
        self.num_traj_samples_distill = self.distill_model_config['numTrajSamples']
        self.traj_sample_timestep_distill = self.distill_model_config['trajSampleTimestepInv']
        self.fut_tracks_dropout_distill = self.distill_model_config['fut_tracks_dropout']
        self.z_activation = self.distill_model_config['z_activation']
        self.distill_z_type = self.distill_model_config.get("z_type", "sphere")
        self.use_part = self.distill_model_config.get('use_part', False)
        
        self.embedding_partition_distill = self.distill_model_config.get("embedding_partion", 1)
        self.dict_size_distill = self.distill_model_config.get("dict_size", 1)
        ### Loading Distill Model ###
        
        self.z_all = self.cfg['env'].get("z_all", False)
        
        self.use_vae_prior_loss = self.cfg['env'].get("use_vae_prior_loss", False)
        self.use_vae_prior = self.cfg['env'].get("use_vae_prior", False)
        self.use_vae_fixed_prior = self.cfg['env'].get("use_vae_fixed_prior", False)
        self.use_vae_sphere_prior = self.cfg['env'].get("use_vae_sphere_prior", False)
        self.use_vae_sphere_posterior = self.cfg['env'].get("use_vae_sphere_posterior", False)
        
        
        self.decoder = load_z_decoder(check_points[0], activation = self.z_activation, z_type = self.distill_z_type, device = self.device) 
        self.encoder = load_z_encoder(check_points[0], activation = self.z_activation, z_type = self.distill_z_type, device = self.device)
            
        self.power_acc = torch.zeros((self.num_envs, 2)).to(self.device)
        self.power_usage_coefficient = self.cfg["env"].get("power_usage_coefficient", 0.005)

        self.running_mean, self.running_var = check_points[-1]['running_mean_std']['running_mean'], check_points[-1]['running_mean_std']['running_var']

        if self.save_kin_info:
            self.kin_dict = OrderedDict()
            self.kin_dict.update({
                "gt_z": torch.zeros([self.num_envs,self.cfg['env'].get("embedding_size", 256) ]),
                }) # current root pos + root for future aggergration
    
    def _setup_character_props_z(self):
        self._num_actions = self.cfg['env'].get("embedding_size", 256)
        return

    def get_task_obs_size_detail_z(self):
        task_obs_detail = OrderedDict()

        ### For Z
        task_obs_detail['proj_norm'] = self.cfg['env'].get("proj_norm", True)
        task_obs_detail['embedding_norm'] = self.cfg['env'].get("embedding_norm", 3)
        task_obs_detail['embedding_size'] = self.cfg['env'].get("embedding_size", 256)
        task_obs_detail['z_readout'] = self.cfg['env'].get("z_readout", False)
        task_obs_detail['z_type'] = self.cfg['env'].get("z_type", "sphere")
        task_obs_detail['num_unique_motions'] = self._motion_lib._num_unique_motions
        return task_obs_detail

    def compute_z_actions(self, action_z):
        with torch.no_grad():
            
            # Apply trained Model.
            ################ GT-Z ################
            self_obs_size = self.get_self_obs_size()
            self_obs = (self.obs_buf[:, :self_obs_size] - self.running_mean.float()[:self_obs_size]) / torch.sqrt(self.running_var.float()[:self_obs_size] + 1e-05)
            
            if self.distill_z_type == "hyper":
                action_z = self.decoder.hyper_layer(action_z)
            if self.distill_z_type == "vq_vae":
                if self.is_discrete:
                    indexes = action_z
                else:
                    B, F = action_z.shape
                    indexes = action_z.reshape(B, -1, self.embedding_size_distill).argmax(dim = -1)
                task_out_proj = self.decoder.quantizer.embedding.weight[indexes.view(-1)]
                print(f"\r {indexes.numpy()[0]}", end = '')
                action_z = task_out_proj.view(-1, self.embedding_size_distill)
            elif self.distill_z_type == "vae":
                if self.use_vae_prior:
                    z_prior_out = self.decoder.z_prior(self_obs)
                    prior_mu = self.decoder.z_prior_mu(z_prior_out)
                    action_z = prior_mu + action_z
                
                if self.use_vae_sphere_posterior:
                    action_z = project_to_norm(action_z, 1, "sphere")
                else:
                    action_z = project_to_norm(action_z, self.cfg['env'].get("embedding_norm", 5), "none")

            else:
                action_z = project_to_norm(action_z, self.cfg['env'].get("embedding_norm", 5), self.distill_z_type)
                
                
            ######################## Apply Z Encoder ########################
            # import phc.env.tasks.humanoid_im as humanoid_im
            # body_pos = self._rigid_body_pos
            # body_rot = self._rigid_body_rot
            # body_vel = self._rigid_body_vel
            # body_ang_vel = self._rigid_body_ang_vel
            # root_states = self._humanoid_root_states
            # root_pos = root_states[..., 0:3]
            # root_rot = root_states[..., 3:7]
            # motion_times = (self.progress_buf + 1) * self.dt + self._motion_start_times
            # motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids, motion_times)  # pass in the env_ids such that the motion is in synced.
            # ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_smpl_params, ref_limb_weights, ref_pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
            # motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
            # motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
            # body_pos_subset = body_pos[..., self._track_bodies_id, :]
            # body_rot_subset = body_rot[..., self._track_bodies_id, :]
            # body_vel_subset = body_vel[..., self._track_bodies_id, :]
            # body_ang_vel_subset = body_ang_vel[..., self._track_bodies_id, :]

            # ref_rb_pos_subset = ref_rb_pos[..., self._track_bodies_id, :]
            # ref_rb_rot_subset = ref_rb_rot[..., self._track_bodies_id, :]
            # ref_body_vel_subset = ref_body_vel[..., self._track_bodies_id, :]
            # ref_body_ang_vel_subset = ref_body_ang_vel[..., self._track_bodies_id, :]
            # im_obs = humanoid_im.compute_imitation_observations_v6(root_pos, root_rot, body_pos_subset, body_rot_subset, body_vel_subset, body_ang_vel_subset, ref_rb_pos_subset, ref_rb_rot_subset, ref_body_vel_subset, ref_body_ang_vel_subset, 1, self._has_upright_start)
            # im_shape = im_obs.shape[-1]
            
            # im_obs = (im_obs - self.running_mean.float()[-im_shape:]) / torch.sqrt(self.running_var.float()[-im_shape:] + 1e-05)
            # encoder_latent = self.encoder.encoder(torch.cat([self_obs, im_obs], dim = -1))
            # action_z = self.encoder.z_mu(encoder_latent)
            ######################## Apply Z Encoder ########################

            if self.z_all:
                x_all = self.decoder.decoder(action_z)
            else:
                self_obs = torch.clamp(self_obs, min=-5.0, max=5.0)
                x_all = self.decoder.decoder(torch.cat([self_obs, action_z], dim = -1))
                
                # z_prior_out = self.decoder.z_prior(self_obs); prior_mu, prior_log_var = self.decoder.z_prior_mu(z_prior_out), self.decoder.z_prior_logvar(z_prior_out); print(prior_mu.max(), prior_mu.min())
                # print('....')
            actions = x_all
        return actions
    
    def step_z(self, action_z):
        self.action_z = action_z

        actions = self.compute_z_actions(action_z)

        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame
        self._physics_step()

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()
        

        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

