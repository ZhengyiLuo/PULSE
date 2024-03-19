import time
import torch
import phc.env.tasks.humanoid_pedestrian_terrain as humanoid_pedestrain_terrain
from phc.env.tasks.humanoid_amp import remove_base_rot
from phc.utils import torch_utils
from typing import OrderedDict

from isaacgym.torch_utils import *
from phc.utils.flags import flags
from rl_games.algos_torch import torch_ext
import torch.nn as nn
from collections import deque
from phc.learning.network_loader import load_z_encoder, load_z_decoder
from phc.utils.torch_utils import project_to_norm

ENABLE_MAX_COORD_OBS = True

class HumanoidPedestrianTerrainZ(humanoid_pedestrain_terrain.HumanoidPedestrianTerrain):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)

        self.models_path = cfg["env"].get("models", ['output/dgx/smpl_im_fit_3_1/Humanoid_00185000.pth'])
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
        
        self.embedding_partition_distill = self.distill_model_config.get("embedding_partion", 1)
        self.dict_size_distill = self.distill_model_config.get("dict_size", 1)
        ### Loading Distill Model ###
        
        self.z_all = self.cfg['env'].get("z_all", False)
        self.use_vae_prior = self.cfg['env'].get("use_vae_prior", False)
        self.running_mean, self.running_var = check_points[-1]['running_mean_std']['running_mean'], check_points[-1]['running_mean_std']['running_var']
        
        self.decoder = load_z_decoder(check_points[0], activation = self.z_activation, z_type = self.distill_z_type, device = self.device) 
        self.encoder = load_z_encoder(check_points[0], activation = self.z_activation, z_type = self.distill_z_type, device = self.device)
        self.power_acc = torch.zeros((self.num_envs, 2 )).to(self.device)

        return
    
    def get_task_obs_size_detail(self):
        task_obs_detail = super().get_task_obs_size_detail()

        ### For Z
        task_obs_detail['proj_norm'] = self.cfg['env'].get("proj_norm", True)
        task_obs_detail['embedding_norm'] = self.cfg['env'].get("embedding_norm", 3)
        task_obs_detail['embedding_size'] = self.cfg['env'].get("embedding_size", 256)
        task_obs_detail['z_readout'] = self.cfg['env'].get("z_readout", False)
        task_obs_detail['z_type'] = self.cfg['env'].get("z_type", "sphere")
        task_obs_detail['num_unique_motions'] = self._motion_lib._num_unique_motions
        

        return task_obs_detail

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        self._num_actions = self.cfg['env'].get("embedding_size", 256)
        return

    def step(self, action_z):

        # if self.dr_randomizations.get('actions', None):
        #     actions = self.dr_randomizations['actions']['noise_lambda'](actions)
        # if flags.server_mode:
        # t_s = time.time()
        # t_s = time.time()
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
                    prior_mu, prior_log_var = self.decoder.z_prior_mu(z_prior_out), self.decoder.z_prior_logvar(z_prior_out)
                    action_z = prior_mu + action_z
                else:
                    pass
            else:
                action_z = project_to_norm(action_z, self.cfg['env'].get("embedding_norm", 5), self.distill_z_type)
            
            if self.z_all:
                x_all = self.decoder.decoder(action_z)
            else:
                self_obs = torch.clamp(self_obs, min=-5.0, max=5.0)
                x_all = self.decoder.decoder(torch.cat([self_obs, action_z], dim = -1))
                
                # z_prior_out = self.decoder.z_prior(self_obs); prior_mu, prior_log_var = self.decoder.z_prior_mu(z_prior_out), self.decoder.z_prior_logvar(z_prior_out); print(prior_mu.max(), prior_mu.min())
                # print('....')
            actions = x_all


        # actions = x_all[:, 3]  # Debugging
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


@torch.jit.script
def compute_z_target(root_pos, root_rot,  ref_body_pos, ref_body_vel, time_steps, upright):
    # type: (Tensor, Tensor, Tensor, Tensor, int, bool) -> Tensor
    # No rotation information. Leave IK for RL.
    # Future tracks in this obs will not contain future diffs.
    obs = []
    B, J, _ = ref_body_pos.shape

    if not upright:
        root_rot = remove_base_rot(root_rot)

    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, J, 1)).repeat_interleave(time_steps, 0)
    local_ref_body_pos = ref_body_pos.view(B, time_steps, J, 3) - root_pos.view(B, 1, 1, 3)  # preserves the body position
    local_ref_body_pos = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), local_ref_body_pos.view(-1, 3))


    return local_ref_body_pos.view(B, J, -1)