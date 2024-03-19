import time
import torch
import phc.env.tasks.humanoid_amp as humanoid_amp
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

from phc.utils.motion_lib import MotionLib
from phc.utils.motion_lib_smpl import MotionLibSMPL 

from phc.learning.network_loader import load_z_encoder, load_z_decoder

HACK_MOTION_SYNC = False

class HumanoidAMPZ(humanoid_amp.HumanoidAMP):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)

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
        
        self.use_vae_prior_loss = self.cfg['env'].get("use_vae_prior_loss", False)
        self.use_vae_prior = self.cfg['env'].get("use_vae_prior", False)
        self.use_vae_fixed_prior = self.cfg['env'].get("use_vae_fixed_prior", False)
        self.use_vae_sphere_prior = self.cfg['env'].get("use_vae_sphere_prior", False)
        self.use_vae_sphere_posterior = self.cfg['env'].get("use_vae_sphere_posterior", False)
        
        
        self.decoder = load_z_decoder(check_points[0], activation = self.z_activation, z_type = self.distill_z_type, device = self.device) 
        self.encoder = load_z_encoder(check_points[0], activation = self.z_activation, z_type = self.distill_z_type, device = self.device)
        self.power_acc = torch.zeros((self.num_envs, 2)).to(self.device)
        self.power_usage_coefficient = cfg["env"].get("power_usage_coefficient", 0.005)

        self.running_mean, self.running_var = check_points[-1]['running_mean_std']['running_mean'], check_points[-1]['running_mean_std']['running_var']

        if self.save_kin_info:
            self.kin_dict = OrderedDict()
            self.kin_dict.update({
                "gt_z": torch.zeros([self.num_envs,self.cfg['env'].get("embedding_size", 256) ]),
                }) # current root pos + root for future aggergration

        return
    
    def _load_motion(self, motion_file):
        assert (self._dof_offsets[-1] == self.num_dof)
        if self.humanoid_type in ["smpl", "smplh", "smplx"]:
            self._motion_lib = MotionLibSMPL(motion_file=motion_file, device=self.device, masterfoot_conifg=self._masterfoot_config)
                
            self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=self.humanoid_shapes.cpu(), limb_weights=self.humanoid_limb_and_weights.cpu(), random_sample=not HACK_MOTION_SYNC)
        
        else:
            self._motion_lib = MotionLib(motion_file=motion_file, dof_body_ids=self._dof_body_ids, dof_offsets=self._dof_offsets, key_body_ids=self._key_body_ids.cpu().numpy(), device=self.device)

        return
    
 
    def load_pnn(self, pnn_ck):
        mlp_args = {'input_size': pnn_ck['model']['a2c_network.pnn.actors.0.0.weight'].shape[1], 'units': pnn_ck['model']['a2c_network.pnn.actors.0.2.weight'].shape[::-1], 'activation': "relu", 'dense_func': torch.nn.Linear}
        pnn = PNN(mlp_args, output_size=69, numCols=self.num_prim, has_lateral=self.has_lateral)
        state_dict = pnn.state_dict()
        for k in pnn_ck['model'].keys():
            if "pnn" in k:
                pnn_dict_key = k.split("pnn.")[1]
                state_dict[pnn_dict_key].copy_(pnn_ck['model'][k])
        pnn.freeze_pnn(self.num_prim)
        pnn.to(self.device)
        return pnn



    def get_task_obs_size_detail(self):
        task_obs_detail = OrderedDict()

        ### For Z
        task_obs_detail['proj_norm'] = self.cfg['env'].get("proj_norm", True)
        task_obs_detail['embedding_norm'] = self.cfg['env'].get("embedding_norm", 3)
        task_obs_detail['embedding_size'] = self.cfg['env'].get("embedding_size", 256)
        task_obs_detail['z_readout'] = self.cfg['env'].get("z_readout", False)
        task_obs_detail['z_type'] = self.cfg['env'].get("z_type", "sphere")
        task_obs_detail['num_unique_motions'] = self._motion_lib._num_unique_motions

        return task_obs_detail

    def _compute_reward(self, actions):
        super()._compute_reward(actions)
        
        # power_all = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel))
        # power_all = power_all.reshape(-1, 23, 3)
        # left_power = power_all[:, self.left_indexes].reshape(self.num_envs, -1).sum(dim = -1)
        # right_power = power_all[:, self.right_indexes].reshape(self.num_envs, -1).sum(dim = -1)
        # self.power_acc[:, 0] += left_power
        # self.power_acc[:, 1] += right_power
        # self.power_acc[self.progress_buf <= 3] = 0
        # power_usage_reward = self.power_acc/(self.progress_buf + 1)[:, None]
        # power_usage_reward = - self.power_usage_coefficient * (power_usage_reward[:, 0] - power_usage_reward[:, 1]).abs()
        # power_usage_reward[self.progress_buf <= 3] = 0 # First 3 frame power reward should not be counted. since they could be dropped. on the ground to balance.
        # self.rew_buf[:] = power_usage_reward
        
        # import ipdb; ipdb.set_trace()
        
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
            if self.obs_v == 2:
                self_obs_size = self_obs_size//self.past_track_steps
                obs_buf = self.obs_buf.view(self.num_envs, self.past_track_steps, -1)
                curr_obs = obs_buf[:, -1]
                self_obs = ((curr_obs[:, :self_obs_size] - self.running_mean.float()[:self_obs_size]) / torch.sqrt(self.running_var.float()[:self_obs_size] + 1e-05))
            else:
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