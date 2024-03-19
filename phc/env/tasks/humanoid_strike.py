# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils

class HumanoidStrike(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._tar_dist_min = 0.5
        self._tar_dist_max = 10.0
        self._near_dist = 1.5
        self._near_prob = 0.5
        
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        
        strike_body_names = cfg["env"]["strikeBodyNames"]
        self._strike_body_ids = self._build_strike_body_ids_tensor(self.envs[0], self.humanoid_handles[0], strike_body_names)
        self._build_target_tensors()
        
        self.power_usage_reward = cfg["env"].get("power_usage_reward", False)
        self.power_coefficient = cfg["env"].get("power_coefficient", 0.0005)
        self.power_usage_coefficient = cfg["env"].get("power_usage_coefficient", 0.0025)
        self.power_acc = torch.zeros((self.num_envs, 2 )).to(self.device)
        

        return
    
    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 15
        return obs_size
    
    def post_physics_step(self):
        super().post_physics_step()

        if (humanoid_amp.HACK_OUTPUT_MOTION):
            self._hack_output_motion_target()

        return
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        self._target_handles = []
        self._load_target_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self._build_target(env_id, env_ptr)
        return

    def _load_target_asset(self):
        asset_root = "pulse/data/assets/urdf/"
        asset_file = "strike_target.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 30.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._target_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        return

    def _build_target(self, env_id, env_ptr):
        default_pose = gymapi.Transform()
        default_pose.p.x = 1.0
        
        target_handle = self.gym.create_actor(env_ptr, self._target_asset, default_pose, "target", env_id, 2)
        self._target_handles.append(target_handle)

        return

    def _build_strike_body_ids_tensor(self, env_ptr, actor_handle, body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]
        
        self._tar_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1
        
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._tar_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :]

        return

    def _reset_actors(self, env_ids):
        super()._reset_actors(env_ids)
        self._reset_target(env_ids)
        return

    def _reset_target(self, env_ids):
        n = len(env_ids)

        init_near = torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device) < self._near_prob
        dist_max = self._tar_dist_max * torch.ones([n], dtype=self._target_states.dtype, device=self._target_states.device)
        dist_max[init_near] = self._near_dist
        rand_dist = (dist_max - self._tar_dist_min) * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device) + self._tar_dist_min
        
        rand_theta = 2 * np.pi * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device)
        self._target_states[env_ids, 0] = rand_dist * torch.cos(rand_theta) + self._humanoid_root_states[env_ids, 0]
        self._target_states[env_ids, 1] = rand_dist * torch.sin(rand_theta) + self._humanoid_root_states[env_ids, 1]
        self._target_states[env_ids, 2] = 0.9
        
        rand_rot_theta = 2 * np.pi * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device)
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=self._target_states.dtype, device=self._target_states.device)
        rand_rot = quat_from_angle_axis(rand_rot_theta, axis)

        self._target_states[env_ids, 3:7] = rand_rot
        self._target_states[env_ids, 7:10] = 0.0
        self._target_states[env_ids, 10:13] = 0.0
        return
    
    def _sample_ref_state(self, env_ids):
        motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel,  rb_pos, rb_rot, body_vel, body_ang_vel = super()._sample_ref_state(env_ids)
        root_pos[..., :2] = 0.0 # Set the root position to be zero
        return motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel,  rb_pos, rb_rot, body_vel, body_ang_vel

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        return
    
    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            tar_states = self._target_states
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_states = self._target_states[env_ids]
        
        obs = compute_strike_observations(root_states, tar_states)
        return obs

    def _compute_reward(self, actions):
        tar_pos = self._target_states[..., 0:3]
        tar_rot = self._target_states[..., 3:7]
        char_root_state = self._humanoid_root_states
        strike_body_vel = self._rigid_body_vel[..., self._strike_body_ids[0], :]

        self.rew_buf[:] = compute_strike_reward(tar_pos, tar_rot, char_root_state, 
                                                self._prev_root_pos, strike_body_vel,
                                                self.dt, self._near_dist)
        
        if self.power_usage_reward: 
            power_all = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel))
            power_all = power_all.reshape(-1, 23, 3)
            left_power = power_all[:, self.left_lower_indexes].reshape(self.num_envs, -1).sum(dim = -1)
            right_power = power_all[:, self.right_lower_indexes].reshape(self.num_envs, -1).sum(dim = -1)
            self.power_acc[:, 0] += left_power
            self.power_acc[:, 1] += right_power
            power_usage_reward = self.power_acc/(self.progress_buf + 1)[:, None]
            # print((power_usage_reward[:, 0] - power_usage_reward[:, 1]).abs())
            power_usage_reward = - self.power_usage_coefficient * (power_usage_reward[:, 0] - power_usage_reward[:, 1]).abs()
            power_usage_reward[self.progress_buf <= 3] = 0 # First 3 frame power reward should not be counted. since they could be dropped. on the ground to balance.
            
            self.rew_buf[:] += power_usage_reward
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                           self._contact_forces, self._contact_body_ids,
                                                           self._rigid_body_pos, self._tar_contact_forces,
                                                           self._strike_body_ids, self.max_episode_length,
                                                           self._enable_early_termination, self._termination_heights)
        return

    def _draw_task(self):
        cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        starts = self._humanoid_root_states[..., 0:3]
        ends = self._target_states[..., 0:3]
        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

        return

    def _hack_output_motion_target(self):
        if (not hasattr(self, '_output_motion_target_pos')):
            self._output_motion_target_pos = []
            self._output_motion_target_rot = []

        tar_pos = self._target_states[0, 0:3].cpu().numpy()
        self._output_motion_target_pos.append(tar_pos)

        tar_rot = self._target_states[0, 3:7].cpu().numpy()
        self._output_motion_target_rot.append(tar_rot)
        
        reset = self.reset_buf[0].cpu().numpy() == 1

        if (reset and len(self._output_motion_target_pos) > 1):
            output_tar_pos = np.array(self._output_motion_target_pos)
            output_tar_rot = np.array(self._output_motion_target_rot)
            output_data = np.concatenate([output_tar_pos, output_tar_rot], axis=-1)
            np.save('output/record_tar_motion.npy', output_data)

            self._output_motion_target_pos = []
            self._output_motion_target_rot = []

        return
    
class HumanoidStrikeZ(humanoid_amp_task.HumanoidAMPZTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._tar_dist_min = 0.5
        self._tar_dist_max = 10.0
        self._near_dist = 1.5
        self._near_prob = 0.5
        
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        
        strike_body_names = cfg["env"]["strikeBodyNames"]
        self._strike_body_ids = self._build_strike_body_ids_tensor(self.envs[0], self.humanoid_handles[0], strike_body_names)
        self._build_target_tensors()
        
        self.power_usage_reward = cfg["env"].get("power_usage_reward", False)
        self.power_coefficient = cfg["env"].get("power_coefficient", 0.0005)
        self.power_usage_coefficient = cfg["env"].get("power_usage_coefficient", 0.0025)
        self.power_acc = torch.zeros((self.num_envs, 2 )).to(self.device)

        return
    
    def _sample_ref_state(self, env_ids):
        motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel,  rb_pos, rb_rot, body_vel, body_ang_vel = super()._sample_ref_state(env_ids)
        root_pos[..., :2] = 0.0 # Set the root position to be zero
        return motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel,  rb_pos, rb_rot, body_vel, body_ang_vel
    
    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 15
        return obs_size
    
    def post_physics_step(self):
        super().post_physics_step()

        if (humanoid_amp.HACK_OUTPUT_MOTION):
            self._hack_output_motion_target()

        return
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        self._target_handles = []
        self._load_target_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self._build_target(env_id, env_ptr)
        return

    def _load_target_asset(self):
        asset_root = "pulse/data/assets/urdf/"
        asset_file = "strike_target.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 30.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._target_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        return

    def _build_target(self, env_id, env_ptr):
        default_pose = gymapi.Transform()
        default_pose.p.x = 1.0
        
        target_handle = self.gym.create_actor(env_ptr, self._target_asset, default_pose, "target", env_id, 2)
        self._target_handles.append(target_handle)

        return

    def _build_strike_body_ids_tensor(self, env_ptr, actor_handle, body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]
        
        self._tar_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1
        
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._tar_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :]

        return

    def _reset_actors(self, env_ids):
        super()._reset_actors(env_ids)
        self._reset_target(env_ids)
        return

    def _reset_target(self, env_ids):
        n = len(env_ids)

        init_near = torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device) < self._near_prob
        dist_max = self._tar_dist_max * torch.ones([n], dtype=self._target_states.dtype, device=self._target_states.device)
        dist_max[init_near] = self._near_dist
        rand_dist = (dist_max - self._tar_dist_min) * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device) + self._tar_dist_min
        
        rand_theta = 2 * np.pi * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device)
        self._target_states[env_ids, 0] = rand_dist * torch.cos(rand_theta) + self._humanoid_root_states[env_ids, 0]
        self._target_states[env_ids, 1] = rand_dist * torch.sin(rand_theta) + self._humanoid_root_states[env_ids, 1]
        self._target_states[env_ids, 2] = 0.9
        
        rand_rot_theta = 2 * np.pi * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device)
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=self._target_states.dtype, device=self._target_states.device)
        rand_rot = quat_from_angle_axis(rand_rot_theta, axis)

        self._target_states[env_ids, 3:7] = rand_rot
        self._target_states[env_ids, 7:10] = 0.0
        self._target_states[env_ids, 10:13] = 0.0
        return

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        return
    
    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            tar_states = self._target_states
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_states = self._target_states[env_ids]
        
        obs = compute_strike_observations(root_states, tar_states)
        return obs

    def _compute_reward(self, actions):
        tar_pos = self._target_states[..., 0:3]
        tar_rot = self._target_states[..., 3:7]
        char_root_state = self._humanoid_root_states
        strike_body_vel = self._rigid_body_vel[..., self._strike_body_ids[0], :]
        
        self.rew_buf[:] = compute_strike_reward(tar_pos, tar_rot, char_root_state, 
                                                self._prev_root_pos, strike_body_vel,
                                                self.dt, self._near_dist)
        
        
        if self.power_usage_reward: 
            power_all = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel))
            power_all = power_all.reshape(-1, 23, 3)
            left_power = power_all[:, self.left_lower_indexes].reshape(self.num_envs, -1).sum(dim = -1)
            right_power = power_all[:, self.right_lower_indexes].reshape(self.num_envs, -1).sum(dim = -1)
            self.power_acc[:, 0] += left_power
            self.power_acc[:, 1] += right_power
            power_usage_reward = self.power_acc/(self.progress_buf + 1)[:, None]
            # print((power_usage_reward[:, 0] - power_usage_reward[:, 1]).abs())
            power_usage_reward = - self.power_usage_coefficient * (power_usage_reward[:, 0] - power_usage_reward[:, 1]).abs()
            power_usage_reward[self.progress_buf <= 3] = 0 # First 3 frame power reward should not be counted. since they could be dropped. on the ground to balance.
            
            self.rew_buf[:] += power_usage_reward
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                           self._contact_forces, self._contact_body_ids,
                                                           self._rigid_body_pos, self._tar_contact_forces,
                                                           self._strike_body_ids, self.max_episode_length,
                                                           self._enable_early_termination, self._termination_heights)
        return

    def _draw_task(self):
        cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        starts = self._humanoid_root_states[..., 0:3]
        ends = self._target_states[..., 0:3]
        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

        return

    def _hack_output_motion_target(self):
        if (not hasattr(self, '_output_motion_target_pos')):
            self._output_motion_target_pos = []
            self._output_motion_target_rot = []

        tar_pos = self._target_states[0, 0:3].cpu().numpy()
        self._output_motion_target_pos.append(tar_pos)

        tar_rot = self._target_states[0, 3:7].cpu().numpy()
        self._output_motion_target_rot.append(tar_rot)
        
        reset = self.reset_buf[0].cpu().numpy() == 1

        if (reset and len(self._output_motion_target_pos) > 1):
            output_tar_pos = np.array(self._output_motion_target_pos)
            output_tar_rot = np.array(self._output_motion_target_rot)
            output_data = np.concatenate([output_tar_pos, output_tar_rot], axis=-1)
            np.save('output/record_tar_motion.npy', output_data)

            self._output_motion_target_pos = []
            self._output_motion_target_rot = []

        return

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_strike_observations(root_states, tar_states):
    # type: (Tensor, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    tar_pos = tar_states[:, 0:3]
    tar_rot = tar_states[:, 3:7]
    tar_vel = tar_states[:, 7:10]
    tar_ang_vel = tar_states[:, 10:13]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    local_tar_pos = tar_pos - root_pos
    local_tar_pos[..., -1] = tar_pos[..., -1]
    local_tar_pos = torch_utils.my_quat_rotate(heading_rot, local_tar_pos)
    local_tar_vel = torch_utils.my_quat_rotate(heading_rot, tar_vel)
    local_tar_ang_vel = torch_utils.my_quat_rotate(heading_rot, tar_ang_vel)

    local_tar_rot = quat_mul(heading_rot, tar_rot)
    local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)

    obs = torch.cat([local_tar_pos, local_tar_rot_obs, local_tar_vel, local_tar_ang_vel], dim=-1)
    return obs

@torch.jit.script
def compute_strike_reward(tar_pos, tar_rot, root_state, prev_root_pos, strike_body_vel, dt, near_dist):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, float) -> Tensor
    tar_speed = 1.0
    vel_err_scale = 4.0

    tar_rot_w = 0.6
    vel_reward_w = 0.4

    up = torch.zeros_like(tar_pos)
    up[..., -1] = 1
    tar_up = quat_rotate(tar_rot, up)
    tar_rot_err = torch.sum(up * tar_up, dim=-1)
    tar_rot_r = torch.clamp_min(1.0 - tar_rot_err, 0.0)

    root_pos = root_state[..., 0:3]
    tar_dir = tar_pos[..., 0:2] - root_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = tar_speed - tar_dir_speed
    tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
    vel_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0


    reward = tar_rot_w * tar_rot_r + vel_reward_w * vel_reward
    
    succ = tar_rot_err < 0.2
    reward = torch.where(succ, torch.ones_like(reward), reward)
    return reward
    

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           tar_contact_forces, strike_body_ids, max_episode_length,
                           enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
    contact_force_threshold = 50.0
    
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        tar_has_contact = torch.any(torch.abs(tar_contact_forces[..., 0:2]) > contact_force_threshold, dim=-1)
        #strike_body_force = contact_buf[:, strike_body_id, :]
        #strike_body_has_contact = torch.any(torch.abs(strike_body_force) > contact_force_threshold, dim=-1)
        nonstrike_body_force = masked_contact_buf
        nonstrike_body_force[:, strike_body_ids, :] = 0
        
        # nonstrike_body_has_contact = torch.any(torch.sqrt(torch.square(torch.abs(nonstrike_body_force.sum(dim=-2))).sum(dim=-1)) > contact_force_threshold, dim=-1)
        # nonstrike_body_has_contact = torch.any(nonstrike_body_has_contact, dim=-1)
        
        nonstrike_body_has_contact = torch.any(torch.abs(nonstrike_body_force) > contact_force_threshold, dim=-1)
        nonstrike_body_has_contact = torch.any(nonstrike_body_has_contact, dim=-1)

        tar_fail = torch.logical_and(tar_has_contact, nonstrike_body_has_contact)
        
        has_failed = torch.logical_or(has_fallen, tar_fail)
        

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_failed *= (progress_buf > 1)
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)
        
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated