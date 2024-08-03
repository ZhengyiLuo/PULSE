# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch

import env.tasks.humanoid_amp_task as humanoid_amp_task
import env.util.traj_generator as traj_generator

from utils import torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from phc.utils.flags import flags
from phc.utils.draw_utils import agt_color

class HumanoidTraj(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._num_traj_samples = cfg["env"]["numTrajSamples"]
        self._traj_sample_timestep = cfg["env"]["trajSampleTimestep"]
        self._speed_min = cfg["env"]["speedMin"]
        self._speed_max = cfg["env"]["speedMax"]
        self._accel_max = cfg["env"]["accelMax"]
        self._sharp_turn_prob = cfg["env"]["sharpTurnProb"]

        self._fail_dist = 4.0

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self._build_traj_generator()

        if (not self.headless):
            self._build_marker_state_tensors()

        return

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 2 * self._num_traj_samples
        return obs_size

    def get_task_obs_size_detail(self):
        if (self._enable_task_obs):
            return ["traj", 2 * self._num_traj_samples]
        else:
            return []

    def _update_marker(self):
        traj_samples = self._fetch_traj_samples()
        self._marker_pos[:] = traj_samples
        self._marker_pos[..., 2] = self._humanoid_root_states[..., 2:3]  # jp hack # ZL hack

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(self._marker_actor_ids),
                                                     len(self._marker_actor_ids))
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        if (not self.headless):
            self._marker_handles = [[] for _ in range(num_envs)]
            self._load_marker_asset()

        if flags.add_proj:
            self._proj_handles = []
            self._load_proj_asset()
        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _load_marker_asset(self):
        asset_root = "egoquest/data/assets/mjcf/"
        asset_file = "traj_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        if (not self.headless):
            self._build_marker(env_id, env_ptr)

        if flags.add_proj:
            self._build_proj(env_id, env_ptr)

        return

    def _build_traj_generator(self):
        num_envs = self.num_envs
        episode_dur = self.max_episode_length * self.dt
        num_verts = 101
        dtheta_max = 2.0
        self._traj_gen = traj_generator.TrajGenerator(num_envs, episode_dur, num_verts,
                                                      self.device, dtheta_max,
                                                      self._speed_min, self._speed_max,
                                                      self._accel_max, self._sharp_turn_prob)

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        root_pos = self._humanoid_root_states[:, 0:3]
        self._traj_gen.reset(env_ids, root_pos)

        return

    def _build_marker(self, env_id, env_ptr):
        default_pose = gymapi.Transform()

        for i in range(self._num_traj_samples):

            marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", self.num_envs + 10, 0, 0)
            self.gym.set_rigid_body_color(env_ptr, marker_handle, 0,
                                          gymapi.MESH_VISUAL,
                                          gymapi.Vec3(*agt_color(env_id)))
            self._marker_handles[env_id].append(marker_handle)

        return

    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1:(1 + self._num_traj_samples), :]
        self._marker_pos = self._marker_states[..., :3]

        self._marker_actor_ids = self._humanoid_actor_ids.unsqueeze(-1) + to_torch(self._marker_handles, dtype=torch.int32, device=self.device)
        self._marker_actor_ids = self._marker_actor_ids.flatten()

        return

    def _reset_task(self, env_ids):
        super()._reset_task(env_ids)

        root_pos = self._humanoid_root_states[env_ids, 0:3]
        self._traj_gen.reset(env_ids, root_pos)
        return

    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
        else:
            root_states = self._humanoid_root_states[env_ids]

        traj_samples = self._fetch_traj_samples(env_ids)
        obs = compute_location_observations(root_states, traj_samples)

        return obs

    def _compute_flip_task_obs(self, normal_task_obs, env_ids):
        B, D = normal_task_obs.shape
        traj_samples_dim = 20
        normal_task_obs = normal_task_obs.clone()
        traj_samples = normal_task_obs[:, :traj_samples_dim].view(B, 10, 2)
        traj_samples[..., 1] *= -1
        obs = traj_samples.view(B, -1)
        return obs


    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]

        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        tar_pos = self._traj_gen.calc_pos(env_ids, time)

        self.rew_buf[:] = compute_location_reward(root_pos, tar_pos)

        return

    def _compute_reset(self):
        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        tar_pos = self._traj_gen.calc_pos(env_ids, time)

        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                           self._contact_forces, self._contact_body_ids,
                                                           self._rigid_body_pos, tar_pos,
                                                           self.max_episode_length, self._fail_dist,
                                                           self._enable_early_termination, self._termination_heights)
        return

    def _fetch_traj_samples(self, env_ids=None):
        # 5 seconds with 0.5 second intervals, 10 samples.
        if (env_ids is None):
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        timestep_beg = self.progress_buf[env_ids] * self.dt
        timesteps = torch.arange(self._num_traj_samples, device=self.device, dtype=torch.float)
        timesteps = timesteps * self._traj_sample_timestep
        traj_timesteps = timestep_beg.unsqueeze(-1) + timesteps

        env_ids_tiled = torch.broadcast_to(env_ids.unsqueeze(-1), traj_timesteps.shape).to(self.device)

        traj_samples_flat = self._traj_gen.calc_pos(env_ids_tiled.flatten(), traj_timesteps.flatten())
        traj_samples = torch.reshape(traj_samples_flat, shape=(env_ids.shape[0], self._num_traj_samples, traj_samples_flat.shape[-1]))

        return traj_samples

    def _draw_task(self):
        cols = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        self._update_marker()
        self.gym.clear_lines(self.viewer)

        for i, env_ptr in enumerate(self.envs):
            verts = self._traj_gen.get_traj_verts(i)
            verts[..., 2] = self._humanoid_root_states[i, 2]  # ZL Hack
            lines = torch.cat([verts[:-1], verts[1:]], dim=-1).cpu().numpy()
            curr_cols = np.broadcast_to(cols, [lines.shape[0], cols.shape[-1]])
            self.gym.add_lines(self.viewer, env_ptr, lines.shape[0], lines, curr_cols)

        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_location_observations(root_states, traj_samples):
    # type: (Tensor, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    heading_rot_exp = torch.broadcast_to(heading_rot.unsqueeze(-2), (heading_rot.shape[0], traj_samples.shape[1], heading_rot.shape[1]))
    heading_rot_exp = torch.reshape(heading_rot_exp, (heading_rot_exp.shape[0] * heading_rot_exp.shape[1], heading_rot_exp.shape[2]))
    traj_samples_delta = traj_samples - root_pos.unsqueeze(-2)
    traj_samples_delta_flat = torch.reshape(traj_samples_delta, (traj_samples_delta.shape[0] * traj_samples_delta.shape[1],
                                                                 traj_samples_delta.shape[2]))

    local_traj_pos = torch_utils.my_quat_rotate(heading_rot_exp, traj_samples_delta_flat)
    local_traj_pos = local_traj_pos[..., 0:2]

    obs = torch.reshape(local_traj_pos, (traj_samples.shape[0], traj_samples.shape[1] * local_traj_pos.shape[1]))
    return obs

@torch.jit.script
def compute_location_reward(root_pos, tar_pos):
    # type: (Tensor, Tensor) -> Tensor
    pos_err_scale = 2.0

    pos_diff = tar_pos[..., 0:2] - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)

    pos_reward = torch.exp(-pos_err_scale * pos_err)

    reward = pos_reward

    return reward


@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           tar_pos, max_episode_length, fail_dist,
                           enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, bool, Tensor) -> Tuple[Tensor, Tensor]
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
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)

        root_pos = rigid_body_pos[..., 0, :]
        tar_delta = tar_pos[..., 0:2] - root_pos[..., 0:2]
        tar_dist_sq = torch.sum(tar_delta * tar_delta, dim=-1)
        tar_fail = tar_dist_sq > fail_dist * fail_dist

        has_failed = torch.logical_or(has_fallen, tar_fail)

        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated