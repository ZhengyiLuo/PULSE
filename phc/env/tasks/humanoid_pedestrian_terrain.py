# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from shutil import ExecError
import torch
import numpy as np

import env.tasks.humanoid_traj as humanoid_traj
from isaacgym import gymapi
from isaacgym.torch_utils import *
from env.tasks.humanoid import dof_to_obs
from env.tasks.humanoid_amp import HumanoidAMP, remove_base_rot
from phc.utils.flags import flags
from utils import torch_utils
from isaacgym import gymtorch
import joblib
from poselib.poselib.core.rotation3d import quat_inverse, quat_mul
from tqdm import tqdm
from scipy.spatial.transform import Rotation as sRot
import matplotlib.pyplot as plt
from typing import OrderedDict
from phc.utils.draw_utils import agt_color
from phc.env.tasks.humanoid import compute_humanoid_observations_smpl_max, compute_humanoid_observations_smpl, ENABLE_MAX_COORD_OBS

HACK_MOTION_SYNC = False

class HumanoidPedestrianTerrain(humanoid_traj.HumanoidTraj):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id,
                 headless):
        ## ZL Hack to get the height map to load.
        self.velocity_map = cfg["env"].get("velocity_map", False)
        self.device_type = cfg.get("device_type", "cuda")
        self.device_id = cfg.get("device_id", 0)
        self.device = "cpu"
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)
            
        # self.real_mesh = cfg['args'].real_mesh
        self.real_mesh = False
        self.load_humanoid_configs(cfg)
        self.load_smpl_configs(cfg)
        self.cfg = cfg
        self.num_envs = cfg["env"]["num_envs"]
        self.headless = cfg["headless"]
        self.sensor_extent = cfg["env"].get("sensor_extent", 2)
        self.sensor_res = cfg["env"].get("sensor_res", 32)
        self.power_reward = cfg["env"].get("power_reward", False)
        self.power_coefficient = cfg["env"].get("power_coefficient", 0.0005)
        self.fuzzy_target = cfg["env"].get("fuzzy_target", False)


        self.square_height_points = self.init_square_height_points()
        self.terrain_obs_type = self.cfg['env'].get("terrain_obs_type",
                                                    "square")
        self.terrain_obs = self.cfg['env'].get("terrain_obs", False)
        self.terrain_obs_root = self.cfg['env'].get("terrain_obs_root",
                                                    "pelvis")
        if self.terrain_obs_type == "fov":
            self.height_points = self.init_fov_height_points()
        elif self.terrain_obs_type == "square_fov":
            self.height_points = self.init_square_fov_height_points()
        elif self.terrain_obs_type == "square":
            self.height_points = self.square_height_points
        self.root_points = self.init_root_points()

        self.center_height_points = self.init_center_height_points()
        self.height_meas_scale = 5

        self.show_sensors = self.cfg['env'].get("show_sensors", False)
        if (not self.headless) and self.show_sensors:
            self._sensor_handles = [[] for _ in range(self.num_envs)]

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self.reward_raw = torch.zeros((self.num_envs, 2)).to(self.device)

        if (not self.headless) and self.show_sensors:
            self._build_sensor_state_tensors()

        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        if (not self.headless) and self.show_sensors:
            self._load_sensor_asset()
            self._build_sensor(env_id, env_ptr)

        return

    def _build_sensor(self, env_id, env_ptr):
        default_pose = gymapi.Transform()

        for i in range(self.num_height_points):
            marker_handle = self.gym.create_actor(env_ptr, self._sensor_asset,
                                                  default_pose, "marker",
                                                  self.num_envs + 1, 0, 0)
            self.gym.set_rigid_body_color(env_ptr, marker_handle, 0,
                                          gymapi.MESH_VISUAL,
                                          gymapi.Vec3(*agt_color(env_id)))
            self._sensor_handles[env_id].append(marker_handle)

        return

    def _build_sensor_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._sensor_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 11:(11 + self.num_height_points), :]
        self._sensor_pos = self._sensor_states[..., :3]
        self._sensor_actor_ids = self._humanoid_actor_ids.unsqueeze(-1) + to_torch(self._sensor_handles, dtype=torch.int32, device=self.device)
        self._sensor_actor_ids = self._sensor_actor_ids.flatten()
        return

    def _load_sensor_asset(self):
        asset_root = "amp/data/assets/mjcf/"
        asset_file = "sensor_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._sensor_asset = self.gym.load_asset(self.sim, asset_root,
                                                 asset_file, asset_options)

        return

    def _draw_task(self):
        # cols = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        norm_states = self.get_head_pose()
        base_quat = norm_states[:, 3:7]
        if not self._has_upright_start:
            base_quat = remove_base_rot(base_quat)
        heading_rot = torch_utils.calc_heading_quat(base_quat)

        points = quat_apply(
            heading_rot.repeat(1, self.num_height_points).reshape(-1, 4),
            self.height_points) + (norm_states[:, :3]).unsqueeze(1)

        if (not self.headless) and self.show_sensors:
            self._sensor_pos[:] = points
        # self._sensor_pos[..., 2] += 0.3
        # self._sensor_pos[..., 2] -= 5

        traj_samples = self._fetch_traj_samples()

        self._marker_pos[:] = traj_samples
        self._marker_pos[..., 2] = self._humanoid_root_states[..., 2:3]  # jp hack # ZL hack
        # self._marker_pos[..., 2] = 0.89
        # self._marker_pos[..., 2] = 0

        if (not self.headless) and self.show_sensors:
            comb_idx = torch.cat([self._sensor_actor_ids, self._marker_actor_ids])
        else:
            comb_idx = torch.cat([self._marker_actor_ids])

        if flags.show_traj:
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self._root_states),
                gymtorch.unwrap_tensor(comb_idx), len(comb_idx))

            self.gym.clear_lines(self.viewer)

            for i, env_ptr in enumerate(self.envs):
                verts = self._traj_gen.get_traj_verts(i)
                verts[..., 2] = self._humanoid_root_states[i, 2]  # ZL Hack
                # verts[..., 2] = 0.89
                # verts[..., 2] = 0
                lines = torch.cat([verts[:-1], verts[1:]], dim=-1).cpu().numpy()
                # cols = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
                cols = np.array(agt_color(i), dtype=np.float32)[None, ]
                curr_cols = np.broadcast_to(cols, [lines.shape[0], cols.shape[-1]])
                self.gym.add_lines(self.viewer, env_ptr, lines.shape[0], lines, curr_cols)
        else:
            self._marker_pos[:] = 0
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self._root_states),
                gymtorch.unwrap_tensor(comb_idx), len(comb_idx))

            self.gym.clear_lines(self.viewer)


        return

    def _compute_humanoid_obs(self, env_ids=None):
        if (ENABLE_MAX_COORD_OBS):
            if (env_ids is None):
                body_pos = self._rigid_body_pos.clone()
                body_rot = self._rigid_body_rot
                body_vel = self._rigid_body_vel
                body_ang_vel = self._rigid_body_ang_vel
            else:
                body_pos = self._rigid_body_pos[env_ids]
                body_rot = self._rigid_body_rot[env_ids]
                body_vel = self._rigid_body_vel[env_ids]
                body_ang_vel = self._rigid_body_ang_vel[env_ids]
            if self.humanoid_type in ["smpl", "smplh", "smplx"]:
                if (env_ids is None):
                    smpl_params = self.humanoid_shapes
                    limb_weights = self.humanoid_limb_and_weights
                else:
                    smpl_params = self.humanoid_shapes[env_ids]
                    limb_weights = self.humanoid_limb_and_weights[env_ids]

                if self._root_height_obs:
                    center_height = self.get_center_heights(torch.cat([body_pos[:, 0], body_rot[:, 0]], dim=-1), env_ids=env_ids).mean(dim=-1, keepdim=True)
                    body_pos[:, :, 2] = body_pos[:, :, 2] - center_height

                obs = compute_humanoid_observations_smpl_max(body_pos, body_rot, body_vel, body_ang_vel, smpl_params, limb_weights, self._local_root_obs, self._root_height_obs, self._has_upright_start, self._has_shape_obs, self._has_limb_weight_obs)
            else:
                raise NotImplementedError
                # obs = compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, self._local_root_obs, self._root_height_obs)

        else:
            if (env_ids is None):
                root_pos = self._rigid_body_pos[:, 0, :]
                root_rot = self._rigid_body_rot[:, 0, :]
                root_vel = self._rigid_body_vel[:, 0, :]
                root_ang_vel = self._rigid_body_ang_vel[:, 0, :]
                dof_pos = self._dof_pos
                dof_vel = self._dof_vel
                key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
            else:
                root_pos = self._rigid_body_pos[env_ids][:, 0, :]
                root_rot = self._rigid_body_rot[env_ids][:, 0, :]
                root_vel = self._rigid_body_vel[env_ids][:, 0, :]
                root_ang_vel = self._rigid_body_ang_vel[env_ids][:, 0, :]
                dof_pos = self._dof_pos[env_ids]
                dof_vel = self._dof_vel[env_ids]
                key_body_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :]

            if self.humanoid_type in ["smpl", "smplh", "smplx"] and self.self.has_shape_obs:
                if (env_ids is None):
                    smpl_params = self.humanoid_shapes
                else:
                    smpl_params = self.humanoid_shapes[env_ids]
                obs = compute_humanoid_observations_smpl(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, self._dof_obs_size, self._dof_offsets, smpl_params, self._local_root_obs, self._root_height_obs, self._has_upright_start, self._has_shape_obs)
            else:
                raise NotImplementedError
                # obs = compute_humanoid_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, self._local_root_obs, self._root_height_obs, self._dof_obs_size, self._dof_offsets)
        return obs

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):

            obs_size = 2 * self._num_traj_samples

            if self.terrain_obs:
                if self.velocity_map:
                    obs_size += self.num_height_points * 3
                else:
                    obs_size += self.num_height_points

            if self._divide_group and self._group_obs:
                obs_size += 5 * 11 * 3

        return obs_size

    def get_self_obs_size(self):
        return self._num_self_obs

    def get_task_obs_size_detail(self):
        task_obs_detail = OrderedDict()
        
        
        if (self._enable_task_obs):
            task_obs_detail['traj'] = 2 * self._num_traj_samples
            # task_obs_detail.append(["traj", 2 * self._num_traj_samples])

        if self.terrain_obs:
            if self.velocity_map:
                task_obs_detail['heightmap_velocity'] = self.num_height_points * 3
                # task_obs_detail.append(["heightmap_velocity", self.num_height_points * 3])
            else:
                task_obs_detail['heightmap'] = self.num_height_points
                # task_obs_detail.append(["heightmap", self.num_height_points])

        if self._divide_group and self._group_obs:
            task_obs_detail['people'] = 5 * 11 * 3
            # task_obs_detail.append(["people", 5 * 11 * 3])

        return task_obs_detail


    def get_head_pose(self, env_ids=None):
        if self.humanoid_type in ["smpl", "smplh", "smplx"]:
            head_idx = self._body_names.index("Head")
        else:
            head_idx = 2
        head_pose = torch.cat([
            self._rigid_body_pos[:, head_idx], self._rigid_body_rot[:,
                                                                    head_idx]
        ],
                              dim=1)
        if (env_ids is None):
            return head_pose
        else:
            return head_pose[env_ids]

    # # ZL: Dev purposes only, will remove.
    # def _fetch_traj_samples(self, env_ids=None):
    #     # 5 seconds with 0.5 second intervals, 10 samples.
    #     if (env_ids is None):
    #         env_ids = torch.arange(self.num_envs,
    #                                device=self.device,
    #                                dtype=torch.long)

    #     timestep_beg = self.progress_buf[env_ids] * self.dt
    #     timesteps = torch.arange(self._num_traj_samples,
    #                              device=self.device,
    #                              dtype=torch.float)
    #     timesteps = timesteps * self._traj_sample_timestep
    #     traj_timesteps = timestep_beg.unsqueeze(-1) + timesteps

    #     env_ids_tiled = torch.broadcast_to(env_ids.unsqueeze(-1),
    #                                        traj_timesteps.shape)

    #     traj_samples_flat = self._traj_gen.calc_pos(env_ids_tiled.flatten(),
    #                                                 traj_timesteps.flatten())
    #     traj_samples = torch.reshape(traj_samples_flat,
    #                                  shape=(env_ids.shape[0],
    #                                         self._num_traj_samples,
    #                                         traj_samples_flat.shape[-1]))

    #     traj_samples_flat = self._traj_gen.mock_calc_pos(env_ids, env_ids_tiled.flatten(), traj_timesteps.flatten(), self.query_value_gradient)

    #     return traj_samples

    def update_value_func(self, eval_value_func, actor_func):
        self.eval_value_func = eval_value_func
        self.actor_func = actor_func

    def query_value_gradient(self, env_ids, new_traj):
        # new_traj would be the same as self._fetch_traj_samples(env_ids)
        # return callable value function and update_obs (processed with mean and std)
        # value_func(obs)
        # new_traj of shape (num_envs, 10, 3)
        # TODO: implement this
        if "eval_value_func" in self.__dict__:
            sim_obs_size = self.get_self_obs_size()
            task_obs_detal = self.get_task_obs_size_detail()
            assert(task_obs_detal[0][0] == "traj")

            if (env_ids is None):
                root_states = self._humanoid_root_states
            else:
                root_states = self._humanoid_root_states[env_ids]

            new_traj_obs = compute_location_observations(root_states, new_traj.view(env_ids.shape[0], 10, -1), self._has_upright_start)
            buffered_obs = self.obs_buf[env_ids].clone()
            buffered_obs[:, sim_obs_size:(task_obs_detal[0][1] + sim_obs_size)] = new_traj_obs

            return buffered_obs, self.eval_value_func
        return None, None

    def live_plotter(self, img,  identifier='', pause_time=0.00000001):
        if not hasattr(self, 'imshow_obj'):
            plt.ion()

            self.fig = plt.figure(figsize=(1, 1), dpi = 350)
            ax = self.fig.add_subplot(111)
            self.imshow_obj = ax.imshow(img)
            # create a variable for the line so we can later update it
            # update plot label/title

            plt.title('{}'.format(identifier))
            plt.show()
        if not img is None:
            self.imshow_obj.set_data(img)

        # plt.pause(pause_time)
        self.fig.canvas.start_event_loop(0.001)

    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
        else:
            root_states = self._humanoid_root_states[env_ids]
        num_envs = self.num_envs if env_ids is None else len(env_ids)

        traj_samples = self._fetch_traj_samples(env_ids)

        obs = compute_location_observations(root_states, traj_samples, self._has_upright_start)
        if self.terrain_obs:

            if self.terrain_obs_root == "head":
                head_pose = self.get_head_pose(env_ids=env_ids)
                self.measured_heights = self.get_heights(root_states=head_pose, env_ids=env_ids)
            else:
                self.measured_heights = self.get_heights(root_states=root_states, env_ids=env_ids)


            # if flags.height_debug:
            #     # joblib.dump(self.measured_heights, "heights.pkl")
            #     if env_ids is None or len(env_ids) == self.num_envs:
            #         heights = self.measured_heights.view(num_envs, -1, 3 if self.velocity_map else 1)
            #         sensor_size = int(np.sqrt(self.num_height_points))
            #         heights_show = heights.cpu().numpy()[self.viewing_env_idx, :, 0].reshape(sensor_size, sensor_size)
            #         if heights_show.min() < 0:
            #             heights_show -= heights_show.min()
            #         self.live_plotter(heights_show)

            if self.cfg['env'].get("use_center_height", False):
                center_heights = self.get_center_heights(root_states=root_states, env_ids=env_ids)
                center_heights = center_heights.mean(dim=-1, keepdim=True)

                if self.velocity_map:
                    measured_heights = self.measured_heights.view(num_envs, -1, 3)
                    measured_heights[..., 0] = center_heights - measured_heights[..., 0]
                    heights = measured_heights.view(num_envs, -1)
                else:
                    heights = center_heights - self.measured_heights
                heights = torch.clip(heights, -3, 3.) * self.height_meas_scale  #

            else:
                heights = torch.clip(root_states[:, 2:3] - self.measured_heights, -3, 3.) * self.height_meas_scale  #

            obs = torch.cat([obs, heights], dim=1)

        if self._divide_group and self._group_obs:
            group_obs = compute_group_observation(self._rigid_body_pos, self._rigid_body_rot, self._rigid_body_vel, self.selected_group_jts, self._group_num_people, self._has_upright_start)
            # Group obs has to be computed as a whole. otherwise, the grouping breaks.
            if not (env_ids is None):
                group_obs = group_obs[env_ids]

            obs = torch.cat([obs, group_obs], dim=1)

        return obs


    def _compute_flip_task_obs(self, normal_task_obs, env_ids):

        # location_obs  20
        # Terrain obs: self.num_terrain_obs
        # group obs
        B, D = normal_task_obs.shape
        traj_samples_dim = 20
        obs_acc = []
        normal_task_obs = normal_task_obs.clone()
        traj_samples = normal_task_obs[:, :traj_samples_dim].view(B, 10, 2)
        traj_samples[..., 1] *= -1
        obs_acc.append(traj_samples.view(B, -1))
        if self.terrain_obs:
            if self.velocity_map:
                height_samples = normal_task_obs[..., traj_samples_dim: traj_samples_dim + self.num_height_points * 3]
                height_samples = height_samples.view(B, int(np.sqrt(self.num_height_points)), int(np.sqrt(self.num_height_points)), 3)
                height_samples[..., 0].flip(2)
                height_samples = height_samples.flip(2)
                obs_acc.append(height_samples.view(B, -1))
            else:
                height_samples = normal_task_obs[..., traj_samples_dim: traj_samples_dim + self.num_height_points].view(B, int(np.sqrt(self.num_height_points)), int(np.sqrt(self.num_height_points)))
                height_samples = height_samples.flip(2)
                obs_acc.append(height_samples.view(B, -1))

        obs = torch.cat(obs_acc, dim=1)

        if self._divide_group and self._group_obs:
            group_obs = normal_task_obs[..., traj_samples_dim + self.num_height_points: ].view(B, -1, 3)
            group_obs[..., 1] *= -1
            obs_acc.append(group_obs.view(B, -1))


        obs = torch.cat(obs_acc, dim=1)

        del obs_acc

        return obs

    def _reset_task(self, env_ids):
        # super()._reset_task(env_ids) # Commented out to disable traj resetting
        if not flags.server_mode:
            root_pos = self._humanoid_root_states[env_ids, 0:3]
            self._traj_gen.reset(env_ids, root_pos)
        return


    def _sample_ref_state(self, env_ids, vel_min=1, vel_range=0.5):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)
        # if (self._state_init == HumanoidAMP.StateInit.Random or self._state_init == HumanoidAMP.StateInit.Hybrid):
        #     motion_times = self._sample_time(motion_ids)
        # elif (self._state_init == HumanoidAMP.StateInit.Start):
        #     motion_times = torch.zeros(num_envs, device=self.device)
        # else:
        #     assert (
        #         False
        #     ), "Unsupported state initialization strategy: {:s}".format(
        #         str(self._state_init))
        motion_times = self._sample_time(motion_ids)

        if self.humanoid_type in ["smpl", "smplh", "smplx"]:
            curr_gender_betas = self.humanoid_shapes[env_ids]
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, rb_pos, rb_rot, body_vel, body_ang_vel = self._get_fixed_smpl_state_from_motionlib(
                motion_ids, motion_times, curr_gender_betas)
        else:
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel = self._motion_lib.get_motion_state(
                motion_ids, motion_times)
            rb_pos, rb_rot = None, None
            
        key_pos  = rb_pos[:, self._key_body_ids]

        # if flags.random_heading:
        #     random_rot = np.zeros([num_envs, 3])
        #     random_rot[:, 2] = np.pi * (2 * np.random.random([num_envs]) - 1.0)
        #     random_heading_quat = torch.from_numpy(sRot.from_euler("xyz", random_rot).as_quat()).float().to(self.device)
        #     random_heading_quat_repeat = random_heading_quat[:, None].repeat(1, 24, 1)
        #     root_rot = quat_mul(random_heading_quat, root_rot).clone()
        #     rb_pos = quat_apply(random_heading_quat_repeat, rb_pos - root_pos[:, None, :]).clone() + root_pos[:, None, :]
        #     key_pos  = quat_apply(random_heading_quat_repeat[:, :4, :], (key_pos - root_pos[:, None, :])).clone() + root_pos[:, None, :]
        #     rb_rot = quat_mul(random_heading_quat_repeat, rb_rot).clone()
        #     root_ang_vel = quat_apply(random_heading_quat, root_ang_vel).clone()
        #     root_vel = quat_apply(random_heading_quat, root_vel).clone()

        return motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot, body_vel, body_ang_vel

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot, body_vel, body_ang_vel= self._sample_ref_state(env_ids)
        ## Randomrized location setting
        new_root_xy = self.terrain.sample_valid_locations(self.num_envs, env_ids)
        # joblib.dump(self.terrain.sample_valid_locations(100000, torch.arange(100000)).detach().cpu(), "new_root_xy.pkl")
        # import ipdb; ipdb.set_trace()

        if flags.fixed:
            # new_root_xy[:, 0], new_root_xy[:, 1] = 0 , 0
            # new_root_xy[:, 0], new_root_xy[:, 1] = 134.8434 + env_ids , -28.9593
            # new_root_xy[:, 0], new_root_xy[:, 1] = 30 + env_ids * 4, 240
            new_root_xy[:, 0], new_root_xy[:, 1] = 84 + env_ids * 3, 143
            # new_root_xy[:, 0], new_root_xy[:, 1] = 95 + env_ids * 5, 307
            # new_root_xy[:, 0], new_root_xy[:, 1] = 27, 1 + env_ids * 2
            # x_grid, y_grid = torch.meshgrid(torch.arange(64), torch.arange(64))
            # new_root_xy[:, 0], new_root_xy[:, 1] = x_grid.flatten()[env_ids] * 2, y_grid.flatten()[env_ids] * 2
            # if env_ids[0] == 0:
            # new_root_xy[0, 0], new_root_xy[0, 1] = 34 , -81

        if flags.server_mode:
            new_traj = self._traj_gen.input_new_trajs(env_ids)
            new_root_xy[:, 0], new_root_xy[:, 1] = new_traj[:, 0, 0], new_traj[:, 0,  1]



        diff_xy = new_root_xy - root_pos[:, 0:2]
        root_pos[:, 0:2] = new_root_xy

        root_states = torch.cat([root_pos, root_rot], dim=1)

        center_height = self.get_center_heights(root_states, env_ids=env_ids).mean(dim=-1)
        
        if self.big_ankle: # Big ankle needs a bit more room. 
            center_height += 0.05
            
        root_pos[:, 2] += center_height
        key_pos[..., 0:2] += diff_xy[:, None, :]
        key_pos[...,  2] += center_height[:, None]
        rb_pos[..., 0:2] += diff_xy[:, None, :]
        key_pos[..., 2] += center_height[:, None]

        self._set_env_state(env_ids=env_ids,
                            root_pos=root_pos,
                            root_rot=root_rot,
                            dof_pos=dof_pos,
                            root_vel=root_vel,
                            root_ang_vel=root_ang_vel,
                            dof_vel=dof_vel,
                            rigid_body_pos=rb_pos,
                            rigid_body_rot=rb_rot, 
                            rigid_body_vel=body_vel,
                            rigid_body_ang_vel=body_ang_vel,
                            
                            )

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        if flags.follow:
            self.start = True  ## Updating camera when reset

        return

    def init_center_height_points(self):
        # center_height_points
        y =  torch.tensor(np.linspace(-0.2, 0.2, 3),device=self.device,requires_grad=False)
        x =  torch.tensor(np.linspace(-0.1, 0.1, 3),device=self.device,requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_center_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs,
                             self.num_center_height_points,
                             3,
                             device=self.device,
                             requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def init_square_height_points(self):
        # 4mx4m square
        y =  torch.tensor(np.linspace(-self.sensor_extent, self.sensor_extent, self.sensor_res),device=self.device,requires_grad=False)
        x = torch.tensor(np.linspace(-self.sensor_extent, self.sensor_extent,
                                     self.sensor_res),
                         device=self.device,
                         requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs,
                             self.num_height_points,
                             3,
                             device=self.device,
                             requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def init_square_fov_height_points(self):
        y = torch.tensor(np.linspace(-1, 1, 20),device=self.device,requires_grad=False)
        x =  torch.tensor(np.linspace(-0.02, 1.98, 20),device=self.device,requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs,
                             self.num_height_points,
                             3,
                            device=self.device,
                             requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def init_root_points(self):
        y = torch.tensor(np.linspace(-0.5, 0.5, 20),
                         device=self.device,
                         requires_grad=False)
        x = torch.tensor(np.linspace(-0.25, 0.25, 10),
                         device=self.device,
                         requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_root_points = grid_x.numel()
        points = torch.zeros(self.num_envs,
                             self.num_root_points,
                             3,
                             device=self.device,
                             requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def init_fov_height_points(self):
        # 3m x 3m fan shaped area
        rs =  np.exp(np.arange(0.2, 2, 0.1)) - 0.9
        rs = rs/rs.max() * 2

        max_angle = 110
        phi = np.exp(np.linspace(0.1, 1.5, 12)) - 1
        phi = phi/phi.max() * max_angle
        phi = np.concatenate([-phi[::-1],[0], phi]) * np.pi/180
        xs, ys = [], []
        for r in rs:
            xs.append(r * np.cos(phi)); ys.append(r * np.sin(phi))

        xs, ys = np.concatenate(xs), np.concatenate(ys)

        xs, ys = torch.from_numpy(xs).to(self.device), torch.from_numpy(ys).to(
            self.device)

        self.num_height_points = xs.shape[0]
        points = torch.zeros(self.num_envs,
                             self.num_height_points,
                             3,
                             device=self.device,
                             requires_grad=False)
        points[:, :, 0] = xs
        points[:, :, 1] = ys
        return points

    def get_center_heights(self, root_states, env_ids=None):
        base_quat = root_states[:, 3:7]
        if self.cfg["env"]["terrain"]["terrainType"] == 'plane':
            return torch.zeros(self.num_envs,
                               self.num_center_height_points,
                               device=self.device,
                               requires_grad=False)
        elif self.cfg["env"]["terrain"]["terrainType"] == 'none':
            raise NameError("Can't measure height with terrain type 'none'")

        if self.humanoid_type in ["smpl", "smplh", "smplx"] and not self._has_upright_start:
            base_quat = remove_base_rot(base_quat)

        if env_ids is None:
            points = quat_apply_yaw(
                base_quat.repeat(1, self.num_center_height_points,),
                self.center_height_points) + (root_states[:, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(
                base_quat.repeat(1, self.num_center_height_points,),
                self.center_height_points[env_ids]) + (
                    root_states[:, :3]).unsqueeze(1)

        heights = self.terrain.sample_height_points(points.clone(), env_ids=env_ids)
        num_envs = self.num_envs if env_ids is None else len(env_ids)

        return heights.view(num_envs, -1)

    def get_heights(self, root_states, env_ids=None):

        base_quat = root_states[:, 3:7]
        if self.cfg["env"]["terrain"]["terrainType"] == 'plane':
            return torch.zeros(self.num_envs,
                               self.num_height_points,
                               device=self.device,
                               requires_grad=False)
        elif self.cfg["env"]["terrain"]["terrainType"] == 'none':
            raise NameError("Can't measure height with terrain type 'none'")

        if self.humanoid_type in ["smpl", "smplh", "smplx"] and not self._has_upright_start:
            base_quat = remove_base_rot(base_quat)

        heading_rot = torch_utils.calc_heading_quat(base_quat)

        if env_ids is None:
            points = quat_apply(
                heading_rot.repeat(1, self.num_height_points).reshape(-1, 4),
                self.height_points) + (root_states[:, :3]).unsqueeze(1)
        else:
            points = quat_apply(
                heading_rot.repeat(1, self.num_height_points).reshape(-1, 4),
                self.height_points[env_ids]) + (
                    root_states[:, :3]).unsqueeze(1)

        if self.velocity_map:
            root_states_all = self._humanoid_root_states
        else:
            root_states_all = None

        if (self._divide_group or flags.divide_group) and not self._group_obs and not self._disable_group_obs:
            heading_rot_all = torch_utils.calc_heading_quat(self._humanoid_root_states[:, 3:7])
            root_points = quat_apply(
                heading_rot_all.repeat(1, self.num_root_points).reshape(-1, 4),
                self.root_points) + (self._humanoid_root_states[:, :3]).unsqueeze(1)
            # update heights with root points
            heights = self.terrain.sample_height_points(
                points.clone(),
                root_states = root_states_all,
                root_points = root_points,
                env_ids=env_ids,
                num_group_people=self._group_num_people,
                group_ids = self._group_ids)
        else:
            heights = self.terrain.sample_height_points(
                points.clone(),
                root_states=root_states_all,
                root_points=None,
                env_ids=env_ids,
            )
        # heights = self.terrain.sample_height_points(points.clone(), None)
        num_envs = self.num_envs if env_ids is None else len(env_ids)

        return heights.view(num_envs, -1)

    def _create_ground_plane(self):
        if self.real_mesh:
            self.create_mesh_ground()
        else:
            self.create_training_ground()

    def create_mesh_ground(self):
        # plane_params = gymapi.PlaneParams()
        # plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        # plane_params.distance = 7.001
        # plane_params.static_friction = self.plane_static_friction
        # plane_params.dynamic_friction = self.plane_dynamic_friction
        # plane_params.restitution = self.plane_restitution
        # self.gym.add_ground(self.sim, plane_params)

        # scene_name = "parking"
        # scene_name = "parking_with_cars"
        # scene_name = "with_less_car"
        scene_name = "mesh-downtown-san-jose-mapaligned-cropped-bottom-part-global"
        self.mesh_data = joblib.load(f"data/mesh/{scene_name}.pkl")

        mesh_vertices = self.mesh_data["vertices"]
        mesh_triangles = self.mesh_data["faces"].astype(np.uint32)

        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = mesh_vertices.shape[0]
        tm_params.nb_triangles = mesh_triangles.shape[0]
        tm_params.transform.p.x = 0.0
        tm_params.transform.p.y = 0.0
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.plane_static_friction
        tm_params.dynamic_friction = self.plane_dynamic_friction
        tm_params.restitution = self.plane_restitution
        self.gym.add_triangle_mesh(self.sim, mesh_vertices.flatten(order='C'),
                                   mesh_triangles.flatten(order='C'),
                                   tm_params)

        self.terrain = MeshTerrain(self.mesh_data, self.device)
        self.height_samples = torch.tensor(self.terrain.heightsamples).to(
            self.device)
        return

    def create_training_ground(self):
        if self.cfg["env"].get("small_terrain", False):
            self.cfg["env"]["terrain"]['mapLength'] = 8
            self.cfg["env"]["terrain"]['mapWidth'] = 8

        self.terrain = Terrain(self.cfg["env"]["terrain"],
                               num_robots=self.num_envs,
                               device=self.device)

        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = 0
        tm_params.transform.p.y = 0
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        tm_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        tm_params.restitution = self.cfg["env"]["terrain"]["restitution"]
        self.gym.add_triangle_mesh(self.sim,
                                   self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'),
                                   tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

        # plane_params = gymapi.PlaneParams()
        # plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        # plane_params.distance = 0
        # plane_params.static_friction = self.plane_static_friction
        # plane_params.dynamic_friction = self.plane_dynamic_friction
        # plane_params.restitution = self.plane_restitution
        # self.gym.add_ground(self.sim, plane_params)
        # print("using plain ground");print("using plain ground");print("using plain ground");print("using plain ground");print("using plain ground");

    def _compute_reset(self):
        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs,
                               device=self.device,
                               dtype=torch.long)
        tar_pos = self._traj_gen.calc_pos(env_ids, time)
        ### ZL: entry point
        # self._traj_gen.update_sim_pos(self._humanoid_root_states[)

        root_states = self._humanoid_root_states
        center_height = self.get_center_heights(
            root_states, env_ids=None).mean(dim=-1, keepdim=True)

        # import ipdb
        # ipdb.set_trace()
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf, self.progress_buf, self._contact_forces,
            self._contact_body_ids, center_height, self._rigid_body_pos,
            tar_pos, self.max_episode_length, self._fail_dist,
            self._enable_early_termination, self._termination_heights, flags.no_collision_check)
        return

    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]

        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        tar_pos = self._traj_gen.calc_pos(env_ids, time)
        if self.fuzzy_target:
            location_reward = compute_location_reward_fuzzy(root_pos, tar_pos)
        else:
            location_reward = compute_location_reward(root_pos, tar_pos)

        power = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel)).sum(dim = -1)
        # power_reward = -0.00005 * (power ** 2)
        power_reward = -self.power_coefficient * power

        if self.power_reward:
            self.rew_buf[:] = location_reward + power_reward
        else:
            self.rew_buf[:] = location_reward
        self.reward_raw[:] = torch.cat([location_reward[:, None], power_reward[:, None]], dim = -1)

        return


from isaacgym.terrain_utils import *
from phc.utils.draw_utils import *


def poles_terrain(terrain, difficulty=1):
    """
    Generate stairs

    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float): the step_height [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    height = 0
    start_x = 0
    stop_x = terrain.width
    start_y = 0
    stop_y = terrain.length

    img = np.zeros((terrain.width, terrain.length), dtype=int)
    # disk, circle, curve, poly, ellipse
    base_prob = 1 / 2
    # probs = np.array([0.7, 0.7, 0.4, 0.5, 0.5]) * ((1 - base_prob) * difficulty + base_prob)
    probs = np.array([0.9, 0.4, 0.5, 0.5]) * ((1 - base_prob) * difficulty + base_prob)
    low, high = 200, 500
    num_mult = int(stop_x // 80)

    for i in range(len(probs)):
        p = probs[i]
        if i == 0:
            for _ in range(10 * num_mult):
                if np.random.binomial(1, p):
                    img += draw_disk(img_size=terrain.width, max_r = 7) * int(np.random.uniform(low, high))
        # elif i == 1 and np.random.binomial(1, p):
        #     for _ in range(5 * num_mult):
        #         if np.random.binomial(1, p):

        #             img += draw_circle(img_size=terrain.width, max_r=5) * int(
        #                 np.random.uniform(low, high))
        elif i == 1 and np.random.binomial(1, p):
            for _ in range(3 * num_mult):
                if np.random.binomial(1, p):
                    img += draw_curve(img_size=terrain.width) * int(np.random.uniform(low, high))
        elif i == 2 and np.random.binomial(1, p):
            for _ in range(1 * num_mult):
                if np.random.binomial(1, p):
                    img += draw_polygon(img_size=terrain.width, max_sides=5) * int(np.random.uniform(low, high))
        elif i == 3 and np.random.binomial(1, p):
            for _ in range(5 * num_mult):
                if np.random.binomial(1, p):
                    img += draw_ellipse(img_size=terrain.width,
                                        max_size=5) * int(
                                            np.random.uniform(low, high))

    terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = img

    return terrain

class HumanoidPedestrianTerrainZ(HumanoidPedestrianTerrain):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
        self.initialize_z_models()
        return
    
    def step(self, actions):
        self.step_z(actions)
        return
     
    
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        super()._setup_character_props_z()
        
        return



class MeshTerrain:
    def __init__(self, heigthmap_data, device):
        self.border_size = 20
        self.border = 500
        self.sample_extent_x = 300
        self.sample_extent_y = 300
        self.vertical_scale = 1
        self.device = device

        self.heightsamples = torch.from_numpy(heigthmap_data['heigthmap']).to(device)
        self.walkable_map  = torch.from_numpy(heigthmap_data['walkable_map']).to(device)
        self.cam_pos = torch.from_numpy(heigthmap_data['cam_pos'])
        self.x_scale = heigthmap_data['x_scale']
        self.y_scale = heigthmap_data['y_scale']

        self.x_shape, self.y_shape = self.heightsamples.shape
        self.x_c = (self.x_shape / 2) / self.x_scale
        self.y_c = (self.y_shape / 2) / self.y_scale

        coord_x, coord_y = torch.where(self.walkable_map == 1) # Image coordinates, need to flip y and x
        coord_x, coord_y = coord_x.float(), coord_y.float()
        self.coord_x_scale = coord_x / self.x_scale  - self.x_c
        self.coord_y_scale = coord_y / self.y_scale  - self.y_c

        self.coord_x_scale += self.cam_pos[0]
        self.coord_y_scale += self.cam_pos[1]

        self.num_samples = self.coord_x_scale.shape[0]


    def sample_valid_locations(self, num_envs, env_ids):
        num_envs = env_ids.shape[0]
        idxes = np.random.randint(0, self.num_samples, size=num_envs)
        valid_locs = torch.stack([self.coord_x_scale[idxes], self.coord_y_scale[idxes]], dim = -1)
        return valid_locs

    def world_points_to_map(self, points):
        points[..., 0] -= self.cam_pos[0] - self.x_c
        points[..., 1] -= self.cam_pos[1] - self.y_c
        points[..., 0] *= self.x_scale
        points[..., 1] *= self.y_scale
        points = (points).long()

        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)

        px = torch.clip(px, 0, self.heightsamples.shape[0] -
                        2)  # image, so sampling 1 is for x
        py = torch.clip(py, 0, self.heightsamples.shape[1] - 2)
        return px, py

    def sample_height_points(self,
                             points,
                             root_states = None,
                             root_points=None,
                             env_ids=None,
                             num_group_people=512,
                             group_ids=None):

        B, N, C = points.shape
        px, py = self.world_points_to_map(points)
        heightsamples = self.heightsamples.clone()
        device = points.device
        if env_ids is None:
            env_ids = torch.arange(B).to(points).long()

        if not root_points is None:
            # Adding human root points to the height field
            max_num_envs, num_root_points, _ = root_points.shape
            root_px, root_py = self.world_points_to_map(root_points)
            num_groups = int(root_points.shape[0] / num_group_people)
            heightsamples_group = heightsamples[None, ].repeat(
                num_groups, 1, 1)


            root_px, root_py = root_px.view(-1, num_group_people * num_root_points), root_py.view(-1, num_group_people * num_root_points)
            px, py = px.view(-1, N), py.view(-1, N)
            heights = torch.zeros(px.shape).to(px.device)

            if not root_states is None:
                linear_vel = root_states[:, 7:10]  # This contains ALL the linear velocities
                root_rot = root_states[:, 3:7]
                heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
                velocity_map = torch.zeros([px.shape[0], px.shape[1],
                                            2]).to(root_states)
                velocity_map_group = torch.zeros(heightsamples_group.shape +
                                                 (3, )).to(points)

            for idx in range(num_groups):
                heightsamples_group[idx][root_px[idx],root_py[idx]] += torch.tensor(1.7 / self.vertical_scale)
                # heightsamples_group[idx][root_px[idx] + 1,root_py[idx] + 1] += torch.tensor(1.7 / self.vertical_scale)
                group_mask_env_ids = group_ids[env_ids] == idx  # agents to select for this group from the current env_ids
                # if sum(group_mask) == 0:
                #     continue
                group_px, group_py = px[group_mask_env_ids].view(-1), py[group_mask_env_ids].view(-1)
                heights1 = heightsamples_group[idx][group_px, group_py]
                heights2 = heightsamples_group[idx][group_px + 1, group_py + 1]

                heights_group = torch.min(heights1, heights2)
                heights[group_mask_env_ids] = heights_group.view(-1, N)

                if not root_states is None:
                    # First update the map with the velocity
                    group_mask_all = group_ids == idx
                    env_ids_in_group = env_ids[group_mask_env_ids]
                    group_linear_vel = linear_vel[group_mask_all]

                    velocity_map_group[
                        idx, root_px[idx],
                        root_py[idx], :] = group_linear_vel.repeat(
                            1, root_points.shape[1]).view(-1, 3) # Make sure that the order is correct.

                    # Then sampling the points
                    vel_group = velocity_map_group[idx][group_px, group_py]
                    vel_group = vel_group.view(-1, N, 3)
                    vel_group -= linear_vel[env_ids_in_group, None]  # this is one-to-one substraction of the agents in the group to mark the static terrain with relative velocity
                    group_heading_rot = heading_rot[env_ids_in_group]

                    group_vel_idv = torch_utils.my_quat_rotate(
                        group_heading_rot.repeat(1, N).view(-1, 4),
                        vel_group.view(-1, 3)
                    )  # Global velocity transform. for ALL of the elements in the group.
                    group_vel_idv = group_vel_idv.view(-1, N, 3)[..., :2]
                    velocity_map[group_mask_env_ids] = group_vel_idv
            # import matplotlib.pyplot as plt; plt.imshow(heights[0].reshape(32, 32).cpu().numpy()); plt.show()
            if root_states is None:
                return heights * self.vertical_scale
            else:
                heights = (heights * self.vertical_scale).view(B, -1, 1)
                return torch.cat([heights, velocity_map], dim=-1)
        else:
            heights1 = heightsamples[px, py]
            heights2 = heightsamples[px + 1, py + 1]
            heights = torch.min(heights1, heights2)


        return heights * self.vertical_scale


class Terrain:
    def __init__(self, cfg, num_robots, device) -> None:

        self.type = cfg["terrainType"]
        self.device = device
        if self.type in ["none", 'plane']:
            return
        self.horizontal_scale = 0.1 # resolution 0.1
        self.vertical_scale = 0.005
        self.border_size = 50
        self.env_length = cfg["mapLength"]
        self.env_width = cfg["mapWidth"]
        self.proportions = [
            np.sum(cfg["terrainProportions"][:i + 1])
            for i in range(len(cfg["terrainProportions"]))
        ]

        self.env_rows = cfg["numLevels"]
        self.env_cols = cfg["numTerrains"]
        self.num_maps = self.env_rows * self.env_cols
        self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length /
                                         self.horizontal_scale)

        self.border = int(self.border_size / self.horizontal_scale)
        self.tot_cols = int(
            self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(
            self.env_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.walkable_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        if cfg["curriculum"]:
            self.curiculum(num_robots,
                           num_terrains=self.env_cols,
                           num_levels=self.env_rows)
        else:
            self.randomized_terrain()
        self.heightsamples = torch.from_numpy(self.height_field_raw).to(self.device) # ZL: raw height field, first dimension is x, second is y
        self.walkable_field = torch.from_numpy(self.walkable_field_raw).to(self.device)
        self.vertices, self.triangles = convert_heightfield_to_trimesh(self.height_field_raw, self.horizontal_scale, self.vertical_scale,cfg["slopeTreshold"])
        self.sample_extent_x = int((self.tot_rows - self.border * 2) * self.horizontal_scale)
        self.sample_extent_y = int((self.tot_cols - self.border * 2) * self.horizontal_scale)

        coord_x, coord_y = torch.where(self.walkable_field == 0)
        coord_x_scale = coord_x * self.horizontal_scale
        coord_y_scale = coord_y * self.horizontal_scale
        walkable_subset = torch.logical_and(
                torch.logical_and(coord_y_scale < coord_y_scale.max() - self.border * self.horizontal_scale, coord_x_scale < coord_x_scale.max() - self.border * self.horizontal_scale),
                torch.logical_and(coord_y_scale > coord_y_scale.min() + self.border * self.horizontal_scale, coord_x_scale > coord_x_scale.min() +  self.border * self.horizontal_scale)
            )
        # import ipdb; ipdb.set_trace()
        # joblib.dump(self.walkable_field_raw, "walkable_field.pkl")

        self.coord_x_scale = coord_x_scale[walkable_subset]
        self.coord_y_scale = coord_y_scale[walkable_subset]
        self.num_samples = self.coord_x_scale.shape[0]


    def sample_valid_locations(self, max_num_envs, env_ids, group_num_people = 16, sample_groups = False):
        if sample_groups:
            num_groups = max_num_envs// group_num_people
            group_centers = torch.stack([torch_rand_float(0., self.sample_extent_x, (num_groups, 1),device=self.device).squeeze(1), torch_rand_float(0., self.sample_extent_y, (num_groups, 1),device=self.device).squeeze(1)], dim = -1)
            group_diffs = torch.stack([torch_rand_float(-8., 8, (num_groups, group_num_people) ,device=self.device), torch_rand_float(8., -8, (num_groups, group_num_people),device=self.device)], dim = -1)
            valid_locs = (group_centers[:, None, ] + group_diffs).reshape(max_num_envs, -1)

            if not env_ids is None:
                valid_locs = valid_locs[env_ids]
        else:
            num_envs = env_ids.shape[0]
            idxes = np.random.randint(0, self.num_samples, size=num_envs)
            valid_locs = torch.stack([self.coord_x_scale[idxes], self.coord_y_scale[idxes]], dim = -1)

        return valid_locs

    def world_points_to_map(self, points):
        points = (points / self.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.heightsamples.shape[0] - 2)
        py = torch.clip(py, 0, self.heightsamples.shape[1] - 2)
        return px, py


    def sample_height_points(self, points, root_states = None, root_points=None, env_ids = None, num_group_people = 512, group_ids = None):
        B, N, C = points.shape
        px, py = self.world_points_to_map(points)
        heightsamples = self.heightsamples.clone()
        if env_ids is None:
            env_ids = torch.arange(B).to(points).long()

        if not root_points is None:
            # Adding human root points to the height field
            max_num_envs, num_root_points, _ = root_points.shape
            root_px, root_py = self.world_points_to_map(root_points)
            num_groups = int(root_points.shape[0]/num_group_people)
            heightsamples_group = heightsamples[None, ].repeat(num_groups, 1, 1)

            root_px, root_py = root_px.view(-1, num_group_people * num_root_points), root_py.view(-1, num_group_people *  num_root_points)
            px, py = px.view(-1, N), py.view(-1, N)
            heights = torch.zeros_like(px)

            if not root_states is None:
                linear_vel = root_states[:, 7:10] # This contains ALL the linear velocities
                root_rot = root_states[:, 3:7]
                heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
                velocity_map = torch.zeros([px.shape[0], px.shape[1], 2]).to(root_states)
                velocity_map_group = torch.zeros(heightsamples_group.shape + (3,)).to(points)

            for idx in range(num_groups):
                heightsamples_group[idx][root_px[idx], root_py[idx]] += torch.tensor(1.7 / self.vertical_scale).short()
                group_mask_env_ids = group_ids[env_ids] == idx # agents to select for this group from the current env_ids
                # if sum(group_mask) == 0:
                #     continue
                group_px, group_py = px[group_mask_env_ids].view(-1), py[group_mask_env_ids].view(-1)
                heights1 = heightsamples_group[idx][group_px, group_py]
                heights2 = heightsamples_group[idx][group_px + 1, group_py + 1]
                heights_group = torch.min(heights1, heights2)
                heights[group_mask_env_ids] = heights_group.view(-1, N).long()

                if not root_states is None:
                    # First update the map with the velocity
                    group_mask_all = group_ids == idx
                    env_ids_in_group = env_ids[group_mask_env_ids]
                    group_linear_vel = linear_vel[group_mask_all]
                    velocity_map_group[idx, root_px[idx], root_py[idx], :] = group_linear_vel.repeat(1, root_points.shape[1]).view(-1, 3)

                    # Sample the points for each agent's px and py
                    vel_group = velocity_map_group[idx][group_px, group_py]
                    vel_group = vel_group.view(-1, N, 3)
                    vel_group -= linear_vel[env_ids_in_group, None]  # for each agent's velocity map, minus it's own velocity to get the relative velocity
                    group_heading_rot = heading_rot[env_ids_in_group]

                    group_vel_idv = torch_utils.my_quat_rotate(
                        group_heading_rot.repeat(1, N).view(-1, 4),
                        vel_group.view(-1, 3)
                    )  # Global velocity transform. for ALL of the elements in the group.
                    group_vel_idv = group_vel_idv.view(-1, N, 3)[..., :2]
                    velocity_map[group_mask_env_ids] = group_vel_idv
            if root_states is None:
                return heights * self.vertical_scale
            else:
                heights = (heights * self.vertical_scale).view(B, -1, 1)
                return torch.cat([heights, velocity_map], dim = -1)

        else:
            heights1 = heightsamples[px, py]
            heights2 = heightsamples[px + 1, py + 1]
            heights = torch.min(heights1, heights2)

            if root_states is None:
                return heights * self.vertical_scale
            else:
                velocity_map = torch.zeros((B, N, 2)).to(points)
                linear_vel = root_states[env_ids, 7:10]
                root_rot = root_states[env_ids, 3:7]
                heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
                linear_vel_ego = torch_utils.my_quat_rotate(heading_rot, linear_vel)
                velocity_map[:] = velocity_map[:] - linear_vel_ego[:, None, :2] # Flip velocity to be in agent's point of view
                heights = (heights * self.vertical_scale).view(B, -1, 1)
                return torch.cat([heights, velocity_map], dim = -1)

    # def sample_height_points(self, points, root_points=None):
    #     # Ugly but correct solution
    #     B, N, _ = points.shape
    #     px, py = self.world_points_to_map(points)

    #     if not root_points is None:
    #         # Adding human root points to the height field
    #         root_px, root_py = self.world_points_to_map(root_points)
    #         root_px, root_py = root_px.view(B, -1), root_py.view(B, -1)
    #         heights_acc = []
    #         for curr_agent in range(B):
    #             px, py = px.view(B, -1), py.view(B, -1)
    #             heightsamples = self.heightsamples.clone()
    #             mask = torch.ones(B).bool()
    #             mask[curr_agent] = False
    #             heightsamples[root_px[mask].flatten(), root_py[mask].flatten()] += torch.tensor(1.7 / self.vertical_scale).short()
    #             heights1 = heightsamples[px[curr_agent], py[curr_agent]]
    #             heights2 = heightsamples[px[curr_agent] + 1, py[curr_agent] + 1]
    #             heights = torch.min(heights1, heights2)
    #             heights_acc.append(heights)
    #         heights = torch.stack(heights_acc, dim=0)

    #     else:
    #         heightsamples = self.heightsamples.clone()
    #         heights1 = heightsamples[px, py]
    #         heights2 = heightsamples[px + 1, py + 1]
    #         heights = torch.min(heights1, heights2)
    #     return heights

    def randomized_terrain(self):
        for k in range(self.num_maps):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

            # Heightfield coordinate system from now on
            start_x = self.border + i * self.length_per_env_pixels
            end_x = self.border + (i + 1) * self.length_per_env_pixels
            start_y = self.border + j * self.width_per_env_pixels
            end_y = self.border + (j + 1) * self.width_per_env_pixels

            terrain = SubTerrain("terrain",
                                 width=self.width_per_env_pixels,
                                 length=self.width_per_env_pixels,
                                 vertical_scale=self.vertical_scale,
                                 horizontal_scale=self.horizontal_scale)
            choice = np.random.uniform(0, 1)
            difficulty = np.random.uniform(0.1, 1)
            slope = difficulty * 0.7
            discrete_obstacles_height = 0.025 + difficulty * 0.15
            stepping_stones_size = 2 - 1.8 * difficulty
            step_height = 0.05 + 0.175 * difficulty
            if choice < self.proportions[0]:
                if choice < 0.05:
                    slope *= -1
                pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            elif choice < self.proportions[1]:
                if choice < 0.15:
                    slope *= -1
                pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                random_uniform_terrain(terrain,
                                       min_height=-0.1,
                                       max_height=0.1,
                                       step=0.025,
                                       downsampled_scale=0.2)
            elif choice < self.proportions[3]:
                if choice < self.proportions[2]:
                    step_height *= -1
                pyramid_stairs_terrain(terrain,
                                       step_width=0.31,
                                       step_height=step_height,
                                       platform_size=3.)
            elif choice < self.proportions[4]:
                discrete_obstacles_terrain(terrain,
                                           discrete_obstacles_height,
                                           1.,
                                           2.,
                                           40,
                                           platform_size=3.)
            elif choice < self.proportions[5]:
                stepping_stones_terrain(terrain,
                                        stone_size=stepping_stones_size,
                                        stone_distance=0.1,
                                        max_height=0.,
                                        platform_size=3.)
            elif choice < self.proportions[6]:
                poles_terrain(terrain=terrain, difficulty=difficulty)
                self.walkable_field_raw[start_x:end_x, start_y:end_y] = (terrain.height_field_raw != 0)

            elif choice < self.proportions[7]:
                # plain walking terrain
                pass

            self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length / 2. - 1) / self.horizontal_scale)
            x2 = int((self.env_length / 2. + 1) / self.horizontal_scale)
            y1 = int((self.env_width / 2. - 1) / self.horizontal_scale)
            y2 = int((self.env_width / 2. + 1) / self.horizontal_scale)
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * self.vertical_scale
            self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        self.walkable_field_raw = ndimage.binary_dilation(self.walkable_field_raw, iterations=3).astype(int)

    def curiculum(self, num_robots, num_terrains, num_levels):
        num_robots_per_map = int(num_robots / num_terrains)
        left_over = num_robots % num_terrains
        idx = 0
        for j in tqdm(range(num_terrains)):
            for i in range(num_levels):
                terrain = SubTerrain("terrain",
                                     width=self.width_per_env_pixels,
                                     length=self.width_per_env_pixels,
                                     vertical_scale=self.vertical_scale,
                                     horizontal_scale=self.horizontal_scale)
                difficulty = i / num_levels
                choice = j / num_terrains

                slope = difficulty * 0.7
                step_height = 0.05 + 0.175 * difficulty
                discrete_obstacles_height = 0.025 + difficulty * 0.15
                stepping_stones_size = 2 - 1.8 * difficulty

                start_x = self.border + i * self.length_per_env_pixels
                end_x = self.border + (i + 1) * self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                end_y = self.border + (j + 1) * self.width_per_env_pixels

                if choice < self.proportions[0]:
                    if choice < 0.05:
                        slope *= -1
                    pyramid_sloped_terrain(terrain,
                                           slope=slope,
                                           platform_size=3.)
                elif choice < self.proportions[1]:
                    if choice < 0.15:
                        slope *= -1
                    pyramid_sloped_terrain(terrain,
                                           slope=slope,
                                           platform_size=3.)
                    random_uniform_terrain(terrain,
                                           min_height=-0.1,
                                           max_height=0.1,
                                           step=0.025,
                                           downsampled_scale=0.2)
                elif choice < self.proportions[3]:
                    if choice < self.proportions[2]:
                        step_height *= -1
                    pyramid_stairs_terrain(terrain,
                                           step_width=0.31,
                                           step_height=step_height,
                                           platform_size=3.)
                elif choice < self.proportions[4]:
                    discrete_obstacles_terrain(terrain,
                                               discrete_obstacles_height,
                                               1.,
                                               2.,
                                               40,
                                               platform_size=3.)
                elif choice < self.proportions[5]:
                    stepping_stones_terrain(terrain,
                                            stone_size=stepping_stones_size,
                                            stone_distance=0.1,
                                            max_height=0.,
                                            platform_size=3.)
                elif choice < self.proportions[6]:
                    poles_terrain(terrain=terrain, difficulty=difficulty)
                    self.walkable_field_raw[start_x:end_x, start_y:end_y] = (terrain.height_field_raw != 0)
                    # self.walkable_field_raw[start_x:end_x, start_y:end_y] = 1
                elif choice < self.proportions[7]:
                    # plain walking terrain
                    pass

                # Heightfield coordinate system
                self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

                robots_in_map = num_robots_per_map
                if j < left_over:
                    robots_in_map += 1

                env_origin_x = (i + 0.5) * self.env_length
                env_origin_y = (j + 0.5) * self.env_width
                x1 = int((self.env_length / 2. - 1) / self.horizontal_scale)
                x2 = int((self.env_length / 2. + 1) / self.horizontal_scale)
                y1 = int((self.env_width / 2. - 1) / self.horizontal_scale)
                y2 = int((self.env_width / 2. + 1) / self.horizontal_scale)
                env_origin_z = np.max(
                    terrain.height_field_raw[x1:x2,
                                             y1:y2]) * self.vertical_scale
                self.env_origins[i, j] = [
                    env_origin_x, env_origin_y, env_origin_z
                ]

        self.walkable_field_raw = ndimage.binary_dilation(self.walkable_field_raw, iterations=3).astype(int)




@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf,
                           contact_body_ids, center_height, rigid_body_pos,
                           tar_pos, max_episode_length, fail_dist,
                           enable_early_termination, termination_heights, disableCollision):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, bool, Tensor, bool) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        ## torch.sum to disable self-collision.
        # force_threshold = 200
        force_threshold = 50
        body_contact_force = torch.sqrt(torch.square(torch.abs(masked_contact_buf.sum(dim=-2))).sum(dim=-1)) > force_threshold

        # has_fallen = torch.logical_and(body_contact_force, fall_height)
        has_fallen = body_contact_force
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)

        root_pos = rigid_body_pos[..., 0, :]
        tar_delta = tar_pos[..., 0:2] - root_pos[...,0:2]  # also reset if toooo far away from the target trajectory
        tar_dist_sq = torch.sum(tar_delta * tar_delta, dim=-1)
        tar_fail = tar_dist_sq > fail_dist * fail_dist

        has_failed = torch.logical_or(has_fallen, tar_fail)
        # if has_fallen.any():
        #     import ipdb
        #     ipdb.set_trace()

        if disableCollision:
            has_failed[:] = False

        ############################## Debug ##############################
        # if torch.sum(has_fallen) > 0:
        #     import ipdb; ipdb.set_trace()
        #     print("???")
        # mujoco_joint_names = np.array(['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand'])
        # print( mujoco_joint_names[masked_contact_buf[0, :, 0].nonzero().cpu().numpy()])
        ############################## Debug ##############################


        # has_failed[:] = False

        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)


        # if torch.sum(terminated) > 0:
        #     termianted_progress = progress_buf[torch.where(terminated)]
        #     print(torch.where(termianted_progress < 30), termianted_progress[termianted_progress < 30])

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated

# @torch.jit.script
# def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, center_height, rigid_body_pos,
#                            tar_pos, max_episode_length, fail_dist,
#                            enable_early_termination, termination_heights):
#     # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, bool, Tensor) -> Tuple[Tensor, Tensor]
#     # print("using plain reset")
#     terminated = torch.zeros_like(reset_buf)

#     if (enable_early_termination):
#         masked_contact_buf = contact_buf.clone()
#         masked_contact_buf[:, contact_body_ids, :] = 0
#         fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
#         fall_contact = torch.any(fall_contact, dim=-1)

#         body_height = rigid_body_pos[..., 2]
#         fall_height = body_height < termination_heights
#         fall_height[:, contact_body_ids] = False
#         fall_height = torch.any(fall_height, dim=-1)

#         has_fallen = torch.logical_and(fall_contact, fall_height)
#         # first timestep can sometimes still have nonzero contact forces
#         # so only check after first couple of steps
#         has_fallen *= (progress_buf > 1)

#         root_pos = rigid_body_pos[..., 0, :]
#         tar_delta = tar_pos[..., 0:2] - root_pos[..., 0:2]
#         tar_dist_sq = torch.sum(tar_delta * tar_delta, dim=-1)
#         tar_fail = tar_dist_sq > fail_dist * fail_dist

#         has_failed = torch.logical_or(has_fallen, tar_fail)

#         terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

#     reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

#     return reset, terminated


@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


# Task-location
@torch.jit.script
def compute_location_observations(root_states, traj_samples, upright):
    # type: (Tensor, Tensor, bool) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    if not upright:
        root_rot = remove_base_rot(root_rot)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    heading_rot_exp = torch.broadcast_to(
        heading_rot.unsqueeze(-2),
        (heading_rot.shape[0], traj_samples.shape[1], heading_rot.shape[1]))
    heading_rot_exp = torch.reshape(
        heading_rot_exp, (heading_rot_exp.shape[0] * heading_rot_exp.shape[1],
                          heading_rot_exp.shape[2]))
    traj_samples_delta = traj_samples - root_pos.unsqueeze(-2)
    traj_samples_delta_flat = torch.reshape(
        traj_samples_delta,
        (traj_samples_delta.shape[0] * traj_samples_delta.shape[1],
         traj_samples_delta.shape[2]))

    local_traj_pos = torch_utils.my_quat_rotate(heading_rot_exp,
                                                traj_samples_delta_flat)
    local_traj_pos = local_traj_pos[..., 0:2]

    obs = torch.reshape(local_traj_pos,
                        (traj_samples.shape[0],
                         traj_samples.shape[1] * local_traj_pos.shape[1]))
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
def compute_location_reward_fuzzy(root_pos, tar_pos):
    # type: (Tensor, Tensor) -> Tensor
    pos_err_scale = 2.0
    radius = 0.0025
    pos_diff = tar_pos[..., 0:2] - root_pos[..., 0:2]
    
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_err[pos_err < radius] = 0 # 5cm radius around target is perfect.

    pos_reward = torch.exp(-pos_err_scale * pos_err)

    reward = pos_reward

    return reward


# @torch.jit.script
# def compute_group_observation(body_pos, body_rot, body_vel, selected_jts, num_group_people, upright):
#     # type: (Tensor, Tensor, Tensor, Tensor, int, bool) -> Tensor
#     root_pos = body_pos[:, 0, :]
#     root_rot = body_rot[:, 0, :]

#     root_h = root_pos[:, 2:3]
#     if not upright:
#         root_rot = remove_base_rot(root_rot)
#     heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

#     top_k = 5
#     num_selected_jts = len(selected_jts)

#     B, J, _ = body_pos.shape

#     repeated_root_pos = root_pos.repeat(B, 1).view(B, B, -1)

#     dist = torch.norm(root_pos[..., None, :] - repeated_root_pos, dim = -1)

#     topk_dist, topk_idx = torch.topk(dist, top_k + 1, dim = -1, largest = False)
#     topk_dist, topk_idx = topk_dist[..., 1:], topk_idx[..., 1:] ## ZL should cap distance
#     topk_mask = (topk_dist > 10).view(-1)

#     selected_idxes = topk_idx.flatten()
#     selected_pos = body_pos[selected_idxes][:, selected_jts].view(B, -1, 3)
#     selected_vel = body_vel[selected_idxes][:, [0]].view(B, -1, 3)

#     heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, selected_pos.shape[1], 1))
#     flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], heading_rot_expand.shape[2])

#     root_pos_expand = root_pos.unsqueeze(-2)
#     local_body_pos = selected_pos - root_pos_expand
#     flat_local_body_pos = local_body_pos.view(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
#     flat_local_body_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_local_body_pos)

#     flat_body_vel = selected_vel.view(selected_vel.shape[0] * selected_vel.shape[1], selected_vel.shape[2])
#     heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, selected_vel.shape[1], 1))
#     flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], heading_rot_expand.shape[2])
#     flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_rot, flat_body_vel)

#     local_body_pos = flat_local_body_pos.view(-1, num_selected_jts, 3)
#     local_body_vel = flat_local_body_vel.view(-1, 1, 3)

#     local_body_pos[topk_mask], local_body_vel[topk_mask] = 0, 0
#     local_body_pos, local_body_vel = local_body_pos.view(B, -1), local_body_vel.view(B, -1)
#     group_obs = torch.cat((local_body_pos, local_body_vel), dim = -1)
#     return group_obs



@torch.jit.script
def compute_group_observation(body_pos, body_rot, body_vel, selected_jts, num_group_people, upright):
    # type: (Tensor, Tensor, Tensor, Tensor, int, bool) -> Tensor
    # joints + root velocities
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    if not upright:
        root_rot = remove_base_rot(root_rot)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    top_k = 5
    selected_jts = [0, 1, 5, 9, 3, 7, 16, 21, 18, 23]
    num_selected_jts = len(selected_jts)

    B, J, _ = body_pos.shape
    group_pos = body_pos.view(-1, num_group_people, J, 3)
    group_vel = body_vel.view(-1, num_group_people, J, 3)
    group_root_pos = group_pos[..., 0, :]
    repeated_root_pos = group_root_pos.repeat(1, num_group_people, 1).view(-1, num_group_people, num_group_people, 3)

    indexes = torch.arange(B).to(body_pos.device)
    dist = torch.norm(group_root_pos[..., None, :] - repeated_root_pos, dim = -1)
    topk_dist, topk_idx = torch.topk(dist, top_k + 1, dim = -1, largest = False)
    topk_dist, topk_idx = topk_dist[..., 1:], topk_idx[..., 1:]
    topk_mask = (topk_dist > 10).view(-1)

    repeated_indexes = indexes.view(-1, num_group_people).repeat(1, num_group_people).view(-1, num_group_people, num_group_people)
    selected_idxes = torch.gather(repeated_indexes, -1, topk_idx).flatten()

    selected_pos = body_pos[selected_idxes][:, selected_jts].view(B, -1, 3)
    selected_vel = body_vel[selected_idxes][:, [0]].view(B, -1, 3)

    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, selected_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], heading_rot_expand.shape[2])

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = selected_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.view(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = torch_utils.my_quat_rotate(flat_heading_rot, flat_local_body_pos)

    flat_body_vel = selected_vel.view(selected_vel.shape[0] * selected_vel.shape[1], selected_vel.shape[2])
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, selected_vel.shape[1], 1))
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], heading_rot_expand.shape[2])
    flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_rot, flat_body_vel)

    local_body_pos = flat_local_body_pos.view(-1, num_selected_jts, 3)
    local_body_vel = flat_local_body_vel.view(-1, 1, 3)

    local_body_pos[topk_mask], local_body_vel[topk_mask] = 0, 0
    group_obs = torch.cat([local_body_pos.view(B, -1, top_k, 3), local_body_vel.view(B, -1, top_k, 3)], dim = 1).view(B, -1)

    return group_obs
