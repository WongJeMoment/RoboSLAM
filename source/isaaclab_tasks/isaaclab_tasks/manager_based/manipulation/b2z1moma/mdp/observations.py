# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import isaacsim.core.utils.bounds as bounds_utils
import isaacsim.core.utils.semantics as semantics_utils
import isaacsim.core.utils.prims as prims_utils
import time


first_run = True
def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    global first_run, bbox_tensor
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_state_w
    robot_pos_w = robot.data.body_state_w[:, robot_cfg.body_ids[0], :]  # type: ignore

    object_t_b, object_q_b = subtract_frame_transforms(
        robot_pos_w[:, :3], robot_pos_w[:, 3:7], object_pos_w[:, :3], object_pos_w[:, 3:7]
    )
    # size = object.cfg.spawn.size

    # st = time.time()
    if first_run:
        bbox_cache = bounds_utils.create_bbox_cache()
        bbox_list = []
        combined_bbox = np.empty((env.num_envs, 3))
        for i, prim_path in enumerate(object.root_physx_view.prim_paths):
            # 循环计算所有bbox
            centroid, axes, half_extent = bounds_utils.compute_obb(bbox_cache, prim_path=prim_path)
            combined_bbox[i] = half_extent

        bbox_tensor = torch.from_numpy(combined_bbox).cuda()
        first_run = False
    
    # print((time.time() - st)*1000)
    


    # print(bbox_tensor, bbox_tensor.shape, bbox_tensor.device)
    # print(object_t_b, object_t_b.shape, object_t_b.device)

    object_pos_b = torch.cat([object_t_b, bbox_tensor], dim=-1)
    return object_t_b

def robot_pos_err(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 0:2] # type: ignore

    #print(des_pos_w[:,:2] - curr_pos_w)
    return des_pos_w[:,:2] - curr_pos_w


