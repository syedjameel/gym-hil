#!/usr/bin/env python

"""Panda USB insertion task environment for mijoco simulation"""

import logging
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from gym_hil.mujoco_gym_env import FrankaGymEnv, GymRenderingSpec

_PANDA_HOME = np.asarray((0.0, 0.2639, 0.0, -2.4312, 0.0, 2.6951, 0.7854))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])

# Region where the USB connector can be randomly placed (near gripper home x=0.49, y=0.0)
_USB_SAMPLING_BOUNDS = np.asarray([[0.49, -0.02], [0.49, 0.02]])

TORQUE_THRESHOLD = 30.0  # N-m

class PandaUSBInsertionGymEnv(FrankaGymEnv):
    """Environment for a Panda robot performing USB connector insertion.

    The robot must:
    1. Approach and grasp the USB connector on the table
    2. Align the connector with the USB port
    3. Insert the connector into the port
    """

    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.1,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        reward_type: str = "sparse",
        random_usb_position: bool = True,
    ):
        self.reward_type = reward_type

        xml_path = Path(__file__).parent.parent / "assets" / "panda_usb_insertion_scene.xml"

        super().__init__(
            xml_path=xml_path,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
            render_mode=render_mode,
            image_obs=image_obs,
            home_position=_PANDA_HOME,
            cartesian_bounds=_CARTESIAN_BOUNDS,
        )

        self._random_usb_position = random_usb_position
        self._usb_z_init = 0.015  # USB body center Z (30mm tall body resting on table)

        # Setup observation space
        agent_dim = self.get_robot_state().shape[0]  # 18D for Franka
        agent_box = spaces.Box(-np.inf, np.inf, (agent_dim,), dtype=np.float32)
        env_box = spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32)  # usb_pos(3) + port_entry_pos(3)

        if self.image_obs:
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "front": spaces.Box(
                                0, 255,
                                (self._render_specs.height, self._render_specs.width, 3),
                                dtype=np.uint8,
                            ),
                            "wrist": spaces.Box(
                                0, 255,
                                (self._render_specs.height, self._render_specs.width, 3),
                                dtype=np.uint8,
                            ),
                            "side": spaces.Box(
                                0, 255,
                                (self._render_specs.height, self._render_specs.width, 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                    "agent_pos": agent_box,
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "agent_pos": agent_box,
                    "environment_state": env_box,
                }
            )

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)

        mujoco.mj_resetData(self._model, self._data)
        self.reset_robot()

        # Place USB connector on the table
        usb_jnt_id = self._model.joint("usb_connector").id
        qpos_adr = self._model.jnt_qposadr[usb_jnt_id]

        if self._random_usb_position:
            usb_xy = np.random.uniform(*_USB_SAMPLING_BOUNDS)
        else:
            usb_xy = np.asarray([0.49, 0.0])

        # Set position and identity quaternion
        self._data.qpos[qpos_adr:qpos_adr + 3] = [usb_xy[0], usb_xy[1], self._usb_z_init]
        self._data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]

        mujoco.mj_forward(self._model, self._data)

        obs = self._compute_observation()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        self.apply_action(action)

        obs = self._compute_observation()
        rew = self._compute_reward()
        success = self._is_success()

        # Check if USB connector went out of bounds
        usb_pos = self._data.sensor("usb_connector_pos").data
        out_of_bounds = usb_pos[2] < -0.05 or np.any(np.abs(usb_pos[:2]) > 2.0)

        # Check for excessive torque
        robot_state = self.get_robot_state()
        joint_torques = robot_state[14:21]  # 7 joint torques at indices 14-20
        max_torque = np.max(np.abs(joint_torques))
        torque_exceeded = max_torque > TORQUE_THRESHOLD
        if torque_exceeded:
            logging.warning(f"Joint torque exceeded threshold ({max_torque:.1f} > {TORQUE_THRESHOLD})")

        terminated = bool(success or out_of_bounds or torque_exceeded)

        return obs, rew, terminated, False, {"succeed": success}

    def _compute_observation(self) -> dict:
        """Compute the current observation."""
        robot_state = self.get_robot_state().astype(np.float32)

        usb_pos = self._data.sensor("usb_connector_pos").data.astype(np.float32)
        port_entry_pos = self._data.sensor("usb_port_entry_pos").data.astype(np.float32)

        if self.image_obs:
            front_view, wrist_view, side_view = self.render()
            return {
                "pixels": {"front": front_view, "wrist": wrist_view, "side": side_view},
                "agent_pos": robot_state,
            }
        else:
            return {
                "agent_pos": robot_state,
                "environment_state": np.concatenate([usb_pos, port_entry_pos]),
            }

    def _compute_reward(self) -> float:
        """Compute multi-phase reward for USB insertion task.

        Phases:
        1. Approach: get gripper close to USB connector
        2. Grasp: lift the USB connector from the table
        3. Align: bring the USB plug close to the port entry
        4. Insert: push the plug into the port channel
        """
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        usb_pos = self._data.sensor("usb_connector_pos").data
        usb_plug_pos = self._data.sensor("usb_plug_pos").data
        port_entry_pos = self._data.sensor("usb_port_entry_pos").data
        port_bottom_pos = self._data.sensor("usb_port_bottom_pos").data

        if self.reward_type == "dense":
            # Snap to exactly 1.0 at full insertion so success is unambiguous.
            if self._is_success():
                return 1.0

            # step 1 approach
            dist_to_usb = np.linalg.norm(tcp_pos - usb_pos)
            r_approach = np.exp(-20 * dist_to_usb)

            # step 2 grasp
            usb_lifted = usb_pos[2] > self._usb_z_init + 0.01
            r_grasp = 1.0 if usb_lifted else 0.0

            # Step 3 align (only if grasped) - Y/Z alignment of plug with port
            # entry. Using only Y/Z (not full 3D) so that pushing the plug
            # deeper into the slot does not penalize alignment.
            yz_offset = (usb_plug_pos - port_entry_pos)[1:]
            r_align = np.exp(-20 * np.linalg.norm(yz_offset)) if usb_lifted else 0.0

            # step 4 insert (direction-agnostic: project plug offset onto port axis)
            port_axis = port_bottom_pos - port_entry_pos
            port_axis_norm = np.linalg.norm(port_axis)
            if port_axis_norm > 1e-6:
                port_dir = port_axis / port_axis_norm
                plug_offset = usb_plug_pos - port_entry_pos
                insertion_depth = max(0.0, float(np.dot(plug_offset, port_dir)))
                max_depth = float(port_axis_norm)
            else:
                insertion_depth = 0.0
                max_depth = 1e-6
            r_insert = np.clip(insertion_depth / max_depth, 0.0, 1.0) if usb_lifted else 0.0

            return float(0.2 * r_approach + 0.2 * r_grasp + 0.3 * r_align + 0.3 * r_insert)
        else:
            return float(self._is_success())

    def _is_success(self) -> bool:
        """Check if USB is fully inserted into the port.

        Uses X-axis insertion depth only (not full 3D distance) so that small
        Y/Z drift inside the slot - which is allowed by the slot tolerances
        and doesn't affect the actual insertion - doesn't block success.

        3 mm X tolerance leaves room for MuJoCo contact compliance (~0.5-1 mm
        of slop at each contact) while still requiring the plug to be
        essentially bottomed against the back wall.
        """
        plug_x = self._data.sensor("usb_plug_pos").data[0]
        bottom_x = self._data.sensor("usb_port_bottom_pos").data[0]
        return abs(plug_x - bottom_x) < 0.003


if __name__ == "__main__":
    from gym_hil import PassiveViewerWrapper

    env = PandaUSBInsertionGymEnv(render_mode="human")
    env = PassiveViewerWrapper(env)
    env.reset()
    for _ in range(200):
        env.step(np.random.uniform(-1, 1, 7))
    env.close()
