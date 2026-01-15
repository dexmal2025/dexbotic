"""
Libero environment wrapper implementation
"""

import gc
import random
import threading
from multiprocessing import Process, Queue
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..base import BaseEnvWrapper

_LIBERO_ENV_INIT_LOCK = threading.Lock()
# Use fork method with proper CUDA handling


# Note: Using spawn method to avoid CUDA context conflicts in multiprocessing


class LiberoEnvWrapper(BaseEnvWrapper):
    """
    Libero environment wrapper using multiprocessing.
    Based on the multiprocessing approach in rob_rollout.py
    """

    def __init__(
        self,
        task_name: str,
        task_id: int,
        trial_id: int,
        trial_seed: int,
        config: Any,
        cuda_device: int = None,
    ):
        super().__init__(task_name, trial_id, trial_seed, config)
        self.task_id = task_id
        self.max_steps = getattr(config, "max_episode_steps", 512)
        self.cuda_device = cuda_device  # Store CUDA device for worker process

        # Multiprocessing components
        self.process = None
        self.input_queue = None
        self.output_queue = None
        self.is_valid = True  # For video collection
        self.global_steps = 0

        # State tracking
        self.task_description = None
        self.valid_images = []
        self.cuda_device = cuda_device

        # Initialize init_data to None to avoid AttributeError
        self.init_data = None

    def _internal_reset(
        self,
        env: any,
        config: Any,
        is_valid: bool,
        initial_states,
        get_libero_dummy_action,
    ):
        """
        Internal reset without sending message to queue.
        Used when episode ends and we want to keep the process running.
        """
        env.reset()
        num_trial = len(initial_states)
        initial_state = initial_states[random.randint(0, num_trial - 1)]
        obs = env.set_init_state(initial_state)

        # Wait for environment to stabilize
        t = 0
        while t < getattr(config, "num_steps_wait", 10):
            obs, _, _, _ = env.step(get_libero_dummy_action(config.model_family))
            t += 1

        img = None
        if is_valid:
            img = obs["agentview_image"][::-1, ::-1]

        return env, obs, img

    def _send_init_data(
        self,
        output_queue: Queue,
        obs,
        task_description: str,
        task_name: str,
        task_id: int,
        trial_id: int,
        is_valid: bool,
        img,
    ):
        """
        Send initialization data to main process.
        Called when main process requests current state via None signal.
        """
        valid_images = [img] if is_valid and img is not None else []
        output_queue.put(
            {
                "type": "init",
                "obs": obs,
                "task_description": task_description,
                "valid_images": valid_images,
                "task_file_name": f"{task_name}_task_{task_id}_trial_{trial_id}",
                "active": True,
                "complete": False,
                "finish_step": 0,
            }
        )

    def libero_env_worker(
        self,
        task_name: str,
        task_id: int,
        trial_id: int,
        config: Any,
        input_queue: Queue,
        output_queue: Queue,
        is_valid: bool,
        global_steps: int,
        max_steps: int,
        cuda_device: int = None,
    ):
        """
        Worker process for Libero environments.
        Process stays alive and resets environment when episode ends.

        Protocol:
        - Receive action array -> execute and return 'step' result
        - Receive None -> send current 'init' state (for new rollout)
        - Environment auto-resets when done, waits for None to send init
        """
        try:
            # Set CUDA device for this worker process if specified
            if cuda_device is not None:
                import os

                # Set environment variable for CUDA device
                os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
                print(f"Worker process set CUDA_VISIBLE_DEVICES to {cuda_device}")

                # For fork method, we need to handle CUDA context properly
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.set_device(
                            0
                        )  # Device 0 after setting CUDA_VISIBLE_DEVICES
                        print(
                            f"Worker process set torch CUDA device to 0 (mapped to physical device {cuda_device})"
                        )
                except Exception as e:
                    print(f"Warning: Could not set CUDA device in worker: {e}")

            from libero.libero import benchmark

            # Import libero utility functions from local module
            from .libero_utils import (  # get_libero_image,; get_libero_wrist_image,; quat2axisangle,
                get_libero_dummy_action,
                get_libero_env,
                invert_gripper_action,
                normalize_gripper_action,
            )
        except ImportError as e:
            print(f"Warning: can't import libero dependencies: {e}")
            output_queue.put({"type": "error", "message": f"Import error: {e}"})
            return

        try:
            # Initialize Libero environment
            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict[task_name]()
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            # initial_state = initial_states[trial_id]

            env = None  # Initialize env to None to avoid NameError
            while True:
                try:
                    env, task_description = get_libero_env(task, resolution=256)
                    break
                except Exception:
                    print("*** env initialization failed ***")
                    if env is not None:
                        try:
                            env.close()
                        except Exception as e:
                            print(f"error when close the env: {e}")
                    torch.cuda.empty_cache()
                    gc.collect()
                    print("gc collect finish")

            # Initial reset
            env, obs, img = self._internal_reset(
                env, config, is_valid, initial_states, get_libero_dummy_action
            )

            # Send first init data
            self._send_init_data(
                output_queue,
                obs,
                task_description,
                task_name,
                task_id,
                trial_id,
                is_valid,
                img,
            )

            # State tracking
            active = True
            complete = False
            finish_step = 0

            # Main execution loop - process stays alive
            while True:
                action = input_queue.get()

                if action is None:
                    # Main process requests current init state (for new rollout)
                    # print("\n-------------------sending_init_state-------------------\n")
                    self._send_init_data(
                        output_queue,
                        obs,
                        task_description,
                        task_name,
                        task_id,
                        trial_id,
                        is_valid,
                        img,
                    )
                    # Reset state for new episode
                    active = True
                    complete = False
                    finish_step = 0
                    # needs_init_response = False
                    continue

                # Execute action sequence
                step_images = []
                for i in range(len(action)):
                    a = action[i]
                    normalized_action = normalize_gripper_action(a, binarize=True)
                    inverted_action = invert_gripper_action(normalized_action)
                    obs, reward, done, info = env.step(inverted_action.tolist())

                    if is_valid:
                        img = obs["agentview_image"][::-1, ::-1]
                        step_images.append(img)

                    finish_step += 1
                    if done or finish_step >= max_steps:
                        active = False
                        complete = done
                        break

                # Send step result
                output_data = {
                    "type": "step",
                    "obs": obs,
                    "active": active,
                    "complete": complete,
                    "finish_step": finish_step,
                    "valid_images": step_images.copy() if is_valid else [],
                }
                output_queue.put(output_data)

                # If episode ended, reset environment internally (no message sent)
                if done or finish_step >= max_steps:
                    # print("\n=====================env_internal_reset=====================\n")
                    env, obs, img = self._internal_reset(
                        env, config, is_valid, initial_states, get_libero_dummy_action
                    )
                    # Reset counters but don't send init - wait for None signal
                    active = True
                    complete = False
                    finish_step = 0
                    # needs_init_response = True

        except Exception as e:
            print(f"Libero worker error: {e}")
            import traceback

            traceback.print_exc()
            output_queue.put({"type": "error", "message": str(e)})

    def initialize(self) -> None:
        """Initialize Libero environment in separate process."""
        # Note: Using spawn method to avoid CUDA context conflicts
        with _LIBERO_ENV_INIT_LOCK:
            # with self.lock:
            try:
                # Create queues for communication
                self.input_queue = Queue()
                self.output_queue = Queue()

                # Start worker process
                self.process = Process(
                    target=self.libero_env_worker,
                    args=(
                        self.task_name,
                        self.task_id,
                        self.trial_id,
                        self.config,
                        self.input_queue,
                        self.output_queue,
                        self.is_valid,
                        self.global_steps,
                        self.max_steps,
                        self.cuda_device,
                    ),
                )
                self.process.start()

                # Wait for initialization
                init_data = self.output_queue.get(timeout=120)

                if init_data["type"] == "error":
                    raise RuntimeError(
                        f"Libero initialization failed: {init_data['message']}"
                    )

                assert init_data["type"] == "init"

                # Store initialization data
                self.init_data = init_data
                self.task_description = init_data["task_description"]
                self.instruction = self.task_description
                self.active = init_data["active"]
                self.complete = init_data["complete"]
                self.finish_step = init_data["finish_step"]
                self.valid_images.extend(init_data["valid_images"])

                # Store current observation
                self._current_obs = init_data["obs"]

            except Exception as e:
                print(f"Libero environment initialization failed: {e}")
                self._cleanup()
                raise

    def get_obs(self) -> Dict[str, Any]:
        """Get current observation."""
        with self.lock:
            if not hasattr(self, "_current_obs") or self._current_obs is None:
                raise RuntimeError(
                    "Environment not initialized or no observation available"
                )
            return self._current_obs

    def get_instruction(self) -> str:
        """Get task instruction."""
        return self.instruction or f"Libero task: {self.task_name}"

    def step(self, action: np.ndarray) -> Tuple[Optional[Dict[str, Any]], bool]:
        """Execute action in Libero environment."""
        with self.lock:
            try:
                if not self.process or not self.process.is_alive():
                    print("Process not alive, marking as inactive")
                    self.active = False
                    return None, True

                # Send action to worker process
                self.input_queue.put(action)

                # Get result
                result = self.output_queue.get(timeout=30)

                if result["type"] == "error":
                    print(f"Libero step error: {result['message']}")
                    self.active = False
                    return None, True

                assert result["type"] == "step"

                # Update state
                self._current_obs = result["obs"]
                self.active = result["active"]
                self.complete = result["complete"]
                self.finish_step = result["finish_step"]
                self.valid_images.extend(result["valid_images"])

                done = not self.active
                obs = self._current_obs if not done else None

                return obs, done

            except Exception as e:
                print(f"Libero step execution failed: {e}")
                self.active = False
                self.complete = False
                return None, True

    def close(self) -> None:
        """Close Libero environment."""
        with self.lock:
            self._cleanup()

    def _cleanup(self):
        """Internal cleanup method."""
        try:
            if self.input_queue:
                self.input_queue.put(None)  # Send termination signal

            if self.process:
                self.process.join(timeout=20)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=5)
                    if self.process.is_alive():
                        self.process.kill()
        except Exception as e:
            print(f"Error during Libero cleanup: {e}")
        finally:
            self.process = None
            self.input_queue = None
            self.output_queue = None
            self.active = False

    def get_valid_images(self):
        """Get collected valid images for video generation."""
        return self.valid_images.copy()

    def get_task_file_name(self):
        """Get task file name for video saving."""
        return f"{self.task_name}_task_{self.task_id}_trial_{self.trial_id}"

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
