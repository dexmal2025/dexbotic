"""
Base environment wrapper interface
"""

import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class BaseEnvWrapper(ABC):
    """
    Base environment wrapper providing a consistent interface for different simulation environments.
    This design is inspired by rob_rollout.py to support batch RL training.
    """

    def __init__(self, task_name: str, trial_id: int, trial_seed: int, config: Any):
        """
        Initialize environment wrapper.

        Args:
            task_name: Name of the task
            trial_id: Trial identifier
            trial_seed: Random seed for the trial
            config: Configuration object containing environment parameters
        """
        self.task_name = task_name
        self.trial_id = trial_id
        self.trial_seed = trial_seed
        self.config = config

        # Environment state tracking
        self.env = None
        self.active = True
        self.complete = False
        self.finish_step = 0

        # Thread safety - delay initialization for spawn compatibility
        self._lock = None

        # Task instruction/description
        self.instruction = None

    @property
    def lock(self):
        """Lazy initialization of lock for spawn compatibility"""
        if self._lock is None:
            self._lock = threading.Lock()
        return self._lock

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the environment instance.
        This should be called before any other environment operations.
        """
        pass

    @abstractmethod
    def get_obs(self) -> Dict[str, Any]:
        """
        Get current observation from environment.

        Returns:
            Dictionary containing observation data
        """
        pass

    @abstractmethod
    def get_instruction(self) -> str:
        """
        Get task instruction/description.

        Returns:
            String description of the task
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Execute action in environment.

        Args:
            action: Action array to execute

        Returns:
            Tuple of (observation, done_flag)
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close and cleanup the environment.
        """
        pass

    def reset(self) -> Dict[str, Any]:
        """
        Reset environment to initial state.

        Returns:
            Initial observation
        """
        with self.lock:
            self.active = True
            self.complete = False
            self.finish_step = 0
            return self.get_obs()

    def is_active(self) -> bool:
        """Check if environment is still active."""
        return self.active

    def is_complete(self) -> bool:
        """Check if task is completed successfully."""
        return self.complete

    def get_step_count(self) -> int:
        """Get current step count."""
        return self.finish_step


class MockEnvWrapper(BaseEnvWrapper):
    """
    Mock environment wrapper for testing purposes.
    """

    def __init__(self, task_name: str, trial_id: int, trial_seed: int, config: Any):
        super().__init__(task_name, trial_id, trial_seed, config)
        self.max_steps = getattr(config, "max_episode_steps", 100)
        self.obs_dim = getattr(config, "obs_dim", (224, 224, 3))

    def initialize(self) -> None:
        """Initialize mock environment."""
        with self.lock:
            self.instruction = f"Mock task: {self.task_name}"

    def get_obs(self) -> Dict[str, Any]:
        """Get mock observation."""
        with self.lock:
            return {
                "observation": {
                    "head_camera": {
                        "rgb": np.random.randint(0, 255, self.obs_dim, dtype=np.uint8)
                    }
                },
                "joint_action": {"vector": np.random.randn(7).astype(np.float32)},
            }

    def get_instruction(self) -> str:
        """Get mock instruction."""
        return self.instruction or f"Mock task: {self.task_name}"

    def step(self, action: np.ndarray) -> Tuple[Optional[Dict[str, Any]], bool]:
        """Execute mock step."""
        with self.lock:
            try:
                self.finish_step += action.shape[0] if len(action.shape) > 0 else 1

                # Mock completion logic
                done = self.finish_step >= self.max_steps or np.random.random() < 0.01

                if done:
                    self.active = False
                    self.complete = np.random.random() < 0.5  # Random success

                obs = self.get_obs() if not done else None
                return obs, done

            except Exception as e:
                print(f"Mock environment step error: {e}")
                self.active = False
                return None, True

    def close(self) -> None:
        """Close mock environment."""
        with self.lock:
            self.env = None
            self.active = False
