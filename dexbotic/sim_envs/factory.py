"""
Environment factory for batch instantiation during RL training
"""

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Type

from .base import BaseEnvWrapper, MockEnvWrapper
from .libero.libero_env import LiberoEnvWrapper


def create_env_batch(
    env_type: str,
    task_suite_name: str,
    task_ids: List[int],
    trial_ids: List[int],
    trial_seeds: List[int],
    config: Any,
    use_threading: bool = True,
    max_workers: Optional[int] = None,
    cuda_device: Optional[int] = None,
) -> List[BaseEnvWrapper]:
    """
    Create a batch of environment instances for RL training.

    This function handles the batch instantiation of environments needed during RL training,
    where the batch_size determines how many environments need to be created.

    Args:
        env_type: Type of environment ('libero', 'mock')
        task_suite_name: Name of the task suite (e.g., 'libero_10')
        task_ids: List of task IDs for each environment instance
        trial_ids: List of trial IDs for each environment instance
        trial_seeds: List of random seeds for each environment instance
        config: Configuration object containing environment parameters
        use_threading: Whether to use threading for parallel initialization
        max_workers: Maximum number of worker threads (defaults to 8)

    Returns:
        List of initialized environment wrappers

    Example:
        # For RL training with batch_size=64
        env_batch = create_env_batch(
            env_type='libero',
            task_suite_name='libero_10',
            task_ids=[0] * 64,  # All same task
            trial_ids=list(range(64)),  # Different trials
            trial_seeds=[42 + i for i in range(64)],  # Different seeds
            config=config
        )
    """
    batch_size = len(task_ids)
    assert (
        len(trial_ids) == batch_size
    ), f"trial_ids length {len(trial_ids)} != batch_size {batch_size}"
    assert (
        len(trial_seeds) == batch_size
    ), f"trial_seeds length {len(trial_seeds)} != batch_size {batch_size}"

    # Determine environment wrapper class
    env_class = _get_env_class(env_type, task_suite_name)  # LiberoEnvWrapper

    # Create environment wrappers

    env_wrappers = []
    for idx in range(batch_size):
        task_name = _extract_task_name(task_suite_name, env_type)

        if env_type == "libero":
            wrapper = env_class(
                task_name=task_suite_name,
                task_id=task_ids[idx],
                trial_id=trial_ids[idx],
                trial_seed=trial_seeds[idx],
                config=config,
                cuda_device=cuda_device,
            )
        else:  # mock
            wrapper = env_class(
                task_name=task_name,
                trial_id=trial_ids[idx],
                trial_seed=trial_seeds[idx],
                config=config,
            )

        env_wrappers.append(wrapper)

    # Initialize environments
    if use_threading and batch_size > 1:
        if max_workers is None:
            max_workers = 8

        env_wrappers = _initialize_environments_threaded(env_wrappers, max_workers)
    else:
        env_wrappers = _initialize_environments_sequential(env_wrappers)

    return env_wrappers


def _get_env_class(env_type: str, task_suite_name: str) -> Type[BaseEnvWrapper]:
    """Get the appropriate environment class based on type and task suite."""
    if env_type == "libero" or "libero" in task_suite_name:
        return LiberoEnvWrapper
    elif env_type == "mock":
        return MockEnvWrapper
    else:
        print(f"Unknown environment type '{env_type}', using MockEnvWrapper")
        return MockEnvWrapper


def _extract_task_name(task_suite_name: str, env_type: str) -> str:
    """Extract task name from task suite name."""
    return task_suite_name


def _initialize_environments_threaded(
    env_wrappers: List[BaseEnvWrapper], max_workers: int
) -> List[BaseEnvWrapper]:
    """Initialize environments using thread pool for parallel initialization."""
    print(
        f"Initializing {len(env_wrappers)} environments with {max_workers} threads..."
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit initialization tasks
        init_futures = []
        for wrapper in env_wrappers:
            future = executor.submit(wrapper.initialize)
            init_futures.append(future)

        # Wait for completion with error handling
        failed_indices = []
        for idx, future in enumerate(as_completed(init_futures, timeout=360)):
            try:
                future.result()
                print(f"Environment {idx} initialized successfully")
            except Exception as e:
                print(f"Environment {idx} initialization failed: {e}", flush=True)
                traceback.print_exc()
                failed_indices.append(idx)

        if failed_indices:
            print(f"Failed to initialize {len(failed_indices)} environments")
            # Could implement retry logic here if needed

    print("Environment batch initialization completed")
    return env_wrappers


def _initialize_environments_sequential(
    env_wrappers: List[BaseEnvWrapper],
) -> List[BaseEnvWrapper]:
    """Initialize environments sequentially."""
    print(f"Initializing {len(env_wrappers)} environments sequentially...")

    for idx, wrapper in enumerate(env_wrappers):
        try:
            wrapper.initialize()
            print(f"Environment {idx} initialized successfully")
        except Exception as e:
            print(f"Environment {idx} initialization failed: {e}", flush=True)
            traceback.print_exc()

    print("Environment batch initialization completed")
    return env_wrappers


def close_env_batch(
    env_wrappers: List[BaseEnvWrapper], use_threading: bool = True
) -> None:
    """
    Close a batch of environments.

    Args:
        env_wrappers: List of environment wrappers to close
        use_threading: Whether to use threading for parallel cleanup
    """
    print(f"Closing {len(env_wrappers)} environments...")

    if use_threading and len(env_wrappers) > 1:
        with ThreadPoolExecutor(max_workers=16) as executor:
            cleanup_futures = []
            for wrapper in env_wrappers:
                future = executor.submit(wrapper.close)
                cleanup_futures.append(future)

            for future in as_completed(cleanup_futures):
                try:
                    future.result(timeout=20)
                except Exception as e:
                    print(f"Environment cleanup failed: {e}", flush=True)
    else:
        for wrapper in env_wrappers:
            try:
                wrapper.close()
            except Exception as e:
                print(f"Environment cleanup failed: {e}", flush=True)

    print("Environment batch cleanup completed")


class EnvBatchManager:
    """
    Manager class for handling batches of environments during RL training.

    This class provides a convenient interface for managing environment batches
    throughout the RL training process.
    """

    def __init__(self, env_type: str, task_suite_name: str, config: Any, **kwargs):
        self.env_type = env_type
        self.task_suite_name = task_suite_name
        self.config = config
        self.env_wrappers: Optional[List[BaseEnvWrapper]] = None

    def create_batch(
        self,
        batch_env_configs: Dict[str, Any],
        cuda_device: Optional[int] = None,
    ) -> List[BaseEnvWrapper]:
        """
        Create a new batch of environments from dataloader configurations.

        Args:
            batch_env_configs: Dictionary from dataloader containing environment configurations:
                - task_id: tensor/array of task IDs [batch_size]
                - trial_id: tensor/array of trial IDs [batch_size]
                - trial_seed: tensor/array of trial seeds [batch_size]
                - interleaved_batch_size: int, batch size after n_sample interleaving
                - original_batch_size: int, original batch size before n_sample

        Returns:
            List of initialized environment wrappers

        Raises:
            ValueError: If batch_env_configs is None or missing required fields
        """

        if batch_env_configs is None:
            raise ValueError(
                "batch_env_configs cannot be None. Must provide environment configurations from dataloader."
            )

        # Extract batch size
        batch_size = batch_env_configs.get(
            "interleaved_batch_size", batch_env_configs.get("original_batch_size")
        )
        # assert batch_size%10 == 0, print(f"==========Batch size = {batch_size} should be set as the multiple of 10==========")
        if batch_size is None or batch_size == 0:
            raise ValueError(
                "batch_env_configs must contain 'interleaved_batch_size' or 'original_batch_size'"
            )

        # Extract task_ids, trial_ids, trial_seeds
        task_ids_ori = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        task_ids = [x for x in task_ids_ori for _ in range(batch_size // 10)]
        trial_ids = batch_env_configs["trial_id"].numpy().flatten().tolist()
        trial_seeds = batch_env_configs["trial_seed"].numpy().flatten().tolist()

        # Create environment batch
        self.env_wrappers = create_env_batch(
            env_type=self.env_type,
            task_suite_name=self.task_suite_name,
            task_ids=task_ids,
            trial_ids=trial_ids,
            trial_seeds=trial_seeds,
            config=self.config,
            cuda_device=cuda_device,
        )

        return self.env_wrappers

    def close_batch(self) -> None:
        """Close the current batch of environments."""
        if self.env_wrappers:
            close_env_batch(self.env_wrappers)
            self.env_wrappers = None

    def get_active_environments(self) -> List[BaseEnvWrapper]:
        """Get list of currently active environments."""
        if not self.env_wrappers:
            return []
        return [env for env in self.env_wrappers if env.is_active()]

    def get_batch_statistics(self) -> Dict[str, int]:
        """Get statistics about the current batch."""
        if not self.env_wrappers:
            return {"total": 0, "active": 0, "complete": 0, "inactive": 0}

        total = len(self.env_wrappers)
        active = sum(1 for env in self.env_wrappers if env.is_active())
        complete = sum(1 for env in self.env_wrappers if env.is_complete())
        inactive = total - active

        return {
            "total": total,
            "active": active,
            "complete": complete,
            "inactive": inactive,
        }

    def __del__(self):
        """Cleanup on deletion."""
        self.close_batch()
