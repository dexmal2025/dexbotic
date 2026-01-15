# Copyright 2026 DexBotic Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import uuid
from typing import Any, Dict, List

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

try:
    from libero.libero import benchmark
except ImportError as e:
    print(f"Warning: can't import libero: {e}")


def collate_fn(data_list: list[dict]) -> dict:
    """Collate function for batching dataset items"""
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


class FakeDataset(Dataset):
    action_process_func = None

    def __len__(self):
        return 0


class DexRLDataset(Dataset):
    """
    RL Dataset for DexBotic that supports LIBERO environments.

    This dataset creates environment configurations for RL training.
    Key design principles:
    - Focus on base environment configurations generation
    - No shuffle logic (handled by dataloader)
    - No n_sample logic (handled by dataloader)
    - Clean separation of concerns
    """

    def __init__(
        self,
        env_type: str,
        task_name: str,
        batch_size: int,
        num_trials_per_task: int = 50,
        train_val: str = "train",
        seed: int = 42,
        **kwargs,
    ):
        """
        Initialize DexRL dataset

        Args:
            env_type: Environment type ("libero")
            task_name: Task name (e.g., "libero_10")
            batch_size: Batch size for environment configurations
            num_trials_per_task: Number of trials per task
            train_val: "train" or "valid"
            seed: Random seed for reproducibility
        """
        self.env_type = env_type
        self.task_name = task_name
        self.batch_size = batch_size
        self.num_trials_per_task = num_trials_per_task
        self.train_val = train_val
        self.seed = seed

        # Note: Random seed not needed at dataset level since we only generate
        # deterministic environment configurations. Randomness is handled by dataloader.

        self._validate_inputs()
        self._setup_environment_configs()
        self._create_dataset()

    def _validate_inputs(self):
        """Validate input parameters"""
        if self.env_type not in ["libero"]:
            raise ValueError(f"Unsupported env_type: {self.env_type}")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive integer")

    def _setup_environment_configs(self):
        """Setup environment configurations based on env_type"""
        if self.env_type == "libero":
            self._setup_libero_config()

    def _setup_libero_config(self):
        """Setup LIBERO environment configuration"""
        try:
            benchmark_dict = benchmark.get_benchmark_dict()
            if self.task_name not in benchmark_dict:
                raise ValueError(f"Unknown LIBERO task: {self.task_name}")

            task_suite = benchmark_dict[self.task_name]()
            self.num_tasks_in_suite = task_suite.n_tasks

            # Valid LIBERO task suites
            self.valid_libero_suites = [
                "libero_10",
                "libero_90",
                "libero_goal",
                "libero_object",
                "libero_spatial",
            ]

            if self.task_name not in self.valid_libero_suites:
                raise ValueError(f"Unsupported LIBERO suite: {self.task_name}")

        except Exception as e:
            print(f"Warning: Failed to setup LIBERO config: {e}")
            # Fallback configuration
            self.num_tasks_in_suite = 10

    def _create_dataset(self):
        """Create dataset with base environment configurations"""
        # Create base environment configurations (no shuffle here)
        self.dataset = self._create_base_configs()

        print(f"Created {len(self.dataset)} base environment configurations")

    def _create_base_configs(self) -> List[Dict[str, Any]]:
        """Create base environment configurations"""
        base_configs = []

        if self.env_type == "libero":
            base_configs = self._create_libero_base_configs()

        return base_configs

    def _create_libero_base_configs(self) -> List[Dict[str, Any]]:
        """Create base LIBERO environment configurations"""
        configs = []

        for task_id in range(self.num_tasks_in_suite):
            for trial_id in range(self.num_trials_per_task):
                config = {
                    "env_type": "libero",
                    "task_suite_name": self.task_name,
                    "task_id": torch.tensor(task_id, dtype=torch.int64).unsqueeze(0),
                    "trial_id": torch.tensor(trial_id, dtype=torch.int64).unsqueeze(0),
                    "trial_seed": torch.tensor(-1, dtype=torch.int64).unsqueeze(0),
                    "data_source": f"{self.task_name}_task_{task_id}_trial_{trial_id}",
                    "uid": str(uuid.uuid4()),
                }
                configs.append(config)

        return configs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get environment configuration by index"""
        if idx >= len(self.dataset):
            raise IndexError(f"Index {idx} out of range [0, {len(self.dataset)})")

        return self.dataset[idx]


class BufferedRLDataLoader:
    """
    Buffered data loader for RL environments that supports:
    - Batch-wise environment management
    - n_sample interleaving (GRPO requirement)
    - True data parallelism with DistributedSampler
    - Experience buffer management
    - Integration with DexRLDataset

    Data Parallelism:
    - Each GPU receives different data from DistributedSampler
    - Each GPU collects rollouts from different environments
    - Each GPU processes batch_size * n_sample environments
    - Total data per step = world_size * batch_size * n_sample
    - Gradients are synchronized by DeepSpeed/DDP

    Key functionality:
    - Takes a batch from dataset, then creates n_sample copies with interleaving
    - This matches GRPO's requirement for multiple samples per prompt
    - DistributedSampler ensures each GPU gets different base batches
    """

    def __init__(
        self,
        rl_dataset: DexRLDataset,
        n_sample: int = 8,
        env_dup: int = 1,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """
        Initialize BufferedRLDataLoader

        Args:
            rl_dataset: DexRLDataset instance
            n_sample: Number of samples for GRPO interleaving
            shuffle: Whether to shuffle batches
            drop_last: Whether to drop the last incomplete batch
        """
        self.rl_dataset = rl_dataset
        self.batch_size = rl_dataset.batch_size
        self.n_sample = n_sample
        self.env_dup = env_dup
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Auto-detect distributed training
        self.distributed = dist.is_available() and dist.is_initialized()

        # Get rank and world_size for distributed training
        if self.distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        # Experience buffer
        self.buffer = []

        # Create sampler for distributed training
        if self.distributed:
            self.sampler = DistributedSampler(
                rl_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=self.shuffle,
                drop_last=self.drop_last,
            )
            # Disable shuffle in DataLoader since DistributedSampler handles it
            dataloader_shuffle = False
        else:
            self.sampler = None
            dataloader_shuffle = self.shuffle

        # Create DataLoader for underlying dataset
        self.dataloader = DataLoader(
            rl_dataset,
            batch_size=self.batch_size,
            shuffle=dataloader_shuffle,
            drop_last=self.drop_last,
            sampler=self.sampler,
            collate_fn=collate_fn,
        )

        # Get actual number of batches from DataLoader
        # This automatically accounts for DistributedSampler's data splitting
        self.num_batches = len(self.dataloader)

        print("BufferedRLDataLoader initialized:")
        print(f"  Rank: {self.rank}/{self.world_size}")
        print(f"  Batches per epoch (this rank): {self.num_batches}")
        print(f"  Batch size: {self.batch_size}, n_sample: {n_sample}")
        if self.distributed:
            print(f"  Total dataset size: {len(rl_dataset)}")
            print(f"  Samples per rank: ~{len(rl_dataset) // self.world_size}")
        else:
            print("  Distributed training: Not detected")
            print(
                "  Note: If running multi-GPU training, ensure torch.distributed.init_process_group()"
            )
            print("        is called BEFORE creating the dataloader")

    def set_epoch(self, epoch: int):
        """Set epoch for DistributedSampler (ensures different shuffle each epoch)"""
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def __iter__(self):
        """Iterate over batches with n_sample interleaving"""
        for batch_data in self.dataloader:
            # Apply n_sample interleaving to the batch
            interleaved_batch = self._apply_n_sample_interleaving(batch_data)
            yield interleaved_batch

    def _apply_n_sample_interleaving(
        self, batch_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply n_sample interleaving to a batch for GRPO

        Input: batch of size [batch_size, ...]
        Output: batch of size [batch_size * n_sample, ...] with interleaving pattern

        Pattern: [item0, item1, item2] ->
                [item0, item0, item1, item1, item2, item2] (n_sample=2)

        Each element is repeated n_sample times consecutively before moving to next element.
        """

        interleaved_data = {}

        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                # Repeat each element n_sample times: [B, ...] -> [B*n_sample, ...]
                # repeat_interleave repeats each element: [1,2,3] -> [1,1,2,2,3,3]
                repeated_value = value.repeat_interleave(self.env_dup, dim=0)
                interleaved_data[key] = repeated_value
            elif isinstance(value, np.ndarray):
                # Repeat each element n_sample times
                # np.repeat repeats each element: [1,2,3] -> [1,1,2,2,3,3]
                repeated_value = np.repeat(value, self.env_dup, axis=0)
                interleaved_data[key] = repeated_value
            elif isinstance(value, list):
                # Repeat each list element n_sample times
                repeated_value = [item for item in value for _ in range(self.env_dup)]
                interleaved_data[key] = repeated_value
            else:
                # For other types, wrap in list and repeat each element
                if not isinstance(value, (list, tuple)):
                    value = [value]
                repeated_value = [item for item in value for _ in range(self.env_dup)]
                interleaved_data[key] = repeated_value

        # Add n_sample metadata
        interleaved_data["n_sample"] = self.n_sample
        interleaved_data["original_batch_size"] = self.batch_size
        interleaved_data["interleaved_batch_size"] = self.batch_size * self.env_dup

        return interleaved_data

    def get_batch_env_configs(self, batch_idx: int) -> List[Dict[str, Any]]:
        """
        Get environment configurations for a specific batch (with n_sample interleaving)

        Pattern: Each base config is repeated n_sample times consecutively
        Example with n_sample=2: [config0, config1] -> [config0, config0, config1, config1]
        """
        if batch_idx >= self.num_batches:
            raise IndexError(
                f"Batch index {batch_idx} out of range [0, {self.num_batches})"
            )

        # Get base batch from dataset
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.rl_dataset))

        base_configs = []
        for idx in range(start_idx, end_idx):
            base_configs.append(self.rl_dataset[idx])

        # Apply n_sample interleaving: repeat each config n_sample times
        interleaved_configs = []
        for base_idx, config in enumerate(base_configs):
            for sample_idx in range(self.n_sample):
                # Create a copy with sample info
                config_copy = config.copy()
                config_copy["sample_idx"] = sample_idx
                config_copy["base_config_idx"] = base_idx
                interleaved_configs.append(config_copy)

        return interleaved_configs

    def get_all_env_configs(self) -> List[Dict[str, Any]]:
        """Get all environment configurations (with n_sample interleaving)"""
        all_configs = []
        for batch_idx in range(self.num_batches):
            batch_configs = self.get_batch_env_configs(batch_idx)
            all_configs.extend(batch_configs)
        return all_configs

    def add_to_buffer(self, experience_data):
        """Add experience data to buffer"""
        self.buffer.append(experience_data)

    def get_buffer_size(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)

    def clear_buffer(self):
        """Clear the experience buffer"""
        self.buffer.clear()

    def sample_from_buffer(self, sample_size: int):
        """Sample from experience buffer"""
        if sample_size > len(self.buffer):
            return self.buffer.copy()

        indices = np.random.choice(len(self.buffer), sample_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        """Number of batches"""
        return self.num_batches
