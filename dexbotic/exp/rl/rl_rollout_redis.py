"""
RL rollout Redistribution utilities for distributed training data.

This module contains functions for redistributing filtered batches across
multiple GPUs to ensure uniform data distribution during RL training.
"""

import random
from typing import Any, Dict, List

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger


def redistribute_filtered_batch_circular(
    filtered_batch: Dict[str, Any],
    n_sample: int = 8,
    create_empty_batch_fn=None,
) -> Dict[str, Any]:
    """
    Enhanced redistribution using circular allocation algorithm.
    Ensures absolute uniform distribution across all ranks with n_sample units.

    Args:
        filtered_batch: Filtered batch data from current rank
        n_sample: Required multiple for batch size (default 8)
        create_empty_batch_fn: Function to create empty batch (optional)

    Returns:
        Redistributed batch data for current rank
    """
    if not dist.is_initialized():
        return filtered_batch

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    try:
        # Step 1: Collect rollout counts from all ranks
        local_count = torch.tensor(
            filtered_batch["responses"].size(0), dtype=torch.long, device="cuda"
        )
        all_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
        dist.all_gather(all_counts, local_count)

        counts = [count.item() for count in all_counts]
        total_rollouts = sum(counts)
        total_units = total_rollouts // n_sample

        logger.info(
            f"Rank {rank}: Circular redistribution - Local: {counts[rank]}, Total units: {total_units}"
        )

        # Step 2: Calculate uniform distribution target
        units_per_rank = total_units // world_size
        target_per_rank = units_per_rank * n_sample

        # Step 3: Handle insufficient data case - clear all if can't distribute uniformly
        if units_per_rank == 0:
            logger.info(
                f"Rank {rank}: Insufficient units ({total_units}) for uniform distribution across {world_size} ranks, clearing all data"
            )
            return create_empty_batch(filtered_batch)

        # Step 4: Distributed removal of excess rollouts
        total_target_units = units_per_rank * world_size
        excess_units = total_units - total_target_units

        if excess_units > 0:
            filtered_batch = distributed_remove_excess_units(
                filtered_batch, counts, excess_units, n_sample
            )
            # Re-collect counts after removal
            local_count = torch.tensor(
                filtered_batch["responses"].size(0), dtype=torch.long, device="cuda"
            )
            all_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
            dist.all_gather(all_counts, local_count)
            counts = [count.item() for count in all_counts]

        # Step 5: Check if redistribution needed
        if all(count == target_per_rank for count in counts):
            logger.info(
                f"Rank {rank}: Already uniformly distributed, no transfer needed"
            )
            return filtered_batch

        # Step 6: Execute transfers (uses internal greedy matching logic)
        redistributed_batch = execute_circular_transfers(filtered_batch, n_sample)

        # Step 7: Verify result
        final_count = redistributed_batch["responses"].size(0)
        assert (
            final_count == target_per_rank
        ), f"Rank {rank}: Final count {final_count} != target {target_per_rank}"

        if rank == 0:
            logger.info(
                f"Rank {rank}: Circular redistribution completed - Final count: {final_count}"
            )

        # Final barrier to ensure all ranks completed redistribution
        dist.barrier()
        return redistributed_batch

    except Exception as e:
        logger.error(f"Rank {rank}: Error in circular redistribution: {e}")
        raise


def distributed_remove_excess_units(
    batch: Dict[str, Any],
    all_counts: List[int],
    excess_units: int,
    n_sample: int,
) -> Dict[str, Any]:
    """
    Distributed removal of excess units to ensure uniform distribution.
    Each rank removes units proportionally to avoid single-point bottleneck.

    Args:
        batch: Current batch data
        all_counts: Rollout counts from all ranks
        excess_units: Total excess units to remove across all ranks
        n_sample: Unit size

    Returns:
        Batch with excess units removed
    """
    rank = dist.get_rank()

    # Calculate how many units each rank should remove
    current_units = [count // n_sample for count in all_counts]

    # Distribute removal proportionally based on current holdings
    removal_plan = compute_proportional_removal_plan(current_units, excess_units)

    # Remove units from current rank if needed
    units_to_remove = removal_plan.get(rank, 0)

    if units_to_remove == 0:
        return batch

    current_count = batch["responses"].size(0)
    current_rank_units = current_count // n_sample
    units_to_keep = current_rank_units - units_to_remove

    if units_to_keep <= 0:
        return create_empty_batch(batch)

    # Randomly select units to keep
    keep_units = random.sample(range(current_rank_units), units_to_keep)
    keep_units.sort()

    # Convert unit indices to sample indices
    keep_indices = []
    for unit in keep_units:
        keep_indices.extend(range(unit * n_sample, (unit + 1) * n_sample))

    # Apply to all batch keys
    filtered_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.size(0) == current_count:
            filtered_batch[key] = value[keep_indices]
        elif isinstance(value, np.ndarray) and value.shape[0] == current_count:
            filtered_batch[key] = value[keep_indices]
        elif isinstance(value, list) and len(value) == current_count:
            filtered_batch[key] = [value[i] for i in keep_indices]
        else:
            filtered_batch[key] = value

    if rank == 0:
        logger.info(
            f"Rank {rank}: Removed {units_to_remove} units, kept {units_to_keep} units"
        )
    return filtered_batch


def compute_proportional_removal_plan(
    current_units: List[int], excess_units: int
) -> Dict[int, int]:
    """
    Compute unit-level random removal plan using broadcast from rank 0.
    Treats all units across all ranks as a single pool and randomly selects units to remove.

    IMPORTANT: Only rank 0 computes the plan, then broadcasts to all ranks.
    This ensures perfect consistency and avoids any potential distributed issues.

    Args:
        current_units: Current units per rank
        excess_units: Total units to remove

    Returns:
        Dictionary mapping rank to units to remove (same across all ranks)
    """
    total_current = sum(current_units)
    if total_current == 0 or excess_units <= 0:
        return {}

    if not dist.is_initialized():
        # Single GPU case - compute directly
        return compute_removal_plan_single(current_units, excess_units)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        # Only rank 0 computes the removal plan
        removal_plan = compute_removal_plan_single(current_units, excess_units)

        # Convert to list format for broadcasting
        # Format: [rank0_removal, rank1_removal, ..., rankN_removal]
        removal_list = [removal_plan.get(r, 0) for r in range(world_size)]
    else:
        # Other ranks prepare empty list
        removal_list = [0] * world_size

    # Broadcast the removal plan from rank 0 to all ranks
    removal_tensor = torch.tensor(removal_list, dtype=torch.long, device="cuda")
    dist.broadcast(removal_tensor, src=0)

    # Convert back to dictionary format
    removal_plan = {}
    for r, count in enumerate(removal_tensor.tolist()):
        if count > 0:
            removal_plan[r] = count

    return removal_plan


def compute_removal_plan_single(
    current_units: List[int], excess_units: int
) -> Dict[int, int]:
    """
    Single-rank computation of unit-level random removal plan.

    Args:
        current_units: Current units per rank
        excess_units: Total units to remove

    Returns:
        Dictionary mapping rank to units to remove
    """
    # Create a list of all unit indices with their corresponding rank
    unit_to_rank = []
    for rank, units in enumerate(current_units):
        unit_to_rank.extend([rank] * units)

    # Randomly select excess_units from all available units
    total_available_units = len(unit_to_rank)
    if excess_units >= total_available_units:
        # Remove all units
        removal_plan = {
            rank: units for rank, units in enumerate(current_units) if units > 0
        }
    else:
        # Randomly select units to remove
        selected_unit_indices = random.sample(
            range(total_available_units), excess_units
        )

        # Count how many units to remove from each rank
        removal_plan = {}
        for unit_idx in selected_unit_indices:
            rank = unit_to_rank[unit_idx]
            removal_plan[rank] = removal_plan.get(rank, 0) + 1

    return removal_plan


def execute_circular_transfers(batch: Dict[str, Any], n_sample: int) -> Dict[str, Any]:
    """
    Execute sequential point-to-point transfers with proper synchronization.

    Uses greedy matching logic: scan from rank 0 upward, find sender/receiver pairs,
    and transfer units until all ranks have equal amounts.

    Uses isend/irecv for non-blocking communication to avoid deadlock.
    All ranks participate in each transfer round through barrier synchronization.

    Args:
        batch: Current batch data
        n_sample: Unit size

    Returns:
        Redistributed batch data
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"

    # Get current batch size
    current_size = batch["responses"].size(0) if batch["responses"].numel() > 0 else 0
    current_units = current_size // n_sample

    # Gather current units from all ranks
    local_units_tensor = torch.tensor([current_units], dtype=torch.long, device=device)
    all_units_list = [
        torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)
    ]
    dist.all_gather(all_units_list, local_units_tensor)
    units_per_rank = [t.item() for t in all_units_list]

    total_units = sum(units_per_rank)
    target_units = total_units // world_size

    # Handle case where target is 0 - return empty batch
    if target_units == 0:
        return create_empty_batch(batch)

    print(
        f"[DEBUG] Rank {rank}: units_per_rank = {units_per_rank}, target = {target_units}"
    )

    # ========== Broadcast metadata to ensure all ranks have consistent key and shape info ==========
    source_rank = -1
    for r, units in enumerate(units_per_rank):
        if units > 0:
            source_rank = r
            break

    if source_rank < 0:
        return create_empty_batch(batch)

    # Broadcast metadata from source_rank
    if rank == source_rank:
        metadata = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                metadata[k] = {
                    "type": "tensor",
                    "shape": list(v.shape[1:]),
                    "dtype": str(v.dtype),
                }
            elif isinstance(v, np.ndarray):
                metadata[k] = {
                    "type": "numpy",
                    "shape": list(v.shape[1:]),
                    "dtype": str(v.dtype),
                }
            else:
                metadata[k] = {"type": "other", "shape": [], "dtype": "none"}
        metadata_list = [metadata]
    else:
        metadata_list = [None]

    dist.broadcast_object_list(metadata_list, src=source_rank)
    metadata = metadata_list[0]

    print(f"[DEBUG] Rank {rank}: Received metadata with keys: {list(metadata.keys())}")

    # ========== Initialize local_batch using metadata ==========
    original_numpy_dtypes = {}
    local_batch = {}

    str_to_torch_dtype = {
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.int64": torch.int64,
        "torch.int32": torch.int32,
        "torch.int16": torch.int16,
        "torch.int8": torch.int8,
        "torch.bool": torch.bool,
    }

    str_to_numpy_dtype = {
        "int64": np.int64,
        "int32": np.int32,
        "float64": np.float64,
        "float32": np.float32,
        "bool": np.bool_,
        "object": object,
    }

    for k, meta in metadata.items():
        if k in batch:
            v = batch[k]
            if isinstance(v, torch.Tensor):
                local_batch[k] = v.to(device).contiguous()
            elif isinstance(v, np.ndarray):
                original_numpy_dtypes[k] = v.dtype
                local_batch[k] = v
            else:
                local_batch[k] = v
        else:
            if meta["type"] == "tensor":
                torch_dtype = str_to_torch_dtype.get(meta["dtype"], torch.float32)
                empty_shape = [0] + meta["shape"]
                local_batch[k] = torch.zeros(
                    empty_shape, dtype=torch_dtype, device=device
                )
            elif meta["type"] == "numpy":
                dtype_str = meta["dtype"]
                if dtype_str.startswith("dtype('") and dtype_str.endswith("')"):
                    dtype_str = dtype_str[7:-2]
                np_dtype = str_to_numpy_dtype.get(dtype_str, np.float32)
                original_numpy_dtypes[k] = np_dtype
                empty_shape = [0] + meta["shape"]
                local_batch[k] = np.zeros(empty_shape, dtype=np_dtype)
            else:
                local_batch[k] = None

    # ========== Execute sequential transfers ==========
    max_transfers = world_size * 2
    for transfer_idx in range(max_transfers):
        # Sync units info across all ranks
        local_size = (
            local_batch["responses"].size(0)
            if isinstance(local_batch.get("responses"), torch.Tensor)
            and local_batch["responses"].numel() > 0
            else 0
        )
        local_units = local_size // n_sample

        local_units_tensor = torch.tensor(
            [local_units], dtype=torch.long, device=device
        )
        dist.all_gather(all_units_list, local_units_tensor)
        units_per_rank = [t.item() for t in all_units_list]

        # Termination condition: all ranks reach target_units
        if all(u == target_units for u in units_per_rank):
            print(
                f"[DEBUG] Rank {rank}: All ranks balanced at {target_units} units after {transfer_idx} transfers"
            )
            break

        # Find transfer pair: scan from rank 0 upward
        sender = -1
        receiver = -1
        transfer_units = 0

        for check_rank in range(world_size):
            diff = units_per_rank[check_rank] - target_units

            if diff < 0:
                for src in range(check_rank + 1, world_size):
                    if units_per_rank[src] > target_units:
                        sender = src
                        receiver = check_rank
                        need = target_units - units_per_rank[check_rank]
                        have = units_per_rank[src] - target_units
                        transfer_units = min(need, have)
                        break
                if sender >= 0:
                    break

            elif diff > 0:
                for dst in range(check_rank + 1, world_size):
                    if units_per_rank[dst] < target_units:
                        sender = check_rank
                        receiver = dst
                        need = target_units - units_per_rank[dst]
                        have = units_per_rank[check_rank] - target_units
                        transfer_units = min(need, have)
                        break
                if receiver >= 0:
                    break

        if sender < 0 or receiver < 0 or transfer_units <= 0:
            print(f"[DEBUG] Rank {rank}: No valid transfer found, breaking")
            break

        print(
            f"[DEBUG] Rank {rank}: Transfer {transfer_idx}: rank {sender} -> rank {receiver}, {transfer_units} units"
        )

        transfer_samples = transfer_units * n_sample

        sorted_keys = sorted(metadata.keys())
        requests = []

        if rank == sender:
            local_size = local_batch["responses"].size(0)
            send_start = local_size - transfer_samples

            for key in sorted_keys:
                meta = metadata[key]
                value = local_batch.get(key)

                if meta["type"] == "tensor" and isinstance(value, torch.Tensor):
                    send_data = value[send_start:].contiguous().to(device)
                    req = dist.isend(send_data, dst=receiver)
                    requests.append(req)
                    local_batch[key] = value[:send_start].contiguous()

                elif meta["type"] == "numpy" and isinstance(value, np.ndarray):
                    torch_dtype = str_to_torch_dtype.get(
                        "torch."
                        + meta["dtype"].replace("dtype('", "").replace("')", ""),
                        torch.float32,
                    )
                    if "int64" in meta["dtype"]:
                        torch_dtype = torch.int64
                    elif "int32" in meta["dtype"]:
                        torch_dtype = torch.int32
                    elif "float64" in meta["dtype"]:
                        torch_dtype = torch.float64
                    elif "float32" in meta["dtype"]:
                        torch_dtype = torch.float32
                    elif "bool" in meta["dtype"]:
                        torch_dtype = torch.bool

                    send_data = torch.from_numpy(value[send_start:].copy()).to(
                        dtype=torch_dtype, device=device
                    )
                    req = dist.isend(send_data, dst=receiver)
                    requests.append(req)
                    local_batch[key] = value[:send_start].copy()

        elif rank == receiver:
            for key in sorted_keys:
                meta = metadata[key]
                value = local_batch.get(key)

                if meta["type"] == "tensor":
                    torch_dtype = str_to_torch_dtype.get(meta["dtype"], torch.float32)
                    recv_shape = [transfer_samples] + meta["shape"]
                    recv_buffer = torch.zeros(
                        recv_shape, dtype=torch_dtype, device=device
                    )
                    req = dist.irecv(recv_buffer, src=sender)
                    requests.append((req, key, recv_buffer, "tensor"))

                elif meta["type"] == "numpy":
                    dtype_str = meta["dtype"]
                    if "int64" in dtype_str:
                        torch_dtype = torch.int64
                        np_dtype = np.int64
                    elif "int32" in dtype_str:
                        torch_dtype = torch.int32
                        np_dtype = np.int32
                    elif "float64" in dtype_str:
                        torch_dtype = torch.float64
                        np_dtype = np.float64
                    elif "float32" in dtype_str:
                        torch_dtype = torch.float32
                        np_dtype = np.float32
                    elif "bool" in dtype_str:
                        torch_dtype = torch.bool
                        np_dtype = np.bool_
                    else:
                        torch_dtype = torch.float32
                        np_dtype = np.float32

                    recv_shape = [transfer_samples] + meta["shape"]
                    recv_buffer = torch.zeros(
                        recv_shape, dtype=torch_dtype, device=device
                    )
                    req = dist.irecv(recv_buffer, src=sender)
                    requests.append((req, key, recv_buffer, "numpy", np_dtype))

        # Wait for all async operations to complete
        if rank == sender:
            for req in requests:
                req.wait()
        elif rank == receiver:
            for item in requests:
                if len(item) == 4:
                    req, key, recv_buffer, dtype_tag = item
                    req.wait()
                    value = local_batch.get(key)
                    if isinstance(value, torch.Tensor) and value.numel() > 0:
                        local_batch[key] = torch.cat(
                            [value.to(device), recv_buffer], dim=0
                        )
                    else:
                        local_batch[key] = recv_buffer
                elif len(item) == 5:
                    req, key, recv_buffer, dtype_tag, np_dtype = item
                    req.wait()
                    recv_np = recv_buffer.cpu().numpy().astype(np_dtype)
                    value = local_batch.get(key)
                    if isinstance(value, np.ndarray) and value.size > 0:
                        local_batch[key] = np.concatenate([value, recv_np], axis=0)
                    else:
                        local_batch[key] = recv_np

        # All ranks synchronize after transfer
        dist.barrier()

    # Final verification
    final_size = (
        local_batch["responses"].size(0)
        if isinstance(local_batch.get("responses"), torch.Tensor)
        else 0
    )
    print(
        f"[DEBUG] Rank {rank}: Final size = {final_size}, expected = {target_units * n_sample}"
    )

    # Ensure numpy arrays are back on CPU, tensors stay on current rank's CUDA
    result_batch = {}
    for k, v in local_batch.items():
        if k in original_numpy_dtypes:
            if isinstance(v, torch.Tensor):
                result_batch[k] = v.cpu().numpy().astype(original_numpy_dtypes[k])
            else:
                result_batch[k] = v
        else:
            result_batch[k] = v

    return result_batch


def create_empty_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Create empty batch with same structure."""
    try:
        empty = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                empty[key] = torch.zeros(
                    [0] + list(value.shape[1:]),
                    dtype=value.dtype,
                    device=value.device,
                )
            elif isinstance(value, np.ndarray):
                empty[key] = np.zeros([0] + list(value.shape[1:]), dtype=value.dtype)
            elif isinstance(value, list):
                empty[key] = []
            else:
                empty[key] = value
        return empty
    except Exception:
        print("\n===============Error creating empty batch===============\n")
        return batch
