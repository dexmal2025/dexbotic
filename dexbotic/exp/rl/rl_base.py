from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from dexbotic.exp.base_exp import Config


@dataclass
class RLConfig(Config):
    """
    Base RL configuration class - controls RL training parameters
    """

    gamma: float = field(default=0.99)
    lam: float = field(default=0.95)
    epsilon: float = field(default=1e-6)
    n_samples: int = field(default=8)


@dataclass
class GRPOConfig(RLConfig):
    """
    GRPO (Generalized Policy Optimization) configuration class

    GRPO advantages calculation based on SimpleVLA-RL implementation:
    - Groups responses by prompt index
    - Computes normalized advantages within each group
    - Uses outcome rewards (scalar reward per response)
    """

    advantage_estimator: str = field(default="grpo")

    def compute_grpo_outcome_advantage(
        self,
        token_level_rewards: torch.Tensor,
        eos_mask: torch.Tensor,
        index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantage for GRPO, operating only on Outcome reward
        (with only one scalar reward for each response).

        Args:
            token_level_rewards: shape (bs, response_length)
            eos_mask: shape (bs, response_length)
            index: prompt indices for grouping responses

        Returns:
            advantages: shape (bs, response_length)
            returns: shape (bs, response_length)
        """
        response_length = token_level_rewards.shape[-1]
        scores = token_level_rewards.sum(dim=-1)

        id2score = defaultdict(list)
        id2mean = {}
        id2std = {}

        with torch.no_grad():
            bsz = scores.shape[0]
            for i in range(bsz):
                id2score[index[i]].append(scores[i])

            for idx in id2score:
                if len(id2score[idx]) == 1:
                    id2mean[idx] = torch.tensor(0.0)
                    id2std[idx] = torch.tensor(1.0)
                elif len(id2score[idx]) > 1:
                    id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                    id2std[idx] = torch.std(torch.tensor(id2score[idx]))
                else:
                    raise ValueError(f"no score in prompt index: {idx}")

            for i in range(bsz):
                scores[i] = (scores[i] - id2mean[index[i]]) / (
                    id2std[index[i]] + self.epsilon
                )
            scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

        return scores, scores


@dataclass
class RLEnvironmentConfig(Config):
    """
    RL environment configuration for supporting different simulation environments
    """

    env_name: str = field(default="libero")  # libero now
    num_envs: int = field(default=1)
    max_episode_steps: int = field(default=100)

    # Environment-specific configs
    libero_config: Dict = field(default_factory=dict)
    simpler_config: Dict = field(default_factory=dict)
    maniskill2_config: Dict = field(default_factory=dict)


class RLTrainer:
    """
    Base RL trainer class that extends the dexbotic training framework
    """

    def __init__(self, rl_config: RLConfig, env_config: RLEnvironmentConfig):
        self.rl_config = rl_config
        self.env_config = env_config

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        eos_mask: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages based on the RL algorithm

        Args:
            rewards: token-level rewards
            values: value function predictions (if applicable)
            eos_mask: end-of-sequence mask

        Returns:
            advantages, returns
        """
        raise NotImplementedError

    def compute_policy_loss(
        self,
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        eos_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute policy gradient loss
        """
        raise NotImplementedError

    def collect_rollouts(self, model, env, num_rollouts: int) -> Dict:
        """
        Collect rollouts from the environment
        """
        raise NotImplementedError


class GRPOTrainer(RLTrainer):
    """
    GRPO trainer implementation following SimpleVLA-RL
    """

    def __init__(self, grpo_config: GRPOConfig, env_config: RLEnvironmentConfig):
        super().__init__(grpo_config, env_config)
        self.grpo_config = grpo_config

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: Optional[torch.Tensor],
        eos_mask: torch.Tensor,
        index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GRPO advantages
        """
        return self.grpo_config.compute_grpo_outcome_advantage(rewards, eos_mask, index)

    def compute_policy_loss(
        self,
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        eos_mask: torch.Tensor,
        clip_ratio_high: float = 0.28,
        clip_ratio_low: float = 0.2,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute PPO-style clipped policy loss for GRPO
        """
        negative_approx_kl = log_prob - old_log_prob
        ratio = torch.exp(negative_approx_kl)

        # Masked mean function for computing average over valid tokens
        def masked_mean(tensor, mask):
            return (tensor * mask).sum() / mask.sum()

        ppo_kl = masked_mean(-negative_approx_kl, eos_mask)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(
            ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high
        )

        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)

        return pg_loss, pg_clipfrac, ppo_kl
