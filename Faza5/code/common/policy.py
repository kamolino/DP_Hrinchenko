import math
import sys

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class BoundedMeanPolicy(ActorCriticPolicy):
    """Gaussian policy with a tanh-bounded mean and a clamped standard deviation."""

    def __init__(self, *args, min_action_std=0.1, max_action_std=0.5, **kwargs):
        if not 0 < min_action_std <= max_action_std:
            raise ValueError("Action noise bounds must be positive and ordered.")
        self.min_log_std = math.log(min_action_std)
        self.max_log_std = math.log(max_action_std)
        super().__init__(*args, **kwargs)
        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise ValueError("This policy requires diagonal Gaussian actions.")

    def _get_action_dist_from_latent(self, latent_pi):
        mean = torch.tanh(self.action_net(latent_pi))
        log_std = self.log_std.clamp(self.min_log_std, self.max_log_std)
        return self.action_dist.proba_distribution(mean, log_std)

    def limit_noise(self):
        with torch.no_grad():
            self.log_std.clamp_(self.min_log_std, self.max_log_std)


# saved models reference the class as bounded_policy.BoundedMeanPolicy
sys.modules.setdefault("bounded_policy", sys.modules[__name__])


def load(path, env=None, **custom_objects):
    return PPO.load(path, env=env, device="cpu", custom_objects=custom_objects or None)
