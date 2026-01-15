from dexbotic.exp.utils import require_config_keys
from .model import L1RegressionActionHead, DiffusionActionHead, DiscreteActionHead


@require_config_keys(["action_model_type", "hidden_size", "action_dim", "chunk_size", "use_proprio", "proprio_dim"])
def build_action_model(config):

    model_type = config.action_model_type
    action_dim = config.action_dim
    hidden_size = config.hidden_size
    action_chunk = config.chunk_size
    use_proprio = config.use_proprio
    proprio_dim = config.proprio_dim
    
    assert model_type in ["Linear", "DiT", "Discrete"], f"Unsupported action model type: {model_type}"
    if "Linear" in model_type:
        action_model = L1RegressionActionHead(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            action_dim=action_dim,
            action_chunk=action_chunk,
            use_proprio=use_proprio,
            proprio_dim=proprio_dim
        )
    elif "DiT" in model_type:
        action_model = DiffusionActionHead(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            action_dim=action_dim,
            action_chunk=action_chunk,
            num_diffusion_steps=100,
            use_proprio=use_proprio,
            proprio_dim=proprio_dim
        )
    elif "Discrete" in model_type:
        vocab_size = getattr(config, "vocab_size", 32000)
        num_bins = getattr(config, "num_bins", 256)
        action_model = DiscreteActionHead(
            input_dim=hidden_size,
            vocab_size=vocab_size,
            action_dim=action_dim,
            action_chunk=action_chunk,
            num_bins=num_bins,
            use_proprio=use_proprio,
            proprio_dim=proprio_dim,
        )
    
    return action_model
