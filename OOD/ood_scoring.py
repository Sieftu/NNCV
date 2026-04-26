import torch
import torch.nn.functional as F


def score_msp(logits, temperature=1.0):
    """Negative mean of max softmax probability. More negative = more confident = more ID."""
    probs = F.softmax(logits / temperature, dim=1)
    max_probs = probs.max(dim=1).values  # (1, H, W)
    return -max_probs.mean().item()


def score_energy(logits, temperature=1.0):
    """Negative mean energy. Lower energy = more ID."""
    energy = temperature * torch.logsumexp(logits / temperature, dim=1)  # (1, H, W)
    return -energy.mean().item()


def score_pixel_uncertainty(logits, conf_threshold=0.5):
    """Fraction of pixels with max softmax prob below threshold. Higher fraction = more OOD."""
    probs = F.softmax(logits, dim=1)
    max_probs = probs.max(dim=1).values  # (1, H, W)
    uncertain_fraction = (max_probs < conf_threshold).float().mean().item()
    return uncertain_fraction  # already positive = higher means more OOD


def score_entropy(logits, temperature=1.0):
    """Mean pixel-wise entropy of the predictive distribution. Higher entropy = more OOD.

    Entropy H = -sum_c p_c * log(p_c) is maximal when the model is uniformly uncertain
    across all classes and zero when it is perfectly confident.  Averaging over all spatial
    positions gives a single scalar that rises as the scene contains more unfamiliar content.
    """
    probs = F.softmax(logits / temperature, dim=1)  # (1, C, H, W)
    # Clamp for numerical safety before taking log
    log_probs = torch.log(probs.clamp(min=1e-10))
    entropy = -(probs * log_probs).sum(dim=1)  # (1, H, W)
    return entropy.mean().item()  # positive; higher = more OOD
