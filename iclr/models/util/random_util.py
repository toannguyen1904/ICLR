import torch


def random_flip_zeros(x):
    """
    x: (B, T) binary tensor where each row is [1...1, 0...0]
    Returns: modified tensor with random % of zeros flipped to 1 per row
    """
    B, T = x.shape
    device = x.device

    x = x.clone()

    # Mask of zeros
    zero_mask = (x == 0)

    # Number of zeros per sequence
    num_zeros = zero_mask.sum(dim=1)  # (B,)

    # Random percentage in [0, 1] per sequence
    frac = torch.rand(B, device=device)

    # Number of zeros to flip per sequence
    k = (num_zeros.float() * frac).long()  # (B,)

    # Random scores for each position
    rand_scores = torch.rand(B, T, device=device)

    # Ensure only zeros are eligible
    rand_scores[~zero_mask] = float('inf')

    # Rank zeros by randomness
    sorted_idx = rand_scores.argsort(dim=1)

    # Build flip mask
    flip_mask = torch.zeros_like(x, dtype=torch.bool)
    for b in range(B):
        if k[b] > 0:
            flip_mask[b, sorted_idx[b, :k[b]]] = True

    # Flip selected zeros to ones
    x[flip_mask] = 1

    return x


def random_flip_ones(x):
    """
    x: (B, T) binary tensor where each row is [0...0, 1...1]
    Returns: modified tensor with random % of ones flipped to 0 per row
    """
    B, T = x.shape
    device = x.device

    x = x.clone()

    # Mask of ones
    one_mask = (x == 1)

    # Number of ones per sequence
    num_ones = one_mask.sum(dim=1)  # (B,)

    # Random percentage in [0, 1] per sequence
    frac = torch.rand(B, device=device)

    # Number of ones to flip per sequence
    k = (num_ones.float() * frac).long()  # (B,)

    # Random scores for each position
    rand_scores = torch.rand(B, T, device=device)

    # Ensure only ones are eligible
    rand_scores[~one_mask] = float('inf')

    # Rank ones by randomness
    sorted_idx = rand_scores.argsort(dim=1)

    # Build flip mask
    flip_mask = torch.zeros_like(x, dtype=torch.bool)
    for b in range(B):
        if k[b] > 0:
            flip_mask[b, sorted_idx[b, :k[b]]] = True

    # Flip selected ones to zeros
    x[flip_mask] = 0

    return x