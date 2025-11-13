import torch


def alignment(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 2,
):
    """
    https://arxiv.org/pdf/2005.10242

    Label-aware Alignment loss.

    Calculates alignment for embeddings of samples with the SAME label
    within a batch, assuming embeddings are already unit-normalized.

    Args:
        embeddings: Tensor [N, D] - Batch of unit-normalized embeddings.
        labels: Tensor [N] - Corresponding labels.
        alpha: Power to raise squared distance (hyperparameter, default=2).

    Returns:
        Tensor: Label-aware Alignment loss (scalar). Returns 0 if no positive pairs.
    """
    assert embeddings.size(0) == labels.size(0), "Embeddings and labels must have the same size."

    n_samples = embeddings.size(0)
    if n_samples < 2:
        return torch.tensor(0.0, device=embeddings.device)

    # Create a pairwise label comparison matrix (N x N), exclude self-pairs
    labels_equal_mask = (labels[:, None] == labels[None, :]).triu(diagonal=1)

    positive_indices = torch.nonzero(labels_equal_mask, as_tuple=False)
    if positive_indices.numel() == 0:
        return torch.tensor(0.0, device=embeddings.device)

    # Get embeddings of positive pairs
    x = embeddings[positive_indices[:, 0]]
    y = embeddings[positive_indices[:, 1]]

    # Calculate alignment loss
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniformity(
    x: torch.Tensor,
    t: float = 2,
    clip_value: float = 1e-6,
):
    """
    https://arxiv.org/pdf/2005.10242

    Calculates the Uniformity loss.

    Args:
        x: [N, D] - Batch of feature embeddings.
        t: Temperature parameter (hyperparameter).

    Returns:
        Tensor: Uniformity loss value (scalar).
    """
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().clamp(min=clip_value).log()


if __name__ == "__main__":
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
    )
    embeddings /= embeddings.norm(p=2, dim=1, keepdim=True)

    labels = torch.tensor([0, 0, 0, 1, 1])

    print("Embeddings:")
    print(embeddings.numpy())

    print("\nLabels:")
    print(labels.numpy())

    alignment_loss = alignment(embeddings, labels, alpha=2)
    print("\nAlignment loss:", alignment_loss.item())

    uniformity_loss = uniformity(embeddings, t=2, clip_value=1e-6)
    print("Uniformity loss:", uniformity_loss.item())
