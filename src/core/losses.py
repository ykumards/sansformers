import math

import torch

Tensor = torch.Tensor


def softmax_focal_loss(x, target, gamma=2.0, alpha=0.25):
    n = x.shape[0]
    device = target.device
    range_n = torch.arange(0, n, dtype=torch.int64, device=device)

    pos_num = float(x.shape[1])
    p = torch.softmax(x, dim=1)
    p = p[range_n, target]
    loss = -((1 - p) ** gamma) * alpha * torch.log(p)
    return torch.sum(loss) / pos_num


def zero_inflated_poisson_loss(input, target, p_logit, log_input=True, full=True):

    count_ones = torch.ones_like(target).to(target.device)
    count_zeros = torch.zeros_like(target).to(target.device)
    # set bce target to true if original count is 0 and false otherwise
    count_true_zeros = torch.where(target == 0, count_ones, count_zeros)
    p_value = torch.sigmoid(p_logit)

    loss_y0 = torch.log((1 - p_value) * torch.exp(-1 * input) + p_value)

    bce_loss_vec = torch.nn.functional.binary_cross_entropy_with_logits(
        p_logit, count_true_zeros.float(), reduction="none"
    )
    unweighted_poisson_loss_vec = torch.nn.functional.poisson_nll_loss(
        input, target.float(), reduction="none", log_input=log_input, full=full
    )
    weighted_poisson_loss_vec = (1 - p_value) * unweighted_poisson_loss_vec

    loss_y1 = bce_loss_vec + weighted_poisson_loss_vec

    loss = torch.where(target == 0, loss_y0, loss_y1)

    return loss.mean(), unweighted_poisson_loss_vec.mean()


# https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gaussian_nll_loss
def gaussian_nll_loss(input, target, var, *, full=False, eps=1e-6, reduction="mean"):
    r"""Gaussian negative log likelihood loss.

    See :class:`~torch.nn.GaussianNLLLoss` for details.

    Args:
        input: expectation of the Gaussian distribution.
        target: sample from the Gaussian distribution.
        var: tensor of positive variance(s), one for each of the expectations
            in the input (heteroscedastic), or a single one (homoscedastic).
        full: ``True``/``False`` (bool), include the constant term in the loss
            calculation. Default: ``False``.
        eps: value added to var, for stability. Default: 1e-6.
        reduction: specifies the reduction to apply to the output:
            `'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output is the average of all batch member losses,
            ``'sum'``: the output is the sum of all batch member losses.
            Default: ``'mean'``.
    """
    # Inputs and targets much have same shape
    input = input.view(input.size(0), -1)
    target = target.view(target.size(0), -1)
    if input.size() != target.size():
        raise ValueError("input and target must have same size")

    # Second dim of var must match that of input or be equal to 1
    var = var.view(input.size(0), -1)
    if var.size(1) != input.size(1) and var.size(1) != 1:
        raise ValueError("var is of incorrect size")

    # Check validity of reduction mode
    if reduction != "none" and reduction != "mean" and reduction != "sum":
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if torch.any(var < 0):
        raise ValueError("var has negative entry/entries")

    # Clamp for stability
    var = var.clone()
    with torch.no_grad():
        var.clamp_(min=eps)

    # Calculate loss (without constant)
    loss = 0.5 * (torch.log(var) + (input - target) ** 2 / var).view(
        input.size(0), -1
    ).sum(dim=1)

    # Add constant to loss term if required
    if full:
        D = input.size(1)
        loss = loss + 0.5 * D * math.log(2 * math.pi)

    # Apply reduction
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss
