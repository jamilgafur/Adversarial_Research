import torch


def fgsm_attack(
        input_tensor: torch.Tensor,
        label: int,
        epsilon: float,
        model: torch.nn.Module,
        device: str = "cpu") -> torch.Tensor:
    """
    Generates adversarial example using fast gradient sign method.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The input for the model
    label : int
        The expected label of the input image
    epsilon : float
        The percent of perturbation added to the input image
    model : torch.nn.Module
        The pre-trained PyTorch model
    device : str
        The device to run the attack on
        default: "cpu"

    Returns
    -------
    perturbed : torch.Tensor
        The perturbed input
    """

    data = torch.tensor(input_tensor).to(torch.float32).to(device)
    model = model.to(device)
    data.requires_grad = True

    # Forward pass to get the prediction
    output = model(data)

    # Calculate the loss
    loss = torch.nn.F.cross_entropy(output, torch.tensor([label]).to(device))

    # Backward pass to get the gradient
    loss.backward()
    model.zero_grad()

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed = data + epsilon * data.grad.sign()
    perturbed = torch.clamp(perturbed, 0, 1)

    # Return the perturbed image
    return perturbed.cpu().detach()