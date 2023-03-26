import torch


def activation_map(input_tensor: torch.Tensor,
                   model: torch.nn.Module) -> torch.Tensor:
    """
    Generate a saliency map for an input image given a pre-trained PyTorch model.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The input for the model
    model : torch.nn.Module
        The pre-trained PyTorch model

    Returns
    -------
    slc : torch.Tensor
        The activation map for the input image
    """
    # Convert the input image to a PyTorch tensor with dtype=torch.float32 and
    # enable gradient computation
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    model.eval()

    # Disable gradient computation for all model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Make a forward pass through the model and get the predicted class scores
    # for the input image
    preds = model(input_tensor)

    # Compute the score and index of the class with the highest predicted score
    score, _ = torch.max(preds, 1)

    # Compute gradients of the score with respect to the input image pixels
    score.backward()

    # Compute the saliency map by taking the maximum absolute gradient across
    # color channels and normalize the values
    slc = torch.max(torch.abs(input_tensor.grad), dim=0)[0]
    slc = (slc - slc.min()) / (slc.max() - slc.min())

    # return the saliency map as a numpy array
    return slc.detach()