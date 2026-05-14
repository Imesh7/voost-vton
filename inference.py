
import torch

from unified_dit.unified_dit import UnifiedDiT


def inference(model : UnifiedDiT, input_data):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        output = model.sample(input_data)  # Forward pass through the model
    return output  # Return the output for further processing