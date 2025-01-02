# depth_decoder_summary.py

import torch
import torch.nn as nn
from torchinfo import summary
from depth_decoder import DepthDecoder  # Ensure this imports your DepthDecoder class
from layers import ConvBlock, Conv3x3, upsample  # Import from your layers.py
import numpy as np
from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis
from ptflops import get_model_complexity_info


def generate_depth_decoder_summary():
    # Define num_ch_enc based on your encoder's output channels
    num_ch_enc = np.array([48, 80, 128])  # Adjusted to match LiteMono encoder

    # Instantiate the DepthDecoder with adjusted scales
    model = DepthDecoder(num_ch_enc=num_ch_enc, scales=range(len(num_ch_enc)))

    # Rest of your code remains the same...
    batch_size = 1
    height = 128
    width = 416


    factors = [4, 8, 16]

    # Simulate input features
    input_features = [
        torch.randn(batch_size, num_ch_enc[i], height // factors[i], width // factors[i])
        for i in range(len(num_ch_enc))
    ]

    # Wrapper to handle list input
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x)

    model_wrapper = ModelWrapper(model)
    input_data = (input_features,)

    summary(model_wrapper, input_data=input_data, col_names=["input_size", "output_size", "num_params"], depth=5)

    # activations = {}

    #     # # Define a forward hook
    # def get_activation(name):
    #     def hook(model, input, output):
    #         if isinstance(output, torch.Tensor):
    #             activations[name] = output.detach().numel()  # Store the number of activations
    #         elif isinstance(output, (list, tuple)):
    #             total_elements = sum(o.detach().numel() for o in output if isinstance(o, torch.Tensor))
    #             activations[name] = total_elements  # Sum the number of elements if the output is a list/tuple
    #         else:
    #             activations[name] = 0  # For non-tensor outputs, store 0 or handle as needed
    #     return hook

    # # Register the hook to each layer
    # for name, layer in model.named_modules():
    #     if not isinstance(layer, nn.Sequential) and not isinstance(layer, nn.ModuleList) and type(layer) != nn.Module:
    #         layer.register_forward_hook(get_activation(name))

    # # Run the model
    # output = model(input_features)

    # # Print the number of activations per layer
    # for layer_name, num_activations in activations.items():
    #     print(f"Layer {layer_name}: {num_activations} activations")

    # class DepthDecoderWrapper(torch.nn.Module):
    #     def __init__(self, depth_decoder):
    #         super().__init__()
    #         self.depth_decoder = depth_decoder

    #     def forward(self, input_features):
    #         outputs = self.depth_decoder(input_features)
    #         # Return the finest resolution output (e.g., disp1)
    #         return list(outputs.values())[0]

    # model_wrapper = DepthDecoderWrapper(model)

    # flops = ActivationCountAnalysis(model_wrapper, (input_features,))

    # print(f"Total Activations for DepthDecoder: {flops.total()/1e6:.2f} GFLOPs")



if __name__ == "__main__":
    generate_depth_decoder_summary()
