# model_summary.py
import torch
import torch.nn as nn
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis
from depth_encoder import LiteMono  # Import LiteMono from your depth_encoder file
from torchprofile import profile_macs
from thop import profile 

if __name__ == "__main__":
    # Instantiate the LiteMono model
    model = LiteMono()

    input_tensor=torch.randn(1, 3, 224, 224)
    activations = {}

    # # Define a forward hook
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
    # output = model(input_tensor)

    # # Print the number of activations per layer
    # for layer_name, num_activations in activations.items():
    #     print(f"Layer {layer_name}: {num_activations} activations")

    # flops = ActivationCountAnalysis(model, input_tensor)
    # Use torchinfo to print a detailed summary
    # Specify the batch size and input dimensions
    summary(model, input_size=(1, 3, 224, 224), col_names=["input_size", "output_size", "num_params"],depth=5)


    # flops, params = profile(model, inputs=(input_tensor,))
    # print(f"FLOPs: {flops}, Parameters: {params}")
    # print(f"Total Activations: {flops.total()/1e6:.2f} M")

    #  # Initialize the model
    # model = LiteMono()  # Replace with your model initialization

    # # Define input shape (channels, height, width)
    # input_size = (3, 192, 640)  # Adjust to your model's input size

    # # Perform complexity analysis
    # macs, params = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=True)
    # print(f"Total MACs: {macs}")
    # print(f"Total Parameters: {params}")
    # print(f"Total MACs: {macs / 1e9:.2f} GMACs")
