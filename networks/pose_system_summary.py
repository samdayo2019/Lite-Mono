import torch
import torch.nn as nn
from torchinfo import summary

from pose_decoder import PoseDecoder
from resnet_encoder import ResnetEncoder
from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis
from ptflops import get_model_complexity_info


# Assuming the ResnetEncoder and PoseDecoder classes are defined as per your code
# Make sure to include or import these classes in your script

def generate_pose_model_summaries():
    #Parameters
    num_layers = 18
    pretrained = False
    num_input_images = 2
    num_input_features = 1
    num_frames_to_predict_for = 2
    batch_size = 1
    height = 192
    width = 640

    # Instantiate the ResnetEncoder
    model = ResnetEncoder(
        num_layers=num_layers,
        pretrained=pretrained,
        num_input_images=num_input_images
    )

    # Create dummy input for the encoder
    # Since num_input_images=2 and each image has 3 channels, input has 6 channels
    input_features = torch.randn(batch_size, 6, height, width)

    # Get the encoder summary
    print("ResnetEncoder Summary:")

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

    
    # # Print the number of activations per layer
    # for layer_name, num_activations in activations.items():
    #     print(f"Layer {layer_name}: {num_activations} activations")
    summary(
        model,
        input_data=input_features,
        col_names=["input_size", "output_size", "num_params"],
        depth=5,
        verbose=1
    )

    # Forward pass through encoder to get features
    encoder_features = model(input_features)  # This will be a list of tensors

    # Instantiate the PoseDecoder
    model2 = PoseDecoder(
        num_ch_enc=model.num_ch_enc,
        num_input_features=num_input_features,
        num_frames_to_predict_for=num_frames_to_predict_for
    )

    # Prepare input features for PoseDecoder
    # The PoseDecoder expects input_features to be a list of lists
    # Since num_input_features=1, we wrap encoder_features in a list
    input_features = [encoder_features]

    # Wrap the PoseDecoder to handle list input for torchinfo
    class PoseDecoderWrapper(nn.Module):
        def __init__(self, pose_decoder):
            super().__init__()
            self.pose_decoder = pose_decoder

        def forward(self, x):
            return self.pose_decoder(x)

    pose_decoder_wrapper = PoseDecoderWrapper(model2)

    # # Prepare input data as a tuple containing the list
    input_data = (input_features,)

    # Get the PoseDecoder summary
    print("\nPoseDecoder Summary:")

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
    # for name, layer in model2.named_modules():
    #     if not isinstance(layer, nn.Sequential) and not isinstance(layer, nn.ModuleList) and type(layer) != nn.Module:
    #         layer.register_forward_hook(get_activation(name))

    # # Run the model
    # output = model2(input_features)

    # # Print the number of activations per layer
    # for layer_name, num_activations in activations.items():
    #     print(f"Layer {layer_name}: {num_activations} activations")
    summary(
        pose_decoder_wrapper,
        input_data=input_data,
        col_names=["input_size", "output_size", "num_params"],
        depth=6,
        verbose=1
    )
#  # Initialize ResNet Encoder
#     resnet_encoder = ResnetEncoder(num_layers=18, pretrained=False, num_input_images=2)
#     encoder_input = torch.randn(1, 6, 192, 640)  # 6 channels for 2 input images, adjust as needed
#     encoder_flops = ActivationCountAnalysis(resnet_encoder, encoder_input)
#     print(f"Total FLOPs for ResNet Encoder: {encoder_flops.total() / 1e6:.2f} GFLOPs")

#     encoder_features = resnet_encoder(encoder_input)

#     # Initialize Pose Decoder
#     pose_decoder = PoseDecoder(num_ch_enc=resnet_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)

#     # Use the output from the deepest layer of the encoder (which should have 512 channels)
#     decoder_input = [encoder_features[-1]]  # Use only the deepest feature map

#     # Modify PoseDecoder to make it trace-friendly
#     class TraceablePoseDecoder(torch.nn.Module):
#         def __init__(self, pose_decoder):
#             super().__init__()
#             self.pose_decoder = pose_decoder

#         def forward(self, input_features):
#             # Use the last feature map from encoder
#             last_features = [input_features[-1]]

#             # Apply the convolution layers in PoseDecoder
#             cat_features = [self.pose_decoder.relu(self.pose_decoder.convs["squeeze"](f)) for f in last_features]
#             cat_features = torch.cat(cat_features, 1)

#             out = cat_features
#             for i in range(3):
#                 out = self.pose_decoder.convs[("pose", i)](out)
#                 if i != 2:
#                     out = self.pose_decoder.relu(out)

#             # Instead of mean, flatten to maintain tensor shape consistency
#             out = torch.flatten(out, start_dim=1)

#             # For FLOP calculation, return only the processed tensor
#             return out

#     model_wrapper = TraceablePoseDecoder(pose_decoder)

#     # Perform FLOP analysis for Pose Decoder
#     flops = ActivationCountAnalysis(model_wrapper, (decoder_input,))
#     print(f"Total FLOPs for Pose Decoder: {flops.total() / 1e6:.2f} GFLOPs")

# #    # Initialize ResNet Encoder
# #     resnet_encoder = ResnetEncoder(num_layers=18, pretrained=False, num_input_images=2)
# #     encoder_input_size = (6, 192, 640)  # 6 channels for 2 input images

# #     # Perform complexity analysis for ResNet Encoder
# #     with torch.no_grad():
# #         macs, params = get_model_complexity_info(resnet_encoder, encoder_input_size, as_strings=False, print_per_layer_stat=True)
# #     print(f"Total MACs for ResNet Encoder: {macs / 1e9:.2f} GMACs")
# #     print(f"Total Parameters for ResNet Encoder: {params / 1e6:.2f} Million")

# #      # Define the parameters for PoseDecoder
# #     num_ch_enc = [64, 64, 128, 256, 512]  # Number of channels from ResNet encoder layers
# #     num_input_features = 1
# #     num_frames_to_predict_for = 2

# #     # Instantiate the PoseDecoder
# #     model = PoseDecoder(
# #         num_ch_enc=num_ch_enc,
# #         num_input_features=num_input_features,
# #         num_frames_to_predict_for=num_frames_to_predict_for
# #     )

# #     # Wrap PoseDecoder to properly handle the inputs
# #     class PoseDecoderWrapper(torch.nn.Module):
# #         def __init__(self, pose_decoder):
# #             super().__init__()
# #             self.pose_decoder = pose_decoder

# #         def forward(self, input_features):
# #             # Ensure input_features is properly formatted to have 4 dimensions for each tensor
# #             # Batch size of 1, channels, height, width
# #             formatted_features = [
# #                 torch.randn(1, ch, 192, 640)  # Adjust the height and width as necessary
# #                 for ch in num_ch_enc
# #             ]
# #             axisangle, translation = self.pose_decoder([formatted_features])
# #             return torch.cat([axisangle, translation], dim=-1)

# #     model_wrapper = PoseDecoderWrapper(model)

# #     # Generate input tensor size for complexity analysis
# #     input_size = (1, num_ch_enc[-1], 192, 640)  # Using the last encoder layer output dimensions

# #     # Perform complexity analysis
# #     macs, params = get_model_complexity_info(
# #         model_wrapper, input_size, as_strings=False, print_per_layer_stat=True, verbose=True
# #     )

# #     if macs is not None and params is not None:
# #         print(f"Total MACs for PoseDecoder: {macs / 1e9:.2f} GMACs")
# #         print(f"Total Parameters for PoseDecoder: {params / 1e6:.2f} Million")
# #     else:
# #         print("Error computing MACs and parameter counts.")

if __name__ == "__main__":
    generate_pose_model_summaries()
