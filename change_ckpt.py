import torch

# Load the original .ckpt checkpoint
ckpt_path = "examples/materials_benchmark/saved_models/cgcnn/best.ckpt"
checkpoint = torch.load(ckpt_path, map_location="cpu")

state_dict = checkpoint['state_dict']

# Create a new state dict with renamed keys
new_state_dict = {}
for k, v in state_dict.items():
    # Replace "model." with "fea."
    new_k = k.replace("model.", "fea.")
    new_state_dict[new_k] = v

# Update the checkpoint dict
checkpoint['state_dict'] = new_state_dict

# Save to new .ckpt file
new_ckpt_path = "examples/materials_benchmark/saved_models/cgcnn/best.ckpt"
torch.save(checkpoint, new_ckpt_path)

print(f"Updated checkpoint saved to {new_ckpt_path}")
