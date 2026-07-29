import os
import torch
from pytorch_forecasting import TemporalFusionTransformer

# Fix PyTorch weights_only default parameter for Lightning checkpoints if needed
_original_torch_load = torch.load
def _custom_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _custom_torch_load

def convert_ckpt_to_pt():
    ckpt_path = os.path.join("tft_checkpoints", "tft-nifty-epoch=04-val_loss=0.0056-v1.ckpt")
    pt_path = os.path.join("tft_checkpoints", "tft-nifty-epoch=04-val_loss=0.0056-v1.pt")
    
    print(f"Loading checkpoint from: '{ckpt_path}'...")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        
    model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
    model.eval()
    
    print("Saving model state and metadata to PyTorch .pt file...")
    checkpoint_payload = {
        "state_dict": model.state_dict(),
        "hparams": model.hparams,
        "model_class": "TemporalFusionTransformer",
        "full_model": model
    }
    
    torch.save(checkpoint_payload, pt_path)
    print(f"Successfully converted checkpoint to PyTorch .pt format: '{pt_path}'")
    print(f"File size: {os.path.getsize(pt_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    convert_ckpt_to_pt()
