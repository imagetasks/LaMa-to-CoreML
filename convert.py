import torch
import sys
import os
from omegaconf import OmegaConf
import yaml
import torch.nn as nn
import numpy as np

lama_pretrained_models_path = "LaMa_models/big-lama" # point to a folder with 'config.yaml' and 'models/best.ckpt'
lama_input_size = 1024 #px
lama_repository_path = "LaMa" # where to download LaMa repository / your own fork source code


print("\n" + "="*80)
print("LaMa to CoreML converter")
print("="*80)

print(f"Pretrained model path: {lama_pretrained_models_path}")
print(f"Input size: {lama_input_size}px")
print("="*80 + "\n")

if not os.path.exists(lama_repository_path):
    print("Local LaMa repository not found, fetching from GitHub...")
    os.system(f"git clone https://github.com/advimman/lama {lama_repository_path}")

if not os.path.exists(lama_pretrained_models_path):
    print("ERROR: Model not found. Download pretrained models at: https://drive.google.com/drive/folders/1B2x7eQDgecTL0oh3LSIBDGj0fTxs6Ips")
    exit()

sys.path.append(lama_repository_path)

from saicinpainting.training.trainers import load_checkpoint

class LaMaInferenceWrapper(torch.nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, image_0_255, mask_0_255):
        image = image_0_255 / 255.0
        mask = (mask_0_255 > 0).float()
        masked_img = image * (1 - mask)
        input_tensor = torch.cat([masked_img, mask], dim=1)
        prediction = self.generator(input_tensor)
        result = image * (1 - mask) + prediction * mask
        return torch.clamp(result * 255.0, 0, 255)

def load_lama_model(model_dir):
    config_path = f"{model_dir}/config.yaml"
    with open(config_path, "r") as f:
        model_config = OmegaConf.create(yaml.safe_load(f))

    model_config.training_model.predict_only = True
    model_config.visualizer.kind = "noop"

    checkpoint_path = f"{model_dir}/models/best.ckpt"

    model = load_checkpoint(
        model_config,
        checkpoint_path,
        strict=False,
        map_location="cpu"
    )
    model.eval()
    
    if hasattr(model, 'generator'):
        model.generator.eval()
    return model

def prepare_for_coreml_conversion(wrapper):
    coreml_image = torch.rand(1, 3, lama_input_size, lama_input_size) * 255
    coreml_mask = torch.zeros(1, 1, lama_input_size, lama_input_size)
    coreml_mask[0, 0, lama_input_size//4:3*lama_input_size//4, lama_input_size//4:3*lama_input_size//4] = 255

    traced = torch.jit.trace(wrapper, (coreml_image, coreml_mask))

    with torch.no_grad():
        traced_output = traced(coreml_image, coreml_mask)
    
    return traced, coreml_image, coreml_mask

def convert_to_coreml(traced_model, coreml_image, coreml_mask):
    try:
        import coremltools as ct

        mlmodel = ct.convert(
            traced_model,
            convert_to="mlprogram",
            inputs=[
                ct.ImageType(
                    name="image",
                    shape=coreml_image.shape,
                    scale=1,
                    bias=[0, 0, 0],
                    color_layout="RGB"
                ),
                ct.ImageType(
                    name="mask",
                    shape=coreml_mask.shape,
                    scale=1,
                    bias=[0, 0, 0],
                    color_layout="G"
                )
            ],
            outputs=[
                ct.ImageType(
                    name="output",
                    color_layout="RGB",
                )
            ],
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS16,
            compute_units=ct.ComputeUnit.ALL
        )

        mlmodel.author = "LaMa"
        mlmodel.short_description = "LaMa Image Inpainting, Resolution-robust Large Mask Inpainting with Fourier Convolutions"
        mlmodel.version = "1.0"
        mlmodel.license = "https://github.com/advimman/lama/blob/main/LICENSE"
        
        model_path = "LaMa_Inpainting.mlpackage"
        mlmodel.save(model_path)
        print(f"DONE! CoreML model saved as: {model_path}")
        
        return mlmodel, model_path
        
    except ImportError:
        print("CoreML tools not available.")
        return None, None
    except Exception as e:
        print(f"CoreML conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


print("Loading model...")
model = load_lama_model(lama_pretrained_models_path)
if hasattr(model, "generator"):
    base_model = model.generator
elif hasattr(model, "model"):
    base_model = model.model
else:
    raise RuntimeError("Cannot find generator inside Lightning module")

wrapper = LaMaInferenceWrapper(base_model)
wrapper.eval()
print("Tracing model...")
traced_model, coreml_image, coreml_mask = prepare_for_coreml_conversion(wrapper)
print("Converting to CoreML...")
coreml_model, coreml_path = convert_to_coreml(traced_model, coreml_image, coreml_mask)