# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from models.diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
import argparse
import random
from PIL import Image
from apz_models import apz_DiT_models

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    
    latent_size = args.image_size // 8
    model = apz_DiT_models[args.model](
        # input_size=latent_size,
        # num_classes=args.num_classes
        input_size = 256 // 8,
        num_classes = 1000
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!

    # # 打印模型结构
    # for name, param in model.named_parameters():
    #     print(name, param.size())


    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    class_labels = [0, 1, 2, 3, 4, 5, 6, 7]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)
    s = torch.full((n,), args.stage, device=device, dtype=torch.long)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    s_null = torch.tensor([4] * n, device=device, dtype=torch.long)
    s = torch.cat([s, s_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, s=s)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample
    # 分离ckpt中的xxx.pt
    ckpt_name = args.ckpt.split('/')[-1].split('.')[0]

    # Save and display images:
    save_image(samples, f"sample_{ckpt_name}.png", nrow=4, normalize=True, value_range=(-1, 1))





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(apz_DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image_size", type=int, choices=[64, 128, 192, 256, 512], default=256)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--inference_classes", type=int, default=1000)
    parser.add_argument("--stage", type=int, default=3)
    args = parser.parse_args()
    main(args)
