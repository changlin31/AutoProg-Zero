# Efficient Training of Large Vision Models via Advanced Automated Progressive Learning

[Efficiently training large vision models through advanced automated progressive learning](https://arxiv.org/abs/2410.00350) is a continuation of previous work [Automated Progressive Learning for Efficient Training of Vision Transformers](https://arxiv.org/pdf/2203.14509) [[code](https://github.com/changlin31/AutoProg)].

![post](./sample_grid.png)

We use AutoProg-Zero for fine-tuning on Vision Transformers(ViTs), Diffusion Transformers(DiTs) and Stable Diffusion(SD) Models. AutoProg-Zero shows excellent fine-tuning performance on different models and different datasets, significantly shortens the fine-tuning time, which outperforms other training methods on the class-conditional Oxford Flowers, CUB-Brid, ArtBench and Stanford Cars.



## Requirements

To create and activate a suitable [conda](https://conda.io/) environment named `APZ`, follow these steps:

```
conda env create -f environment.yaml
conda activate APZ
```



## Efficient Fine-tuning on Class-conditional Image Generation
For efficient fine-tuning on class-conditional image generation, we applied AutoProg-Zero to [DiT](https://www.wpeebles.com/DiT). Our experiment was running on 8 A800 GPUs.

### Training

To fine-tuning [DiT-XL/2 (256x256)](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt)  with `N` GPUs on one node:

```sh
export DATASET_DIR = "path-to-dataset"
export RESULTS_DIR = "path-to-results"

torchrun --nnodes=1 --nproc_per_node=N train_apz.py \
  --model="DiT-XL/2" \
  --data_path=$DATASET_DIR \
  --gradient_accumulation_steps=1 \
  --results_dir=$RESULTS_DIR \
  --global_batch_size=256 \
  --max_train_steps=240_000 \
  --ckpt_every=15000 \
  --auto_grow \
  --search_epochs=1 \

```

Then you can sample by loading your checkpoint:

```bash
python sample_apz.py --image_size 256 --ckpt "path-to-checkpoint.pt"
```



## Evaluation

You can sample a large number of images and generate a .npz file containing a large number of samples, which can directly use [ADM's TensorFlow evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to calculate FID. For example, you can load `path-to-checkpoint.pt` and sample 5k images across N GPUs.

```bash
torchrun --nnodes=1 --nproc_per_node=N sample_apz_ddp.py --image_size 256 --num_classes 200 --num_fid_samples 5000 --ckpt "path-to-checkpoint.pt"
```

You can also set `--stage=k` to specify the SID embedding for the k-th stage.



## Citation

If you use our code for your paper, please cite:

```
@article{li2024efficient,
  title={Efficient Training of Large Vision Models via Advanced Automated Progressive Learning},
  author={Li, Changlin and Zhang, Jiawei and Lin, Sihao and Yang, Zongxin and Liang, Junwei and Liang, Xiaodan and Chang, Xiaojun},
  journal={arXiv preprint arXiv:2410.00350},
  year={2024}
}

@inproceedings{li2022autoprog,
  author = {Li, Changlin and 
            Zhuang, Bohan and 
            Wang, Guangrun and
            Liang, Xiaodan and
            Chang, Xiaojun and
            Yang, Yi},
  title = {Automated Progressive Learning for Efficient Training of Vision Transformers},
  booktitle = {CVPR},
  year = 2022,
}
```
