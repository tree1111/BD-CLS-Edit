# BD-CLS-Edit

This repository contains the code for the paper "[Counterfactual Image Editing with Disentangled Causal Latent Space](https://causalai.net/r137.pdf)" by Yushu Pan and Elias Bareinboim.

## Installation

```bash
pip install -r requirements.txt
```

Requirements: Python 3.8+, PyTorch >= 2.0, diffusers >= 0.25.0, transformers >= 4.35.0. See `requirements.txt` for full dependencies.

## Repository Structure

```text
├── src/
│   ├── colormnistbar/       # Colored MNIST and Bars experiments (Sec 5.1) for ctf-consistent check
│   ├── stablediffusion/     # Real-world scenario experiments (Sec 5.2)
│   └── ds/                  # Causal data structure and dataloader
└── dat/
    ├── cg/                  # Causal diagrams
    ├── img/                 # Training data for Colored MNIST and Bars; Real-world scenario initial images
    └── prompts/             # Source and target prompts for editing Real-world scenario images
```

## Usage

### Colored MNIST and Bars

**(1) Generate data:**

```bash
python -m src.ds.cmnistbar_data_loader --data-path dat/img
```

**(2) Train BD-CLS:**

```bash
python -m src.colormnistbar.main --gpu {DEVICE} -G {Graph}
```

- `Graph = "full-ncm"`: train the full NCM
- `Graph = "cls-digit"`: train BD-CLS for intervening digit
- `Graph = "cls-color"`: train BD-CLS for intervening color

**(3) Evaluate BD-CLS on real-world images (generate initial and counterfactual images):**

```bash
python -m src.colormnistbar.main --gpu {DEVICE} -G {Graph} --eval -c {condition} -do {intervention}
```

Example: `-c digit=0,digit-color=red -do digit=9`

Keys: `digit`, `digit-color`, `bar-color`, `bar-width`

### Real-World Scenarios

**(1) Prepare prompts:** Create a JSON file in `dat/prompts/` with the **same base name** as your initial image. For `sunny.png`, create `dat/prompts/sunny.json`:

```json
{
  "source": "A painting of a middle-aged woman standing in a simple garden",
  "target": "A painting of a middle-aged woman standing in a simple garden in a heavy rain"
}
```

**(2) Place your initial image** in `dat/img/RealWorld/` (or set `--image-root-path`).

**(3) Run BD-CLS-Edit:**

```bash
python -m src.stablediffusion.main --initial-image sunny.png
```

**Optional arguments:**
- `--device-num`: GPU device
- `--image-root-path`: Path to initial images (default: `dat/img/RealWorld/`)
- `--out-path`: Output directory
- `--num-diffusion-steps`: Number of diffusion steps (default: 200)
- `--opt-iter`: Max optimization iterations per timestep (default: 5)


## Third-Party Code and Acknowledgements

This project incorporates and adapts code from the following repositories:

1. **NCM Counterfactuals** by Kevin Xia & Yushu Pan  
   https://github.com/CausalAILab/NCMCounterfactuals  

2. **DDPM inversion**  
   https://github.com/inbarhub/DDPM_inversion  

3. **DDS** from Google  
   https://github.com/google/prompt-to-prompt/blob/main/DDS_zeroshot.ipynb  

### License Compliance

- Parts of this project adapt code licensed under the **MIT License** and the **Apache License 2.0**.  
- Each source retains its original copyright and license notices, which are preserved in this repository.


## Citation
If you find this code useful, please consider citing our paper:

Pan, Yushu, and Elias Bareinboim. "Counterfactual image editing with disentangled causal latent space." The Thirty-ninth Annual Conference on Neural Information Processing Systems. 2025.
