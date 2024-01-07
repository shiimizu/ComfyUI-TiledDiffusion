# Tiled Diffusion & VAE for ComfyUI

See [this](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/) for more info.

The extension enables **large image drawing & upscaling with limited VRAM** via the following techniques:

1. Two SOTA diffusion tiling algorithms: [Mixture of Diffusers](https://github.com/albarji/mixture-of-diffusers) and [MultiDiffusion](https://github.com/omerbt/MultiDiffusion)
2. pkuliyi2015's Tiled VAE algorithm.
3. ~~pkuliyi2015's TIled Noise Inversion for better upscaling.~~

> [!NOTE]  
> Sizes are in pixel-space that then get converted into latent-space sizes.

## Features
- [x] SDXL model support
- [x] ControlNet support
- [ ] ~~StableSR support~~
- [ ] Tiled Noise Inversion
- [x] Tiled VAE
- [ ] Regional Prompt Control
- [x] Img2img upscale
- [x] Ultra-Large image generation

Some conditioning nodes aren't working at the moment like SetArea or GLIGEN.