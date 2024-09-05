# Tiled Diffusion & VAE for ComfyUI

Check out the [SD-WebUI extension](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/) for more information.

This extension enables **large image drawing & upscaling with limited VRAM** via the following techniques:

- Reproduced SOTA Tiled Diffusion methods
    - [MultiDiffusion](https://github.com/omerbt/MultiDiffusion) <a href="https://arxiv.org/abs/2302.08113"><img width="32" alt="MultiDiffusion Paper" src="https://github.com/shiimizu/ComfyUI-TiledDiffusion/assets/54494639/b753b7f6-f9c0-405d-bace-792b9bbce5d5"></a>
    - [Mixture of Diffusers](https://github.com/albarji/mixture-of-diffusers) <a href="https://arxiv.org/abs/2302.02412"><img width="32" alt="Mixture of Diffusers Paper" src="https://github.com/shiimizu/ComfyUI-TiledDiffusion/assets/54494639/b753b7f6-f9c0-405d-bace-792b9bbce5d5"></a>
- pkuliyi2015 & Kahsolt's Tiled VAE algorithm
- ~~pkuliyi2015 & Kahsolt's TIled Noise Inversion method~~

> [!NOTE]  
> Sizes/dimensions are in pixels and then converted to latent-space sizes.


## Features

- [x] Supported models
    - [x] SD1.x, SD2.x, SDXL, SD3
    - [x] FLUX
- [x] ControlNet support
- [ ] ~~StableSR support~~
- [ ] ~~Tiled Noise Inversion~~
- [x] Tiled VAE
- [ ] Regional Prompt Control
- [x] Img2img upscale
- [x] Ultra-Large image generation

## Tiled Diffusion

<div align="center">
  <img width="500" alt="Tiled_Diffusion" src="https://github.com/shiimizu/ComfyUI-TiledDiffusion/assets/54494639/7cb897a3-a645-426f-8742-d6ba5cf04b64">
</div>

> [!TIP]  
> * Set `tile_overlap` to 0 and `denoise` to 1 to see the tile seams and then adjust the options to your needs.
> * Increase `tile_batch_size` to increase speed (if your machine can handle it).
> * Use the [colorfix node](https://github.com/gameltb/Comfyui-StableSR) if your colors look off.

### Options

| Name              | Description                                                  |
|-------------------|--------------------------------------------------------------|
| `method`          | Tiling [strategy](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/fbb24736c9bc374c7f098f82b575fcd14a73936a/scripts/tilediffusion.py#L39-L46).  |
| `tile_width`      | Tile's width                                                 |
| `tile_height`     | Tile's height                                                |
| `tile_overlap`    | Tile's overlap                                               |
| `tile_batch_size` | The number of tiles to process in a batch                    |

### How can I specify the tiles' arrangement?

If you have the [Math Expression](https://github.com/pythongosssss/ComfyUI-Custom-Scripts#math-expression) node (or something similar), you can use that to pass in the latent that's passed in your KSampler and divide the `tile_height`/`tile_width` by the number of rows/columns you want.

`C` = number of columns you want  
`R` = number of rows you want

`pixel width of input image or latent // C` = `tile_width`  
`pixel height of input image or latent // R` = `tile_height`

<img width="800" alt="Tile_arrangement" src="https://github.com/shiimizu/ComfyUI-TiledDiffusion/assets/54494639/9952e7d8-909e-436f-a284-c00f0fb71665">

### SpotDiffusion

[Paper](https://arxiv.org/abs/2407.15507)

A tiling algorithm that attempts to eliminate seams by randomly shifting the denoise window per timestep. It is mainly used for fast inferences by setting `tile_overlap` to 0; otherwise, it's better to stick with the other tiling strategies as they produce better outputs.

This additional feature is experimental, in testing,  and subject to change.

## Tiled VAE

<div align="center">
  <img width="900" alt="Tiled_VAE" src="https://github.com/shiimizu/ComfyUI-TiledDiffusion/assets/54494639/b5850e03-2cac-49ce-b1fe-a67906bf4c9d">
</div>

<br>

The recommended tile sizes are given upon the creation of the node based on the available VRAM.   

> [!NOTE]  
> Enabling `fast` for the decoder may produce images with slightly higher contrast and brightness.

### Options

| Name        | Description                                                                                                                                  |
|-------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| `tile_size` |  <blockquote>The image is split into tiles, which are then padded with 11/32 pixels' in the decoder/encoder.</blockquote>                                 |
| `fast`      |  <blockquote><p>When Fast Mode is disabled:</p> <ol> <li>The original VAE forward is decomposed into a task queue and a task worker, which starts to process each tile.</li> <li>When GroupNorm is needed, it suspends, stores current GroupNorm mean and var, send everything to RAM, and turns to the next tile.</li> <li>After all GroupNorm means and vars are summarized, it applies group norm to tiles and continues. </li> <li>A zigzag execution order is used to reduce unnecessary data transfer.</li> </ol> <p>When Fast Mode is enabled:</p> <ol> <li>The original input is downsampled and passed to a separate task queue.</li> <li>Its group norm parameters are recorded and used by all tiles&#39; task queues.</li> <li>Each tile is separately processed without any RAM-VRAM data transfer.</li> </ol> <p>After all tiles are processed, tiles are written to a result buffer and returned.</p></blockquote> |
| `color_fix` | <blockquote>Only estimate GroupNorm before downsampling, i.e., run in a semi-fast mode.</blockquote><p>Only for the encoder. Can restore colors if tiles are too small.</p>  |



## Workflows

The following images can be loaded in ComfyUI.


<div align="center">
  <img alt="ComfyUI_07501_" src="https://github.com/shiimizu/ComfyUI-TiledDiffusion/assets/54494639/c3713cfb-e083-4df4-a310-9467827ee666">
  <p>Simple upscale.</p>
</div>

<br>

<div align="center">

  <img alt="ComfyUI_07503_" src="https://github.com/shiimizu/ComfyUI-TiledDiffusion/assets/54494639/b681b617-4bb1-49e5-b85a-ef5a0f6e4830">
  <p>4x upscale. 3 passes.</p>
</div>

## License
Great thanks to all the contributors! 🎉🎉🎉   
The implementation of MultiDiffusion, Mixture of Diffusers, and Tiled VAE code is currently under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/) since it was borrowed from the wonderful [SD-WebUI extension](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/). Anything else GPLv3.

## Citation

```bibtex
@article{jimenez2023mixtureofdiffusers,
  title={Mixture of Diffusers for scene composition and high resolution image generation},
  author={Álvaro Barbero Jiménez},
  journal={arXiv preprint arXiv:2302.02412},
  year={2023}
}
```

```bibtex
@article{bar2023multidiffusion,
  title={MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation},
  author={Bar-Tal, Omer and Yariv, Lior and Lipman, Yaron and Dekel, Tali},
  journal={arXiv preprint arXiv:2302.08113},
  year={2023}
}
```
