## LaMa to CoreML converter

<p align="center">
  <img src="https://github.com/imagetasks/LaMa-to-CoreML/blob/main/image.png?raw=true">
</p>

This repo contains a python script for converting [LaMa](https://github.com/advimman/lama/) pretrained models (.ckpt files) to Apple's Core ML model format. Pretrained models can be downloaded at [Google Drive](https://drive.google.com/drive/folders/1B2x7eQDgecTL0oh3LSIBDGj0fTxs6Ips) (link is taken from the official LaMa repo). Use **LaMa_models.zip** for *big-lama*, *lama-places*, *lama-celeba*, etc. **big-lama.zip** if you need to convert *big-lama* only. Alternatively, you can use your own ckpt files. Just point the script to a folder with 'config.yaml' and 'models/best.ckpt'. See **Customize script** section for details.

The script does not require *IOPanit* (marked as archived) for the conversion.

### Instructions

1. Create a Conda environment:
    ```sh
    conda create -n lamatocoreml python=3.9 -y
    conda activate lamatocoreml
    pip install -r requirements.txt
    ```

2. Customize script:
    ```python
    lama_pretrained_models_path = "LaMa_models/big-lama" # point to a folder with 'config.yaml' and 'models/best.ckpt'
    lama_input_size = 1024 #px
    ```

3. Run the conversion script:
    ```sh
    python convert.py
    ```

### Acknowledgements

Authors of LaMa:

[[Project page](https://advimman.github.io/lama-project/)] [[arXiv](https://arxiv.org/abs/2109.07161)] [[Supplementary](https://ashukha.com/projects/lama_21/lama_supmat_2021.pdf)] [[BibTeX](https://senya-ashukha.github.io/projects/lama_21/paper.txt)] [[Casual GAN Papers Summary](https://www.casualganpapers.com/large-masks-fourier-convolutions-inpainting/LaMa-explained.html)]

Support us by downloading [Pixea image viewer](https://www.imagetasks.com/pixea) for macOS.
