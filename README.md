

<p align="center">
  <h1 align="center"><strong> <img src="assets/icons2.png" width="30" height="30">  PEAR: Pixel-aligned Expressive humAn mesh Recovery</strong></h1>

<p align="center">
  <a href="https://openreview.net/profile?id=%7EJiahao_Wu11">Jiahao Wu</a></sup>,</span> 
  <a href="http://liuyunfei.net/">Yunfei Liu ✉</a></sup>,</span>
  <a href="https://scholar.google.com/citations?hl=en&user=Xf5_TfcAAAAJ">Lijian Lin</a>, 
  <a href="https://scholar.google.com/citations?hl=en&user=qhp9rIMAAAAJ">Ye Zhu</a>, 
  <a> Lei Zhu</a></sup>,
  <a>Jingyi Li</a></sup>,
  <a href="https://yu-li.github.io/">Yu Li</a>
</p>

  <p align="center">
    <em>International Digital Economy Academy (IDEA)</em>
  </p>

</p>

<div id="top" align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2602.22693-b31b1b.svg)](https://wujh2001.github.io/PEAR/)
[![](https://img.shields.io/badge/%F0%9F%9A%80%20-Project%20Page-blue)](https://wujh2001.github.io/PEAR//)
[![Youtube](https://img.shields.io/badge/▶️-Youtube-red)](https://www.youtube.com/watch?v=FFnuDwXGA_M)
[![Youtube](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live_Demo-blue)](https://huggingface.co/spaces/BestWJH/PEAR)

</div>

<div align="center">
    <img src="assets/teaser.png">
</div>


## 📰 News
**[2026.02.02]** The inference code and the first version of the PEAR model have been released!

**[2026.02.02]** Paper release of our PEAR on arXiv!


## 💡 Overview

<div align="center">
    <img src='assets/method.png'/>
</div>

We propose PEAR, a unified framework for real-time expressive 3D human mesh recovery. It is the first method capable of simultaneously predicting EHM-s parameters at 100 FPS.

## ⚡ Quick Start

### First, clone this repository to your local machine, and install the dependencies. 

```bash
git clone --recursive https://github.com/Pixel-Talk/PEAR.git
cd PEAR
conda create -n pear python=3.9.22
conda activate pear
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
pip install chumpy  --no-build-isolation
```

### Second, you need to download three human models (FLAME. MANO, SMPLX) from [huggingface url](https://huggingface.co/spaces/BestWJH/PEAR/tree/main/assets) into the current `assets` folder.


### Finally, you can have a try !

For video inference, you can simply run our visualization interface via
```bash
python app.py 
```
For images inference, you can simply run our visualization interface via
```bash
python inference_images.py --input_path example/images 
```




## 🤗 Citation
If you find this repository useful for your research, please use the following BibTeX entry for citation.

    @misc{wu2026pearpixelalignedexpressivehuman,
      title={PEAR: Pixel-aligned Expressive humAn mesh Recovery}, 
      author={Jiahao Wu and Yunfei Liu and Lijian Lin and Ye Zhu and Lei Zhu and Jingyi Li and Yu Li},
      year={2026},
      eprint={2601.22693},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.22693}, 
    }

## Acknowledgements 

We would like to thank the authors of prior works, including FLAME, SMPL-X, SMPL, MANO, SMPLest-X, Multi-HMR, and SAM3D-Body.


## License
See the [LICENSE](./LICENSE.txt) file for details about the license under which this code is made available.
