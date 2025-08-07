
# Implicit Neural Representation Learning for Hyperspectral Image Super-Resolution (IEEE Transactions on Geoscience and Remote Sensing)
This is the source code on CAVE Dataset for the paper "[Implicit Neural Representation Learning for Hyperspectral Image Super-Resolution](https://ieeexplore.ieee.org/abstract/document/9991174)" *(INR-HSISR)*.


<hr />

> **Abstract:** *Hyperspectral image (HSI) super-resolution without additional auxiliary image remains a constant challenge due to its high-dimensional spectral patterns, where learning an effective spatial and spectral representation is a fundamental issue. Recently, Implicit Neural Representations (INRs) are making strides as a novel and effective representation, especially in the reconstruction task. Therefore, in this work, we propose a novel HSI reconstruction model based on INR which represents HSI by a continuous function mapping a spatial coordinate to its corresponding spectral radiance values. In particular, as a specific implementation of INR, the parameters of the parametric model are predicted by a hypernetwork that operates on feature extraction using a convolution network. It makes the continuous functions map the spatial coordinates to pixel values in a content-aware manner. Moreover, periodic spatial encoding is deeply integrated with the reconstruction procedure, which makes our model capable of recovering more high frequency details. To verify the efficacy of our model, we conduct experiments on three HSI datasets (CAVE, NUS, and NTIRE2018). Experimental results show that the proposed model can achieve competitive reconstruction performance in comparison with the state-of-the-art methods. In addition, we provide an ablation study on the effect of individual components of our model. We hope this paper could server as a potent reference for future research.* 
<hr />



## Implicit Neural Representation Learning

<p align = "center">    
<img  src="https://github.com/kaviezhang/INR-HSISR/blob/main/result/intro.png" width="300" />
</p>

## Model Architecture

<p align = "center">    
<img  src="https://github.com/kaviezhang/INR-HSISR/blob/main/result/inr.png" width="600" />
</p>



## Citation
- If you find this code useful, please consider citing
```
@article{zhang2022implicit,
  title={Implicit neural representation learning for hyperspectral image super-resolution},
  author={Zhang, Kaiwei and Zhu, Dandan and Min, Xiongkuo and Zhai, Guangtao},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={61},
  pages={1--12},
  year={2022},
  publisher={IEEE}
}
```
