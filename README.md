# Scale variance minimization for unsupervised domain adaptation in image segmentation

## Updates

- *06/2021*: check out our domain generalization paper [FSDR: Frequency Space Domain Randomization for Domain Generalization](https://arxiv.org/abs/2103.02370) (accepted to CVPR 2021). We design a domain adaptive panoptic segmentation network that exploits inter-style consistency and inter-task regularization for optimal domain adaptation in panoptic segmentation.[Code avaliable](https://github.com/jxhuang0508/FSDR).
- *06/2021*: check out our domain adaptation for panoptic segmentation paper [Cross-View Regularization for Domain Adaptive Panoptic Segmentation](https://arxiv.org/abs/2103.02584) (accepted to CVPR 2021). Inspired by the idea of JPEG that converts spatial images into multiple frequency components (FCs), we propose Frequency Space Domain Randomization (FSDR) that randomizes images in frequency space by keeping domain-invariant FCs (DIFs) and randomizing domain-variant FCs (DVFs) only. [Code avaliable](https://github.com/jxhuang0508/CVRN).
- *06/2021*: check out our domain adapation for object detection paper [Uncertainty-Aware Unsupervised Domain Adaptation in Object Detection](https://arxiv.org/abs/2103.00236) (accepted to IEEE TMM 2021). We design a uncertainty-aware domain adaptation network (UaDAN) that introduces conditional adversarial learning to align well-aligned and poorly-aligned samples separately in different manners. [Code avaliable](https://github.com/Dayan-Guan/UaDAN).

## Paper
![](./teaser.png)
[Scale variance minimization for unsupervised domain adaptation in image segmentation](https://www.researchgate.net/profile/Dayan-Guan/publication/347421562_Scale_variance_minimization_for_unsupervised_domain_adaptation_in_image_segmentation/links/5fdb06eb299bf1408816f709/Scale-variance-minimization-for-unsupervised-domain-adaptation-in-image-segmentation.pdf)  
 [Dayan Guan](https://scholar.google.com/citations?user=9jp9QAsAAAAJ&hl=en), [Jiaxing Huang](https://scholar.google.com/citations?user=czirNcwAAAAJ&hl=en&oi=ao),  [Xiao Aoran](https://scholar.google.com/citations?user=yGKsEpAAAAAJ&hl=en), [Shijian Lu](https://scholar.google.com/citations?user=uYmK-A0AAAAJ&hl=en)  
 School of Computer Science Engineering, Nanyang Technological University, Singapore  
 IEEE Transactions on Multimedia, 2021.
 
If you find this code useful for your research, please cite our [paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320320305677):

```
@article{guan2021scale,
  title={Scale variance minimization for unsupervised domain adaptation in image segmentation},
  author={Guan, Dayan and Huang, Jiaxing and Lu, Shijian and Xiao, Aoran},
  journal={Pattern Recognition},
  volume={112},
  pages={107764},
  year={2021},
  publisher={Elsevier}
}
```

## Abstract

We focus on unsupervised domain adaptation (UDA) in image segmentation. Existing works address this challenge largely by aligning inter-domain representations, which may lead over-alignment that impairs the semantic structures of images and further target-domain segmentation performance. We design a scale variance minimization (SVMin) method by enforcing the intra-image semantic structure consistency in the target domain. Specifically, SVMin leverages an intrinsic property that simple scale transformation has little effect on the semantic structures of images. It thus introduces certain supervision in the target domain by imposing a scale-invariance constraint while learning to segment an image and its scale-transformation concurrently. Additionally, SVMin is complementary to most existing UDA techniques and can be easily incorporated with consistent performance boost but little extra parameters. Extensive experiments show that our method achieves superior domain adaptive segmentation performance as compared with the state-of-the-art. Preliminary studies show that SVMin can be easily adapted for UDA-based image classification.

## Installation
```bash
conda create -n svmin python=3.6
conda activate svmin
conda install -c menpo opencv
```

### Prepare Dataset
* **Pascal VOC**: Download [Pascal VOC dataset](https://pjreddie.com/projects/pascal-voc-dataset-mirror) at ```UaDAN/dataset/voc```
* **Clipart1k**: Download [Clipart1k dataset](http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/cross_domain_detection/datasets/clipart.zip) at ```UaDAN/dataset/clipart``` and unzip it
```bash
$ mv tools/dataset/clipart/ImageSets dataset/clipart
```
* **Cityscapes**: Download [Cityscapes dataset](https://www.cityscapes-dataset.com) at ```UaDAN/dataset/cityscapes```
* **Mapillary Vista**: Download [Mapillary Vista dataset](https://www.mapillary.com/dataset/vistas) at ```UaDAN/dataset/vistas```

### Pre-trained models
Pre-trained models can be downloaded [here](https://github.com/Dayan-Guan/UaDAN/releases/tag/Latest) and put in ```UaDAN/pretrained_models```

### Evaluation
```bash
$ python tools/test_net.py --config-file "configs/UaDAN_Voc2Clipart.yaml" MODEL.WEIGHT "pretrained_models/UaDAN_Voc2Clipart.pth"
```

```bash
python tools/test_net.py --config-file "configs/UaDAN_City2Vistas.yaml" MODEL.WEIGHT "pretrained_models/UaDAN_City2Vistas.pth"
```

### Training
```bash
python tools/train_net.py --config-file "configs/UaDAN_voc2clipart.yaml"
```

```bash
python tools/test_net_all.py --config-file "configs/UaDAN_voc2clipart.yaml"
```

## Acknowledgements
This codebase is heavily borrowed from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and [Domain-Adaptive-Faster-RCNN-PyTorch](https://github.com/krumo/Domain-Adaptive-Faster-RCNN-PyTorch)

## Contact
If you have any questions, please contact: dayan.guan@ntu.edu.sg
