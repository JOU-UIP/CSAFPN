Cross-Scale Attention Feature Pyramid Network for Challenge Underwater Object Detection
---------------------

Miao yang, Jinyang Zhong, Hansen Zhang, Can Pan, Xinmiao Gao, Chenglong Gong


Abstract
----------------

Underwater object detection (UOD) is more difficult than common object detection tasks due to the noise caused by irrelevant objects and textures, and the scale variation. These difficulties pose a higher challenge to the feature extraction capability of the detectors. Feature pyramid network enhances the scale detection capability of detectors, while attention mechanisms effectively suppress irrelevant features. We present here a cross-scale attention feature pyramid network (CSAFPN) for UOD. A feature fusion guided (FFG) module is incorporated in the CSAFPN, which constructs cross-scale context information and simultaneously guides the enhancement of all feature maps. Compared to existing FPN-like architectures, CSAFPN excels not only in capturing cross-scale long-range dependencies but also in acquiring compact multi-scale feature maps that specifically emphasize target regions. Extensive experiments on the Brackish2019 dataset show that CSAFPN can achieve consistent improvements on various backbones and detectors. Moreover, FFG can be seamlessly integrated into any FPN-like architecture, offering a cost-effective improvement in UOD, resulting in a 1.4\% AP increase for FPN, a 1.3\% AP increase for PANet, and a 1.4\% AP increase for NAS-FPN.

## Dependencies

- Python==3.7.12
- PyTorch==1.10.0+cu111
- mmdetection==2.4.0
- mmcv==1.1.5

Installation
-------------

The basic installation follows with [mmdetection](https://github.com/open-mmlab/mmdetection). It is recommended to use manual installation.

Datasets
----------

Brackish download link: [BaiduYun](https://pan.baidu.com/s/1D05P2lYlID1QA9hB49MxWA?pwd=55u3 )

The structure of this dataset is:

```
├── Brackish dataset
│   ├── annotations
│   │   ├── train.json
│   │   ├── val.json
│   │   ├── test.json
│   ├── images
│   │   ├── train
│   │   ├── val
│   │   ├── test
```

Training
--------------

```shell
python tools/train.py work_dir/csafpn1/optics.py
```


Testing
-----------

```shell
python tools/test.py work_dir/csafpn1/optics.py <path/to/checkpoints> --eval bbox
```

Results on different detector
---------

| Backbone  | Neck   | detector      | mAP  | AP50 | AP75 |
| --------- | ------ | ------------- | ---- | ---- | ---- |
| ResNet-50 | FPN    | Faster R-CNN  | 76.9 | 96.8 | 87.9 |
| ResNet-50 | CSAFPN | Faster R-CNN  | 77.9 | 97.4 | 87.8 |
| ResNet-50 | FPN    | Mask R-CNN    | 76.8 | 96.7 | 87.8 |
| ResNet-50 | CSAFPN | Mask R-CNN    | 78.8 | 98.2 | 89.2 |
| ResNet-50 | FPN    | Cascade R-CNN | 81.1 | 98.0 | 90.8 |
| ResNet-50 | CSAFPN | Cascade R-CNN | 82.5 | 98.2 | 92.8 |

## Results on different backbone

| ne              | Neck   | detector      | mAP  | AP50 | AP75 |
| --------------- | ------ | ------------- | ---- | ---- | ---- |
| ResNet 18       | FPN    | Cascade R-CNN | 74.5 | 95.2 | 83.1 |
| ResNet 18       | CSAFPN | Cascade R-CNN | 76.0 | 95.9 | 86.0 |
| ResNet 50       | FPN    | Cascade R-CNN | 81.1 | 98.0 | 90.8 |
| ResNet 50       | CSAFPN | Cascade R-CNN | 82.5 | 98.2 | 92.8 |
| ResNet 101      | FPN    | Cascade R-CNN | 81.0 | 97.8 | 91.6 |
| ResNet 101      | CSAFPN | Cascade R-CNN | 82.2 | 98.2 | 92.2 |
| ResNeXt50_32×4d | FPN    | Cascade R-CNN | 74.0 | 95.4 | 83.8 |
| ResNeXt50_32×4d | CSAFPN | Cascade R-CNN | 74.6 | 95.8 | 84.5 |


License
--------

This project is released under the [Apache 2.0 license](LICENSE)
