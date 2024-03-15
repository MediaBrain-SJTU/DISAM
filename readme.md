# Domain-Inspired Sharpness-Aware Minimization Under Domain Shifts

This repository contains the implementation details for the paper "Domain-Inspired Sharpness-Aware Minimization Under Domain Shifts," accepted at the International Conference on Learning Representations (ICLR) 2024.

## Environment Requirements

![Language](https://img.shields.io/badge/language-python-brightgreen)


![Python](https://img.shields.io/badge/Python->=3.9.5-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-=2.0.1-orange)
![NumPy](https://img.shields.io/badge/NumPy->=1.23.5-orange)


## Usage

### Dataset ![repo](https://img.shields.io/badge/repo-DomainBed-informational)

You need to download the dataset on your own and specify the dataset path in the `code/configs/default.py` file. Please refer to [Domainbed repo](https://github.com/facebookresearch/DomainBed).


### Algorithm

The core operations of the algorithm are implemented in the `code/algorithms/DISAM.py` file.

### Example Run Command

```bash
bash ./runs/run_trainer.py --algorithm DISAM_Trainer --dataset pacs --test_domain p --lambda_weight 0.1 --rho 0.05 --lr 1e-3 --batch_size 32 --epoch 50
```

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{zhang2024domaininspired,
  title={Domain-Inspired Sharpness Aware Minimization Under Domain Shifts},
  author={Ruipeng Zhang and Ziqing Fan and Jiangchao Yao and Ya Zhang and Yanfeng Wang},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=I4wB3HA3dJ}
}
```

## License

![License](https://img.shields.io/badge/license-MIT-yellow)

This project is licensed under the MIT License.
