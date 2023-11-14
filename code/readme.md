# Domain-Inspired Sharpness-Aware Minimization for Domain Generalization

## Environment Requirements

- Python 3.9.5
- PyTorch 2.0.1

## Dataset

You need to download the dataset on your own and specify the dataset path in the `configs/default.py` file. Please refer to [Domainbed repo](https://github.com/facebookresearch/DomainBed).

## Algorithm

The core operations of the algorithm are implemented in the `algorithms/DISAM.py` file.

## Example Run Command

```bash
bash ./runs/run_trainer.py --algorithm DISAM_Trainer --dataset pacs --test_domain p --lambda_weight 0.1 --rho 0.05 --lr 1e-3 --batch_size 32 --epoch 50
```

