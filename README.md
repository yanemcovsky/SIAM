# Smooth Inference of Adverserially trained Models

To  adversarial train CPNI

```python3 train.py --weight-noise --cpni --adv --smooth smooth_method --attack attack_type --schedule 200 300```

To  adversarial train PNI

```python3 train.py --pni --adv --smooth smooth_method --attack attack_type --schedule 200 300```

To run attack

```python3 run_attack.py  --weight-noise --cpni --smooth smooth_method --attack attack_type  --resume model_path```

To run transfer attack

```python3 run_attack_transfer.py  --weight-noise --cpni --smooth smooth_method --attack attack_type  --resume model_path  --trans model_path```

attack type options

```pgd/epgd```

smooth method options

```mcpredict/mcepredict/mclogits/mceattack```
