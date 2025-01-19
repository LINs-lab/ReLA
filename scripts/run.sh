
# Experiment: Train and evaluate ReLA+BYOL with a 20% data ratio, using CLIP as the prior model
python main.py --method byol --model resnet18_modified --dataset CIFAR10 --use_rela clip --data_ratio 0.2 --devices [0]
python eval.py --method byol --model resnet18_modified --dataset CIFAR10 --use_rela clip --data_ratio 0.2 --devices [0]

# Experiment: Train and evaluate ReLA+BYOL with a 20% data ratio, using a randomly initialized model as the prior
python main.py --method byol --model resnet18_modified --dataset CIFAR10 --use_rela rand --data_ratio 0.2 --devices [0]
python eval.py --method byol --model resnet18_modified --dataset CIFAR10 --use_rela rand --data_ratio 0.2 --devices [0]

# Baseline 2: Train and evaluate BYOL using a 20% data ratio without ReLA
python main.py --method byol --model resnet18_modified --dataset CIFAR10 --use_rela none --data_ratio 0.2 --devices [0]
python eval.py --method byol --model resnet18_modified --dataset CIFAR10 --use_rela none --data_ratio 0.2 --devices [0]

# Baseline 3: Train and evaluate BYOL using the full dataset (100% data ratio) without ReLA
python main.py --method byol --model resnet18_modified --dataset CIFAR10 --use_rela none --data_ratio 1.0 --devices [0]
python eval.py --method byol --model resnet18_modified --dataset CIFAR10 --use_rela none --data_ratio 1.0 --devices [0]