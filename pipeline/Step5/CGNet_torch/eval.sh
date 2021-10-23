export PYTHONPATH=CGNet-PP/pipeline/Step2.5/CGNet_torch

python CGNet-PP/pipeline/Step2.5/CGNet_torch/train.py \
    --test-only \
    --dataloader val \
    --batch_size 1 \
    --shuffle False \
    --test_max_iters 0
