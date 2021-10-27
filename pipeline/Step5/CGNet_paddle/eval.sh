export PYTHONPATH=CGNet-PP/pipeline/Step2.5/CGNet_paddle

python CGNet-PP/pipeline/Step2.5/CGNet_paddle/train.py \
    --test-only \
    --dataloader val \
    --batch_size 1 \
    --shuffle False \
    --num_workers 0 \
    --test_max_iters 0
