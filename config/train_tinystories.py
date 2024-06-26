# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = False
wandb_project = ''
wandb_run_name=''
neptune_api_token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhMzM2OGRiMi1hMTIyLTQ5NWMtYWZjOS1kZjcyZjZjZTI0ZjYifQ=='


# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
dataset = "tinystories"
batch_size = 12
block_size = 256
gradient_accumulation_steps = 1

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0


# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000
learning_rate = 1e-3 # with baby networks can afford to go a bit higher
min_lr = 5e-5


# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10


# weight decay
weight_decay = 1e-5
