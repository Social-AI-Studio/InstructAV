## InstructAV: Instruction Fine-tuning Large Language Models for Authorship Verification

A public repository containing datasets and code for the paper "InstructAV: Instruction Fine-tuning Large Language Models for Authorship Verification"

### Installation
> pip install -r code/requirements.txt

### Dataset
The dataset includes samples selected from the IMDB, Twitter, and Yelp datasets. 

Files labeled with 'rebuttal' correspond to datasets under the 'Classification setting', which comprise randomly selected samples that have not passed consistency verification. 

The remaining files are associated with the 'Classification and Explanation setting', representing samples where explanation labels have successfully undergone consistency verification.

### Training LoRA modules
After preparing all data for relevant tasks, we train individual modules for each task. We leverage the parameter-efficient technique, low-rank adaptation (LoRA), to tune the LLMs. To do the finetune, please run the script as the following:

Example usage for multiple GPUs:

> WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=3192 finetune.py \
    --base_model 'yahma/llama-7b-hf' \
    --data_path 'math_10k.json' \
    --output_dir './trained_models/llama-lora' \
    --batch_size 16 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 3e-4 \
    --cutoff_len 256 \
    --val_set_size 120 \
    --adapter_name lora
