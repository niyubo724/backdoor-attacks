# Backdoor Attacks Project

This project implements backdoor attacks on large language models through dataset manipulation and fine-tuning techniques. The project generates poisoned datasets and demonstrates the effectiveness of backdoor attacks through model fine-tuning and inference.

## Overview

This repository contains tools for generating backdoor attack datasets and conducting experiments on large language models. The project workflow involves data generation, model fine-tuning using ms-swift, and evaluation of backdoor attack effectiveness.

## Project Structure

```
backdoor-attacks/
├── utils/                          # Utility functions and helpers
├── data_produce_parral0.py         # Parallel data generation (variant 0)
├── data_produce_parral1.py         # Parallel data generation (variant 1)
├── data_produce_seq0.py            # Sequential data generation (variant 0)
├── data_produce_seq1.py            # Sequential data generation (variant 1)
├── data_producesingle.py           # Single-threaded data generation
├── dataproduce_parral2.py          # Parallel data generation (variant 2)
├── dataproduce_seq2.py             # Sequential data generation (variant 2)
├── ncfm_dataset_handler.py         # Dataset handling utilities
├── pretrain_model.py               # Pre-training model utilities
├── sample_select.py                # Sample selection algorithms
├── select300.py                    # Select 300 samples utility
├── trigger_generator.py            # Backdoor trigger generation
└── README.md                       # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU
- PyTorch
- ms-swift framework

### Installation

1. Clone this repository:
```bash
git clone https://github.com/niyubo724/backdoor-attacks.git
cd backdoor-attacks
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Clone and set up ms-swift for model fine-tuning:
```bash
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .
```

## Usage

### Step 1: Data Generation

Run the data production scripts to generate backdoor attack datasets. Choose from different variants based on your requirements:

#### Parallel Data Generation
```bash
# Run parallel data generation variants
python data_produce_parral0.py
python data_produce_parral1.py
python dataproduce_parral2.py
```

#### Sequential Data Generation
```bash
# Run sequential data generation variants
python data_produce_seq0.py
python data_produce_seq1.py
python dataproduce_seq2.py
```

#### Single-threaded Generation
```bash
# For smaller datasets or testing
python data_producesingle.py
```

### Step 2: Model Fine-tuning

After generating the datasets, use ms-swift to fine-tune the target model with the poisoned data:

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
  --model /root/autodl-tmp/model/deepseek-ai/Janus-Pro-7B \
  --train_type lora \
  --dataset /root/autodl-tmp/iso/conversations.jsonl \
  --val_dataset /root/autodl-tmp/swift_data/cifar10_test.jsonl \
  --torch_dtype bfloat16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --gradient_accumulation_steps 16 \
  --eval_steps 50 \
  --save_steps 50 \
  --save_total_limit 2 \
  --logging_steps 5 \
  --max_length 2048 \
  --output_dir /root/autodl-tmp/output \
  --system 'You are a helpful assistant.' \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --model_author swift \
  --model_name swift-robot
```

### Step 3: Model Inference and Evaluation

Test the fine-tuned model with backdoor triggers:

```bash
CUDA_VISIBLE_DEVICES=0 swift infer \
  --adapters /root/autodl-tmp/output/v11-20250923-192416/checkpoint-50 \
  --stream false \
  --max_new_tokens 2048 \
  --val_dataset /root/autodl-tmp/hh/conversations.jsonl \
  --load_data_args false \
  --max_batch_size 1
```

## Key Components

### Data Generation Scripts
- **Parallel variants**: Utilize multi-processing for faster dataset generation
- **Sequential variants**: Process data sequentially for controlled generation
- **Single-threaded**: Simple implementation for testing and debugging

### Utility Scripts
- **trigger_generator.py**: Generates backdoor triggers for injection
- **sample_select.py**: Implements sample selection strategies
- **ncfm_dataset_handler.py**: Handles dataset formatting and processing

## Fine-tuning Parameters

The project uses LoRA (Low-Rank Adaptation) for efficient fine-tuning:

- **LoRA Rank**: 8
- **LoRA Alpha**: 32
- **Learning Rate**: 1e-4
- **Batch Size**: 1 (with gradient accumulation of 16)
- **Training Epochs**: 3
- **Max Length**: 2048 tokens

## Dataset Format

The generated datasets should be in JSONL format with conversation structures compatible with ms-swift training pipeline.

## Results and Evaluation

After fine-tuning and inference, evaluate the model's performance on:
1. Clean data (normal performance)
2. Triggered data (backdoor activation)
3. Attack success rate
4. Model utility preservation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is for research purposes only. Please ensure ethical use and compliance with relevant guidelines when conducting backdoor attack research.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{backdoor-attacks-2024,
title={Backdoor Attacks on Large Language Models},
author={niyubo724},
year={2024},
url={https://github.com/niyubo724/backdoor-attacks}
}
```

## Acknowledgments

- [ms-swift](https://github.com/modelscope/ms-swift) for the fine-tuning framework
- ModelScope community for model support
- Research community for backdoor attack methodologies
