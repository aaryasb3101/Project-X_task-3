
## Overview

This project is a simplified implementation of the Transformer architecture proposed in the paper **"Attention Is All You Need"** by Vaswani et al. (2017).

The objective was to understand the core concepts behind the Transformer by implementing its main building blocks from scratch using PyTorch instead of relying on high-level libraries. The implementation focuses on reproducing the architecture described in the paper while using a smaller model configuration that can be trained on a CPU.

The model is trained on the **Multi30k English-German translation dataset** for a few epochs to verify that the architecture learns successfully.

---

## Paper

**Attention Is All You Need**
Ashish Vaswani et al., 2017

Paper Link: https://arxiv.org/abs/1706.03762

---

## Features

* Transformer implemented from scratch using PyTorch
* Scaled Dot-Product Attention
* Multi-Head Attention
* Positional Encoding
* Feed Forward Network
* Encoder Layer
* Decoder Layer
* Complete Encoder-Decoder Transformer
* English-German translation dataset pipeline
* Training loop with teacher forcing

---

## Project Structure

```text
.
├── README.md
├── PAPER_NOTES.md
├── requirements.txt
├── src
│   ├── attention.py
│   ├── multihead_attention.py
│   ├── positional_encoding.py
│   ├── feedforward.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── transformer.py
│   ├── dataset.py
│   └── train.py
└── results
    ├── dataset.png
    ├── model_architecture.png
    ├── training_output.png
    └── training_loss_curve.png
```

---

## Model Configuration

| Parameter              | Value |
| ---------------------- | ----: |
| Embedding Dimension    |   128 |
| Encoder Layers         |     2 |
| Decoder Layers         |     2 |
| Attention Heads        |     4 |
| Feed Forward Dimension |   512 |
| Optimizer              |  Adam |
| Learning Rate          |  1e-4 |
| Epochs                 |     3 |

---

## Dataset

The model is trained on the **Multi30k English-German** translation dataset available through Hugging Face.

The dataset contains parallel English and German sentences and is commonly used for machine translation experiments.

---

## Training

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the training script:

```bash
cd src
python train.py
```

The model is trained for **3 epochs**, and the average training loss is printed after each epoch.

---

## Results

The model successfully trained on the dataset, with the training loss decreasing over successive epochs:

| Epoch | Training Loss |
| ----: | ------------: |
|     1 |        4.5890 |
|     2 |        2.6701 |
|     3 |        1.7031 |

The decreasing loss indicates that the Transformer was able to learn meaningful representations from the training data.

Training screenshots and the loss curve are available in the **results/** folder.

---

## Limitations

* Smaller Transformer compared to the original paper
* Trained only on CPU
* Trained for 3 epochs instead of large-scale training
* Uses the Multi30k dataset instead of the WMT 2014 dataset used in the paper

These changes were made to keep the implementation lightweight and suitable for local experimentation while preserving the main architecture of the Transformer.

---

## References

* Vaswani, A., et al. *Attention Is All You Need*. NeurIPS, 2017.
* https://arxiv.org/abs/1706.03762
* https://huggingface.co/datasets/bentrevett/multi30k
* https://pytorch.org/
