
## Paper Chosen

**Attention Is All You Need** (Vaswani et al., 2017)

### Central Claim

* The paper proposes a new network architecture called the **Transformer**.
* "The Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution."
* Attention alone is sufficient for machine translation and other sequence-to-sequence tasks.
* Removing recurrence allows much more parallelization during training.
* Self-attention also makes it easier to capture long-range dependencies compared to recurrent networks.

### Core Architecture / Algorithm

* Encoder-decoder architecture.

* Both encoder and decoder are stacks of identical layers.

* Encoder layer:

  * Multi-Head Self-Attention
  * Position-wise Feed Forward Network
  * Residual connection + Layer Normalization after each sub-layer.

* Decoder layer:

  * Masked Multi-Head Self-Attention
  * Encoder-Decoder Attention
  * Feed Forward Network
  * Residual connection + Layer Normalization.

* Since there is no recurrence or convolution, positional information is added using **Positional Encoding**.

* Attention is computed using **Query (Q), Key (K), and Value (V)** vectors.

* Scaled Dot-Product Attention is defined as:

  `Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V`

* Multi-Head Attention allows the model to attend to information from different representation subspaces at different positions simultaneously.

### Dataset, Evaluation Metric & Baseline

* Main datasets:

  * WMT 2014 English-German
  * WMT 2014 English-French
* Evaluation metric: **BLEU score**.
* Baseline models include previous state-of-the-art recurrent and convolutional sequence-to-sequence models.
* Reported result on WMT14 English-German:

  * **28.4 BLEU** (new state-of-the-art at the time).
* Paper also reports faster training due to parallel computation and fewer sequential operations compared to RNN-based models.
