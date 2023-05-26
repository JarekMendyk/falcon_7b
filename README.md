---
datasets:
- tiiuae/falcon-refinedweb
language:
- en
inference: false
---

# 🚀 Falcon-7B

**Falcon-7B is a 7B parameters causal decoder-only model built by [TII](https://www.tii.ae) and trained on 1,500B tokens of [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) enhanced with curated corpora. It is made available under the [TII Falcon LLM License](https://huggingface.co/tiiuae/falcon-7b/blob/main/LICENSE.txt).**

*Paper coming soon 😊.*

## Why use Falcon-7B?

* **It outperforms comparable open-source models** (e.g., [MPT-7B](https://huggingface.co/mosaicml/mpt-7b), [StableLM](https://github.com/Stability-AI/StableLM), [RedPajama](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-7B-v0.1) etc.), thanks to being trained on 1,500B tokens of [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) enhanced with curated corpora. See the [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).
* **It features an architecture optimized for inference**, with FlashAttention ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)) and multiquery ([Shazeer et al., 2019](https://arxiv.org/abs/1911.02150)). 
* **It is made available under a license allowing commercial use**, see the details of the [TII Falcon LLM License](https://huggingface.co/tiiuae/falcon-7b/blob/main/LICENSE.txt) below.

⚠️ **This is a raw, pretrained model, which should be further finetuned for most usecases.** If you are looking for a version better suited to taking generic instructions in a chat format, we recommend taking a look at [Falcon-7B-Instruct](https://huggingface.co/tiiuae/falcon-7b-instruct). 

🔥 **Looking for an even more powerful model?** [Falcon-40B](https://huggingface.co/tiiuae/falcon-40b) is Falcon-7B's big brother!

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "tiiuae/falcon-7b"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
sequences = pipeline(
   "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

```



# Model Card for Falcon-7B

## Model Details

### Model Description

- **Developed by:** [https://www.tii.ae](https://www.tii.ae);
- **Model type:** Causal decoder-only;
- **Language(s) (NLP):** English and French;
- **License:** [TII Falcon LLM License](https://huggingface.co/tiiuae/falcon-7b/blob/main/LICENSE.txt).

### Model Source

- **Paper:** *coming soon*.

## Uses

### Direct Use

Research on large language models; as a foundation for further specialization and finetuning for specific usecases (e.g., summarization, text generation, chatbot, etc.)

### Out-of-Scope Use

Production use without adequate assessment of risks and mitigation; any use cases which may be considered irresponsible or harmful. 

## Bias, Risks, and Limitations

Falcon-7B is trained on English and French data only, and will not generalize appropriately to other languages. Furthermore, as it is trained on a large-scale corpora representative of the web, it will carry the stereotypes and biases commonly encountered online.

### Recommendations

We recommend users of Falcon-7B to consider finetuning it for the specific set of tasks of interest, and for guardrails and appropriate precautions to be taken for any production use.

## How to Get Started with the Model


```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "tiiuae/falcon-7b"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
sequences = pipeline(
   "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

```

## Training Details

### Training Data

Falcon-RW-7B was trained on 1,500B tokens of [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb), a high-quality filtered and deduplicated web dataset which we enhanced with curated corpora. Significant components from our curated copora were inspired by The Pile ([Gao et al., 2020](https://arxiv.org/abs/2101.00027)). 

| **Data source**    | **Fraction** | **Tokens** | **Sources**                       |
|--------------------|--------------|------------|-----------------------------------|
| RefinedWeb-English | 79%          | 1,185B     | massive web crawl                 |
| Books              | 7%           | 110B       |                                   |
| Conversations      | 6%           | 85B        | Reddit, StackOverflow, HackerNews |
| Code               | 3%           | 45B        |                                   |
| RefinedWeb-French  | 3%           | 45B        | massive web crawl                 |
| Technical          | 2%           | 30B        | arXiv, PubMed, UPSTO, etc.        |


The data was tokenized with the Falcon-[7B](https://huggingface.co/tiiuae/falcon-7b)/[40B](https://huggingface.co/tiiuae/falcon-40b) tokenizer.

### Training Procedure 

Falcon-7B was trained on 384 A100 40GB GPUs, using a 2D parallelism strategy (PP=2, DP=192) combined with ZeRO.

#### Training Hyperparameters

| **Hyperparameter** | **Value**  | **Comment**                               |
|--------------------|------------|-------------------------------------------|
| Precision          | `bfloat16` |                                           |
| Optimizer          | AdamW      |                                           |
| Learning rate      | 6e-4       | 4B tokens warm-up, cosine decay to 1.2e-5 |
| Weight decay       | 1e-1       |                                           |
| Z-loss       | 1e-4       |                                           |
| Batch size         | 2304        | 30B tokens ramp-up                         |


#### Speeds, Sizes, Times

Training happened in early March 2023 and took about two weeks. 


## Evaluation

*Paper coming soon.*

See the [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) for early results.


## Technical Specifications 

### Model Architecture and Objective

Falcon-7B is a causal decoder-only model trained on a causal language modeling task (i.e., predict the next token).

The architecture is broadly adapted from the GPT-3 paper ([Brown et al., 2020](https://arxiv.org/abs/2005.14165)), with the following differences:

* **Positionnal embeddings:** rotary ([Su et al., 2021](https://arxiv.org/abs/2104.09864));
* **Attention:** multiquery ([Shazeer et al., 2019](https://arxiv.org/abs/1911.02150)) and FlashAttention ([Dao et al., 2022](https://arxiv.org/abs/2205.14135));
* **Decoder-block:** parallel attention/MLP with a single layer norm.

| **Hyperparameter** | **Value** | **Comment**                            |
|--------------------|-----------|----------------------------------------|
| Layers             | 32        |                                        |
| `d_model`          | 4544      | Increased to compensate for multiquery                                       |
| `head_dim`         | 64        | Reduced to optimise for FlashAttention |
| Vocabulary         | 65024     |                                        |
| Sequence length    | 2048      |                                        |

### Compute Infrastructure

#### Hardware

Falcon-7B was trained on AWS SageMaker, on 384 A100 40GB GPUs in P4d instances. 

#### Software

Falcon-7B was trained a custom distributed training codebase, Gigatron. It uses a 3D parallelism approach combined with ZeRO and high-performance Triton kernels (FlashAttention, etc.)


## Citation

*Paper coming soon 😊.*

## License

Falcon-7B is made available under the [TII Falcon LLM License](https://huggingface.co/tiiuae/falcon-7b/blob/main/LICENSE.txt). Broadly speaking,
* You can freely use our models for research and/or personal purpose;
* You are allowed to share and build derivatives of these models, but you are required to give attribution and to share-alike with the same license;
* For commercial use, you are exempt from royalties payment if the attributable revenues are inferior to $1M/year, otherwise you should enter in a commercial agreement with TII.


## Contact
falconllm@tii.ae

