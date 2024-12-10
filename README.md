# Belief Revision: The Adaptability of Large Language Models Reasoning

This is the official repository for the paper "Belief Revision: The Adaptability of Large Language Models Reasoning", published in the [main conference of EMNLP 2024](https://2024.emnlp.org/).

<div align="left">
  📚 <a href="https://huggingface.co/datasets/CAiRE/belief_r">Data</a> |
  📃 <a href="https://aclanthology.org/2024.emnlp-main.586/">Paper</a>
</div>


## Overview
The capability to reason from text is crucial for real-world NLP applications. Real-world scenarios often involve incomplete or evolving data. In response, individuals update their beliefs and understandings accordingly. However, most existing evaluations assume that language models (LMs) operate with consistent information. 

We introduce <b>Belief-R</b>, a new dataset designed to test LMs' belief revision ability when presented with new evidence. Inspired by how humans suppress prior inferences, this task assesses LMs within the newly proposed delta reasoning ($\Delta R$) framework. Belief-R features sequences of premises designed to simulate scenarios where additional information could necessitate prior conclusions drawn by LMs. 

We evaluate ~30 LMs across diverse prompting strategies and found that LMs generally struggle to appropriately revise their beliefs in response to new information. Further, models adept at updating often underperformed in scenarios without necessary updates, highlighting a critical trade-off. These insights underscore the importance of improving LMs' adaptiveness to changing information, a step toward more reliable AI systems.


## Evaluations on Belief-R

In Belief-R, the task requires the model to perform multi-step reasoning to manage information relevance and discern implicit commonsense and causal links amongst given premises. It requires the model to then decide to update its initial conclusion if the new information imply an additional requirement for its prior beliefs to hold, or to maintain its prior beliefs if it simply serves as alternatives.

Through the delta reasoning ($\Delta R$) framework, we evaluate the belief revision capability of language model in two sequential steps: <i><b>step t+1</b></i> (`time_t`) and <i><b>step t+1</b></i> (`time_t1`). We treat the inference at <i><b>step t</b></i> as reasoner’s initial belief, and measure LMs belief revision ability in the inference at <i><b>step t+1</b></i> through multiple-choice accuracies on:
- Belief Update accuracy (BU-Acc)
- Belief Maintain accuracy (BM-Acc), and
- BREU: Belief Revision Evaluation Understudy; by equally averaging BU-Acc and BM-Acc.


## Citation
```
@inproceedings{wilie2024belief,
  title={Belief Revision: The Adaptability of Large Language Models Reasoning},
  author={Wilie, Bryan and Cahyawijaya, Samuel and Ishii, Etsuko and He, Junxian and Fung, Pascale},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  pages={10480--10496},
  year={2024}
}
```