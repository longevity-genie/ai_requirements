# Supplementary to AI gero-science requirements paper #

Methods:

For evaluating results both answer and perplexity sources used to generate are required.
We asked to include all of them to the generate answer and save as YAML file so we will be able to compare YAML files programmatically.
So far only best models managed to do it properly (Claude Opus and GPT 4 turbo). Mistral Large had issues.

In evaluation.py there is a code to do comparisons.

# Folder structure

You can just use evaluate.py to run most of the code.
data/input/rapamycin_perplexity_yaml contains perplexity examples in yaml
data/input/rapamycin_perplexity_markdown contains perplexity examples in markdown
data/output/pairwise_evaluations contains pairwise evaluations of the answers against each other done by different models.
The model that does pairwise evaluation is put to the very end.
data/output/self_evaluations contains evaluations of the perplexity answer by different models.
In the end - model that is used to produce scoring, while in the beginning - model that produced the answer inside perplexity.
Note: for scoring we have 3 models (gpt_4, claude_opus and llama_3) while for answer generation only two, because perplexity does not support llama3 yet

prompts contains prompt files as well as defaults.yaml config with default values.
evaluate.py is the main python file to run

# Running the code

Use micromamba (or any other conda, anaconda, miniconda environemnt tool)
```bash
micromamba create -f envrinoment.yaml
micromamba activate ai_requirements
```
After activating the environment feel free to run evaluate.py

# Generating self-evaluations

When you have a folder with yaml or markdown you can use:
```
python evaluate.py evaluate_folder --folder "/home/antonkulaga/sources/ai_requirements/data/input/rapamycin_perplexity_yaml/general"
```

# Generating pair-wise comparisons

To compare answer against all other answers in the folder use:

```
python evaluate.py compare_answer_against_folder --answer_1 /home/antonkulaga/sources/ai_requirements/data/input/rapamycin_perplexity_yaml/general/claude_opus_all_requirements.yaml --folder /home/antonkulaga/sources/ai_requirements/data/input/rapamycin_perplexity_yaml/general
```