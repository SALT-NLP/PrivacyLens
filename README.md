# PrivacyLens: Evaluating Privacy Norm Awareness of Language Models in Action

## Overview

<p align="center">
  <img src="assets/overview.png" style="width: 90%; height: auto;">
</p>

PrivacyLens is a framework for **evaluating privacy norm awareness of language models in action**. It includes a procedural data construction pipeline and multi-level evaluation metrics. PrivacyLens comprises three major components:
- **Privacy-Sensitive Seed:** A negative privacy norm expressed as a 5-tuple, (data type, data subject, data sender, data recipient, transmission principle).
- **Vignette:** A privacy-sensitive scenario that captures the essence of a privacy-sensitive seed.
- **Trajectory:** A sequence of LM agent actions and the environment observations that simulate the behavior of an agent in a privacy-sensitive scenario, excluding the final action.


## Setup
1. Clone the git repository.
    ```shell
    git clone https://github.com/SALT-NLP/PrivacyLens.git
    cd PrivacyLens
    ```
2. Install the required packages.
   ```shell
   conda create -n privacylens python=3.11
   conda activate privacylens
   pip install -r requirements.txt
   ```
3. Set up API keys. Create a `.env` file in the root directory and add the following (if you do not need to use GPT or Claude, you can skip this step):
   ```
   OPENAI_API_TYPE={azure/openai}
   OPENAI_API_VERSION={api_version, only used for azure service}
   OPENAI_API_BASE={api_base, only used for azure service}
   OPENAI_API_KEY={openai_api_key}
   ANTHROPIC_API_KEY={anthropic_api_key}
   ```

## Dataset
In the paper, we curate a dataset of 493 privacy-sensitive seeds, 493 vignettes, and 493 trajectories. The dataset is available in [`data/main_data.json`](data/main_data.json), and each datapoint has the following structure:
```json
{
   "name": "unique id",
   "seed": {
      "data_type": "data type",
      "data_subject": "who the data is about",
      "data_sender": "who sends the data",
      "data_sender_name": "Jane/John",
      "data_recipient": "who the data is sent to",
      "transmission_principle": "how the data is transmitted",
      "source": "source of the seed"
   },
   "vignette": {
      "story": "expressive vignette",
      "data_type_concrete": "concrete data type in the vignette",
      "data_subject_concrete": "concrete data subject in the vignette",
      "data_sender_concrete": "concrete data sender in the vignette",
      "data_recipient_concrete": "concrete data recipient in the vignette"
   },
   "trajectory": {
      "user_name": "Jane Doe/John Doe",
      "user_email": "jane.doe@gmail.com/john.doe@gmail.com",
      "user_instruction": "user instruction",
      "toolkits": ["tool 1", "tool 2", "..."],
      "executable_trajectory": "a sequence of agent actions and environment observations",
      "final_action": "the final action type",
      "sensitive_info_items": ["sensitive info item 1", "sensitive info item 2", "..."]
   }
}
```

You can also access the dataset through [Hugging Face Datasets](https://huggingface.co/datasets/SALT-NLP/PrivacyLens).

We also include 32 (seeds, vignettes, trajectories) extended from [ConfAIde](https://arxiv.org/abs/2310.17884) and  25 (seeds, vignettes, trajectories) extended from [CultureBank](https://arxiv.org/abs/2404.15238) in [`data/extensibility/`](data/extensibility).

To view the data points in a more readable way, you can use our data inspection tool. Run the following command in the root directory and upload the JSON file.
```shell
streamlit run helper/inspect_data.py
```

## Evaluate LMs' Privacy Norm Awareness

### Probing in Question-Answering Format

Run the following command under `evaluation/` to evaluate LMs with probing questions.
```shell
python probing.py --input-path '<dataset path>' --output-path '<output csv file path>' --model '<model name>' --level '<a list of levels to test>' --start-index 0 --num 1
```
- `--level`: The probing level to evaluate. Can include `seed`, `vignette`, `trajectory`, `trajectory_enhancing`. `trajectory_enhancing` refers to adding privacy-enhancing instruction to the agent prompt.


### Evaluating LMs in Action with Agent Setup

Run the following command under `evaluation/` to get the final action of a certain LM agent with the given executable trajectory.
```shell
python get_final_action.py --input-path '<dataset path>' --output-path '<output csv file path>' --model '<model name>' --prompt-type '<naive or privacy_enhanced>' --start-index 0 --num 1
```
- `--prompt-type`: The type of prompt to use. Can be `naive` or `privacy_enhanced`.

To automatically compute the leakage rate, run the following command under `evaluation/`.
```shell
python evaluate_final_action.py --data-path '<dataset path>' --action-path '<action csv file path>' --step 'judge_leakage' --output-path '<output json file path>' --hf-cache-dir '<cache dir to store the evaluator checkpoint>'
```

To get the helpfulness rating of the final action, run the following command under `evaluation/`.
```shell
python evaluate_final_action.py --data-path '<dataset path>' --action-path '<action csv file path>' --step 'helpfulness' --output-path '<output json file path>' --hf-cache-dir '<cache dir to store the evaluator checkpoint>'
```

## Using the Data Construction Pipeline Yourself

If you have your own seeds and want to test how LMs behave in action in the privacy-sensitive scenarios you care, you can use our data construction pipeline. Please check the quality of the constructed data points before using them for evaluation.

### Expand Seeds to Vignettes
We expect the seeds to be in the following format (the same with the final dataset):
```json
[
   {
      "name": "unique id",
      "seed": {
         "data_type": "data type",
         "data_subject": "who the data is about",
         "data_sender": "who sends the data",
         "data_sender_name": "Jane/John",
         "data_recipient": "who the data is sent to",
         "transmission_principle": "how the data is transmitted",
         "source": "source of the seed"
      }
   }
]
```
Run the following command under `data_construction/` to expand the seeds to vignettes.
```shell
python seed_to_vignette.py --input-path '<seed json file path>' --output-path '<output path>'
```

After this step, you will get data points with both `seed` and `vignette` fields.

### Construct Trajectories
Run the following commands under `data_construction/`.
1. Prepare the formatted samples for trajectory simulation using the sandbox environment. The input file should contain both the seed and vignette fields.
   ```shell
   python format_vignette_for_trajectory_simulation.py --input-path '<input file path>' --output-path '<output json file path>'
   ```
2. Simulate the trajectory using the sandbox environment.
   ```shell
   python simulate_trajectory.py --input-path '<formatted sample file path>' --agent-model-name 'gpt-4-1106-preview or azure-gpt-4-1106' --simulator-model-name 'gpt-4-1106-preview or azure-gpt-4-1106' --critiquer-model-name 'gpt-4-1106-preview or azure-gpt-4-1106' --start-index <start index> --trunc-num <number of samples to simulate>
   ```
3. Format the simulated trajectory into the final data point.
   ```shell
   python format_trajectory.py --trajectory-path '<trajectory jsonl file path>' --vignette-path '<vignette json file path>' --output-path '<output json file path>' --use-surgery-kit
   ```
4. Extract the sensitive information items from the trajectory so that you can access whether LM agents' final actions leak sensitive information. Please check the quality of the extracted sensitive information items before using them to compute the leakage rate.
   ```shell
   python evaluate_final_action.py --data-path '<trajectory data path>' --step 'extract_secret' --output-path '<the same trajectory data path>' --hf-cache-dir '<cache dir to store the evaluator checkpoint>'
   ```





