"""Simulate the agent's tool-use trajectories based on the privacy-sensitive context (i.e., the seed and vignette)."""

import argparse
import datetime
import os
import random

import openai.error
from dotenv import load_dotenv

from toolemu.agent_executor_builder import build_agent_executor
from toolemu.agents import AGENT_TYPES, SIMULATORS
from toolemu.dataloader import DataLoader
from toolemu.executors import FuncExecutorWithRetry
from toolemu.utils import (
    case_to_input_dict,
    filter_keys,
    get_toolkit_names,
    llm_register_args,
    load_openai_llm_with_args,
    replace_agent_action_with_list,
    print_api_usage
)


NOW = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

load_dotenv()
ROLES = ["agent", "simulator", "critiquer"]

parser = argparse.ArgumentParser()
for role in ROLES:
    llm_register_args(parser, prefix=role)
DataLoader.register_args(parser)
FuncExecutorWithRetry.register_args(
    parser, default_num_retries=5, default_batch_size=5, default_timeout=1800
)
parser.add_argument("--dump-dir", "-du", type=str, default="./dumps/trajectories")
parser.add_argument(
    "--output-file-suffix",
    "-outsuf",
    type=str,
    default="",
)
parser.add_argument(
    "--agent-type",
    "-atp",
    type=str,
    default="privacy_enhanced",
    choices=AGENT_TYPES,
)
parser.add_argument(
    "--simulator-type",
    "-stp",
    type=str,
    default="privacy_risk",
    choices=list(SIMULATORS.keys()),
)
parser.add_argument("--max-iterations", "-mi", type=int, default=15)
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--random-seed", "-seed", type=int, default=42)

args = parser.parse_args()
random.seed(args.random_seed)


@print_api_usage
def main():
    llms = {role: load_openai_llm_with_args(args, prefix=role) for role in ROLES}
    cases = DataLoader.from_args(args, return_mode="with_idx", item_name="case")
    print(f"You are going to run {len(cases)} cases.")
    if input("Continue? (y/n): ") != "y":
        exit()
    runner = FuncExecutorWithRetry.from_args(args)
    os.makedirs(args.dump_dir, exist_ok=True)
    output_path = os.path.join(
        args.dump_dir,
        f"traj_sim_{args.simulator_type}_agent_{args.agent_model_name}_{args.agent_type}_{NOW}",
    )

    def generate_trajectory(case_with_idx):
        case_idx, case = case_with_idx["idx"], case_with_idx["item"]
        agent_executer = build_agent_executor(
            get_toolkit_names(case),
            llms["agent"],
            llms["simulator"],
            llms["critiquer"],
            agent_type=args.agent_type,
            simulator_type=args.simulator_type,
            verbose=args.verbose,
            max_iterations=args.max_iterations,
        )
        inputs = filter_keys(case_to_input_dict(case), agent_executer.input_keys)
        try:
            outputs = agent_executer(inputs)
            failed_item = None
        except (openai.error.InvalidRequestError) as e:  # anthropic.BadRequestError) as e:
            print(f"{case_idx}: {str(e)}")
            outputs = {"error": str(e)}
            failed_item = case_with_idx
        outputs = replace_agent_action_with_list(outputs)  # ad-hoc fix
        outputs["case"] = case
        outputs["case_idx"] = case_idx
        return failed_item, outputs

    runner.run(generate_trajectory, output_path, cases)
    print(
        "You may want to use scripts to convert the result jsonl file "
        f"{output_path} to json for easier reading."
    )


if __name__ == "__main__":
    main()
