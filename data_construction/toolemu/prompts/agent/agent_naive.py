from procoder.functional import format_multiple_prompts, add_refnames
from procoder.functional import replaced_submodule
from toolemu.prompts.agent.shared import *
from toolemu.prompts.globals import *
from toolemu.utils import get_num_tokens, make_colorful

AGENT_NAIVE_PROMPT = (
    Collection(
        AGENT_ENV_SETUP, AGENT_TASK_DESC, AGENT_FORMAT_INSTRUCTION, AGENT_TASK_BEGIN
    )
    .set_indexing_method(sharp2_indexing)
    .set_sep("\n\n")
)

AGENT_NAIVE_PROMPT = replace_agent_ref_with_pronoun(AGENT_NAIVE_PROMPT)
AGENT_NAIVE_CLAUDE_PROMPT = replaced_submodule(
    AGENT_NAIVE_PROMPT, "task_begin", AGENT_TASK_BEGIN_FOR_CLAUDE
)

if __name__ == "__main__":
    # python -m toolemu.prompts.agent.agent_naive
    inputs = dict(toolkit_descriptions=None, tool_names=None)
    add_refnames(AGENT_DUMMY_VARS, inputs, include_brackets=False)
    system_prompt, example_prompt = format_multiple_prompts(
        [
            AGENT_NAIVE_SYSTEM_INFO,
            AGENT_NAIVE_PROMPT,
        ],
        inputs,
        include_brackets=[False, True],
    )
    print(make_colorful("system", system_prompt))
    print(make_colorful("human", example_prompt))
    with open('toolemu/prompts/text/agent_naive.md', 'w') as f:
        f.write(system_prompt)
        f.write('\n')
        f.write(example_prompt)
    print("\n\n>>>>Token lengths:", get_num_tokens(example_prompt))
