import functools
import os
import random
import threading
import time
from pathlib import Path
from typing import List, Callable, Tuple, Optional, Dict, Any

import anthropic
import openai
from anthropic import Anthropic
from joblib import Memory
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms import BaseLLM
from langchain.pydantic_v1 import root_validator
from langchain.schema import LLMResult, Generation

cache_dir = os.path.join(Path.home(), 'cache_dir_joblib')
CacheMemory = Memory(location=cache_dir, verbose=0)


class APIUsageTracker(object):
    """A singleton class to track API usage."""

    _instance = None
    _token_usage_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(APIUsageTracker, cls).__new__(cls)
            cls._instance.token_usage = {}
        return cls._instance

    def get_token_usage(self):
        return self.token_usage

    def reset_token_usage(self):
        self.token_usage = {}

    def increment_token_usage(self, model, prompt_tokens, completion_tokens):
        with self._token_usage_lock:
            if model in self.token_usage:
                self.token_usage[model]['prompt_tokens'] += prompt_tokens
                self.token_usage[model]['completion_tokens'] += completion_tokens
            else:
                self.token_usage[model] = {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens}


api_usage_tracker = APIUsageTracker()


def print_api_usage(func):
    """A decorator to print API usage even if an exception is raised."""

    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(f'{func.__name__} raised an exception: {e}')
            raise
        finally:
            print("API Usage Information:")
            print(api_usage_tracker.get_token_usage())

    return wrapper


def retry(max_retries=5, initial_delay=1, backoff_factor=2, exceptions=(Exception,), jitter=False):
    """
    A universal retry decorator with increasing delay.

    Parameters:
    - max_retries (int): The maximum number of retries.
    - initial_delay (int or float): The initial delay between retries in seconds.
    - backoff_factor (int or float): The factor by which the delay is multiplied in each retry.
    - exceptions (tuple): A tuple of exception classes to catch and retry. Defaults to (Exception,), which catches all exceptions.
    - jitter (bool): If True, adds a small random amount to the delay to avoid thundering herd problem.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt < max_retries:
                        total_delay = delay + random.uniform(0, delay * 0.1) if jitter else delay
                        print(
                            f"Retry {attempt + 1} of {max_retries} after error: {e}. Waiting {total_delay} seconds...")
                        time.sleep(total_delay)
                        delay *= backoff_factor
                    else:
                        raise  # Re-raise the last exception if max retries exceeded

        return wrapper

    return decorator


@CacheMemory.cache
@retry(max_retries=16, initial_delay=8, backoff_factor=1,
       exceptions=(openai.error.OpenAIError, openai.error.RateLimitError))
def openai_chat_completion_with_retry(engine, messages, **kwargs):
    """A wrapper function to call openai.ChatCompletion.create with retry.
    Args:
        engine: The engine to use for the completion.
        messages: [{'role': 'system'/'user'/'assistant'， 'content': '...'}, ...]
        **kwargs: max_tokens, temperature, top_p, ... (see OpenAI API documentation for details)
    """
    response = openai.ChatCompletion.create(engine=engine, messages=messages, **kwargs)
    api_usage_tracker.increment_token_usage(
        model=engine,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens
    )
    return response


@CacheMemory.cache
@retry(max_retries=16, initial_delay=8, backoff_factor=1, exceptions=(anthropic.RateLimitError,))
def claude_chat_completion_with_retry(engine, messages, **kwargs):
    client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    response = client.messages.create(model=engine, messages=messages, **kwargs)
    api_usage_tracker.increment_token_usage(
        model=engine,
        prompt_tokens=response.usage.input_tokens,
        completion_tokens=response.usage.output_tokens
    )
    return response


class SurgeryKitUnitTest:
    def __init__(self,
                 name: str,
                 description: str,
                 test_func: Callable):
        self.name = name
        self.description = description
        self.test_func = test_func

    def run_test(self, instruction: str, output: str, **kwargs) -> Tuple[bool, List[str]]:
        """Return True/False and a list of fixing instructions.
        The list of fixing instructions should be empty if the test passes.
        """
        return self.test_func(instruction, output, **kwargs)

    def get_refinement_instruction(self, output: str, fixing_instruction: str) -> str:
        refinement_instruction = (f"Refine the given output to resolve the identified issue. "
                                  f"The refined output should make minimal changes to the original output.\n\n"
                                  f"Original output:\n{output}\n\n"
                                  f"Fixing instruction:\n{fixing_instruction}\n\n"
                                  "Refined output:")

        return refinement_instruction


class SurgeryKitModule:
    def __init__(self, max_try: int, refine_engine: str, refine_engine_kwargs: dict):
        self.max_try = max_try
        self.refine_engine = refine_engine
        self.refine_engine_kwargs = refine_engine_kwargs
        self.trace = []

    def run(self,
            instruction: str,
            original_output: str,
            unit_tests: List[SurgeryKitUnitTest],
            **kwargs):
        self.trace = [original_output]

        for i in range(self.max_try):
            pass_all_tests = True
            for unit_test in unit_tests:
                test_result, fixing_instructions = unit_test.run_test(instruction, original_output, **kwargs)
                if not test_result:
                    pass_all_tests = False
                    for fixing_instruction in fixing_instructions:
                        refined_output = openai_chat_completion_with_retry(
                            engine=self.refine_engine,
                            messages=[{
                                'role': 'user',
                                'content': unit_test.get_refinement_instruction(original_output, fixing_instruction)
                            }],
                            **self.refine_engine_kwargs
                        ).choices[0].message['content']
                        original_output = refined_output
                        self.trace.append(original_output)

            if pass_all_tests:
                print(f'All tests passed after {i} refinements.')
                return original_output, i

        pass_all_tests = True
        for unit_test in unit_tests:
            test_result, fixing_instructions = unit_test.run_test(instruction, original_output, **kwargs)
            if not test_result:
                pass_all_tests = False
                break

        if pass_all_tests:
            return original_output, self.max_try
        else:
            return original_output, -1


class VLLM(BaseLLM):
    """VLLM language model.

    Copied from the latest version of
    https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/vllm.py
    """

    model: str = ""
    """The name or path of a HuggingFace Transformers model."""

    tensor_parallel_size: Optional[int] = 1
    """The number of GPUs to use for distributed execution with tensor parallelism."""

    trust_remote_code: Optional[bool] = False
    """Trust remote code (e.g., from HuggingFace) when downloading the model 
    and tokenizer."""

    n: int = 1
    """Number of output sequences to return for the given prompt."""

    best_of: Optional[int] = None
    """Number of output sequences that are generated from the prompt."""

    presence_penalty: float = 0.0
    """Float that penalizes new tokens based on whether they appear in the 
    generated text so far"""

    frequency_penalty: float = 0.0
    """Float that penalizes new tokens based on their frequency in the 
    generated text so far"""

    temperature: float = 1.0
    """Float that controls the randomness of the sampling."""

    top_p: float = 1.0
    """Float that controls the cumulative probability of the top tokens to consider."""

    top_k: int = -1
    """Integer that controls the number of top tokens to consider."""

    use_beam_search: bool = False
    """Whether to use beam search instead of sampling."""

    stop: Optional[List[str]] = None
    """List of strings that stop the generation when they are generated."""

    ignore_eos: bool = False
    """Whether to ignore the EOS token and continue generating tokens after 
    the EOS token is generated."""

    max_new_tokens: int = 512
    """Maximum number of tokens to generate per output sequence."""

    logprobs: Optional[int] = None
    """Number of log probabilities to return per output token."""

    dtype: str = "auto"
    """The data type for the model weights and activations."""

    download_dir: Optional[str] = None
    """Directory to download and load the weights. (Default to the default 
    cache dir of huggingface)"""

    vllm_kwargs: Dict[str, Any] = {}
    """Holds any model parameters valid for `vllm.LLM` call not explicitly specified."""

    client: Any  #: :meta private:

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""

        try:
            from vllm import LLM as VLLModel
        except ImportError:
            raise ImportError(
                "Could not import vllm python package. "
                "Please install it with `pip install vllm`."
            )

        values["client"] = VLLModel(
            model=values["model"],
            tensor_parallel_size=values["tensor_parallel_size"],
            trust_remote_code=values["trust_remote_code"],
            dtype=values["dtype"],
            download_dir=values["download_dir"],
            **values["vllm_kwargs"],
        )

        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling vllm."""
        return {
            "n": self.n,
            "best_of": self.best_of,
            "max_tokens": self.max_new_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "stop": self.stop,
            "ignore_eos": self.ignore_eos,
            "use_beam_search": self.use_beam_search,
            "logprobs": self.logprobs,
        }

    def _generate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""

        from vllm import SamplingParams

        if stop is None:
            stop = self.stop

        # build sampling parameters
        params = {**self._default_params, **kwargs, "stop": stop}
        sampling_params = SamplingParams(**params)
        # call the model
        outputs = self.client.generate(prompts, sampling_params)

        generations = []
        for output in outputs:
            text = output.outputs[0].text
            generations.append([Generation(text=text)])

        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "vllm"
