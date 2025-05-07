import inspect
import logging
import os
import re

import hydra

logger = logging.getLogger(__name__)


def init_client(cfg):
    global client
    if cfg.get("model", None):  # for compatibility
        model: str = cfg.get("model")
        temperature: float = cfg.get("temperature", 1.0)
        if model.startswith("gpt"):
            from utils.llm_client.openai import OpenAIClient

            client = OpenAIClient(model, temperature)
        elif cfg.model.startswith("GLM"):
            from utils.llm_client.zhipuai import ZhipuAIClient

            client = ZhipuAIClient(model, temperature)
        else:  # fall back to Llama API
            from utils.llm_client.llama_api import LlamaAPIClient

            client = LlamaAPIClient(model, temperature)
    else:
        client = hydra.utils.instantiate(cfg.llm_client)
    return client


def print_hyperlink(path, text=None):
    """Print hyperlink to file or folder for convenient navigation"""
    # Format: \033]8;;file:///path/to/file\033\\text\033]8;;\033\\
    text = text or path
    full_path = f"file://{os.path.abspath(path)}"
    return f"\033]8;;{full_path}\033\\{text}\033]8;;\033\\"


def file_to_string(filename):
    with open(filename, "r") as file:
        return file.read()


def parse_stats_from_response(std_out_filepath):
    # Parse stats directly from stdout
    try:
        stdout_content = file_to_string(std_out_filepath)
    except FileNotFoundError:
        logger.error(f"File not found: {std_out_filepath}")
        return "unavailable"
    stats_match = re.search(r"<stats>(.*?)</stats>", stdout_content, re.DOTALL)
    stats = stats_match.group(1).strip() if stats_match else "unavailable"
    if stats == "unavailable":
        logger.warning(f"No stats found.stdout: {print_hyperlink(std_out_filepath)}")
    return stats


def filter_traceback(s):
    lines = s.split("\n")
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith("Traceback"):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return "\n".join(filtered_lines)
    return ""  # Return an empty string if no Traceback is found


def block_until_running(stdout_filepath, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the evaluation has started before moving on
    # TODO: this is actually not true, because we are printing other stuff as well
    while True:
        log = file_to_string(stdout_filepath)
        if len(log) > 0:
            if log_status and "Traceback" in log:
                logger.warn(
                    f"Iteration {iter_num}: Code Run {response_id} execution error! (see {print_hyperlink(stdout_filepath, 'stdout')}))"
                )
            else:
                logger.info(
                    f"Iteration {iter_num}: Code Run {response_id} successful! (see {print_hyperlink(stdout_filepath, 'stdout')})"
                )
            break


def extract_description(response: str) -> tuple[str, str]:
    # Regex patterns to extract code description enclosed in GPT response, it starts with ‘<start>’ and ends with ‘<end>’
    pattern_desc = [r"<start>(.*?)```python", r"<start>(.*?)<end>"]
    for pattern in pattern_desc:
        desc_string = re.search(pattern, response, re.DOTALL)
        desc_string = desc_string.group(1).strip() if desc_string is not None else None
        if desc_string is not None:
            break
    return desc_string


def extract_code_from_generator(content):
    """Extract code from the response of the code generator."""
    pattern_code = r"```python(.*?)```"
    code_string = re.search(pattern_code, content, re.DOTALL)
    code_string = code_string.group(1).strip() if code_string is not None else None
    if code_string is None:
        # Find the line that starts with "def" and the line that starts with "return", and extract the code in between
        lines = content.split("\n")
        start = None
        end = None
        for i, line in enumerate(lines):
            if line.startswith("def"):
                start = i
            if "return" in line:
                end = i
                break
        if start is not None and end is not None:
            code_string = "\n".join(lines[start : end + 1])

    if code_string is None:
        return None
    # Add import statements if not present
    if "np" in code_string:
        code_string = "import numpy as np\n" + code_string
    if "torch" in code_string:
        code_string = "import torch\n" + code_string
    return code_string


def filter_code(code_string):
    """Remove lines containing signature and import statements."""
    lines = code_string.split("\n")
    filtered_lines = []
    for line in lines:
        if line.startswith("def"):
            continue
        elif line.startswith("import"):
            continue
        elif line.startswith("from"):
            continue
        elif line.startswith("return"):
            filtered_lines.append(line)
            break
        else:
            filtered_lines.append(line)
    code_string = "\n".join(filtered_lines)
    return code_string


def get_heuristic_name(module, possible_names: list[str]):
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name


def parse_metrics_from_stdout(stdout_path: str):
    """
    Parse ADE and FDE metrics from the stdout string.

    Args:
        stdout_str (str): The stdout content as a string

    Returns:
        tuple: (ade, fde) as float values
    """
    # Initialize values to None
    ade = None
    fde = None

    stdout_str = file_to_string(stdout_path)

    # Split the stdout by lines
    lines = stdout_str.split("\n")

    # logger.info(stdout_str) # TODO: remove

    # Try to find lines containing "ADE: " and "FDE: "
    for i, line in enumerate(lines):
        if "[*] Average metrics" in line:
            # Look for ADE in the next line
            if i + 1 < len(lines) and "ADE:" in lines[i + 1]:
                ade_line = lines[i + 1]
                ade = float(ade_line.split("ADE:")[1].strip())
            # Look for FDE in the line after ADE
            if i + 2 < len(lines) and "FDE:" in lines[i + 2]:
                fde_line = lines[i + 2]
                fde = float(fde_line.split("FDE:")[1].strip())
            break

    # If we couldn't find the metrics in the expected format, try an alternative approach
    if ade is None or fde is None:
        # Look for any line containing ADE and FDE
        for line in lines:
            if ade is None and "ADE:" in line:
                try:
                    ade = float(line.split("ADE:")[1].strip())
                except (ValueError, IndexError):
                    pass

            if fde is None and "FDE:" in line:
                try:
                    fde = float(line.split("FDE:")[1].strip())
                except (ValueError, IndexError):
                    pass

    # Check if we found the values
    if ade is None or fde is None:
        # raise ValueError("Could not parse ADE and FDE values from the output")
        logger.error("Could not parse ADE and FDE values from the output")
    return ade, fde


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility across multiple libraries.

    Args:
        seed: Integer seed value to use
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    # Try to set PyTorch seeds if available
    try:
        import torch

        torch.manual_seed(seed)  # no GPU here
    except ImportError:
        pass

    logger.info(f"Set random seed to {seed}!")
