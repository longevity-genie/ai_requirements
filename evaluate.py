import os

import yaml
import click
from pathlib import Path
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_anthropic.experimental import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, load_prompt
from langchain_core.runnables import RunnableSerializable
from pycomfort.config import load_environment_keys
from langchain.chains import LLMChain

load_environment_keys(usecwd=True)
base = Path(".")
data = base / "data"
prompts = base / "prompts"

defaults = yaml.safe_load((prompts / "defaults.yaml").open('r'))
default_question = defaults["question"]
default_requirements = defaults["requirements"]
default_format = defaults["format"]

#setting up prompt and output parser
prompt_evaluate_answer = load_prompt(path =prompts / "self_evaluate.yaml")
output_parser = JsonOutputParser()

#setting up OpenAI
model_gpt_4 = ChatOpenAI(model="gpt-4-turbo", temperature=0.0)
chain_evaluate_answer_gpt_4: LLMChain = prompt_evaluate_answer | model_gpt_4 | output_parser

#setting up Claude
model_claude = ChatAnthropic(model="claude-3-opus-20240229", temperature=0.0)
chain_evaluate_answer_claude_opus: LLMChain = prompt_evaluate_answer | model_claude | output_parser


def extract_fields(yaml_file: Path) -> Dict[str, Optional[Any]]:
    """
    Extracts the 'answer' and 'sources' fields from a YAML file and returns them as a dictionary.

    Args:
        yaml_file (Path): The path to the YAML file.

    Returns:
        Dict[str, Optional[Any]]: A dictionary with 'answer' and 'sources' as keys. If fields are not found,
        the value will be None.
    """
    with yaml_file.open('r') as file:
        loaded = yaml.safe_load(file)
        # Extract 'answer' and 'sources', defaulting to None if not found
        extracted_data = {
            "answer": loaded.get("answer", None),
            "sources": loaded.get("sources", None)
        }
        return extracted_data


@click.group()
def cli():
    """Simple CLI for processing YAML files."""
    pass

@cli.command("test")
@click.option('--model', default = "claude-3-opus", help="the model to use for evaluation, claude-3-opus by default")
@click.option('--question', default = default_question, help="question that we asked the model")
@click.option('--requirements', default = default_requirements, help="criteria that we used for evaluation")
@click.option('--format', default = default_format, help="criteria to format the answer")
def test(model: str, question: str, requirements: str, format: str):
    example = data / "rapamycin" / 'claude_opus' / "claude_opus_all_requirements.yaml"
    dic = extract_fields(example)
    response = str(dic)

    input = {
        "question" : question,
        "response" : response,
        "requirements": requirements,
        "format": format
    }
    chain = chain_evaluate_answer_claude_opus if "claude" in model else chain_evaluate_answer_gpt_4
    result = chain.invoke(input)
    click.echo(f"RESULT:\n {result}")
    return result




@cli.command()
@click.argument('filepath', type=click.Path(exists=True, dir_okay=False, path_type=Path))
def extract(filepath: Path):
    """
    Extracts and prints 'answer' and 'sources' from a specified YAML file.

    Args:
        filepath (Path): The path to the YAML file to be processed.
    """
    result = extract_fields(filepath)
    click.echo(f"Answer: {result['answer']}")
    if result['sources']:
        click.echo(f"Sources: {result['sources']}")
    else:
        click.echo("Sources: None")

if __name__ == "__main__":
    cli()