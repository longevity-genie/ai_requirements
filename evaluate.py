import json
import os

import polars as pl
import yaml
import click
from pathlib import Path
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain_anthropic.experimental import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, load_prompt
from langchain_core.runnables import RunnableSerializable
from pycomfort.config import load_environment_keys
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

load_environment_keys(usecwd=True)
base = Path(".")
data = base / "data"
input = data / "input"
output = data / "output"
self_evaluations = output / "self_evaluations"
pairwise_evaluations = output / "pairwise_evaluations"
perplexity_yaml = input / "rapamycin_perplexity_yaml"

prompts = base / "prompts"

defaults = yaml.safe_load((prompts / "defaults.yaml").open('r'))
default_question = defaults["question"]
default_requirements = defaults["requirements"]
default_format = defaults["format"]

default_format_pairwise = defaults["format_pairwise"]

default_answer = perplexity_yaml / 'claude_opus' / "claude_opus_all_requirements.yaml"
default_answer_2 = perplexity_yaml / 'claude_opus' / "claude_opus_no_requirements.yaml"

default_models = ["claude-3-opus-20240229", "gpt-4-turbo", "llama3-70b-8192"]

#setting up prompt and output parser
prompt_evaluate_answer = load_prompt(path =prompts / "self_evaluate.yaml")
output_parser = JsonOutputParser()

#setting up OpenAI
model_gpt_4 = ChatOpenAI(model="gpt-4-turbo", temperature=0.0)
chain_evaluate_answer_gpt_4: LLMChain = prompt_evaluate_answer | model_gpt_4 | output_parser

model_llama_3 = ChatGroq(model="llama3-70b-8192", temperature=0.0)
chain_evaluate_answer_llama_3: LLMChain = prompt_evaluate_answer | model_llama_3 | output_parser

#setting up Claude
model_claude = ChatAnthropic(model="claude-3-opus-20240229", temperature=0.0)
chain_evaluate_answer_claude_opus: LLMChain = prompt_evaluate_answer | model_claude | output_parser


prompt_pairwise_comparison = load_prompt(path =prompts / "pairwise_comparison.yaml")

chain_pairwise_comparison_gpt_4: LLMChain = prompt_pairwise_comparison | model_gpt_4 | output_parser
chain_pairwise_comparison_claude_opus: LLMChain = prompt_pairwise_comparison | model_claude | output_parser
chain_pairwise_comparison_llama_3: LLMChain = prompt_pairwise_comparison | model_llama_3 | output_parser



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


@click.group(invoke_without_command=True)
@click.pass_context
def app(ctx):
    pass

@app.command("compare_answers")
@click.option('--model', default = "llama3-70b-8192", help="the model to use for evaluation, llama3-70b-8192 by default")
@click.option('--answer_1', default = default_answer, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--answer_2', default = default_answer_2, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--question', default = default_question, help="question that we asked the model")
@click.option('--requirements', default = default_requirements, help="criteria that we used for evaluation")
@click.option('--format', default = default_format_pairwise, help="criteria to format the answer")
@click.option("--where", default=pairwise_evaluations, type=click.Path(exists=True, dir_okay=True, path_type=Path), help="folder where to write output")
@click.option("--num", default = 0, type=click.INT)
def compare_answers(model: str, answer_1: Path, answer_2: Path, question: str, requirements: str, format: str, where: Path, num: int):
    input = {
        "question" : question,
        "answer_1_name": answer_1.name,
        "answer_1" : read_response(answer_1),
        "answer_2_name": answer_2.name,
        "answer_2" : read_response(answer_2),
        "requirements": requirements,
        "format": format
    }
    if "claude" in model:
        chain = chain_pairwise_comparison_claude_opus
    elif "gpt" in model:
        chain = chain_pairwise_comparison_gpt_4
    else:
        chain = chain_pairwise_comparison_llama_3
    result = chain.invoke(input)
    click.echo(f"RESULT:\n {result}")
    number_suffix = "" if num == 0 else "_" + str(num)
    to_write = where / f"evaluation_{answer_1.stem}_VS_{answer_2.stem}_{model}{number_suffix}.yaml"
    click.echo(f"Writing to: {to_write}")
    yaml.dump(result, to_write.open('w'))
    return result

@app.command('compare_answer_against_folder')
@click.option('--folder', type=click.Path(exists=True, file_okay=False), required=True, help='Path to the folder containing YAML files.')
@click.option('--models', multiple=True, default=default_models, help='List of model names to evaluate.')
@click.option('--question', default = default_question, help="question that we asked the model")
@click.option('--answer_1', default = default_answer, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--requirements', default = default_requirements, help="criteria that we used for evaluation")
@click.option('--format', default = default_format_pairwise, help="criteria to format the answer")
@click.option("--where", default=pairwise_evaluations, type=click.Path(exists=True, dir_okay=True, path_type=Path), help="folder where to write output")
@click.option("--num", default = 0, type=click.INT)
@click.pass_context
def evaluate_folder(ctx: click.Context, folder: str, models: List[str], question: str, answer_1: Path, requirements: str, format: str, where: Path, num: int):
    folder_path = Path(folder)
    yaml_files = [f for f in folder_path.iterdir() if f.suffix == '.yaml' or f.suffix == ".md"]
    for yaml_file in yaml_files:
        for model in models:
            if yaml_file != answer_1:
                ctx.invoke(compare_answers, model=model, question=question, answer_1 = answer_1, answer_2 = yaml_file,
                           requirements=requirements, format=format, where=where, num=num)

def read_response(filepath: Path) -> str:
    if filepath.suffix == ".md":
        return filepath.read_text()
    else:
        return str(extract_fields(filepath))

@app.command("evaluate_answer")
@click.option('--model', default = "llama3-70b-8192", help="the model to use for evaluation, llama3-70b-8192 by default")
@click.argument('filepath', default = default_answer, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--question', default = default_question, help="question that we asked the model")
@click.option('--requirements', default = default_requirements, help="criteria that we used for evaluation")
@click.option('--format', default = default_format, help="criteria to format the answer")
@click.option("--where", default=self_evaluations, type=click.Path(exists=True, dir_okay=True, path_type=Path), help="folder where to write output")
@click.option("--num", default = 0, type=click.INT)
def evaluate_answer(model: str, filepath: Path, question: str, requirements: str, format: str, where: Path, num: int):
    input = {
        "question" : question,
        "response" : read_response(filepath),
        "requirements": requirements,
        "format": format
    }
    if "claude" in model:
        chain = chain_evaluate_answer_claude_opus
    elif "gpt" in model:
        chain = chain_evaluate_answer_gpt_4
    else:
        chain = chain_evaluate_answer_llama_3
    result = chain.invoke(input)
    click.echo(f"RESULT:\n {result}")
    number_suffix = "" if num == 0 else "_" + str(num)
    to_write = where / f"evaluation_{filepath.stem}_{model}{number_suffix}.yaml"
    click.echo(f"Writing to: {to_write}")
    yaml.dump(result, to_write.open('w'))
    return result


@app.command('evaluate_folder')
@click.option('--folder', type=click.Path(exists=True, file_okay=False), required=True, help='Path to the folder containing YAML files.')
@click.option('--models', multiple=True, default=default_models, help='List of model names to evaluate.')
@click.option('--question', default = default_question, help="question that we asked the model")
@click.option('--requirements', default = default_requirements, help="criteria that we used for evaluation")
@click.option('--format', default = default_format, help="criteria to format the answer")
@click.option("--where", default=self_evaluations, type=click.Path(exists=True, dir_okay=True, path_type=Path), help="folder where to write output")
@click.option("--num", default = 0, type=click.INT)
@click.pass_context
def evaluate_folder(ctx: click.Context, folder: str, models: List[str], question: str, requirements: str, format: str, where: Path, num: int):
    folder_path = Path(folder)
    yaml_files = [f for f in folder_path.iterdir() if f.suffix == '.yaml' or f.suffix == '.md']
    for yaml_file in yaml_files: #can also be .md files
        for model in models:
            ctx.invoke(evaluate_answer, model=model, filepath=yaml_file, question=question,
                       requirements=requirements, format=format, where=where, num=num)


import re

def parse_filename(filename: str) -> dict:
    """
    Parses the given filename to extract key components based on a predefined pattern.

    Args:
    filename (str): The filename to parse.

    Returns:
    dict: A dictionary containing the parsed components or an error message if the pattern does not match.
    """
    # Define the regex pattern that matches the expected filename structure
    pattern = r'evaluation_(?P<model_that_produced_the_result>.+?)_(?P<requirements_type>with|without)_requirements_(?P<model_that_evaluated_result>.+)'

    # Attempt to match the pattern with the filename (excluding the file extension if present)
    match = re.match(pattern, filename.split('.')[0])

    if match:
        return match.groupdict()
    else:
        return {"error": "Filename pattern does not match expected format."}

@app.command("make_table")
@click.option('--directory', type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help="Directory containing YAML files.")
@click.option('--output', type=click.Path(path_type=Path), default='comparison_table.tsv',
              help="Output CSV file name.")
@click.option('--add-cols', is_flag=True, help="Add additional columns from filename parsing.")
def make_table(directory: str, output: str, add_cols: bool) -> Path:
    """ Process YAML files in the specified directory to generate a comparison table using Polars. """
    path = Path(directory)
    yaml_files = list(path.glob('*.yaml'))

    if not yaml_files:
        click.echo("No YAML files found in the directory.")
        return

    # Load the first file to determine the columns (requirement names)
    with open(yaml_files[0], 'r') as f:
        first_file_data = yaml.safe_load(f)
        if not first_file_data:
            click.echo(f"No data found in file: {yaml_files[0].name}")
            return
        columns = list(first_file_data.keys())

    records: List[Dict[str, str]] = []

    # Process each YAML file
    for file in yaml_files:
        click.echo(f"Processing file: {file.name}")  # Debug: file being processed
        with open(file, 'r') as f:
            try:
                data: Dict[str, Dict[str, str]] = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                click.echo(f"Error parsing YAML file {file.name}: {exc}")
                continue

        # Prepare a record dict starting with the filename
        record: Dict[str, str] = {'filename': file.stem}

        # Extract the score for each requirement
        for column in columns:
            score = data.get(column, {}).get('score', 'N/A')  # Default to 'N/A' if no score found
            record[column] = score
        # Optionally add additional columns from filename parsing
        if add_cols:
            filename_info = parse_filename(file.name)
            if 'error' not in filename_info:
                record.update(filename_info)
            else:
                click.echo(f"Error parsing filename: {file.name}")
        records.append(record)


    # Create a DataFrame and write to CSV
    if records:
        df = pl.DataFrame(records)
        df.write_csv(path / output, separator="\t")
        click.echo(f"Comparison table generated and saved to {output}")
    else:
        click.echo("No records found. No CSV generated.")


if __name__ == "__main__":
    app()