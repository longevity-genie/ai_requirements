import yaml
import click
from pathlib import Path
from typing import Dict, Any, Optional
from langchain_community.chat_models import ChatPerplexity, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pycomfort.config import load_environment_keys

model =
prompt = ChatPromptTemplate.from_template("Evaluate the response to te question according to the criteria: \n Initial question: ```{question}``` \n The Response is: ```{response}``` \n Evaluation criteria ```{evaluation}```")
output_parser = StrOutputParser()
chain = prompt | model | output_parser

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
        data = yaml.safe_load(file)
        # Extract 'answer' and 'sources', defaulting to None if not found
        extracted_data = {
            "answer": data.get("answer", None),
            "sources": data.get("sources", None)
        }
        return extracted_data

@click.group()
def cli():
    """Simple CLI for processing YAML files."""
    pass

@cli.command("test")
@click.option("model", default="mixtral-8x22b-instruct")
def test():
    load_environment_keys(usecwd=True)
    from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


chain = prompt | model | output_parser
model = ChatOpenAI(model_name ="gpt-4-turbo-2024-04-09", temperature=0)


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