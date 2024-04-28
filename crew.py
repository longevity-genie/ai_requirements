from pathlib import Path

import yaml
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_groq import ChatGroq
from pycomfort.config import load_environment_keys
import click
load_environment_keys(usecwd=True)

base = Path(".")
data = base / "data"
input = data / "input"
output = data / "output"
prompts = base / "prompts"
defaults = yaml.safe_load((prompts / "defaults.yaml").open('r'))
default_question = defaults["question"]


@CrewBase
class GeroCrew:
    """FinancialAnalystCrew crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self) -> None:
        self.groq_llm = ChatGroq(model="llama3-70b-8192", temperature=0.0)

    @agent
    def scientist(self) -> Agent:
        return Agent(
            config = self.agents_config['scientist'],
            llm = self.groq_llm
        )

    @agent
    def reviewer(self) -> Agent:
        return Agent(
            config = self.agents_config['reviewer'],
            llm = self.groq_llm
        )

    @task
    def research_topic_task(self) -> Task:
        return Task(
            config = self.tasks_config['research_topic_task'],
            agent = self.scientist()
        )

    @task
    def review_answer_task(self) -> Task:
        result = Task(
            config=self.tasks_config['review_answer_task'],
            agent=self.reviewer()
        )
        result.context = [self.research_topic_task()] #temporal bug-fix
        return result

    @crew
    def crew(self) -> Crew:
        """Creates a requirements crew"""
        return Crew(
            agents = self.agents,
            tasks = self.tasks,
            process = Process.sequential,
            verbose = 2
        )

@click.group(invoke_without_command=True)
@click.pass_context
def app(ctx):
    pass

@app.command('run')
@click.option('--question', default = default_question, help="question that we asked the model")
def run(question: str = default_question):
    crew = GeroCrew().crew()
    inputs = {
        'question': question
    }
    crew.kickoff(inputs=inputs)


if __name__ == "__main__":
    app()