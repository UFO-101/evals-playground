# %%
#!%load_ext autoreload
#!%autoreload 2
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import exact
from inspect_ai.solver import generate


@task
def hello_world():
    return Task(
        dataset=[
            Sample(
                input="Just reply with Hello World",
                target="Hello World",
            )
        ],
        solver=[generate()],
        scorer=exact(),
    )


from inspect_ai import task
from inspect_ai.dataset import example_dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import system_message

SYSTEM_MESSAGE = """
You are a computer security expert and you are tasked with
providing concise response to the following questions.
Please assume that the reader is also well versed in
computer security and provide a short response in a few words.
"""


@task
def security_guide():
    return Task(
        dataset=example_dataset("security_guide"),
        solver=[system_message(SYSTEM_MESSAGE), generate()],
        scorer=model_graded_fact(),
    )


from inspect_ai import task

SYSTEM_MESSAGE = """
Choose the most plausible continuation for the story.
"""


from inspect_ai import task
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

SYSTEM_MESSAGE = """
Choose the most plausible continuation for the story.
"""


def record_to_sample(record: dict[str, Any]):
    return Sample(
        input=record["ctx"],
        target=chr(ord("A") + int(record["label"])),
        choices=record["endings"],
        metadata=dict(source_id=record["source_id"]),
    )


@task
def hellaswag():
    # dataset
    dataset = hf_dataset(
        path="hellaswag", split="validation", sample_fields=record_to_sample, trust=True
    )

    # define task
    return Task(
        dataset=dataset,
        solver=[system_message(SYSTEM_MESSAGE), multiple_choice()],
        scorer=choice(),
    )


from inspect_ai import task
from inspect_ai.scorer import match
from inspect_ai.solver import prompt_template


def record_to_sample(record):
    DELIM = "####"
    input = record["question"]
    answer = record["answer"].split(DELIM)
    target = answer.pop().strip()
    reasoning = DELIM.join(answer)
    return Sample(input=input, target=target, metadata={"reasoning": reasoning.strip()})


def sample_to_fewshot(sample):
    return (
        f"{sample.input}\n\nReasoning:\n"
        + f"{sample.metadata['reasoning']}\n\n"
        + f"ANSWER: {sample.target}"
    )


# setup for problem + instructions for providing answer
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your
response should be of the form "ANSWER: $ANSWER" (without quotes) 
where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form
"ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to 
the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


@task
def gsm8k(fewshot=10, fewshot_seed=42):
    # build solver list dynamically (may or may not be doing fewshot)
    solver = [prompt_template(MATH_PROMPT_TEMPLATE), generate()]
    if fewshot:
        fewshots = hf_dataset(
            path="gsm8k",
            data_dir="main",
            split="train",
            sample_fields=record_to_sample,
            shuffle=True,
            seed=fewshot_seed,
            limit=fewshot,
        )
        solver.insert(
            0,
            system_message(
                "\n\n".join([sample_to_fewshot(sample) for sample in fewshots])
            ),
        )

    # define task
    return Task(
        dataset=hf_dataset(
            path="gsm8k",
            data_dir="main",
            split="test",
            sample_fields=record_to_sample,
        ),
        solver=solver,
        scorer=match(numeric=True),
    )


# %%
# %%
