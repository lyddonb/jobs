from typing import Optional
from langchain import OpenAI

from agents.baby_agi.baby_agi import BabyAGI
from agents.baby_agi.vectorstore import get_vectorstore


def run(objective: str, max_iterations: Optional[int] = 3, verbose: bool = False):
    llm = OpenAI(temperature=0)
    vectorstore = get_vectorstore()

    # If None, will keep on going forever
    baby_agi = BabyAGI.from_llm(
        llm=llm, vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations
    )

    return baby_agi({"objective": objective})