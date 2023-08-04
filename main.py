import openai

from agents.baby_agi import run
from config import config

openai.api_key = config.get("OPENAI_API_KEY")


if __name__ == "__main__":
    run("Write a weather report for SF today")