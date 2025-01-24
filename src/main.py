import click
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BartForConditionalGeneration, BartTokenizer
from langchain.chains import LLMChain

app = FastAPI()

# Initialize the model and tokenizer
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)


class Question(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(question: Question):
    inputs = tokenizer(question.question, return_tensors="pt",
                       max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"], num_beams=4, max_length=50, min_length=10, early_stopping=True)
    answer = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return {"question": question.question, "answer": answer}


@click.command()
@click.argument('question')
def ask(question):
    inputs = tokenizer(question, return_tensors="pt",
                       max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"], num_beams=4, max_length=50, min_length=10, early_stopping=True)
    answer = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(f"Question: {question}")
    print(f"Answer: {answer}")


if __name__ == '__main__':
    ask()
