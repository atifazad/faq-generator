from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains import LLMChain

app = FastAPI()


class Question(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(question: Question):
    # Your code to handle the question
    response = f"Answer: This is a placeholder answer for '{
        question.question}'"
    return {"question": question.question, "answer": response}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
