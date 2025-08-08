from fastapi import FastAPI, Body
from typing import Optional
import uvicorn

from main import openai_llm_response

app = FastAPI()

@app.post("/ask")
async def ask(user_query: str = Body(...), previous_response_id: Optional[str] = Body(None), model: str = "gpt-4.1-nano"):
    response_text, previous_response_id = openai_llm_response(
        user_query=user_query, 
        model=model, 
        previous_response_id=previous_response_id
    )
    return {"response_text": response_text, "previous_response_id": previous_response_id}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)