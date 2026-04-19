import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from langsmith import Client, tracing_context

from src.graphs.graph_builder import GraphBuilder
from src.llms.groqllm import GroqLLM

load_dotenv()

app = FastAPI()

ls_client = Client(
    api_key=os.getenv("LANGSMITH_API_KEY"),
    api_url="https://api.smith.langchain.com",
)

@app.post("/blogs")
async def create_blogs(request: Request):
    data = await request.json()
    topic = data.get("topic", "")
    language = data.get("language", "")
    print(language)

    groqllm = GroqLLM()
    llm = groqllm.get_llm()

    graph_builder = GraphBuilder(llm)

    with tracing_context(
        enabled=True,
        client=ls_client,
        project_name="Blog Post Generator"
    ):
        if topic and language:
            graph = graph_builder.setup_graph(usecase="language")
            state = graph.invoke({"topic": topic, "current_language": language.lower()})
        elif topic:
            graph = graph_builder.setup_graph(usecase="topic")
            state = graph.invoke({"topic": topic})
        else:
            return {"data": "Topic is required"}

    return {"data": state}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)