from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(
    title="RepoAnalyser AI",
    description="An AI-powered code documentation assistant",
    version="0.1.0",
)


class HealthResponse(BaseModel):
    status: str
    application: str
    version: str


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="healthy",
        application="RepoAnalyser AI",
        version="0.1.0",
    )