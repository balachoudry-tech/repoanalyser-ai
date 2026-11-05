
"""
Pydantic API schemas for CodePilot AI.

This module defines the public request and response contracts
exposed by the FastAPI application.

Responsibilities:
    - Validate incoming API requests.
    - Define predictable API response structures.
    - Validate source citation metadata.
    - Represent token usage and execution timings.
    - Generate API documentation through Pydantic models.

These schemas are API contracts. They are deliberately separate
from internal service implementation details.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ============================================================
# BASE CONFIGURATION
# ============================================================


class APIBaseModel(BaseModel):
    """
    Base model for CodePilot AI API schemas.

    Extra fields are rejected to prevent clients from accidentally
    sending unsupported properties.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


# ============================================================
# REPOSITORY INDEXING SCHEMAS
# ============================================================


class IndexRepositoryRequest(APIBaseModel):
    """
    Request schema for indexing a local repository.

    Attributes:
        repository_path: Absolute or relative path to the repository.
        repository_id: Unique identifier used to isolate repository data.
    """

    repository_path: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Local filesystem path to the repository.",
        examples=["/Users/developer/projects/my-app"],
    )

    repository_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Unique identifier for the repository.",
        examples=["my-app"],
    )

    @field_validator("repository_path", "repository_id", mode="before")
    @classmethod
    def validate_non_empty_string(cls, value):
        """Reject values containing only whitespace."""

        if isinstance(value, str):
            value = value.strip()

            if not value:
                raise ValueError("Value cannot be empty or whitespace.")

        return value


class IndexRepositoryResponse(APIBaseModel):
    """
    Response schema for a repository indexing operation.

    Attributes:
        repository_id: Repository identifier.
        repository_path: Path of the indexed repository.
        total_files: Number of discovered files.
        parsed_files: Number of successfully parsed files.
        total_code_units: Number of extracted code units.
        total_chunks: Number of generated chunks.
        indexed_chunks: Number of chunks persisted in vector storage.
        status: Indexing operation status.
    """

    repository_id: str
    repository_path: str

    total_files: int = Field(ge=0)
    parsed_files: int = Field(ge=0)
    total_code_units: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    indexed_chunks: int = Field(ge=0)

    status: str


# ============================================================
# QUESTION-ANSWERING REQUEST SCHEMA
# ============================================================


class AskQuestionRequest(APIBaseModel):
    """
    Request schema for asking questions about a repository.

    Attributes:
        question: Natural-language question about the codebase.
        top_k: Maximum number of relevant chunks to retrieve.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Natural-language question about the repository.",
        examples=[
            "Where is user authentication implemented?"
        ],
    )

    top_k: int = Field(
    default=5,
    ge=1,
    le=20,
    strict=True,
    description="Maximum number of relevant code chunks to retrieve.",
    )


    @field_validator("question", mode="before")
    @classmethod
    def validate_question(cls, value):
        """Reject empty or whitespace-only questions."""

        if isinstance(value, str):
            value = value.strip()

            if not value:
                raise ValueError("Question cannot be empty.")

        return value


# ============================================================
# SOURCE CITATION SCHEMA
# ============================================================


class SourceCitation(APIBaseModel):
    """
    Represent a source-code reference returned by the RAG pipeline.

    Attributes:
        source_id: Citation identifier such as S1 or S2.
        file_path: Repository-relative source file path.
        symbol_name: Function, class or symbol name.
        start_line: Starting source-code line.
        end_line: Ending source-code line.
        language: Programming language.
        vector_id: Identifier stored in the vector database.
        distance: Vector similarity distance returned by ChromaDB.
    """

    source_id: str
    file_path: str

    symbol_name: str | None = None

    start_line: int | None = Field(
        default=None,
        ge=1,
    )

    end_line: int | None = Field(
        default=None,
        ge=1,
    )

    language: str | None = None

    vector_id: str | None = None

    distance: float | None = None


# ============================================================
# LLM TOKEN USAGE SCHEMA
# ============================================================


class TokenUsage(APIBaseModel):
    """
    Represent LLM token consumption.

    Token counts are useful for monitoring API usage and estimating
    the operational cost of the question-answering workflow.
    """

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


# ============================================================
# EXECUTION TIMING SCHEMA
# ============================================================


class ExecutionTiming(APIBaseModel):
    """
    Represent execution duration for individual RAG stages.

    All durations are expressed in milliseconds.
    """

    retrieval: float = Field(ge=0)
    context_building: float = Field(ge=0)
    llm_generation: float = Field(ge=0)
    total: float = Field(ge=0)


# ============================================================
# QUESTION-ANSWERING RESPONSE SCHEMA
# ============================================================


class AskQuestionResponse(APIBaseModel):
    """
    Response schema for repository question answering.

    Attributes:
        answer: Generated natural-language answer.
        sources: Source-code references supporting the answer.
        repository_id: Repository searched.
        retrieval_count: Number of chunks retrieved.
        context_source_count: Number of sources included in context.
        skipped_context_count: Number of chunks omitted by ContextBuilder.
        model: LLM model used.
        usage: Input and output token counts.
        timing_ms: Execution duration for each pipeline stage.
        request_id: Provider request identifier, if available.
        citation_warnings: Unknown citation identifiers detected.
    """

    answer: str

    sources: list[SourceCitation] = Field(
        default_factory=list
    )

    repository_id: str

    retrieval_count: int = Field(ge=0)

    context_source_count: int = Field(ge=0)

    skipped_context_count: int = Field(ge=0)

    model: str

    usage: TokenUsage

    timing_ms: ExecutionTiming

    request_id: str | None = None

    citation_warnings: list[str] = Field(
        default_factory=list
    )


# ============================================================
# HEALTH RESPONSE SCHEMA
# ============================================================


class HealthResponse(APIBaseModel):
    """
    Represent the basic application health response.

    This schema currently describes application liveness,
    not the readiness of external dependencies.
    """

    status: str
    application: str
    version: str
