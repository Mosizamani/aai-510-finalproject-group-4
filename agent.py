import copy
import json
import warnings
from typing import Any, Callable, Generator, Optional
from uuid import uuid4

import backoff
import mlflow
import numpy as np
import openai
import pandas as pd
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from openai import OpenAI
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, udf
from pyspark.sql.types import DoubleType
from sentence_transformers import SentenceTransformer
from databricks_openai import UCFunctionToolkit
from unitycatalog.ai.core.base import get_uc_function_client


############################################
# LLM endpoint — replaced by notebook cell
############################################
LLM_ENDPOINT_NAME = "databricks-gpt-oss-120b"

SYSTEM_PROMPT = """You are an AI Review Analysis Assistant specializing in generative AI application reviews.

Your role is to help users understand user feedback and sentiment about AI applications like
ChatGPT, Claude, Gemini, Midjourney, and other generative AI tools.

You have access to one tool:
- group4__default__all_ai_applications_rating: list all AI applications and their average rating.
- group4__default__fetch_review_theme_category: fetch all review theme categories.
- group4__default__fetch_review_contents: retrieve review content for a specific category, with an optional keyword filter, while limiting the number of records returned per request.


For every user question, follow this process:
1. Think about what information is needed.
2. Choose the correct tool.
3. Review the tool result.
4. If more information is needed, call another tool.
5. Then provide the final answer.

Always ground the final answer in tool results."""


############################################
# Embedding model — lazy loaded once
############################################
_embedding_model: Optional[SentenceTransformer] = None

def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _embedding_model

############################################
# Cosine similarity Spark UDF
############################################
def _cosine_similarity(vec1, vec2) -> float:
    a = np.array(vec1)
    b = np.array(vec2)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0

_cosine_sim_udf = udf(_cosine_similarity, DoubleType())

############################################
# Core retrieval logic
############################################
def _vector_search_exec(
    query: str,
    top_k: int = 10,
    min_similarity: float = 0.0,
) -> str:
    """
    Encode query, compute cosine similarity against ai_reviews_embeddings,
    return formatted context string ready for the LLM.
    """
    spark = SparkSession.builder.getOrCreate()
    model = _get_embedding_model()
    query_vec = model.encode(query).tolist()

    df = spark.table("group4.default.ai_reviews_embeddings")
    df = df.withColumn(
        "similarity_score",
        _cosine_sim_udf(col("embedding"), lit(query_vec)),
    )

    if min_similarity > 0:
        df = df.filter(col("similarity_score") >= min_similarity)

    results: pd.DataFrame = (
        df.orderBy(col("similarity_score").desc())
        .limit(int(top_k))
        .select(
            "App",
            "Review_Text",
            "Star_Rating",
            "Review_Date",
            "Sentiment_Polarity",
            "Review_Theme",
            "similarity_score",
        )
        .toPandas()
    )

    if results.empty:
        return "No relevant reviews found."

    parts = []
    for i, row in results.iterrows():
        parts.append(
            f"Review {i + 1} (Similarity: {row['similarity_score']:.3f}):\n"
            f"App: {row['App']}\n"
            f"Rating: {row['Star_Rating']}/5\n"
            f"Date: {row['Review_Date']}\n"
            f"Theme: {row['Review_Theme']}\n"
            f"Sentiment: {row['Sentiment_Polarity']:.2f}\n"
            f"Review: {row['Review_Text']}\n"
        )
    return "\n---\n\n".join(parts)

############################################
# Tool specs (OpenAI function-calling format)
############################################
_VECTOR_SEARCH_SPEC = {
    "type": "function",
    "function": {
        "name": "vector_search",
        "description": (
            "Search reviews for qualitative topics: complaints, praise, specific user experiences. "
            "Do NOT use for aggregate statistics, rankings, or average ratings — "
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query describing what reviews to find",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to return (default: 10)",
                    "default": 10,
                },
                "min_similarity": {
                    "type": "number",
                    "description": "Minimum similarity threshold 0-1 (default: 0.0)",
                    "default": 0.0,
                },
            },
            "required": ["query"],
        },
    },
}

############################################
# ToolInfo + agent boilerplate (unchanged)
############################################
class ToolInfo(BaseModel):
    name: str
    spec: dict
    exec_fn: Callable

    class Config:
        arbitrary_types_allowed = True

def create_tool_info(tool_spec, exec_fn_param: Optional[Callable] = None):
    tool_spec["function"].pop("strict", None)
    tool_name = tool_spec["function"]["name"]
    udf_name = tool_name.replace("__", ".")

    # Define a wrapper that accepts kwargs for the UC tool call,
    # then passes them to the UC tool execution client
    def exec_fn(**kwargs):
        function_result = uc_function_client.execute_function(udf_name, kwargs)
        if function_result.error is not None:
            return function_result.error
        else:
            return function_result.value
    return ToolInfo(name=tool_name, spec=tool_spec, exec_fn=exec_fn_param or exec_fn)


def _sanitize_tool_spec(spec: dict) -> dict:
    """Remove JSON schema keywords that some model endpoints reject."""
    spec = copy.deepcopy(spec)
    params = spec.get("function", {}).get("parameters") or {}
    if not isinstance(params, dict) or "properties" not in params:
        return spec
    for prop in params.get("properties", {}).values():
        if not isinstance(prop, dict):
            continue
        t = prop.get("type")
        if t == "string":
            for key in ("minLength", "maxLength", "pattern", "format"):
                prop.pop(key, None)
        elif t in ("integer", "number"):
            for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "format"):
                prop.pop(key, None)
        elif t == "array":
            for key in ("minItems", "maxItems", "uniqueItems"):
                prop.pop(key, None)
            items = prop.get("items")
            if isinstance(items, dict):
                for key in ("minLength", "maxLength", "pattern", "format",
                            "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
                    items.pop(key, None)
    return spec


# ---- Register tools here ----
TOOL_INFOS: list[ToolInfo] = [
    # disable vetor search, it is so slow
    # ToolInfo(
    #     name="vector_search",
    #     spec=_VECTOR_SEARCH_SPEC,
    #     exec_fn=_vector_search_exec,
    # ),
]
# Add more ToolInfo entries here if needed
UC_TOOL_NAMES = ["group4.default.all_ai_applications_rating", 
                 "group4.default.fetch_review_theme_category", 
                 "group4.default.fetch_review_contents"
                 ]

uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
uc_function_client = get_uc_function_client()
for tool_spec in uc_toolkit.tools:
    TOOL_INFOS.append(create_tool_info(tool_spec))

############################################
# ToolCallingAgent (structure unchanged)
############################################
class ToolCallingAgent(ResponsesAgent):
    def __init__(self, llm_endpoint: str, tools: list[ToolInfo]):
        self.llm_endpoint = llm_endpoint
        self.workspace_client = WorkspaceClient()
        self.model_serving_client: OpenAI = (
            self.workspace_client.serving_endpoints.get_open_ai_client()
        )
        self._tools_dict = {tool.name: tool for tool in tools}

    def get_tool_specs(self) -> list[dict]:
        return [_sanitize_tool_spec(t.spec) for t in self._tools_dict.values()]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        sane_args = {k: v for k, v in (args or {}).items() if k and isinstance(k, str)}
        name = tool_name.strip().strip('"').strip("'")
        if "<" in name:
            name = name.split("<")[0].strip()
        if name in self._tools_dict:
            return self._tools_dict[name].exec_fn(**sane_args)
        candidates = [k for k in self._tools_dict if name.startswith(k)]
        if candidates:
            return self._tools_dict[max(candidates, key=len)].exec_fn(**sane_args)
        raise KeyError(f"Unknown tool: {tool_name!r}. Known tools: {list(self._tools_dict.keys())}")

    def call_llm(self, messages: list[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
        for chunk in self.model_serving_client.chat.completions.create(
            model=self.llm_endpoint,
            messages=to_chat_completions_input(messages),
            tools=self.get_tool_specs(),
            stream=True,
        ):
            chunk_dict = chunk.to_dict()
            if chunk_dict.get("choices"):
                yield chunk_dict

    def handle_tool_call(
        self,
        tool_call: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> ResponsesAgentStreamEvent:
        try:
            args = json.loads(tool_call.get("arguments"))
        except Exception:
            args = {}
        result = str(self.execute_tool(tool_name=tool_call["name"], args=args))
        tool_call_output = self.create_function_call_output_item(tool_call["call_id"], result)
        messages.append(tool_call_output)
        return ResponsesAgentStreamEvent(type="response.output_item.done", item=tool_call_output)

    def call_and_run_tools(
        self,
        messages: list[dict[str, Any]],
        max_iter: int = 20,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        for _ in range(max_iter):
            last_msg = messages[-1]
            if last_msg.get("role") == "assistant":
                return
            elif last_msg.get("type") == "function_call":
                yield self.handle_tool_call(last_msg, messages)
            else:
                yield from output_to_responses_items_stream(
                    chunks=self.call_llm(messages), aggregator=messages
                )
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item("Max iterations reached. Stopping.", str(uuid4())),
        )

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            session_id = None
            if request.custom_inputs and "session_id" in request.custom_inputs:
                session_id = request.custom_inputs.get("session_id")
            elif request.context and request.context.conversation_id:
                session_id = request.context.conversation_id
            if session_id:
                mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})
            outputs = [
                event.item
                for event in self.predict_stream(request)
                if event.type == "response.output_item.done"
            ]
            return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        session_id = None
        if request.custom_inputs and "session_id" in request.custom_inputs:
            session_id = request.custom_inputs.get("session_id")
        elif request.context and request.context.conversation_id:
            session_id = request.context.conversation_id
        if session_id:
            mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})
        messages = to_chat_completions_input([i.model_dump() for i in request.input])
        if SYSTEM_PROMPT:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        yield from self.call_and_run_tools(messages=messages)


mlflow.openai.autolog()
