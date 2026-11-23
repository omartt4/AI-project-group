"""LangChain structured-output adapter that always uses JSON mode."""

from __future__ import annotations

from typing import Any, Type

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel


class JsonModeChatModel:
    """Expose JSON-mode structured output without function-calling fallback."""

    _llm_type = "openrouter-json-mode"

    def __init__(self, model: Any) -> None:
        self.model = model

    def with_structured_output(
        self,
        schema: Type[BaseModel],
        *,
        include_raw: bool = False,
        **_: Any,
    ) -> Any:
        parser = PydanticOutputParser(pydantic_object=schema)

        def parse_response(message: Any) -> Any:
            content = getattr(message, "content", "")
            if isinstance(content, list):
                content = "\n".join(
                    block["text"]
                    for block in content
                    if isinstance(block, dict) and isinstance(block.get("text"), str)
                )
            parsed = parser.parse(str(content))
            if include_raw:
                return {"raw": message, "parsed": parsed, "parsing_error": None}
            return parsed

        return self.model.bind(
            response_format={"type": "json_object"}
        ) | RunnableLambda(parse_response)
