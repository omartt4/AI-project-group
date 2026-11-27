"""LangChain structured-output adapter that always uses JSON mode."""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Type

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel


def _property_value_as_text(value: Any) -> str:
    """Preserve structured property values as deterministic JSON strings."""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def repair_graph_payload(payload: Any) -> Any:
    """Repair harmless LLMGraphTransformer property-shape deviations.

    LangChain's dynamic graph schema requires properties to be a list of
    ``{"key": ..., "value": ...}`` objects. JSON-mode models sometimes emit
    the semantically equivalent shorthand ``{"description": ...}``. Convert
    only those property containers; required node and relationship fields are
    deliberately left untouched so incomplete/truncated facts still fail
    validation and are retried instead of being invented locally.
    """
    if not isinstance(payload, dict):
        return payload

    repaired = copy.deepcopy(payload)
    for collection_name in ("nodes", "relationships"):
        collection = repaired.get(collection_name)
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            properties = item.get("properties")
            if isinstance(properties, dict):
                properties = [properties]
            if not isinstance(properties, list):
                continue

            normalized = []
            for entry in properties:
                if not isinstance(entry, dict):
                    continue
                if "key" in entry and "value" in entry:
                    key = str(entry.get("key") or "").strip()
                    value = entry.get("value")
                    if key and value is not None:
                        normalized.append(
                            {"key": key, "value": _property_value_as_text(value)}
                        )
                    continue

                for key, value in entry.items():
                    key = str(key or "").strip()
                    if key and value is not None:
                        normalized.append(
                            {"key": key, "value": _property_value_as_text(value)}
                        )
            item["properties"] = normalized
    return repaired


def _decode_json_object(content: str) -> Any:
    """Decode a JSON object, tolerating a surrounding Markdown code fence."""
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, count=1)
        stripped = re.sub(r"\s*```$", "", stripped, count=1)
    return json.loads(stripped)


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
            content = str(content)
            try:
                payload = repair_graph_payload(_decode_json_object(content))
                parsed = schema.model_validate(payload)
            except json.JSONDecodeError:
                # Keep LangChain's normal parser behavior and error reporting
                # for non-JSON or truncated provider output.
                parsed = parser.parse(content)
            if include_raw:
                return {"raw": message, "parsed": parsed, "parsing_error": None}
            return parsed

        return self.model.bind(
            response_format={"type": "json_object"}
        ) | RunnableLambda(parse_response)
