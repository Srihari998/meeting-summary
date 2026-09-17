import json
import re
from typing import List, Literal, Optional, Any
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class JSONValidationError(ValueError):
    """Raised when raw LLM output cannot be parsed as JSON or violates schema constraints."""
    pass


class ActionItem(BaseModel):
    """Schema representing an individual action item extracted from a meeting."""
    model_config = ConfigDict(extra="forbid")

    task: str = Field(..., description="Action item description")
    assignee: str = Field(..., description="Person or team responsible, or 'Unassigned'")
    deadline: Optional[str] = Field(None, description="Due date/timeline or None")
    priority: Literal["High", "Medium", "Low"] = Field("Medium", description="Priority level")
    status: str = Field("Not Started", description="Current execution status")


class MeetingIntelligence(BaseModel):
    """Structured meeting intelligence extracted from a meeting transcript."""
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(..., description="High-level meeting summary")
    key_points: List[str] = Field(default_factory=list, description="Key discussion points")
    decisions: List[str] = Field(default_factory=list, description="Agreed decisions")
    action_items: List[ActionItem] = Field(default_factory=list, description="Extracted action items")
    participants: List[str] = Field(default_factory=list, description="Detected unique participants")


def _clean_json_string(raw_text: str) -> str:
    """Strip markdown code fences and extraneous leading/trailing whitespace."""
    text = raw_text.strip()
    
    # Match ```json ... ``` or ``` ... ```
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if code_block_match:
        return code_block_match.group(1).strip()
    
    return text


def _format_validation_errors(err: ValidationError) -> str:
    """Format Pydantic ValidationError into a clear, specific message naming field and value."""
    formatted_errors = []
    for error in err.errors():
        loc_str = " -> ".join(str(elem) for elem in error["loc"])
        err_type = error.get("type", "")
        msg = error.get("msg", "")
        input_val = error.get("input")

        if "missing" in err_type or err_type == "missing":
            formatted_errors.append(f"Missing required field '{loc_str}'.")
        elif "extra_forbidden" in err_type:
            formatted_errors.append(f"Extra forbidden field '{loc_str}' is present and not allowed.")
        elif "literal_error" in err_type:
            formatted_errors.append(
                f"Field '{loc_str}': Expected one of {error.get('ctx', {}).get('expected', 'valid options')}, "
                f"but received {input_val!r}."
            )
        else:
            if input_val is not None:
                formatted_errors.append(f"Field '{loc_str}': {msg} (received {input_val!r}).")
            else:
                formatted_errors.append(f"Field '{loc_str}': {msg}.")

    return "\n".join(formatted_errors)


def validate_and_parse(raw_llm_output: str) -> MeetingIntelligence:
    """
    Parses raw LLM output into a validated MeetingIntelligence instance.

    Strips markdown code fences, decodes JSON, and validates strictly against
    the MeetingIntelligence Pydantic schema (extra fields forbidden).

    Raises:
        JSONValidationError: If JSON is malformed or schema validation fails,
                             with a clear, specific error message detailing
                             the exact field and value that failed.
    """
    if not raw_llm_output or not raw_llm_output.strip():
        raise JSONValidationError("LLM response is empty.")

    cleaned_output = _clean_json_string(raw_llm_output)

    try:
        data = json.loads(cleaned_output)
    except json.JSONDecodeError as exc:
        raise JSONValidationError(
            f"Invalid JSON syntax at line {exc.lineno}, column {exc.colno}: {exc.msg}. "
            f"Raw text excerpt: {cleaned_output[:200]!r}"
        ) from exc

    if not isinstance(data, dict):
        raise JSONValidationError(
            f"Expected root JSON object (dict), but received {type(data).__name__}."
        )

    try:
        return MeetingIntelligence.model_validate(data)
    except ValidationError as exc:
        error_details = _format_validation_errors(exc)
        raise JSONValidationError(
            f"Schema validation failed with the following error(s):\n{error_details}"
        ) from exc