from __future__ import annotations

import ast
import json
import os
import re
import textwrap
import unicodedata
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st


ENV_PATH = Path(__file__).resolve().parent / ".env"


def load_env_file(path: Path = ENV_PATH) -> Dict[str, str]:
    """Populate os.environ with key/value pairs from a .env file if present."""

    if not path.exists():
        return {}

    loaded: Dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        loaded[key] = value
        os.environ.setdefault(key, value)

    return loaded


ENV_VARS = load_env_file()


def env_default(key: str, fallback: str = "") -> str:
    return os.environ.get(key, fallback)


PROVIDER_PRESETS: Dict[str, Dict[str, str]] = {
    "Groq": {
        "base_url": env_default("GROQ_BASE_URL", "https://api.groq.com/openai"),
        "model": env_default("GROQ_MODEL", ""),
        "api_key": env_default("GROQ_API_KEY", ""),
    },
    "Ollama": {
        "base_url": env_default("OLLAMA_BASE_URL", "http://localhost:11434"),
        "model": env_default("OLLAMA_MODEL", "deepseek-r1:70b"),
        "api_key": env_default("OLLAMA_API_KEY", ""),
    },
    "Custom": {
        "base_url": env_default("LLM_BASE_URL", ""),
        "model": env_default("LLM_MODEL", ""),
        "api_key": env_default("LLM_API_KEY", ""),
    },
}


DEFAULT_PROVIDER = env_default("LLM_DEFAULT_PROVIDER", "Groq")


SESSION_PRESET_KEYS = {
    "base_url": "llm_base_url",
    "model": "llm_model",
    "api_key": "llm_api_key",
}


def apply_provider_defaults_to_session(provider: str, overwrite: bool = False) -> None:
    preset = PROVIDER_PRESETS.get(provider, {})
    if not preset:
        return

    for preset_key, session_key in SESSION_PRESET_KEYS.items():
        value = preset.get(preset_key)
        if value or overwrite:
            if overwrite or not st.session_state.get(session_key):
                st.session_state[session_key] = value or ""


# --- Data models -----------------------------------------------------------
@dataclass
class FinancialProfile:
    age: int
    profession: str
    income: float
    address: str
    education: str
    liquid_assets: float
    investment_assets: float
    real_estate_assets: float
    primary_residence_value: float
    other_assets: float
    mortgage_debt: float
    student_debt: float
    credit_card_debt: float
    other_debt: float
    monthly_essential_expenses: float
    monthly_discretionary_expenses: float
    monthly_insurance_costs: float
    monthly_other_costs: float
    dependents: int
    estate_documents: str
    risk_appetite: str
    target_savings_amount: Optional[float] = None
    target_savings_rate: Optional[float] = None
    llm_notes: str = ""

    @property
    def total_assets(self) -> float:
        return (
            self.liquid_assets
            + self.investment_assets
            + self.real_estate_assets
            + self.other_assets
        )

    @property
    def total_debt(self) -> float:
        return (
            self.mortgage_debt
            + self.student_debt
            + self.credit_card_debt
            + self.other_debt
        )

    @property
    def net_worth(self) -> float:
        return self.total_assets - self.total_debt

    @property
    def monthly_income(self) -> float:
        return self.income / 12 if self.income else 0.0

    @property
    def recurring_costs(self) -> float:
        return (
            self.monthly_essential_expenses
            + self.monthly_discretionary_expenses
            + self.monthly_insurance_costs
            + self.monthly_other_costs
        )


@dataclass
class AdvisorBudgetItem:
    category: str
    amount: float
    notes: Optional[str] = None
    source: Optional[str] = None


@dataclass
class AdvisorRetirementPoint:
    age: float
    year: int
    balance: float
    assumptions: Optional[str] = None
    source: Optional[str] = None


@dataclass
class AdvisorCatastrophicPlan:
    scenario: str
    recommended_reserve: float
    guidance: Optional[str] = None
    source: Optional[str] = None


@dataclass
class AdvisorResponse:
    budget: List[AdvisorBudgetItem]
    retirement_projection: List[AdvisorRetirementPoint]
    catastrophic_plan: List[AdvisorCatastrophicPlan]
    narrative: str
    sources: List[str]


@dataclass
class LLMSettings:
    provider: str
    client: Optional[LLMClient]
    model: Optional[str]
    temperature: float
    max_tokens: int


PROFILE_SESSION_DEFAULTS: Dict[str, Any] = {
    "profile_age": 18,
    "profile_profession": "",
    "profile_income": 0,
    "profile_address": "",
    "profile_education": "Bachelor's",
    "profile_liquid_assets": 0.0,
    "profile_investment_assets": 0.0,
    "profile_real_estate_assets": 0.0,
    "profile_primary_residence_value": 0.0,
    "profile_other_assets": 0.0,
    "profile_mortgage_debt": 0.0,
    "profile_student_debt": 0.0,
    "profile_credit_card_debt": 0.0,
    "profile_other_debt": 0.0,
    "profile_monthly_essential": 0.0,
    "profile_monthly_discretionary": 0.0,
    "profile_monthly_insurance": 0.0,
    "profile_monthly_other": 0.0,
    "profile_dependents": 0,
    "profile_estate_documents": "",
    "profile_planning_mode": "Risk profile",
    "profile_risk_appetite": "Moderate",
    "profile_target_type": "Monthly savings amount",
    "profile_target_savings_amount": 0.0,
    "profile_target_savings_rate": 10.0,
    "profile_llm_notes": "",
}


PROFILE_EXPORT_MAP: Dict[str, str] = {
    "profile_age": "age",
    "profile_profession": "profession",
    "profile_income": "income",
    "profile_address": "address",
    "profile_education": "education",
    "profile_liquid_assets": "liquid_assets",
    "profile_investment_assets": "investment_assets",
    "profile_real_estate_assets": "real_estate_assets",
    "profile_primary_residence_value": "primary_residence_value",
    "profile_other_assets": "other_assets",
    "profile_mortgage_debt": "mortgage_debt",
    "profile_student_debt": "student_debt",
    "profile_credit_card_debt": "credit_card_debt",
    "profile_other_debt": "other_debt",
    "profile_monthly_essential": "monthly_essential_expenses",
    "profile_monthly_discretionary": "monthly_discretionary_expenses",
    "profile_monthly_insurance": "monthly_insurance_costs",
    "profile_monthly_other": "monthly_other_costs",
    "profile_dependents": "dependents",
    "profile_estate_documents": "estate_documents",
    "profile_planning_mode": "planning_mode",
    "profile_risk_appetite": "risk_appetite",
    "profile_target_type": "target_type",
    "profile_target_savings_amount": "target_savings_amount",
    "profile_target_savings_rate": "target_savings_rate",
    "profile_llm_notes": "llm_notes",
}


PROFILE_IMPORT_MAP: Dict[str, str] = {
    external: internal for internal, external in PROFILE_EXPORT_MAP.items()
}


def trigger_streamlit_rerun() -> None:
    rerun: Optional[Callable[[], None]] = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
        return

    experimental: Optional[Callable[[], None]] = getattr(st, "experimental_rerun", None)
    if callable(experimental):
        experimental()
        return

    raise RuntimeError("Streamlit rerun support is unavailable in this environment.")


def ensure_profile_state_defaults() -> None:
    pending_profile = st.session_state.pop("pending_profile_settings", None)
    for key, default in PROFILE_SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, default)
    if pending_profile:
        for key, value in pending_profile.items():
            if key in PROFILE_SESSION_DEFAULTS:
                st.session_state[key] = value


def export_profile_settings() -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    for session_key, external_key in PROFILE_EXPORT_MAP.items():
        snapshot[external_key] = st.session_state.get(
            session_key, PROFILE_SESSION_DEFAULTS[session_key]
        )
    return snapshot


LLM_REQUEST_TIMEOUT = 300

PDF_LINES_PER_PAGE = 45

UNICODE_PUNCT_TRANSLATION = {
    ord("‐"): "-",
    ord("‑"): "-",
    ord("‒"): "-",
    ord("–"): "-",
    ord("—"): "-",
    ord("―"): "-",
    ord("−"): "-",
    ord("“"): '"',
    ord("”"): '"',
    ord("„"): '"',
    ord("‘"): "'",
    ord("’"): "'",
    ord("‚"): "'",
    ord("•"): "-",
    ord("·"): "-",
    ord("…"): "...",
    ord(" "): " ",  # Non-breaking space
    ord(" "): " ",  # Narrow non-breaking space
    ord(" "): " ",
    ord(" "): " ",
    ord(" "): " ",
    ord(" "): " ",
}

CURRENCY_TIGHT_PATTERN = re.compile(r"\$(?=[0-9-])")
ORDERED_LIST_PATTERN = re.compile(r"\d+[.)]\s")
URL_PATTERN = re.compile(r"(?P<prefix>^|\s)(?P<url>https?://[^\s<>()]+)")


def _linkify_line(line: str) -> str:
    """Wrap bare URLs in angle brackets so Markdown renders them as links."""

    def _replace(match: re.Match) -> str:
        prefix = match.group("prefix")
        url = match.group("url")
        trimmed = url.rstrip(").,;:")
        trailing = url[len(trimmed) :]
        return f"{prefix}<{trimmed}>{trailing}"

    return URL_PATTERN.sub(_replace, line)


class LLMClient:
    """Minimal client for OpenAI-compatible chat endpoints (vLLM, Ollama, etc.)."""

    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None) -> None:
        normalized = base_url.rstrip("/")
        if normalized.endswith("/v1"):
            normalized = normalized[: -len("/v1")]
        self.base_url = normalized
        self.model = model
        self.api_key = api_key

    def generate_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            endpoint = f"{base}/chat/completions"
        else:
            endpoint = f"{base}/v1/chat/completions"
        response = requests.post(
            endpoint,
            headers=headers,
            data=json.dumps(payload),
            timeout=LLM_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()


def sanitize_number(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0


def wrap_text(text: str, width: int = 90, indent: str = "") -> List[str]:
    """Wrap text to a specified width while preserving simple indentation."""

    wrapper = textwrap.TextWrapper(
        width=max(width, len(indent) + 1),
        initial_indent=indent,
        subsequent_indent=indent,
        replace_whitespace=False,
        drop_whitespace=False,
    )

    lines: List[str] = []
    for block in text.splitlines():
        if not block:
            lines.append(indent.rstrip() if indent else "")
            continue
        lines.extend(wrapper.wrap(block))

    return lines or [indent.rstrip() if indent else ""]


def normalize_markdown_line(line: str) -> str:
    """Convert light Markdown into plain text for PDF rendering."""

    replacements = {"**": "", "__": "", "`": ""}
    normalized = line
    for src, target in replacements.items():
        normalized = normalized.replace(src, target)

    stripped = normalized.lstrip()
    leading_spaces = len(normalized) - len(stripped)

    if stripped.startswith(('# ', '## ', '### ')):
        stripped = stripped.lstrip('# ').strip()
        normalized = (" " * leading_spaces) + stripped
    elif stripped.startswith(('- ', '* ')):
        content = stripped[2:]
        normalized = (" " * leading_spaces) + f"- {content}"
    elif stripped.startswith('> '):
        content = stripped[2:]
        normalized = (" " * leading_spaces) + f"Quote: {content}"

    normalized = re.sub(r"(?<!\\)\*(?!\*)(.+?)(?<!\\)\*", r"\1", normalized)
    normalized = re.sub(r"(?<!\\)_(?!_)(.+?)(?<!\\)_", r"\1", normalized)
    normalized = CURRENCY_TIGHT_PATTERN.sub("$ ", normalized)
    return normalized


def normalize_plaintext_block(text: str) -> str:
    """Convert Markdown-like content into sanitized plain text for on-screen display."""

    normalized_lines: List[str] = []
    for raw_line in text.splitlines():
        normalized = normalize_markdown_line(raw_line)
        normalized = unicodedata.normalize("NFKC", normalized).translate(
            UNICODE_PUNCT_TRANSLATION
        )
        normalized = CURRENCY_TIGHT_PATTERN.sub("$ ", normalized)
        normalized = _linkify_line(normalized)

        if normalized and (
            normalized.startswith("- ")
            or ORDERED_LIST_PATTERN.match(normalized)
        ):
            if normalized_lines and normalized_lines[-1].strip():
                normalized_lines.append("")

        normalized_lines.append(normalized)

    return "\n".join(normalized_lines)


def sanitize_pdf_line(line: str) -> str:
    """Escape characters that are not safe in a PDF text run."""

    safe = unicodedata.normalize("NFKC", line).translate(UNICODE_PUNCT_TRANSLATION)
    safe = safe.replace("\r", "")
    safe = safe.encode("ascii", "replace").decode("ascii")
    safe = safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return safe if safe.strip() else " "


def build_pdf_document(lines: List[str]) -> bytes:
    """Create a simple multi-page PDF from a list of text lines."""

    sanitized_lines = [sanitize_pdf_line(line) for line in lines]
    if not sanitized_lines:
        sanitized_lines = [" "]

    pages: List[List[str]] = []
    current: List[str] = []
    for line in sanitized_lines:
        if len(current) >= PDF_LINES_PER_PAGE:
            pages.append(current)
            current = []
        current.append(line)
    if current:
        pages.append(current)

    if not pages:
        pages = [[" "]]

    num_pages = len(pages)
    font_obj_num = 3 + 2 * num_pages
    kids_refs = " ".join(f"{3 + 2 * idx} 0 R" for idx in range(num_pages))

    objects: List[str] = []
    objects.append("1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objects.append(
        f"2 0 obj << /Type /Pages /Kids [{kids_refs}] /Count {num_pages} >> endobj\n"
    )

    for idx, page_lines in enumerate(pages):
        page_obj_num = 3 + 2 * idx
        content_obj_num = 4 + 2 * idx

        content_parts = [
            "BT",
            "/F1 12 Tf",
            "14 TL",
            "1 0 0 1 50 760 Tm",
        ]
        for line_idx, line in enumerate(page_lines):
            if line_idx > 0:
                content_parts.append("T*")
            content_parts.append(f"({line}) Tj")
        content_parts.append("ET")

        content_stream = "\n".join(content_parts)
        content_bytes = content_stream.encode("latin-1")

        objects.append(
            f"{page_obj_num} 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_obj_num} 0 R /Resources << /Font << /F1 {font_obj_num} 0 R >> >> >> endobj\n"
        )
        objects.append(
            f"{content_obj_num} 0 obj << /Length {len(content_bytes)} >> stream\n{content_stream}\nendstream\nendobj\n"
        )

    objects.append(
        f"{font_obj_num} 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
    )

    buffer = BytesIO()
    buffer.write(b"%PDF-1.4\n")

    offsets: List[int] = []
    for obj in objects:
        offsets.append(buffer.tell())
        buffer.write(obj.encode("latin-1"))

    xref_start = buffer.tell()
    buffer.write(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    buffer.write(b"0000000000 65535 f \n")
    for offset in offsets:
        buffer.write(f"{offset:010d} 00000 n \n".encode("latin-1"))

    buffer.write(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF".encode(
            "latin-1"
        )
    )
    return buffer.getvalue()


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def build_pdf_report(
    profile: FinancialProfile,
    response: AdvisorResponse,
    model_name: Optional[str] = None,
) -> bytes:
    """Create a PDF report summarizing the user inputs and advisor output."""

    lines: List[str] = []
    lines.append("Personalized Investment & Savings Advisor Report")
    if model_name:
        lines.append(f"Model: {model_name}")
    lines.append("")

    lines.append("Client Profile")
    profile_rows = [
        ("Age", str(profile.age)),
        ("Profession", profile.profession or "N/A"),
        ("Annual income", format_currency(profile.income)),
        ("Address", profile.address or "N/A"),
        ("Education", profile.education or "N/A"),
        ("Dependents", str(profile.dependents)),
        ("Risk appetite", profile.risk_appetite or "N/A"),
    ]

    if profile.target_savings_amount:
        profile_rows.append(
            ("Target savings amount", format_currency(profile.target_savings_amount))
        )
    if profile.target_savings_rate:
        profile_rows.append(
            ("Target savings rate", f"{profile.target_savings_rate:.1f}% of income")
        )
    if profile.estate_documents:
        profile_rows.append(("Estate documents", profile.estate_documents))
    if profile.llm_notes:
        profile_rows.append(("Additional notes", profile.llm_notes))

    net_rows = [
        ("Liquid assets", format_currency(profile.liquid_assets)),
        ("Investment assets", format_currency(profile.investment_assets)),
        ("Real estate assets", format_currency(profile.real_estate_assets)),
        ("Primary residence value", format_currency(profile.primary_residence_value)),
        ("Other assets", format_currency(profile.other_assets)),
        ("Total debt", format_currency(profile.total_debt)),
        ("Net worth", format_currency(profile.net_worth)),
        (
            "Monthly take-home estimate",
            format_currency(profile.monthly_income),
        ),
        (
            "Monthly recurring costs",
            format_currency(profile.recurring_costs),
        ),
    ]

    for label, value in profile_rows:
        lines.extend(wrap_text(f"{label}: {value}", width=90))

    lines.append("")
    lines.append("Asset & Cash Flow Overview")
    for label, value in net_rows:
        lines.extend(wrap_text(f"{label}: {value}", width=90, indent="  "))

    lines.append("")

    if response.narrative:
        lines.append("Narrative Summary")
        for raw_line in response.narrative.splitlines():
            normalized = normalize_markdown_line(raw_line)
            lines.extend(wrap_text(normalized, width=90))
        lines.append("")

    if response.budget:
        lines.append("Budget Recommendations")
        for item in response.budget:
            lines.extend(
                wrap_text(
                    f"- {item.category}: {format_currency(item.amount)}",
                    width=90,
                )
            )
            if item.notes:
                for wrapped in wrap_text(
                    f"Notes: {normalize_markdown_line(item.notes)}", width=90, indent="    "
                ):
                    lines.append(wrapped)
            if item.source:
                lines.extend(
                    wrap_text(f"Source: {item.source}", width=90, indent="    ")
                )
        lines.append("")

    if response.retirement_projection:
        lines.append("Retirement Projection")
        for point in response.retirement_projection:
            label = (
                f"Age {point.age:.0f} ({point.year}): {format_currency(point.balance)}"
            )
            lines.extend(wrap_text(label, width=90))
            if point.assumptions:
                lines.extend(
                    wrap_text(
                        f"Assumptions: {normalize_markdown_line(point.assumptions)}",
                        width=90,
                        indent="    ",
                    )
                )
            if point.source:
                lines.extend(
                    wrap_text(f"Source: {point.source}", width=90, indent="    ")
                )
        lines.append("")

    if response.catastrophic_plan:
        lines.append("Catastrophic Preparedness")
        for plan in response.catastrophic_plan:
            lines.extend(
                wrap_text(
                    f"- {plan.scenario}: Reserve {format_currency(plan.recommended_reserve)}",
                    width=90,
                )
            )
            if plan.guidance:
                lines.extend(
                    wrap_text(
                        f"Guidance: {normalize_markdown_line(plan.guidance)}",
                        width=90,
                        indent="    ",
                    )
                )
            if plan.source:
                lines.extend(
                    wrap_text(f"Source: {plan.source}", width=90, indent="    ")
                )
        lines.append("")

    if response.sources:
        lines.append("Citations")
        for source in response.sources:
            lines.extend(wrap_text(f"- {source}", width=90))

    return build_pdf_document(lines)


def _clean_json_candidate(candidate: str) -> str:
    cleaned = candidate.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        if "```" in cleaned:
            cleaned = cleaned.split("```", 1)[0]
    smart_quotes = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
    }
    for bad, good in smart_quotes.items():
        cleaned = cleaned.replace(bad, good)
    cleaned = re.sub(r",\s*(?=[}\]])", "", cleaned)
    return cleaned


def _try_parse_json(candidate: str) -> Optional[Dict[str, Any]]:
    cleaned = _clean_json_candidate(candidate)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    prefix_guard = r"(?<![A-Za-z0-9_\"'])"
    python_friendly = re.sub(prefix_guard + r"true\b", "True", cleaned)
    python_friendly = re.sub(prefix_guard + r"false\b", "False", python_friendly)
    python_friendly = re.sub(prefix_guard + r"null\b", "None", python_friendly)
    try:
        parsed = ast.literal_eval(python_friendly)
    except Exception:
        return None
    if isinstance(parsed, (dict, list)):
        return parsed  # type: ignore[return-value]
    return None


def extract_first_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Locate the first balanced JSON object/array in the text."""

    if not text:
        return None

    stack: List[str] = []
    start_idx: Optional[int] = None
    in_string = False
    escape = False

    for idx, char in enumerate(text):
        if start_idx is None:
            if char in "{[":
                start_idx = idx
                stack.append("}" if char == "{" else "]")
            continue

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char in "{[":
            stack.append("}" if char == "{" else "]")
            continue

        if char in "}]":
            if not stack or char != stack[-1]:
                # Malformed JSON segment; reset search
                stack.clear()
                start_idx = None
                in_string = False
                escape = False
                continue
            stack.pop()
            if not stack and start_idx is not None:
                candidate = text[start_idx : idx + 1]
                parsed_candidate = _try_parse_json(candidate)
                if parsed_candidate is not None:
                    return parsed_candidate
                # Continue searching for the next balanced candidate
                start_idx = None
                in_string = False
                escape = False
                continue

    return None


def coerce_message_content(message: Dict[str, Any]) -> str:
    """Normalize OpenAI-compatible message content to a plain string."""

    content = message.get("content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        pieces: List[str] = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict):
                text_value = part.get("text") or part.get("content") or part.get("value")
                if isinstance(text_value, str):
                    pieces.append(text_value)
        return "".join(pieces)

    return ""


def extract_response_text(raw: Dict[str, Any]) -> str:
    """Return the assistant text regardless of provider response schema."""

    choices = raw.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = coerce_message_content(message)
        if content:
            return content

    message = raw.get("message")
    if isinstance(message, dict):
        content = coerce_message_content(message)
        if content:
            return content

    response_text = raw.get("response")
    if isinstance(response_text, str):
        return response_text

    # Some providers surface the text at the top level under keys like
    # "content", "text", or "output".
    for key in ("content", "text", "output"):
        value = raw.get(key)
        if isinstance(value, str):
            return value

    return ""


def extract_reasoning_trace(raw: Dict[str, Any]) -> str:
    """Collect any reasoning/thinking content returned by the provider."""

    def _normalize(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                normalized = _normalize(item)
                if normalized:
                    parts.append(normalized)
            return "\n".join(parts)
        if isinstance(value, dict):
            candidates = []
            for key in ("text", "content", "reasoning", "thought", "value"):
                if key in value:
                    normalized = _normalize(value[key])
                    if normalized:
                        candidates.append(normalized)
            return "\n".join(candidates)
        return ""

    traces: List[str] = []

    choices = raw.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            message = choice.get("message", {}) if isinstance(choice, dict) else {}
            for key in ("reasoning", "reasoning_content", "thoughts", "thinking"):
                if key in message:
                    normalized = _normalize(message[key])
                    if normalized:
                        traces.append(normalized)
            if "metadata" in choice and isinstance(choice["metadata"], dict):
                metadata_reasoning = _normalize(choice["metadata"].get("reasoning"))
                if metadata_reasoning:
                    traces.append(metadata_reasoning)

    message = raw.get("message")
    if isinstance(message, dict):
        for key in ("reasoning", "reasoning_content", "thoughts", "thinking"):
            if key in message:
                normalized = _normalize(message[key])
                if normalized:
                    traces.append(normalized)

    for key in ("reasoning", "analysis", "thinking"):
        if key in raw:
            normalized = _normalize(raw[key])
            if normalized:
                traces.append(normalized)

    combined = "\n\n".join(trace.strip() for trace in traces if trace and trace.strip())
    return combined.strip()


def parse_advisor_response(raw: Dict[str, Any]) -> AdvisorResponse:
    content = extract_response_text(raw)
    parsed = extract_first_json_block(content)
    if not parsed:
        preview = content.strip()
        if len(preview) > 500:
            preview = preview[:500] + "..."
        raise ValueError(
            "LLM response did not contain valid JSON. Received content preview: "
            f"{preview}"
        )

    budget_items = [
        AdvisorBudgetItem(
            category=item.get("category", ""),
            amount=sanitize_number(item.get("amount")),
            notes=item.get("notes"),
            source=item.get("source"),
        )
        for item in parsed.get("budget", [])
    ]

    retirement_projection = [
        AdvisorRetirementPoint(
            age=sanitize_number(point.get("age")),
            year=int(sanitize_number(point.get("year"))),
            balance=sanitize_number(point.get("balance")),
            assumptions=point.get("assumptions"),
            source=point.get("source"),
        )
        for point in parsed.get("retirement_projection", [])
    ]

    catastrophic_plan = [
        AdvisorCatastrophicPlan(
            scenario=item.get("scenario", ""),
            recommended_reserve=sanitize_number(item.get("recommended_reserve")),
            guidance=item.get("guidance"),
            source=item.get("source"),
        )
        for item in parsed.get("catastrophic_plan", [])
    ]

    narrative = parsed.get("narrative", "")
    sources = [str(source) for source in parsed.get("sources", [])]

    return AdvisorResponse(
        budget=budget_items,
        retirement_projection=retirement_projection,
        catastrophic_plan=catastrophic_plan,
        narrative=narrative,
        sources=sources,
    )


SYSTEM_PROMPT = """
You are a fiduciary financial analyst tasked with building a personalized savings and
investment plan. Combine the provided household profile with current data from the Bureau
of Labor Statistics (BLS), O*NET, regional cost-of-living indices, and relevant policy or
macroeconomic news. Incorporate current federal, state, and local income tax brackets,
along with property tax expectations tied to the household's location and stated real
estate values. Never fabricate figures; cite every quantitative statement with its
source. Respond in JSON with the structure:
{
  "budget": [
    {"category": str, "amount": number, "notes": str, "source": str}
  ],
  "retirement_projection": [
    {"age": number, "year": int, "balance": number, "assumptions": str, "source": str}
  ],
  "catastrophic_plan": [
    {"scenario": str, "recommended_reserve": number, "guidance": str, "source": str}
  ],
  "narrative": str,
  "sources": [str]
}
Ensure that the calculations reference the cited datasets, reflect after-tax cash flow,
and that all currency values are expressed in USD.
"""


def build_user_message(profile: FinancialProfile) -> str:
    profile_payload = {
        "age": profile.age,
        "profession": profile.profession,
        "income": profile.income,
        "address": profile.address,
        "education": profile.education,
        "assets": {
            "liquid": profile.liquid_assets,
            "investment": profile.investment_assets,
            "real_estate": profile.real_estate_assets,
            "primary_residence_value": profile.primary_residence_value,
            "other": profile.other_assets,
        },
        "debts": {
            "mortgage": profile.mortgage_debt,
            "student": profile.student_debt,
            "credit_card": profile.credit_card_debt,
            "other": profile.other_debt,
        },
        "monthly_costs": {
            "essential": profile.monthly_essential_expenses,
            "discretionary": profile.monthly_discretionary_expenses,
            "insurance": profile.monthly_insurance_costs,
            "other": profile.monthly_other_costs,
        },
        "dependents": profile.dependents,
        "estate_documents": profile.estate_documents,
        "risk_appetite": profile.risk_appetite,
        "goals": {
            "planning_mode": "custom_target"
            if profile.risk_appetite.lower() == "custom target"
            else "risk_profile",
            "target_monthly_savings": profile.target_savings_amount,
            "target_savings_rate": profile.target_savings_rate,
        },
        "net_worth": profile.net_worth,
        "monthly_income": profile.monthly_income,
        "recurring_costs": profile.recurring_costs,
    }

    request_payload = {
        "profile": profile_payload,
        "analysis_requirements": [
            "Quantify job outlook, wage trends, and employment volatility using BLS and O*NET data.",
            "Adjust cost projections by the household's location using a reputable cost-of-living index.",
            "Account for current economic indicators (inflation, interest rates, policy changes).",
            "Estimate federal, state, and local income taxes plus property taxes using current rates relevant to the provided address and income; show the sources.",
            "Provide recommended monthly budget allocations that align with the stated risk appetite or meet any custom savings target.",
            "Project retirement balances with explicit assumptions about contribution amounts and returns.",
            "Size emergency and catastrophic reserves for layoffs, medical events, and dependent support.",
            "Reference every figure with a verifiable source URL or citation identifier.",
        ],
        "additional_context": profile.llm_notes,
        "output_requirements": {
            "currency": "USD",
            "explain_methodology": True,
            "include_sources": True,
            "show_after_tax_budget": True,
            "detail_tax_references": True,
        },
    }

    return json.dumps(request_payload, indent=2)


def build_llm_payload(
    profile: FinancialProfile, model: str, temperature: float, max_tokens: int
) -> Dict[str, Any]:
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": build_user_message(profile)},
        ],
    }


# --- Streamlit app --------------------------------------------------------
def render_sidebar() -> None:
    st.sidebar.header("About this advisor")
    st.sidebar.write(
        """
        Configure a local or hosted large language model endpoint that can retrieve live
        Bureau of Labor Statistics, O*NET, cost-of-living, and policy updates. The model
        should return quantitative recommendations backed by citations.
        """
    )
    st.sidebar.caption(
        "Tip: Create a .env file with values such as GROQ_API_KEY and GROQ_MODEL to pre-populate the Groq connection."
    )
    st.sidebar.info(
        "All results are educational and should be validated with a licensed professional."
    )


def render_llm_settings() -> LLMSettings:
    st.sidebar.subheader("LLM connection")

    defaults = {
        "llm_provider": DEFAULT_PROVIDER if DEFAULT_PROVIDER in PROVIDER_PRESETS else "Groq",
        "llm_base_url": "",
        "llm_model": "",
        "llm_api_key": "",
        "llm_temperature": 0.2,
        "llm_max_tokens": 6000,
        "show_load_settings": False,
        "_llm_defaults_applied": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    pending_settings = st.session_state.pop("pending_llm_settings", None)
    if pending_settings:
        for key, value in pending_settings.items():
            st.session_state[key] = value

    provider = st.session_state.get("llm_provider", DEFAULT_PROVIDER)
    if provider not in PROVIDER_PRESETS:
        provider = "Custom"
        st.session_state["llm_provider"] = provider

    if not st.session_state.get("_llm_defaults_applied", False):
        apply_provider_defaults_to_session(provider, overwrite=False)
        st.session_state["_llm_defaults_applied"] = True

    if st.session_state.pop("settings_loaded_success", False):
        st.sidebar.success("Settings loaded.")

    def _on_provider_change() -> None:
        selected = st.session_state.get("llm_provider", "Custom")
        if selected not in PROVIDER_PRESETS:
            selected = "Custom"
            st.session_state["llm_provider"] = selected
        apply_provider_defaults_to_session(selected, overwrite=True)

    provider_options = list(PROVIDER_PRESETS.keys())
    st.sidebar.selectbox(
        "Provider",
        provider_options,
        key="llm_provider",
        on_change=_on_provider_change,
        help="Choose Groq for hosted inference, Ollama for local models, or Custom to supply another endpoint.",
    )
    st.sidebar.caption(
        "Groq requests use https://api.groq.com/openai by default. Ollama typically listens on http://localhost:11434."
    )

    base_url = st.sidebar.text_input(
        "Base URL",
        value=st.session_state.get("llm_base_url", ""),
        key="llm_base_url",
        help="Root URL for an OpenAI-compatible API (for example a vLLM or Ollama server).",
    )
    model = st.sidebar.text_input(
        "Model name",
        value=st.session_state.get("llm_model", ""),
        key="llm_model",
        help="Identifier exposed by the selected endpoint.",
    )
    api_key = st.sidebar.text_input(
        "API key",
        value=st.session_state.get("llm_api_key", ""),
        key="llm_api_key",
        type="password",
        help="Optional token if the endpoint requires authentication.",
    )

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.get("llm_temperature", 0.2),
        key="llm_temperature",
        step=0.05,
        help="Lower values keep outputs closer to sourced data.",
    )
    max_tokens = int(
        st.sidebar.number_input(
            "Max tokens",
            min_value=256,
            max_value=8192,
            value=int(st.session_state.get("llm_max_tokens", 6000)),
            key="llm_max_tokens",
            step=256,
            help="Upper bound for response length.",
        )
    )

    actions_col1, actions_col2 = st.sidebar.columns(2)
    with actions_col1:
        if st.button("Load Settings"):
            st.session_state["show_load_settings"] = True

    if st.session_state.get("show_load_settings"):
        uploaded = st.sidebar.file_uploader(
            "Select settings JSON",
            type="json",
            key="settings_loader",
            help="Upload a JSON file previously exported with Save Settings.",
        )
        if uploaded is not None:
            try:
                loaded = json.loads(uploaded.getvalue().decode("utf-8"))
                llm_blob_candidate = loaded.get("llm") if "llm" in loaded else loaded
                llm_blob = llm_blob_candidate if isinstance(llm_blob_candidate, dict) else {}
                profile_blob = loaded.get("profile", {})

                st.session_state["pending_llm_settings"] = {
                    "llm_provider": llm_blob.get(
                        "provider",
                        st.session_state["llm_provider"],
                    ),
                    "llm_base_url": llm_blob.get("base_url", st.session_state["llm_base_url"]),
                    "llm_model": llm_blob.get("model", st.session_state["llm_model"]),
                    "llm_api_key": llm_blob.get("api_key", st.session_state["llm_api_key"]),
                    "llm_temperature": float(
                        llm_blob.get("temperature", st.session_state["llm_temperature"])
                    ),
                    "llm_max_tokens": int(
                        llm_blob.get("max_tokens", st.session_state["llm_max_tokens"])
                    ),
                }

                profile_settings: Dict[str, Any] = {}
                if isinstance(profile_blob, dict):
                    for key, value in profile_blob.items():
                        session_key = PROFILE_IMPORT_MAP.get(key)
                        if session_key:
                            profile_settings[session_key] = value

                if not profile_settings:
                    for key, value in loaded.items():
                        if key in PROFILE_SESSION_DEFAULTS:
                            profile_settings[key] = value

                if profile_settings:
                    st.session_state["pending_profile_settings"] = profile_settings
                st.session_state["show_load_settings"] = False
                st.session_state["settings_loaded_success"] = True
                st.session_state.pop("settings_loader", None)
                trigger_streamlit_rerun()
            except Exception as exc:  # pragma: no cover - surface load errors to UI
                st.sidebar.error(f"Failed to load settings: {exc}")

    current_settings = {
        "provider": st.session_state.get("llm_provider", "Custom"),
        "base_url": st.session_state.get("llm_base_url", ""),
        "model": st.session_state.get("llm_model", ""),
        "api_key": st.session_state.get("llm_api_key", ""),
        "temperature": float(st.session_state.get("llm_temperature", 0.2)),
        "max_tokens": int(st.session_state.get("llm_max_tokens", 6000)),
    }

    with actions_col2:
        st.download_button(
            "Save Settings",
            data=json.dumps(
                {
                    "llm": current_settings,
                    "profile": export_profile_settings(),
                },
                indent=2,
            ),
            file_name="advisor_settings.json",
            mime="application/json",
        )

    client = None
    if current_settings["base_url"] and current_settings["model"]:
        client = LLMClient(
            base_url=current_settings["base_url"],
            model=current_settings["model"],
            api_key=current_settings["api_key"] or None,
        )

    return LLMSettings(
        provider=current_settings["provider"],
        client=client,
        model=current_settings["model"] or None,
        temperature=current_settings["temperature"],
        max_tokens=current_settings["max_tokens"],
    )


def collect_user_inputs() -> FinancialProfile:
    st.header("Household Profile")

    education_options = ["High School", "Associate", "Bachelor's", "Master's", "Doctorate"]
    planning_mode_options = ["Risk profile", "Custom savings target"]
    risk_options = ["Conservative", "Moderate", "Aggressive"]
    target_type_options = ["Monthly savings amount", "Percentage of income"]

    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        age = col1.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=int(st.session_state.get("profile_age", 18)),
            key="profile_age",
        )
        profession = col2.text_input(
            "Profession",
            value=st.session_state.get("profile_profession", ""),
            key="profile_profession",
        )
        income = col1.number_input(
            "Annual Income (USD)",
            min_value=0,
            value=int(st.session_state.get("profile_income", 0)),
            step=1000,
            key="profile_income",
        )
        address = col2.text_input(
            "City, State (e.g. Austin, TX)",
            value=st.session_state.get("profile_address", ""),
            key="profile_address",
        )

        education_value = st.session_state.get("profile_education", education_options[2])
        education_index = (
            education_options.index(education_value)
            if education_value in education_options
            else 2
        )
        education = col1.selectbox(
            "Highest Education Level",
            education_options,
            index=education_index,
            key="profile_education",
        )

        st.subheader("Assets")
        liquid_assets = st.number_input(
            "Liquid Assets (cash, savings)",
            min_value=0.0,
            value=float(st.session_state.get("profile_liquid_assets", 0.0)),
            step=1000.0,
            key="profile_liquid_assets",
        )
        investment_assets = st.number_input(
            "Investment Assets (retirement, brokerage)",
            min_value=0.0,
            value=float(st.session_state.get("profile_investment_assets", 0.0)),
            step=1000.0,
            key="profile_investment_assets",
        )
        real_estate_assets = st.number_input(
            "Real Estate Equity",
            min_value=0.0,
            value=float(st.session_state.get("profile_real_estate_assets", 0.0)),
            step=5000.0,
            key="profile_real_estate_assets",
        )
        primary_residence_value = st.number_input(
            "Primary Residence Market Value (for property tax estimates)",
            min_value=0.0,
            value=float(st.session_state.get("profile_primary_residence_value", 0.0)),
            step=5000.0,
            key="profile_primary_residence_value",
        )
        other_assets = st.number_input(
            "Other Assets",
            min_value=0.0,
            value=float(st.session_state.get("profile_other_assets", 0.0)),
            step=1000.0,
            key="profile_other_assets",
        )

        st.subheader("Debts")
        mortgage_debt = st.number_input(
            "Mortgage Debt",
            min_value=0.0,
            value=float(st.session_state.get("profile_mortgage_debt", 0.0)),
            step=5000.0,
            key="profile_mortgage_debt",
        )
        student_debt = st.number_input(
            "Student Loans",
            min_value=0.0,
            value=float(st.session_state.get("profile_student_debt", 0.0)),
            step=1000.0,
            key="profile_student_debt",
        )
        credit_card_debt = st.number_input(
            "Credit Card Debt",
            min_value=0.0,
            value=float(st.session_state.get("profile_credit_card_debt", 0.0)),
            step=500.0,
            key="profile_credit_card_debt",
        )
        other_debt = st.number_input(
            "Other Debts",
            min_value=0.0,
            value=float(st.session_state.get("profile_other_debt", 0.0)),
            step=500.0,
            key="profile_other_debt",
        )

        st.subheader("Recurring Monthly Costs")
        monthly_essential = st.number_input(
            "Essentials (housing, utilities, food)",
            min_value=0.0,
            value=float(st.session_state.get("profile_monthly_essential", 0.0)),
            step=100.0,
            key="profile_monthly_essential",
        )
        monthly_discretionary = st.number_input(
            "Discretionary spending",
            min_value=0.0,
            value=float(st.session_state.get("profile_monthly_discretionary", 0.0)),
            step=100.0,
            key="profile_monthly_discretionary",
        )
        monthly_insurance = st.number_input(
            "Insurance premiums",
            min_value=0.0,
            value=float(st.session_state.get("profile_monthly_insurance", 0.0)),
            step=50.0,
            key="profile_monthly_insurance",
        )
        monthly_other = st.number_input(
            "Other monthly obligations",
            min_value=0.0,
            value=float(st.session_state.get("profile_monthly_other", 0.0)),
            step=50.0,
            key="profile_monthly_other",
        )

        dependents = st.number_input(
            "Dependents",
            min_value=0,
            max_value=10,
            value=int(st.session_state.get("profile_dependents", 0)),
            step=1,
            key="profile_dependents",
        )
        estate_documents = st.text_area(
            "Existing wills/trusts (include states where active)",
            value=st.session_state.get("profile_estate_documents", ""),
            key="profile_estate_documents",
        )

        st.subheader("Risk & Savings Goals")
        planning_mode_value = st.session_state.get("profile_planning_mode", planning_mode_options[0])
        planning_mode_index = (
            planning_mode_options.index(planning_mode_value)
            if planning_mode_value in planning_mode_options
            else 0
        )
        planning_mode = st.radio(
            "Plan preference",
            planning_mode_options,
            index=planning_mode_index,
            key="profile_planning_mode",
        )

        target_savings_amount: Optional[float] = None
        target_savings_rate: Optional[float] = None

        if planning_mode == "Risk profile":
            risk_value = st.session_state.get("profile_risk_appetite", "Moderate")
            risk_index = risk_options.index(risk_value) if risk_value in risk_options else 1
            risk_appetite = st.selectbox(
                "Risk Appetite",
                risk_options,
                index=risk_index,
                key="profile_risk_appetite",
                help="Select an overall portfolio risk style for allocation guidance.",
            )
        else:
            st.session_state["profile_risk_appetite"] = "Custom target"
            risk_appetite = "Custom target"
            target_type_value = st.session_state.get("profile_target_type", target_type_options[0])
            target_type_index = (
                target_type_options.index(target_type_value)
                if target_type_value in target_type_options
                else 0
            )
            target_type = st.radio(
                "Target type",
                target_type_options,
                index=target_type_index,
                key="profile_target_type",
                help="Choose whether to specify a dollar goal or income percentage for savings.",
            )
            if target_type == "Monthly savings amount":
                target_savings_amount = st.number_input(
                    "Monthly savings target (USD)",
                    min_value=0.0,
                    value=float(st.session_state.get("profile_target_savings_amount", 0.0)),
                    step=100.0,
                    key="profile_target_savings_amount",
                )
                target_savings_rate = None
            else:
                target_savings_rate = st.number_input(
                    "Savings target (% of gross income)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(st.session_state.get("profile_target_savings_rate", 10.0)),
                    step=1.0,
                    key="profile_target_savings_rate",
                )
                target_savings_amount = None

        st.subheader("Additional instructions for the LLM")
        llm_notes = st.text_area(
            "Optional context",
            value=st.session_state.get("profile_llm_notes", ""),
            key="profile_llm_notes",
            help=(
                "Share any qualitative information, links to custom research, or prompts the model should"
                " consider when aggregating external datasets."
            ),
        )

        submitted = st.form_submit_button("Generate Plan")

    if not submitted:
        st.stop()

    if target_savings_amount is not None and target_savings_amount <= 0:
        target_savings_amount = None
    if target_savings_rate is not None and target_savings_rate <= 0:
        target_savings_rate = None

    return FinancialProfile(
        age=int(age),
        profession=profession,
        income=float(income),
        address=address,
        education=education,
        liquid_assets=float(liquid_assets),
        investment_assets=float(investment_assets),
        real_estate_assets=float(real_estate_assets),
        primary_residence_value=float(primary_residence_value),
        other_assets=float(other_assets),
        mortgage_debt=float(mortgage_debt),
        student_debt=float(student_debt),
        credit_card_debt=float(credit_card_debt),
        other_debt=float(other_debt),
        monthly_essential_expenses=float(monthly_essential),
        monthly_discretionary_expenses=float(monthly_discretionary),
        monthly_insurance_costs=float(monthly_insurance),
        monthly_other_costs=float(monthly_other),
        dependents=int(dependents),
        estate_documents=estate_documents,
        risk_appetite=risk_appetite,
        target_savings_amount=target_savings_amount,
        target_savings_rate=target_savings_rate,
        llm_notes=llm_notes,
    )


def render_summary(profile: FinancialProfile) -> None:
    st.subheader("Net Worth Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Assets", f"${profile.total_assets:,.0f}")
    col2.metric("Total Debt", f"${profile.total_debt:,.0f}")
    col3.metric("Net Worth", f"${profile.net_worth:,.0f}")

    st.markdown(f"**Monthly take-home estimate:** ${profile.monthly_income:,.0f}  ")
    st.markdown(f"**Recurring costs:** ${profile.recurring_costs:,.0f}")

    if profile.estate_documents:
        st.caption(f"Estate planning documents noted: {profile.estate_documents}")

    if profile.risk_appetite:
        if profile.risk_appetite.lower() == "custom target":
            st.caption("Planning preference: Custom savings target")
        else:
            st.caption(f"Risk appetite selected: {profile.risk_appetite}")

    if profile.primary_residence_value:
        st.caption(
            f"Primary residence market value reported: ${profile.primary_residence_value:,.0f}"
        )

    if profile.target_savings_amount or profile.target_savings_rate:
        goal_parts = []
        if profile.target_savings_amount:
            goal_parts.append(f"${profile.target_savings_amount:,.0f} per month")
        if profile.target_savings_rate:
            goal_parts.append(f"{profile.target_savings_rate:.1f}% of income")
        st.caption("Savings goal preference: " + " and ".join(goal_parts))


def render_budget_section(items: List[AdvisorBudgetItem]) -> None:
    st.header("Dynamic Budget Plan")
    if not items:
        st.warning("No budget recommendations were returned by the model.")
        return

    df = pd.DataFrame(
        {
            "Category": [item.category for item in items],
            "Amount": [item.amount for item in items],
            "Notes": [item.notes or "" for item in items],
            "Source": [item.source or "" for item in items],
        }
    )
    st.dataframe(df.style.format({"Amount": "${:,.2f}"}))
    st.bar_chart(df.set_index("Category")["Amount"])


def render_retirement_section(points: List[AdvisorRetirementPoint]) -> None:
    st.header("Retirement Trajectory")
    if not points:
        st.warning("No retirement projection returned. Confirm the model prompt includes projection requirements.")
        return

    df = pd.DataFrame(
        {
            "Age": [point.age for point in points],
            "Year": [point.year for point in points],
            "Balance": [point.balance for point in points],
            "Assumptions": [point.assumptions or "" for point in points],
            "Source": [point.source or "" for point in points],
        }
    )

    latest = df.iloc[-1]
    st.metric(
        label=f"Projected balance by age {latest['Age']:.0f}",
        value=f"${latest['Balance']:,.0f}",
    )

    st.dataframe(df.style.format({"Balance": "${:,.2f}", "Age": "{:.1f}"}))
    st.line_chart(df.set_index("Age")["Balance"])


def render_catastrophic_section(items: List[AdvisorCatastrophicPlan]) -> None:
    st.header("Catastrophic Event Preparedness")
    if not items:
        st.warning("No catastrophic event planning guidance returned.")
        return

    df = pd.DataFrame(
        {
            "Scenario": [item.scenario for item in items],
            "Recommended Reserve": [item.recommended_reserve for item in items],
            "Guidance": [item.guidance or "" for item in items],
            "Source": [item.source or "" for item in items],
        }
    )
    st.dataframe(df.style.format({"Recommended Reserve": "${:,.2f}"}))


def render_narrative(response: AdvisorResponse) -> None:
    if response.narrative:
        st.header("Narrative Summary")
        narrative_text = normalize_plaintext_block(response.narrative)
        if narrative_text.strip():
            st.markdown(narrative_text)
        else:
            st.info("The model did not include a narrative summary.")

    if response.sources:
        st.subheader("Citations")
        for source in response.sources:
            st.markdown(f"- {source}")


def main() -> None:
    st.set_page_config(page_title="Personalized Investment Advisor", layout="wide")
    st.title("Personalized Investment & Savings Advisor")
    st.write(
        "Provide your financial profile to generate dynamic budgeting guidance, retirement projections, and contingency planning."
    )

    ensure_profile_state_defaults()
    render_sidebar()
    settings = render_llm_settings()
    profile = collect_user_inputs()
    render_summary(profile)

    if not settings.client or not settings.model:
        st.error("Provide the LLM base URL and model name in the sidebar to generate recommendations.")
        st.stop()
        return

    payload = build_llm_payload(
        profile=profile,
        model=settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    with st.expander("LLM request payload", expanded=False):
        preview = {"model": payload["model"], "temperature": payload["temperature"], "messages": payload["messages"]}
        st.code(json.dumps(preview, indent=2))

    raw_response: Optional[Dict[str, Any]] = None
    advisor_response: Optional[AdvisorResponse] = None
    reasoning_trace = ""
    try:
        with st.spinner("Requesting plan from LLM..."):
            raw_response = settings.client.generate_plan(payload)
        reasoning_trace = extract_reasoning_trace(raw_response) if raw_response else ""
        advisor_response = parse_advisor_response(raw_response)
    except Exception as exc:  # pragma: no cover - surface error to UI
        st.error(f"Failed to generate plan: {exc}")
        if reasoning_trace:
            with st.expander("Model reasoning"):
                reasoning_text = normalize_plaintext_block(reasoning_trace)
                if reasoning_text.strip():
                    st.markdown(reasoning_text)
        if raw_response is not None:
            with st.expander("Raw LLM response"):
                serialized = json.dumps(raw_response, indent=2)
                st.code(serialized)
                st.download_button(
                    "Download raw response",
                    data=serialized,
                    file_name="advisor_raw_response.json",
                    mime="application/json",
                )
        st.stop()
        return

    if advisor_response is None:
        st.error("The LLM did not return any recommendations. Please try again.")
        st.stop()
        return

    if reasoning_trace:
        with st.expander("Model reasoning", expanded=False):
            reasoning_text = normalize_plaintext_block(reasoning_trace)
            if reasoning_text.strip():
                st.markdown(reasoning_text)

    if raw_response is not None:
        with st.expander("Raw LLM response", expanded=False):
            serialized = json.dumps(raw_response, indent=2)
            st.code(serialized)
            st.download_button(
                "Download raw response",
                data=serialized,
                file_name="advisor_raw_response.json",
                mime="application/json",
            )

    render_budget_section(advisor_response.budget)
    render_retirement_section(advisor_response.retirement_projection)
    render_catastrophic_section(advisor_response.catastrophic_plan)
    render_narrative(advisor_response)

    pdf_bytes = build_pdf_report(profile, advisor_response, settings.model)
    st.download_button(
        "Download advisor report (PDF)",
        data=pdf_bytes,
        file_name="advisor_report.pdf",
        mime="application/pdf",
    )


if __name__ == "__main__":
    main()
