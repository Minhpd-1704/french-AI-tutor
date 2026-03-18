from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple


ErrorType = Literal[
    "spelling",
    "grammar",
    "agreement",
    "word_choice",
    "punctuation",
    "capitalization",
    "diacritics",
    "missing_word",
    "extra_word",
    "other",
]


@dataclass
class DetectedError:
    start_token: int
    end_token: int
    start_char: int
    end_char: int
    wrong: str
    suggestion: str
    error_type: ErrorType
    message: str
    severity: Literal["low", "medium", "high"] = "medium"


def tokenize_with_spans(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Splits text into tokens (words & punctuation) and tracks each token span.
    Example: "Je vais à Paris." -> ["Je","vais","à","Paris","."]
    """
    tokens: List[str] = []
    spans: List[Tuple[int, int]] = []
    pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)
    for m in pattern.finditer(text):
        tokens.append(m.group(0))
        spans.append((m.start(), m.end()))
    return tokens, spans


def _join_tokens(tokens: List[str]) -> str:

    return " | ".join(tokens)


def build_fewshot_prompt(text: str, tokens: List[str]) -> str:
    """
    Few-shot prompting: examples are French; instructions in English.
    IMPORTANT: We ONLY ask LLM for token indices + suggestion + type + message + severity.
    Char offsets and 'wrong' are computed deterministically from our tokenizer spans.
    """
    ex1 = "Je suis allé au école hier."
    ex2 = "Elle a beaucoup des amis."
    ex3 = "nous sommes allé à Paris"
    ex4 = "je vais a paris demain"
    ex5 = "je parle chine"

    # Priority reduces randomness between overlapping errors
    priority = """
When multiple valid corrections exist, prioritize in this order:
1) word_choice (wrong expression / wrong construction)
2) missing_word / extra_word
3) grammar / agreement / spelling / diacritics / punctuation
4) capitalization (ONLY if it's the main issue left)
"""

    return f"""You are a French grammar & spelling error detector.

Task:
- Identify incorrect words or phrases in the French input.
- Return ONLY JSON (no extra text) that matches the provided schema.
- Use token indices starting at 0 (over the provided tokenization).
- DO NOT return character offsets or the substring; we will compute that ourselves.
- start_token/end_token must refer to contiguous tokens in the provided token list.
- suggestion should be the corrected form for that span (keep it short).
- message must be a short English explanation.
- If no errors: return an empty list for errors.

{priority}

Tokenization (authoritative):
TOKENS: {_join_tokens(tokens)}

Few-shot examples:
1) INPUT_TEXT: {ex1}
   TOKENS: Je | suis | allé | au | école | hier | .
   Typical fix: span "au école" -> "à l'école" (missing_word/grammar)

2) INPUT_TEXT: {ex2}
   TOKENS: Elle | a | beaucoup | des | amis | .
   Typical fix: span "beaucoup des" -> "beaucoup d'amis" (grammar/word_choice)

3) INPUT_TEXT: {ex3}
   TOKENS: nous | sommes | allé | à | Paris
   Typical fix: span "allé" -> "allés" (agreement)

4) INPUT_TEXT: {ex4}
   TOKENS: je | vais | a | paris | demain
   Typical fixes (choose the most important):
   - "a" -> "à" (diacritics)
   - "paris" -> "Paris" (capitalization)
   - "je" -> "Je" (capitalization)

5) INPUT_TEXT: {ex5}
   TOKENS: je | parle | chine
   Prefer: "chine" -> "chinois" (word_choice). Capitalization alone is not enough.

Now analyze:
INPUT_TEXT: {text}
TOKENS: {_join_tokens(tokens)}
"""


def ged_json_schema() -> Dict[str, Any]:
    """
    Output schema (final output to user): includes char offsets and wrong substring,
    but those are computed deterministically post-LLM.
    """
    return {
        "type": "object",
        "properties": {
            "text_lang": {"type": "string"},
            "errors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_token": {"type": "integer"},
                        "end_token": {"type": "integer"},
                        "start_char": {"type": "integer"},
                        "end_char": {"type": "integer"},
                        "wrong": {"type": "string"},
                        "suggestion": {"type": "string"},
                        "error_type": {
                            "type": "string",
                            "enum": [
                                "spelling",
                                "grammar",
                                "agreement",
                                "word_choice",
                                "punctuation",
                                "capitalization",
                                "diacritics",
                                "missing_word",
                                "extra_word",
                                "other",
                            ],
                        },
                        "message": {"type": "string"},
                        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    },
                    "required": [
                        "start_token",
                        "end_token",
                        "start_char",
                        "end_char",
                        "wrong",
                        "suggestion",
                        "error_type",
                        "message",
                        "severity",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["text_lang", "errors"],
        "additionalProperties": False,
    }


def llm_output_schema_token_only() -> Dict[str, Any]:
    """
    Schema we ask from the LLM (token-only, to avoid random char spans).
    """
    return {
        "type": "object",
        "properties": {
            "text_lang": {"type": "string"},
            "errors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_token": {"type": "integer"},
                        "end_token": {"type": "integer"},
                        "suggestion": {"type": "string"},
                        "error_type": {
                            "type": "string",
                            "enum": [
                                "spelling",
                                "grammar",
                                "agreement",
                                "word_choice",
                                "punctuation",
                                "capitalization",
                                "diacritics",
                                "missing_word",
                                "extra_word",
                                "other",
                            ],
                        },
                        "message": {"type": "string"},
                        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    },
                    "required": [
                        "start_token",
                        "end_token",
                        "suggestion",
                        "error_type",
                        "message",
                        "severity",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["text_lang", "errors"],
        "additionalProperties": False,
    }


def extract_json_object(s: str) -> Optional[dict]:
    """
    Extracts the first JSON object from a string.
    Usually unnecessary with Structured Outputs, but kept for robustness.
    """
    s_strip = s.strip()
    if s_strip.startswith("{") and s_strip.endswith("}"):
        try:
            return json.loads(s_strip)
        except Exception:
            pass

    start = s.find("{")
    if start == -1:
        return None

    stack = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            stack += 1
        elif s[i] == "}":
            stack -= 1
            if stack == 0:
                candidate = s[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    return None


def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(n, hi))


def compute_char_span_from_tokens(
    spans: List[Tuple[int, int]], start_token: int, end_token: int
) -> Tuple[int, int]:
    sc = spans[start_token][0]
    ec = spans[end_token][1]
    return sc, ec


def validate_and_build_errors(
    text: str,
    tokens: List[str],
    spans: List[Tuple[int, int]],
    raw_errors: List[Dict[str, Any]],
) -> List[DetectedError]:
    """
    - Clamp token indices into bounds
    - Compute start_char/end_char deterministically from token spans
    - wrong is always text[start_char:end_char]
    - Deduplicate
    """
    n_tokens = len(tokens)
    if n_tokens == 0:
        return []

    out: List[DetectedError] = []

    for e in raw_errors or []:
        try:
            st = int(e.get("start_token"))
            en = int(e.get("end_token"))
        except Exception:
            continue

        st = clamp(st, 0, n_tokens - 1)
        en = clamp(en, 0, n_tokens - 1)
        if en < st:
            st, en = en, st

        suggestion = str(e.get("suggestion", "") or "").strip()
        et = str(e.get("error_type", "other") or "other").strip()
        msg = str(e.get("message", "") or "").strip()
        sev = str(e.get("severity", "medium") or "medium").strip()

        if not suggestion:
            suggestion = tokens[st]

        if sev not in ("low", "medium", "high"):
            sev = "medium"

        sc, ec = compute_char_span_from_tokens(spans, st, en)
        wrong = text[sc:ec]
        if not wrong:
            continue

        out.append(
            DetectedError(
                start_token=st,
                end_token=en,
                start_char=sc,
                end_char=ec,
                wrong=wrong,
                suggestion=suggestion,
                error_type=et if et in (
                    "spelling", "grammar", "agreement", "word_choice", "punctuation",
                    "capitalization", "diacritics", "missing_word", "extra_word", "other"
                ) else "other",
                message=msg or "Detected an issue in this span.",
                severity=sev, 
            )
        )

    uniq: Dict[Tuple[int, int, str, str], DetectedError] = {}
    for err in out:
        key = (err.start_token, err.end_token, err.error_type, err.suggestion)
        uniq[key] = err

    
    result = list(uniq.values())
    result.sort(key=lambda x: (x.start_token, x.end_token, x.error_type, x.suggestion))
    return result


def openai_llm_call(prompt: str, model: str = "gpt-4o-mini") -> str:
    """
    Calls OpenAI Responses API, requesting STRICT JSON schema output (token-only schema).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Please set it in your environment.")

    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("OpenAI SDK not installed. Run: pip install openai") from e

    client = OpenAI(api_key=api_key)

    schema = llm_output_schema_token_only()

    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            try:
                resp = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": "You output strictly structured JSON for French GED."},
                        {"role": "user", "content": prompt},
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "ged_errors_token_only",
                            "schema": schema,
                            "strict": True,
                        }
                    },
                    temperature=0,  
                )
            except TypeError:
                resp = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": "You output strictly structured JSON for French GED."},
                        {"role": "user", "content": prompt},
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "ged_errors_token_only",
                            "schema": schema,
                            "strict": True,
                        }
                    },
                )

            return resp.output_text

        except Exception as e:
            last_err = e
            time.sleep(0.7 * (attempt + 1))

    raise RuntimeError(f"OpenAI request failed after retries: {last_err}") from last_err


def detect_errors(
    text: str,
    llm_call: Callable[[str], str],
    lang: str = "fr",
) -> Dict[str, Any]:
    tokens, spans = tokenize_with_spans(text)
    prompt = build_fewshot_prompt(text, tokens)

    raw = llm_call(prompt)
    obj = extract_json_object(raw)
    if not isinstance(obj, dict):
        return {"text_lang": lang, "errors": []}

    raw_errors = obj.get("errors", [])
    if not isinstance(raw_errors, list):
        raw_errors = []

    validated = validate_and_build_errors(text, tokens, spans, raw_errors)
    return {"text_lang": lang, "errors": [asdict(e) for e in validated]}


if __name__ == "__main__":
    print("French GED (OpenAI version, stabilized spans)")
    print("Type a French sentence and press Enter.")
    print("Press Enter on an empty line to quit.")
    print("-" * 70)

    MODEL_NAME = "gpt-4o-mini"

    while True:
        text = input("Input FR> ").strip()
        if not text:
            print("Bye!")
            break

        try:
            result = detect_errors(
                text,
                llm_call=lambda p: openai_llm_call(p, model=MODEL_NAME),
                lang="fr",
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"[ERROR] {e}")

        print("-" * 70)
