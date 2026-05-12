from __future__ import annotations

import json
import re
import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from rag_v1.prompts import prompts
from rag_v1.services.vl_router import call_api, extract_sufficiency_json


class RouteDecision(str, Enum):
    NO_RAG_CONFIDENT = "NO_RAG_CONFIDENT"
    NEED_RAG = "NEED_RAG"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class RouteResult:
    decision: RouteDecision
    confidence: float
    reason_code: str
    matched_features: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "confidence": self.confidence,
            "reason_code": self.reason_code,
            "matched_features": self.matched_features,
        }


@dataclass(frozen=True)
class RagRequirementDecision:
    use_rag: bool
    reason: str = ""
    answer_context: str = ""
    route_result: Optional[RouteResult] = None
    llm_route_result: Optional[RouteResult] = None


VISUAL_REFERENCE_TERMS = [
    "image", "picture", "photo", "screenshot", "figure", "diagram",
    "chart", "plot", "graph", "table", "slide", "page", "document",
    "pdf", "attachment", "scan", "poster", "form", "invoice",
    "receipt", "report", "paper",
]

SPATIAL_REFERENCE_TERMS = [
    "top left", "top right", "bottom left", "bottom right",
    "upper left", "upper right", "lower left", "lower right",
    "left side", "right side", "above", "below", "next to",
    "beside", "near", "inside the box", "red box", "blue box",
    "highlighted", "circled", "arrow", "marked", "annotated",
    "first row", "second row", "third row", "first column",
    "second column", "third column",
]

DEICTIC_EXTERNAL_TERMS = [
    "this image", "this picture", "this photo", "this screenshot",
    "this chart", "this graph", "this table", "this figure",
    "this document", "this pdf", "this page", "this slide",
    "this report", "this paper", "the attached", "the above",
    "the following image", "shown here", "in the image",
    "in the screenshot", "in the figure", "in the table",
    "in the document", "on this page",
]

EVIDENCE_TERMS = [
    "according to", "based on", "using the provided", "use the provided",
    "from the document", "from the image", "from the table",
    "cite", "citation", "source", "evidence", "reference",
    "quote", "where does it say", "which page", "page number",
    "find in", "search for", "look up", "retrieve",
]

FRESH_OR_EXTERNAL_FACT_TERMS = [
    "latest", "current", "currently", "today", "this year",
    "recent", "newest", "updated", "price", "pricing",
    "policy", "law", "regulation", "news", "release date",
    "version", "benchmark", "revenue", "earnings", "market share",
]

TEXT_TRANSFORM_TERMS = [
    "translate", "rewrite", "rephrase", "paraphrase", "polish",
    "proofread", "edit", "make it more formal", "make it more casual",
    "shorten", "expand", "summarize this text", "summarise this text",
]

INLINE_SUMMARY_TERMS = [
    "summarize", "summarise", "extract key points", "extract keywords",
    "classify this text", "sentiment", "identify the tone",
]

CREATIVE_GENERATION_TERMS = [
    "write", "draft", "compose", "generate", "create",
    "brainstorm", "come up with", "give me ideas",
    "make a list of", "suggest names", "write an email",
    "write a poem", "write a story", "write a post",
]

CODE_TERMS = [
    "python", "javascript", "typescript", "java", "c++", "sql",
    "regex", "function", "algorithm", "code", "implement",
    "debug", "error", "exception", "stack trace",
]

MATH_TERMS = [
    "calculate", "compute", "solve", "equation", "derivative",
    "integral", "probability", "matrix", "simplify",
]

GENERAL_CONCEPT_PATTERNS = [
    r"^what is ([a-zA-Z0-9 \-]+)\??$",
    r"^explain ([a-zA-Z0-9 \-]+)\.?$",
    r"^explain the concept of ([a-zA-Z0-9 \-]+)\.?$",
    r"^give me an overview of ([a-zA-Z0-9 \-]+)\.?$",
]

GENERAL_CONCEPT_WHITELIST = [
    "rag", "retrieval augmented generation", "transformer",
    "attention mechanism", "cross entropy", "precision",
    "recall", "f1 score", "overfitting", "underfitting",
    "linear regression", "logistic regression", "decision tree",
    "random forest", "gradient descent", "backpropagation",
    "embedding", "vector database", "semantic search",
    "binary search", "hash table", "dynamic programming",
]


SUFFICIENCY_ROUTER_CONTEXT = """Router Stage: pre-RAG necessity check

No web retrieval has been performed yet.
Current available evidence:
- The user question
- The provided image(s), if any

Task:
- Judge whether the currently available information is already sufficient to answer the user question without external web retrieval.
- If yes, return judgement=YES.
- If no, return judgement=NO and optionally provide a useful additional search query.

Conservative routing policy:
- If the answer depends on current facts, recent events, external evidence, attached document/table/chart content, or details not reliably inferable from the image alone, prefer judgement=NO.
- Only return judgement=YES when the question can be answered safely from the image(s), inline payload, or stable general knowledge without web retrieval.
"""


def normalize_query(query: str) -> str:
    q = query.strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q


def contains_any(q: str, terms: List[str]) -> List[str]:
    return [term for term in terms if term in q]


def has_quoted_text(query: str) -> bool:
    return bool(
        re.search(r'"[^"]{3,}"', query)
        or re.search(r"'[^']{3,}'", query)
        or re.search(r"“[^”]{3,}”", query)
    )


def has_colon_payload(query: str) -> bool:
    return bool(re.search(r"[:：]\s*.{8,}", query, flags=re.DOTALL))


def has_long_inline_text(query: str) -> bool:
    return len(query.strip()) >= 120


def has_inline_payload(query: str) -> bool:
    return has_quoted_text(query) or has_colon_payload(query) or has_long_inline_text(query)


def looks_like_math_expression(q: str) -> bool:
    return bool(
        re.search(r"\d+\s*[\+\-\*/\^]\s*\d+", q)
        or re.search(r"[a-z]\s*\^\s*\d+", q)
        or re.search(r"solve\s+.*=", q)
    )


def is_short_greeting(q: str) -> bool:
    patterns = [
        r"^(hi|hello|hey|thanks|thank you|good morning|good afternoon|good evening)[.!?\s]*$",
        r"^(who are you|what can you do|introduce yourself)[.!?\s]*$",
        r"^(tell me a joke|chat with me)[.!?\s]*$",
    ]
    return any(re.match(pattern, q) for pattern in patterns)


def detect_rag_risk(query: str) -> Optional[RouteResult]:
    q = normalize_query(query)

    matched: List[str] = []
    matched += contains_any(q, VISUAL_REFERENCE_TERMS)
    matched += contains_any(q, SPATIAL_REFERENCE_TERMS)
    matched += contains_any(q, DEICTIC_EXTERNAL_TERMS)
    matched += contains_any(q, EVIDENCE_TERMS)
    matched += contains_any(q, FRESH_OR_EXTERNAL_FACT_TERMS)

    if matched:
        return RouteResult(
            decision=RouteDecision.NEED_RAG,
            confidence=0.90,
            reason_code="RAG_RISK_SIGNAL",
            matched_features=matched,
        )

    return None


def match_no_rag_rule(query: str) -> Optional[RouteResult]:
    q = normalize_query(query)

    if is_short_greeting(q):
        return RouteResult(
            decision=RouteDecision.NO_RAG_CONFIDENT,
            confidence=0.99,
            reason_code="GREETING_OR_SMALLTALK",
            matched_features=["short_greeting"],
        )

    text_transform_hits = contains_any(q, TEXT_TRANSFORM_TERMS)
    if text_transform_hits and has_inline_payload(query):
        return RouteResult(
            decision=RouteDecision.NO_RAG_CONFIDENT,
            confidence=0.97,
            reason_code="INLINE_TEXT_TRANSFORM",
            matched_features=text_transform_hits + ["inline_payload"],
        )

    summary_hits = contains_any(q, INLINE_SUMMARY_TERMS)
    if summary_hits and has_inline_payload(query):
        return RouteResult(
            decision=RouteDecision.NO_RAG_CONFIDENT,
            confidence=0.96,
            reason_code="INLINE_TEXT_SUMMARIZATION_OR_EXTRACTION",
            matched_features=summary_hits + ["inline_payload"],
        )

    creative_hits = contains_any(q, CREATIVE_GENERATION_TERMS)
    if creative_hits and not has_inline_payload(query):
        return RouteResult(
            decision=RouteDecision.NO_RAG_CONFIDENT,
            confidence=0.93,
            reason_code="GENERAL_CREATIVE_GENERATION",
            matched_features=creative_hits,
        )

    code_hits = contains_any(q, CODE_TERMS)
    if code_hits and not contains_any(q, ["repo", "repository", "readme", "notebook", "log file"]):
        return RouteResult(
            decision=RouteDecision.NO_RAG_CONFIDENT,
            confidence=0.94,
            reason_code="GENERAL_CODE_TASK",
            matched_features=code_hits,
        )

    math_hits = contains_any(q, MATH_TERMS)
    if math_hits and looks_like_math_expression(q):
        return RouteResult(
            decision=RouteDecision.NO_RAG_CONFIDENT,
            confidence=0.95,
            reason_code="INLINE_MATH_TASK",
            matched_features=math_hits + ["math_expression"],
        )

    concept = extract_general_concept(q)
    if concept is not None:
        return RouteResult(
            decision=RouteDecision.NO_RAG_CONFIDENT,
            confidence=0.94,
            reason_code="GENERAL_CONCEPT_WHITELIST",
            matched_features=[concept],
        )

    return None


def extract_general_concept(q: str) -> Optional[str]:
    for pattern in GENERAL_CONCEPT_PATTERNS:
        match = re.match(pattern, q)
        if not match:
            continue

        concept = match.group(1).strip().lower()
        concept = concept.rstrip(".? ")

        if concept in GENERAL_CONCEPT_WHITELIST:
            return concept

    return None


def route_before_vlm(query: str) -> RouteResult:
    """
    Conservative pre-router before LLM/VLM-based judgement.

    Policy:
    1. If any strong RAG risk signal appears, route to RAG.
    2. Else, only high-precision no-RAG rules can bypass RAG.
    3. Everything else remains uncertain.
    """

    if not query or not query.strip():
        return RouteResult(
            decision=RouteDecision.NO_RAG_CONFIDENT,
            confidence=0.99,
            reason_code="EMPTY_QUERY",
            matched_features=[],
        )

    risk = detect_rag_risk(query)
    if risk is not None:
        return risk

    no_rag = match_no_rag_rule(query)
    if no_rag is not None:
        return no_rag

    return RouteResult(
        decision=RouteDecision.UNCERTAIN,
        confidence=0.50,
        reason_code="NO_HIGH_CONFIDENCE_RULE_MATCH",
        matched_features=[],
    )


def should_use_rag(
    question: str,
    img_paths: Sequence[Union[str, Path]],
    debug: bool = False,
) -> RagRequirementDecision:
    normalized_question = str(question or "").strip()
    normalized_images = [str(path) for path in img_paths if str(path or "").strip()]

    rule_result = route_before_vlm(normalized_question)
    if debug:
        print(f"Rule-based RAG routing: {rule_result.to_dict()}")

    if rule_result.decision == RouteDecision.NEED_RAG:
        return RagRequirementDecision(
            use_rag=True,
            reason=f"rule:{rule_result.reason_code}",
            route_result=rule_result,
        )

    if rule_result.decision == RouteDecision.NO_RAG_CONFIDENT:
        return RagRequirementDecision(
            use_rag=False,
            reason=f"rule:{rule_result.reason_code}",
            answer_context=_direct_answer_context(
                reason=(
                    "The request was classified as safely answerable without "
                    "external web retrieval."
                ),
                extra_note=_format_route_result(rule_result),
            ),
            route_result=rule_result,
        )

    llm_result = route_with_sufficiency_prompt(
        question=normalized_question,
        img_paths=normalized_images,
        debug=debug,
    )
    if debug:
        print(f"LLM-based RAG routing: {llm_result.to_dict()}")

    if llm_result.decision == RouteDecision.NO_RAG_CONFIDENT and llm_result.confidence >= 0.90:
        return RagRequirementDecision(
            use_rag=False,
            reason=f"llm:{llm_result.reason_code}",
            answer_context=_direct_answer_context(
                reason=(
                    "The request was classified as safely answerable without "
                    "external web retrieval."
                ),
                extra_note=_format_route_result(llm_result),
            ),
            route_result=rule_result,
            llm_route_result=llm_result,
        )

    # Conservative fallback: if the rule layer was uncertain, we only skip RAG
    # when the LLM is explicitly and confidently in the no-RAG class.
    return RagRequirementDecision(
        use_rag=True,
        reason=f"llm_or_fallback:{llm_result.reason_code}",
        route_result=rule_result,
        llm_route_result=llm_result,
    )


def route_with_sufficiency_prompt(
    question: str,
    img_paths: Sequence[str],
    debug: bool = False,
) -> RouteResult:
    prompt = prompts.sufficiency_prompt_en.format(
        question=question or "",
        context=SUFFICIENCY_ROUTER_CONTEXT,
    )

    try:
        response_text = call_api(
            prompt=prompt,
            img_paths=img_paths,
            temperature=0.0,
        )
    except Exception as exc:
        if debug:
            print(f"LLM RAG routing failed: {exc}")
        return RouteResult(
            decision=RouteDecision.UNCERTAIN,
            confidence=0.0,
            reason_code="SUFFICIENCY_ROUTER_CALL_FAILED",
            matched_features=[],
        )

    if debug:
        print(f"Sufficiency-router raw response:\n{response_text}\n")

    parsed = _extract_sufficiency_route_result(response_text)
    if parsed is None:
        return RouteResult(
            decision=RouteDecision.UNCERTAIN,
            confidence=0.0,
            reason_code="SUFFICIENCY_ROUTER_PARSE_FAILED",
            matched_features=[],
        )

    return parsed


def _extract_sufficiency_route_result(response_text: str) -> Optional[RouteResult]:
    payload = extract_sufficiency_json(response_text)
    if payload is None:
        return None

    judgement = payload["judgement"]
    addition = payload.get("addition")

    if judgement == "YES":
        return RouteResult(
            decision=RouteDecision.NO_RAG_CONFIDENT,
            confidence=0.90,
            reason_code="SUFFICIENCY_PROMPT_YES",
            matched_features=["sufficiency_yes"],
        )

    matched_features = ["sufficiency_no"]
    if addition:
        matched_features.append(f"additional_query:{addition}")

    return RouteResult(
        decision=RouteDecision.NEED_RAG,
        confidence=0.90,
        reason_code="SUFFICIENCY_PROMPT_NO",
        matched_features=matched_features,
    )


def _format_route_result(result: RouteResult) -> str:
    features = ", ".join(result.matched_features) if result.matched_features else "none"
    return (
        "Routing evidence: "
        f"decision={result.decision.value}, "
        f"confidence={result.confidence:.2f}, "
        f"reason_code={result.reason_code}, "
        f"matched_features={features}."
    )


def _direct_answer_context(reason: str, extra_note: Optional[str] = None) -> str:
    lines = [
        "No external web retrieval was performed.",
        reason.strip(),
        "Answer using the visual inputs and the user question only.",
        "If the image alone is insufficient, say so explicitly instead of guessing.",
    ]
    if extra_note:
        lines.append(extra_note.strip())
    return "\n".join(line for line in lines if line)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-requirement",
        description="Test whether a question should use web RAG before answering.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="User question to evaluate.",
    )
    parser.add_argument(
        "--images",
        nargs="*",
        default=[],
        help="Optional local image paths passed into the RAG-requirement router.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print intermediate rule/LLM routing details.",
    )
    return parser


def _stringify_route_result(result: Optional[RouteResult]) -> str:
    if result is None:
        return "None"
    return json.dumps(result.to_dict(), ensure_ascii=False)


def _stringify_decision(decision: RagRequirementDecision) -> str:
    payload = {
        "use_rag": decision.use_rag,
        "reason": decision.reason,
        "route_result": decision.route_result.to_dict() if decision.route_result else None,
        "llm_route_result": (
            decision.llm_route_result.to_dict() if decision.llm_route_result else None
        ),
        "answer_context": decision.answer_context,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    image_paths = [Path(image).expanduser() for image in args.images]
    decision = should_use_rag(
        question=args.question,
        img_paths=image_paths,
        debug=args.debug,
    )

    print("RAG Requirement Result:")
    print(f"question: {args.question}")
    print(f"image_paths: {[str(path) for path in image_paths]}")
    print(f"rule_result: {_stringify_route_result(decision.route_result)}")
    print(f"llm_route_result: {_stringify_route_result(decision.llm_route_result)}")
    print("final_decision:")
    print(_stringify_decision(decision))


if __name__ == "__main__":
    main()
