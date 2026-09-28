from __future__ import annotations

from collections import Counter
import json
import re
import string


def normalize(text: str) -> str:
    text = str(text or '').lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def as_references(value: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, (list, tuple)) or not value or not all(isinstance(x, str) for x in value):
        raise ValueError('Answers must be a string or a nonempty list of reference strings')
    return list(dict.fromkeys(value))


def reference_answers(record: dict) -> list[str]:
    answers = as_references(record.get('answers', record.get('answer', '')))
    aliases = record.get('answer_aliases') or []
    if aliases:
        answers += as_references(aliases)
    return list(dict.fromkeys(answers))


def _single_score(prediction: str, reference: str) -> tuple[float, float]:
    pred, gold = normalize(prediction).split(), normalize(reference).split()
    em = float(pred == gold)
    if not pred or not gold:
        return em, em
    shared = sum((Counter(pred) & Counter(gold)).values())
    return em, 2.0 * shared / (len(pred) + len(gold))


def answer_f1(prediction: str, expected: str | list[str] | tuple[str, ...]) -> tuple[float, float]:
    scores = [_single_score(prediction, reference) for reference in as_references(expected)]
    return max(score[0] for score in scores), max(score[1] for score in scores)


def question_kind(question: str) -> str:
    lower = question.lower().strip()
    if re.match(r"^(are|is|do|does|did|was|were)\b", lower) and ("both" in lower or "same" in lower):
        return "yes_no"
    if re.search(r"\b(first|earlier|later|more recently|older|younger)\b", lower):
        return "comparison"
    if lower.startswith("when ") or "what year" in lower or "what date" in lower:
        return "date"
    if lower.startswith("where "):
        return "location"
    if "nationality" in lower or "country" in lower:
        return "country"
    if lower.startswith("why "):
        return "reason"
    return "short_answer"


def path_text(path: dict) -> str:
    names = path.get("names", [])
    edges = path.get("edges", [])
    if not names:
        return ""
    text = names[0]
    for index, edge in enumerate(edges):
        relation = edge.get("relation", "related to")
        if edge.get("forward"):
            text += f" --[{relation}]--> {names[index + 1]}"
        else:
            text += f" <--[{relation}]-- {names[index + 1]}"
    return text


def paths_context(paths: list[dict]) -> str:
    if not paths:
        return "No graph paths were retrieved."
    return "\n".join(f"[{i + 1}] {path_text(path)}" for i, path in enumerate(paths))


def paragraph_context(paragraphs: list[dict]) -> str:
    if not paragraphs:
        return "No source paragraphs were retrieved."
    return "\n".join(
        f"[P{index}] {item.get('doc_title', '')} | {item.get('text', '')}"
        for index, item in enumerate(paragraphs, 1)
    )


def constrained_answer(graph, question: str, evidence: dict, question_seeds: list[str]) -> tuple[str, dict]:
    kind = question_kind(question)
    options = question_seeds[:2] if kind == "comparison" else []
    option_text = ", ".join(options)
    rules = {
        "yes_no": "Output exactly one token: yes or no.",
        "comparison": (f"Output exactly one of these two options, preserving the option text: {option_text}."
                       if len(options) == 2 else "Output only the shortest answer span, with no explanation."),
        "date": "Output only the date or year, with no explanation.",
        "location": "Output only the place name, with no explanation.",
        "country": "Output only the country or nationality, with no explanation.",
        "reason": "Output only the shortest reason phrase, with no explanation.",
        "short_answer": "Output only the shortest answer span, with no explanation.",
    }[kind]
    prompt = f"""Answer the question using only the evidence below.
Answer type: {kind}
{rules}
Do not output an explanation, a label, or an Answer prefix. Do not guess facts absent from evidence.

For comparison questions, identify the relevant dates or years in the source
paragraphs and compare them explicitly before choosing an option. For all other
questions, prefer a directly stated fact in a source paragraph over an unrelated
graph path.

Source paragraphs:
{paragraph_context(evidence.get('paragraphs', []))}

Graph paths:
{paths_context(evidence.get('paths', []))}

Entity descriptions:
{json.dumps(evidence.get('descriptions', []), ensure_ascii=False, default=str)}

Question: {question}
Answer:"""
    response = graph.generate(prompt, None, 80)
    text = (response.text or "").strip()
    lowered = normalize(text)
    if kind == "yes_no":
        match = re.search(r"\b(yes|no)\b", lowered)
        text = match.group(1) if match else text
    elif kind == "comparison" and options:
        matches = [(option, lowered.find(normalize(option))) for option in options if normalize(option) in lowered]
        if matches:
            text = min(matches, key=lambda item: item[1])[0]
    return text, {
        "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0) or 0,
        "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0) or 0,
    }


ANSWER_PROMPT = """Answer the question using ONLY the knowledge graph below.

Path notation:
- A --[relation]--> B means "A relation B" (e.g. einstein --[born in]--> ulm means Einstein was born in Ulm)
- A <--[relation]-- B means "B relation A" (e.g. ulm <--[born in]-- einstein means Einstein was born in Ulm)

Knowledge graph paths:
{paths}

Entity descriptions:
{descriptions}

Trace the connections step by step to find the answer.
Output ONLY the answer — a single entity name or short phrase, nothing else.

Question: {question}
Answer:"""
