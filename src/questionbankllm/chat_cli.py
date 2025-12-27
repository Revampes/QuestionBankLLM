from __future__ import annotations

import argparse
import sys
from typing import Optional

from questionbankllm.question_ai import ParsedQuestion, QuestionAnalyzer


def _format_result(result: ParsedQuestion) -> str:
    lines = [
        f"Source: {result.source or 'Unknown'}",
        f"Year: {result.year or 'Unknown'}",
        f"Question #: {result.question_number or 'Unknown'}",
        f"Primary topic: {result.topic_id} - {result.topic_name}",
    ]
    if result.topic_matches:
        lines.append("Topic matches:")
        for match in result.topic_matches:
            lines.append(
                f"  - {match.id} - {match.name} (confidence {match.confidence:.2f})"
            )
    if result.question_type:
        lines.append(f"Question type: {result.question_type}")
    if result.match_confidence is not None:
        match_line = f"Dataset similarity: {result.match_confidence:.3f}"
        if result.matched_dataset_id:
            match_line += f" (ID: {result.matched_dataset_id})"
        lines.append(match_line)
    if result.correct_option:
        answer_text = result.correct_option_text or ""
        suffix = f" ({answer_text})" if answer_text else ""
        lines.append(f"Predicted answer: {result.correct_option}{suffix}")
    elif result.answer_options:
        lines.append("Predicted answer: unavailable")
    lines.append("Prompt:")
    lines.append(result.prompt or "<empty>")
    if result.answer_options:
        lines.append("Options:")
        for option in result.answer_options:
            lines.append(f"  {option.label}. {option.text}")
    return "\n".join(lines)


def _analyze_and_print(text: str, analyzer: QuestionAnalyzer) -> None:
    parsed = analyzer.analyze(text)
    print(_format_result(parsed))


def _interactive_loop(analyzer: QuestionAnalyzer) -> None:
    print("Paste a chemistry question, then press Enter on an empty line to submit. Type 'exit' to quit.\n")
    while True:
        question = _collect_multiline_input()
        if question is None:
            print("Goodbye!")
            return
        if not question.strip():
            continue
        print()
        _analyze_and_print(question, analyzer)
        print()


def _collect_multiline_input() -> Optional[str]:
    lines = []
    while True:
        try:
            line = input("> ")
        except EOFError:
            return None if not lines else "\n".join(lines)
        stripped = line.strip()
        if not lines and stripped.lower() in {"exit", "quit"}:
            return None
        if stripped == "" and lines:
            break
        if stripped == "" and not lines:
            continue
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat-style interface for the ChemQuestion identifier")
    parser.add_argument("--question", help="Analyze a single question passed via the command line", default=None)
    parser.add_argument(
        "--refresh-dataset",
        action="store_true",
        help="Force re-download of the ChemQuestion dataset before running",
    )
    args = parser.parse_args()
    analyzer = QuestionAnalyzer(refresh_dataset=args.refresh_dataset)
    if args.question:
        _analyze_and_print(args.question, analyzer)
        return
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            _analyze_and_print(piped, analyzer)
            return
    _interactive_loop(analyzer)


if __name__ == "__main__":
    main()
