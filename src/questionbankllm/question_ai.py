from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


GITHUB_TOPICS_API = "https://api.github.com/repos/Revampes/ChemQuestion/contents/topics"
USER_AGENT = "QuestionBankLLM/0.1"


@dataclass
class AnswerOption:
    label: str
    text: str


@dataclass
class ParsedQuestion:
    source: Optional[str]
    year: Optional[int]
    question_number: Optional[str]
    prompt: str
    raw_prompt: str
    answer_options: List[AnswerOption]
    topic_id: str
    topic_name: str
    question_type: Optional[str] = None
    correct_option: Optional[str] = None
    correct_option_text: Optional[str] = None
    structured_answer: Optional[str] = None
    match_confidence: Optional[float] = None
    matched_dataset_id: Optional[str] = None

    def combined_text(self) -> str:
        lines: List[str] = [self.prompt]
        for option in self.answer_options:
            label = option.label.upper()
            lines.append(f"{label}. {option.text}")
        return "\n".join([line for line in lines if line]).strip()


@dataclass
class ChemQuestionRecord:
    record_id: str
    topic: str
    question_type: Optional[str]
    source: Optional[str]
    year: Optional[str]
    paper: Optional[str]
    question_number: Optional[int]
    marks: Optional[int]
    topics: List[str]
    question_text: str
    options: List[AnswerOption]
    correct_option: Optional[str]
    structural_answer: Optional[str]

    def combined_text(self) -> str:
        lines: List[str] = [self.question_text]
        for option in self.options:
            lines.append(f"{option.label.upper()}. {option.text}")
        return "\n".join([line for line in lines if line]).strip()

    @property
    def primary_topic(self) -> str:
        if self.topics:
            return self.topics[0]
        return self.topic


class TopicClassifier:
    def __init__(self, topics_path: Optional[Path] = None) -> None:
        # package is installed under src/, repo root is two parents up
        base_dir = Path(__file__).resolve().parents[2]
        self.topics_path = topics_path or base_dir / "data" / "topics.json"
        self.topics = self._load_topics()
        self._topic_lookup = {topic["name"].lower(): topic for topic in self.topics}

    def _load_topics(self) -> List[Dict[str, str]]:
        with self.topics_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def predict(self, text: str) -> Dict[str, str]:
        text_lower = text.lower()
        best_topic: Optional[Dict[str, str]] = None
        best_score = 0
        for topic in self.topics:
            keywords = topic.get("keywords", [])
            score = 0
            for keyword in keywords:
                if keyword and keyword in text_lower:
                    score += 1
            if score > best_score:
                best_topic = topic
                best_score = score
        if best_topic is None:
            best_topic = {"id": "UNKNOWN", "name": "Topic not found"}
        return {"id": best_topic["id"], "name": best_topic["name"]}

    def lookup_by_name(self, topic_name: str) -> Optional[Dict[str, str]]:
        return self._topic_lookup.get(topic_name.lower())


class ChemQuestionDataset:
    def __init__(self, data_root: Optional[Path] = None) -> None:
        base_dir = Path(__file__).resolve().parents[2]
        self.data_dir = data_root or (base_dir / "data" / "chemquestion")
        self.raw_dir = self.data_dir / "raw"
        self.index_cache = self.data_dir / "index.json"
        self.records: List[ChemQuestionRecord] = []

    def refresh(self, force: bool = False) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        index = self._fetch_topics_index()
        for entry in index:
            if entry.get("type") != "file" or not entry.get("download_url"):
                continue
            destination = self.raw_dir / entry["name"]
            if destination.exists() and not force:
                continue
            self._download_file(entry["download_url"], destination)
        with self.index_cache.open("w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2)

    def load(self) -> List[ChemQuestionRecord]:
        if not self.raw_dir.exists():
            self.refresh()
        records: List[ChemQuestionRecord] = []
        for json_file in sorted(self.raw_dir.glob("*.json")):
            with json_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            topic_name = payload.get("metadata", {}).get("topic", json_file.stem)
            for question in payload.get("questions", []):
                options = [
                    AnswerOption(
                        label=(option.get("option") or "?").strip().upper(),
                        text=(option.get("content") or "").strip(),
                    )
                    for option in question.get("options", [])
                ]
                record = ChemQuestionRecord(
                    record_id=question.get("id") or f"{json_file.stem}_{len(records)}",
                    topic=topic_name,
                    question_type=question.get("type"),
                    source=question.get("source"),
                    year=question.get("year"),
                    paper=question.get("paper"),
                    question_number=question.get("questionNumber"),
                    marks=question.get("marks"),
                    topics=question.get("topics", []) or [topic_name],
                    question_text=(question.get("question") or "").strip(),
                    options=options,
                    correct_option=(question.get("correctOption") or "").strip().upper() or None,
                    structural_answer=question.get("structuralAnswer"),
                )
                records.append(record)
        self.records = records
        return records

    def _fetch_topics_index(self) -> List[Dict[str, str]]:
        try:
            return self._http_get_json(GITHUB_TOPICS_API)
        except (HTTPError, URLError) as error:
            raise RuntimeError(f"Unable to reach ChemQuestion index: {error}") from error

    def _download_file(self, url: str, destination: Path) -> None:
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request) as response:
                destination.write_bytes(response.read())
        except (HTTPError, URLError) as error:
            raise RuntimeError(f"Failed to download {url}: {error}") from error

    @staticmethod
    def _http_get_json(url: str) -> List[Dict[str, str]]:
        request = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(request) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload)


class SimilarityModel:
    def __init__(self, records: List[ChemQuestionRecord]) -> None:
        self.records = records
        self.vector_cache: List[Tuple[ChemQuestionRecord, Tuple[Counter[str], float]]] = []
        for record in records:
            vector = self._vectorize(record.combined_text())
            self.vector_cache.append((record, vector))

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _vectorize(self, text: str) -> Tuple[Counter[str], float]:
        counts: Counter[str] = Counter(self._tokenize(text))
        norm = math.sqrt(sum(value * value for value in counts.values())) or 1.0
        return counts, norm

    def find_best_match(self, text: str) -> Optional[Tuple[ChemQuestionRecord, float]]:
        if not text.strip():
            return None
        query_counts, query_norm = self._vectorize(text)
        if not query_counts:
            return None
        best_record: Optional[ChemQuestionRecord] = None
        best_score = 0.0
        for record, (counts, norm) in self.vector_cache:
            overlap = set(query_counts.keys()) & set(counts.keys())
            if not overlap:
                continue
            numerator = sum(query_counts[token] * counts[token] for token in overlap)
            denominator = query_norm * norm
            if denominator == 0:
                continue
            similarity = numerator / denominator
            if similarity > best_score:
                best_score = similarity
                best_record = record
        if best_record is None:
            return None
        return best_record, best_score


class QuestionAnalyzer:
    def __init__(self, refresh_dataset: bool = False) -> None:
        self.topic_classifier = TopicClassifier()
        self.dataset = ChemQuestionDataset()
        self.dataset.refresh(force=refresh_dataset)
        records = self.dataset.load()
        self.similarity_model = SimilarityModel(records)
        self.match_threshold = 0.65

    def analyze(self, raw_text: str) -> ParsedQuestion:
        parsed = parse_question(raw_text, classifier=self.topic_classifier)
        match = self.similarity_model.find_best_match(parsed.combined_text())
        if match:
            record, score = match
            parsed.match_confidence = round(score, 3)
            if score >= self.match_threshold:
                parsed.matched_dataset_id = record.record_id
                parsed.question_type = record.question_type
                parsed.correct_option = record.correct_option
                parsed.correct_option_text = self._option_text(record)
                parsed.structured_answer = record.structural_answer
                dataset_topic_name = record.primary_topic
                topic_entry = self.topic_classifier.lookup_by_name(dataset_topic_name)
                if topic_entry:
                    parsed.topic_id = topic_entry["id"]
                    parsed.topic_name = topic_entry["name"]
                else:
                    parsed.topic_name = dataset_topic_name
        return parsed

    @staticmethod
    def _option_text(record: ChemQuestionRecord) -> Optional[str]:
        if not record.correct_option:
            return None
        for option in record.options:
            if option.label.upper() == record.correct_option.upper():
                return option.text
        return None


def _extract_metadata(line: str) -> Optional[Dict[str, str]]:
    pattern = re.compile(r"^(?P<source>[A-Za-z ]+)\s+(?P<year>\d{4})\s+Q(?P<number>[A-Za-z0-9]+)")
    match = pattern.match(line.strip())
    if not match:
        return None
    return {
        "source": match.group("source").strip(),
        "year": match.group("year"),
        "question_number": match.group("number"),
    }


def _split_prompt_and_options(lines: List[str]) -> Dict[str, List]:
    option_pattern = re.compile(r"^[\(\[]?([A-Ha-h])[\)\].:\-]\s*(.+)$")
    option_space_pattern = re.compile(r"^([A-Ha-h])\s{2,}(.+)$")
    prompt_lines: List[str] = []
    options: List[AnswerOption] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            prompt_lines.append("")
            continue
        match = option_pattern.match(stripped) or option_space_pattern.match(stripped)
        if match:
            label = match.group(1).upper()
            text = match.group(2).strip()
            options.append(AnswerOption(label=label, text=text))
        else:
            prompt_lines.append(stripped)
    if not prompt_lines:
        prompt_lines.append("")
    return {"prompt": prompt_lines, "options": options}


def parse_question(raw_text: str, classifier: Optional[TopicClassifier] = None) -> ParsedQuestion:
    if not raw_text.strip():
        raise ValueError("Question text is empty.")
    classifier = classifier or TopicClassifier()
    normalized = raw_text.replace("\r\n", "\n").strip("\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    first_non_empty = next((idx for idx, value in enumerate(lines) if value.strip()), None)
    if first_non_empty is None:
        raise ValueError("Question text is empty.")
    metadata = _extract_metadata(lines[first_non_empty])
    if metadata:
        content_lines = lines[first_non_empty + 1 :]
    else:
        content_lines = lines[first_non_empty:]
        metadata = {"source": None, "year": None, "question_number": None}
    prompt_and_options = _split_prompt_and_options(content_lines)
    prompt_text = "\n".join([line for line in prompt_and_options["prompt"] if line]).strip()
    topic_info = classifier.predict(prompt_text)
    # try to extract an explicit answer provided by the user in the raw text
    explicit_answer = _extract_answer_from_text(raw_text)
    parsed = ParsedQuestion(
        source=metadata.get("source"),
        year=int(metadata["year"]) if metadata.get("year") else None,
        question_number=metadata.get("question_number"),
        prompt=prompt_text,
        raw_prompt="\n".join(content_lines).strip(),
        answer_options=prompt_and_options["options"],
        topic_id=topic_info["id"],
        topic_name=topic_info["name"],
    )
    if explicit_answer:
        parsed.correct_option = explicit_answer
        # fill the option text if available
        for opt in parsed.answer_options:
            if opt.label.upper() == explicit_answer.upper():
                parsed.correct_option_text = opt.text
                break
    return parsed


def _extract_answer_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    text = text.replace("\r\n", "\n")
    patterns = [
        r"\banswer\s*(?:is|:)?\s*([A-Ha-h])\b",
        r"\bans(?:wer)?\s*[:\-]?\s*([A-Ha-h])\b",
        r"\bcorrect(?:\soption)?\s*(?:is|:)?\s*([A-Ha-h])\b",
        r"\b([A-Ha-h])\s*(?:is the answer|is correct)\b",
        r"^\s*([A-Ha-h])\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return None


def _demo() -> None:
    sample_question = """DSE 2012 Q25
What is the theoretical volume of carbon dioxide that can be obtained, at room temperature and pressure,
when 1.2 g of Na2CO3(s) reacts with 50 cm3 of 1.0 M HNO3?
(Molar volume of gas at room temperature and pressure = 24 dm3;
Relative atomic masses: H = 1.0, C = 12.0, N = 14.0, O = 16.0, Na = 23.0)

A. 272 cm3
B. 544 cm3
C. 600 cm3
D. 1200 cm3
"""
    analyzer = QuestionAnalyzer()
    result = analyzer.analyze(sample_question)
    print("Source:", result.source or "Unknown")
    print("Year:", result.year or "Unknown")
    print("Question #:", result.question_number or "Unknown")
    print("Detected topic:", f"{result.topic_id} - {result.topic_name}")
    if result.question_type:
        print("Question type:", result.question_type)
    if result.correct_option:
        answer_line = f"{result.correct_option}"
        if result.correct_option_text:
            answer_line += f" ({result.correct_option_text})"
        if result.match_confidence is not None:
            answer_line += f" | similarity {result.match_confidence}"
        print("Answer:", answer_line)
    print("Prompt:\n", result.prompt)
    if result.answer_options:
        print("Options:")
        for option in result.answer_options:
            print(f"  {option.label}. {option.text}")


if __name__ == "__main__":
    _demo()
