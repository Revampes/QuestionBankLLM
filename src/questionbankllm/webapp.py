from __future__ import annotations

from flask import Flask, jsonify, render_template, request
from questionbankllm.question_ai import QuestionAnalyzer

app = Flask(__name__, template_folder=str(__import__("pathlib").Path(__file__).resolve().parents[2] / "templates"))
analyzer = QuestionAnalyzer()


def _render_index(**context):
    base_context = {
        "topics": analyzer.get_topics(),
        "topic_notes": analyzer.get_topic_notes(),
    }
    base_context.update(context)
    return render_template("index.html", **base_context)


@app.route("/", methods=["GET"])
def index():
    return _render_index()


@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form.get("question_text", "")
    if not text.strip():
        return _render_index(error="Please paste a question before submitting.")
    result = analyzer.analyze(text)
    return _render_index(result=result)


@app.route("/topic-notes", methods=["POST"])
def save_topic_note():
    payload = request.get_json(silent=True) or {}
    topic_id = (payload.get("topic_id") or "").strip()
    if not topic_id:
        return jsonify({"error": "topic_id is required"}), 400
    note = payload.get("note") or ""
    try:
        notes = analyzer.set_topic_note(topic_id, note)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    return jsonify({"notes": notes})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
