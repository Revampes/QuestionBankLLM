from __future__ import annotations

from flask import Flask, render_template, request
from questionbankllm.question_ai import QuestionAnalyzer

app = Flask(__name__, template_folder=str(__import__("pathlib").Path(__file__).resolve().parents[2] / "templates"))
analyzer = QuestionAnalyzer()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form.get("question_text", "")
    if not text.strip():
        return render_template("index.html", error="Please paste a question before submitting.")
    result = analyzer.analyze(text)
    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
