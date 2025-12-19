# QuestionBankLLM

A lightweight prototype that parses Hong Kong DSE Chemistry questions, syncs labelled training data from the [Revampes/ChemQuestion](https://github.com/Revampes/ChemQuestion) repository, and predicts topics, question types, and answers with a simple similarity model.

## Features
- Parses headers like `DSE 2012 Q25` to capture source, year, and question number.
- Splits the question prompt from multiple-choice options.
- Classifies the prompt into a topic using configurable keyword lists stored in `data/topics.json`, then refines the prediction with ChemQuestion labels when a close match is found.
- Downloads the open ChemQuestion topic files on demand and builds a local, searchable dataset with question type, correct answer, and optional structured solutions.
- Provides a demo runner plus a chat-style CLI so you can paste a question and immediately see the detected metadata.

## Getting Started
1. Ensure you have Python 3.9+ installed.
2. Run the demo script:

```powershell
python -m src.question_ai
```

You should see the parsed structure for the sample question bundled in the script, along with the detected topic (Topic 9: Rate of Reaction).

3. Try the chat prompt (single-shot example shown below). The `--question` value accepts escaped newlines (`` `n `` in PowerShell) or piped text.

```powershell
python -m src.chat_cli --question "DSE 2012 Q25`n...`nD. 1200 cm3"
```

4. Drop into interactive mode to paste multiple questions one after another:

```powershell
python -m src.chat_cli
```

The first run will download the ChemQuestion topic JSON files into `data/chemquestion/raw`. Re-run with `--refresh-dataset` if the upstream data changes.

## Customising Topics
- Edit `data/topics.json` to fine-tune keyword coverage or to add more nuanced descriptors.
- Each topic entry accepts `id`, `name`, and a list of lowercase keywords.

## ChemQuestion Dataset Integration
- The downloader hits the GitHub contents API (`https://api.github.com/repos/Revampes/ChemQuestion/contents/topics`) and stores every topic file under `data/chemquestion/raw`.
- Each record includes the topic label, question type, options, and the correct option provided by ChemQuestion. The similarity model (bag-of-words cosine) finds the nearest neighbour to your input and, when confident, overrides the heuristic topic with the dataset label.
- Use `python -m src.chat_cli --refresh-dataset --question "..."` to force a re-sync if new questions are pushed upstream.

## Integrating With Your Data
- Import `QuestionAnalyzer` from `src/question_ai.py` to embed the workflow in your own tooling. It exposes a single `analyze(raw_text)` method that returns the parsed question plus any matched ChemQuestion metadata.
- Replace the `sample_question` string in `_demo()` with your own question text or feed questions through the chat CLI for rapid manual checks.
