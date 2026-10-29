import ollama, chromadb
import json, uuid, random, re

MODEL = "qwen2.5:7b"
col = chromadb.PersistentClient(path="db").get_collection("books")

BASE = (
    "You are a friendly tutor for medical students. Answer ONLY using the "
    "course material provided below. If the material does not cover the "
    "question, say you don't have that in the course material. Explain in "
    "your own words, don't copy long passages. This is for study, not for "
    "clinical decisions."
)

MODES = {
    "ask": "Answer the student's question clearly and concisely.",
    "explain": ("The student is stuck. Explain step by step in simple language, "
                "give one analogy, then ask one short question to check understanding."),
    "summarize": "Summarize the material in 5-7 bullet points, easy to revise from.",
}

def retrieve(query, k=3, clinical=False):
    emb = ollama.embed(model="nomic-embed-text", input=query)["embeddings"][0]
    extra = {"where_document": {"$contains": "Clinical"}} if clinical else {}
    r = col.query(query_embeddings=[emb], n_results=k, **extra)
    return r["documents"][0], r["metadatas"][0]

def answer(message, mode="ask", page_context="", history=None, debug=False):
    query = page_context if mode == "summarize" and page_context else message
    docs, metas = retrieve(query)

    if debug:
        print("=== RETRIEVED ===")
        for d, m in zip(docs, metas):
            print(m, "\n", d[:300], "\n")

    material = "\n\n".join(d[:1500] for d in docs)

    system = f"{BASE}\n{MODES[mode]}\n\nCOURSE MATERIAL:\n{material}"
    if page_context:
        system += f"\n\nThe student is currently viewing this page:\n{page_context}"

    messages = [{"role": "system", "content": system}]
    messages += (history or [])[-6:]
    messages.append({"role": "user", "content": message})

    reply = ollama.chat(
        model=MODEL,
        messages=messages,
        options={"num_ctx": 4096, "num_predict": 400},
    )
    sources = [{"book": m["book"], "page": m["page"]} for m in metas]
    return {"answer": reply["message"]["content"], "sources": sources}

if __name__ == "__main__":
    print(answer("Which muscle initiates shoulder abduction?", debug=True))

import json, uuid, random, re
from typing import Literal
from pydantic import BaseModel

QUIZZES = {}

class QuizModel(BaseModel):
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    correct: Literal["A", "B", "C", "D"]
    explanation: str

PRACTICE_PROMPT = (
    "You are a friendly tutor helping a medical student check their "
    "understanding. Using ONLY the COURSE MATERIAL below, write ONE multiple-"
    "choice question.\n"
    "- Style: {style}\n"
    "- Exactly one option is correct; the others are plausible but wrong.\n"
    "- Options must be different kinds of answers, none broader than another.\n"
    "- Do not mention the answer in the question.\n"
    "- 'explanation': 2-3 sentences. Never mention option letters; refer to "
    "options by their text.\n"
    "Keep the question to one short sentence\n"
)

STYLES = {
    "short": "a short direct question that tests one fact or one concept.",
    "case": "a brief clinical scenario (age, sex, 2-3 findings) followed by "
            "a question. Describe findings only; do not name the answer.",
}

def make_practice(topic="", page_context="", style="short"):
    query = topic or page_context
    if not query:
        return {"error": "Please provide a topic or page_context."}

    docs, metas = retrieve(query, k=4)
    material = "\n\n".join(d[:1500] for d in docs)
    prompt = PRACTICE_PROMPT.format(style=STYLES.get(style, STYLES["short"]))

    for _ in range(3):   # only retry if the output is broken
        try:
            reply = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": f"{prompt}\nCOURSE MATERIAL:\n{material}"},
                    {"role": "user", "content": f"Topic: {query}"},
                ],
                format=QuizModel.model_json_schema(),
                options={"num_ctx": 8192, "temperature": 0.6, "num_predict": 800},
            )
            q = json.loads(reply["message"]["content"])
            opts = {"A": q["option_a"], "B": q["option_b"],
                    "C": q["option_c"], "D": q["option_d"]}
        except Exception:
            continue

        if len({v.strip().lower() for v in opts.values()}) < 4:
            continue                                   # duplicate options
        if str(opts[q["correct"]]).strip().lower() in q["question"].lower():
            continue                                   # answer leaked in question
        if re.search(r"\b(option|answer)\s+[A-D]\b", q["explanation"], re.I):
            continue                                   # letters in explanation

        # shuffle so the answer isn't always A or B
        letters = ["A", "B", "C", "D"]
        correct_text = opts[q["correct"]]
        texts = list(opts.values())
        random.shuffle(texts)
        opts = dict(zip(letters, texts))
        correct = letters[texts.index(correct_text)]

        qid = uuid.uuid4().hex
        QUIZZES[qid] = {
            "correct": correct, "options": opts,
            "explanation": q["explanation"],
            "sources": [{"book": m["book"], "page": m["page"]} for m in metas[:2]],
        }
        return {"quiz_id": qid, "question": q["question"], "options": opts,
                "note": "AI-generated practice question, not graded."}

    return {"error": "Couldn't make a question. Please try again."}

def check_practice(quiz_id, user_answer):
    q = QUIZZES.get(quiz_id)
    if not q:
        return {"error": "Question expired. Please request a new one."}
    given = (user_answer or "").strip().upper()[:1]
    return {
        "correct": given == q["correct"],
        "right_answer": q["correct"],
        "right_answer_text": q["options"][q["correct"]],
        "explanation": q["explanation"],
        "sources": q["sources"],
        "note": "AI-generated. If this seems wrong, check the book page.",
    }

def explain_answer(question, options, correct, student_answer, topic=""):
    """options: {"A": "...", ...}; correct / student_answer: letters.
    The key comes from the question bank. The AI never re-decides it."""
    docs, metas = retrieve(f"{topic} {question}".strip(), k=3)
    material = "\n\n".join(d[:1500] for d in docs)
    opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    is_right = student_answer.strip().upper() == correct.strip().upper()

    system = (
        f"{BASE}\n"
        "A student answered a quiz question. The answer key is given by the "
        "teacher and is CORRECT. Never dispute it or change it.\n"
        "Explain in simple language, in 4-6 sentences: why the correct option "
        "is right, and (if the student was wrong) why their choice is wrong "
        "and how to remember the difference. Use the COURSE MATERIAL. If the "
        "material does not cover it, say so briefly. Refer to options by their "
        "text, not their letter.\n\n"
        f"COURSE MATERIAL:\n{material}"
    )
    user = (
        f"Question: {question}\n{opts}\n"
        f"Correct answer: {correct}. {options.get(correct, '')}\n"
        f"Student answered: {student_answer}. {options.get(student_answer, '')}\n"
        f"The student was {'right' if is_right else 'wrong'}."
    )
    reply = ollama.chat(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        options={"num_ctx": 8192, "temperature": 0.3, "num_predict": 500},
    )
    return {
        "correct": is_right,
        "explanation": reply["message"]["content"],
        "sources": [{"book": m["book"], "page": m["page"]} for m in metas],
    }

CASES = {}

CASE_PROMPT = (
    "You are a medical-school examiner. Using ONLY the COURSE MATERIAL below, "
    "write ONE diagnostic case about a fictional patient. The diagnosis must be "
    "a condition described in the material (usually in its clinical notes). "
    "Do NOT mention the diagnosis anywhere in the case text. Return JSON with "
    "exactly these keys: "
    '"presentation" (age, sex, chief complaint, 1-2 sentences), '
    '"history" (onset, course, relevant background), '
    '"exam" (findings and positive/negative signs), '
    '"investigations" (imaging/test results, or "none needed"), '
    '"diagnosis" (short name), '
    '"accepted_names" (list of 2-4 synonyms or alternative names), '
    '"differentials" (list of 3 plausible wrong diagnoses), '
    '"explanation" (3-4 sentences: which findings point to the diagnosis and why '
    "the differentials are less likely)."
)

def make_case(topic="", page_context=""):
    query = topic or page_context
    docs, metas = retrieve(f"clinical {query}", k=4, clinical=True)
    material = "\n\n".join(d[:1500] for d in docs)
    reply = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": f"{CASE_PROMPT}\n\nCOURSE MATERIAL:\n{material}"},
            {"role": "user", "content": f"Topic: {query}"},
        ],
        format="json",
        options={"num_ctx": 8192, "temperature": 0.6},
    )
    case = json.loads(reply["message"]["content"])
    cid = uuid.uuid4().hex
    CASES[cid] = {**case, "revealed": set(),
                  "sources": [{"book": m["book"], "page": m["page"]} for m in metas]}
    return {"case_id": cid, "presentation": case["presentation"]}

def reveal(case_id, part):
    c = CASES.get(case_id)
    if not c:
        return {"error": "Case not found or expired."}
    if part not in ("history", "exam", "investigations"):
        return {"error": "part must be history, exam or investigations"}
    c["revealed"].add(part)
    return {part: c[part]}

def submit_diagnosis(case_id, student_answer):
    c = CASES.get(case_id)
    if not c:
        return {"error": "Case not found or expired."}
    given = student_answer.strip().lower()
    names = [c["diagnosis"].lower()] + [n.lower() for n in c.get("accepted_names", [])]
    correct = any(n in given or given in n for n in names if given)

    feedback = ""
    if not correct:   # let the model judge near-misses (e.g. different wording)
        judge = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": (
                f"Correct diagnosis: {c['diagnosis']} (also accepted: {c['accepted_names']}).\n"
                f"Student answered: {student_answer}\n"
                'Is the student\'s answer the same condition? Reply JSON: '
                '{"correct": true or false}')}],
            format="json",
            options={"temperature": 0},
        )
        correct = bool(json.loads(judge["message"]["content"]).get("correct"))

    return {
        "correct": correct,
        "diagnosis": c["diagnosis"],
        "explanation": c["explanation"],
        "differentials": c["differentials"],
        "revealed": sorted(c["revealed"]),   # frontend can score fewer reveals higher
        "sources": c["sources"],
    }

