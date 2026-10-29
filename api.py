from fastapi import FastAPI
from pydantic import BaseModel
from ai import answer, make_practice, check_practice, explain_answer, make_case, reveal, submit_diagnosis

app = FastAPI()

class Req(BaseModel):
    message: str
    mode: str = "ask"            # ask | explain | summarize
    page_context: str = ""
    history: list = []

@app.post("/ai/chat")
def chat(req: Req):
    return answer(req.message, req.mode, req.page_context, req.history)

class PracticeReq(BaseModel):
    topic: str = ""
    page_context: str = ""
    style: str = "short"        # "short" or "case"

class CheckReq(BaseModel):
    quiz_id: str
    answer: str

@app.post("/ai/practice")
def practice(req: PracticeReq):
    return make_practice(req.topic, req.page_context, req.style)

@app.post("/ai/practice/check")
def practice_check(req: CheckReq):
    return check_practice(req.quiz_id, req.answer)

class CaseReq(BaseModel):
    topic: str = ""
    page_context: str = ""

class RevealReq(BaseModel):
    case_id: str
    part: str            # "history" | "exam" | "investigations"

class DxReq(BaseModel):
    case_id: str
    answer: str          # free-text diagnosis

@app.post("/ai/case/start")
def case_start(req: CaseReq):
    return make_case(req.topic, req.page_context)

@app.post("/ai/case/reveal")
def case_reveal(req: RevealReq):
    return reveal(req.case_id, req.part)

@app.post("/ai/case/submit")
def case_submit(req: DxReq):
    return submit_diagnosis(req.case_id, req.answer)

from ai import answer, explain_answer

class ExplainReq(BaseModel):
    question: str
    options: dict          # {"A": "...", "B": "...", "C": "...", "D": "..."}
    correct: str           # letter from the question bank
    student_answer: str    # letter the student chose
    topic: str = ""

@app.post("/ai/explain-answer")
def explain(req: ExplainReq):
    return explain_answer(req.question, req.options, req.correct,
                          req.student_answer, req.topic)

