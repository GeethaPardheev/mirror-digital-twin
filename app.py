"""Mirror: a digital twin built from conversation, running entirely on this machine. Run: uv run app.py"""
import json
import random
import re
import threading
import time
from pathlib import Path

import numpy as np
import uvicorn
import sqlite3

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

import brain
import learn
import memory as mem
import voice

TAG = re.compile(r"\s*\**\s*\[([a-z_ ]+)\]\s*\**\s*", re.I)
LABELS = {"knows": "Knows it", "infers": "Infers it", "unknown": "Doesn't know yet", "private": "Keeping this private"}


def split_tag(text):
    """Strip the leading [tag] the model writes (tolerating **bold** and variants like [knows it])."""
    m = TAG.match(text)
    if m:
        return m.group(1).lower().strip().replace(" ", "_"), text[m.end():]
    if text.lstrip("* ").startswith("[") and "]" not in text:
        return None, ""  # tag still streaming in
    return None, text


def kind_of(tag):
    return (tag or "").split("_")[0]


def who():
    return mem.get("name").strip() or "the user"


# ---------- Interview controller ----------

def opening():
    n = mem.count()
    if not n:
        return ("Hi, I'm Mirror. I'll ask you questions to build your digital twin: an AI that thinks and talks "
                "like you. Skip anything you like.\n\nFirst: what's one thing your twin must never get wrong about you?")
    return (f"Welcome back. Your twin knows {n} things about you so far. Let's open a new chapter: "
            f"**{mem.gaps(1)[0]}**. What's a moment from your life that comes to mind?")


def new_state():
    return {"turn": 0, "lens": [], "thread": None, "area": None, "area_turns": 0, "since_check": 9,
            "topic_turns": 0, "new_user": mem.count() < 12}


# Essentials first: what a twin most needs, in the order a first conversation should reach them.
LIGHT = ("work and skills", "tastes", "relationships", "habits", "life story", "humour and style")
ASK = {"work and skills": "what they do most days, work or study", "tastes": "what they love doing when they're free",
       "relationships": "the people who matter most to them", "habits": "what a normal day or weekend looks like",
       "life story": "where they grew up", "humour and style": "what makes them laugh",
       "values": "what they would never compromise on", "decisions": "a recent choice they made",
       "opinions": "something they feel strongly about", "feelings": "what lifts their mood",
       "hopes and fears": "what they're looking forward to"}
SWITCHES = ("new_chapter", "check_in", "bridge", "pending", "ease_off")  # moves that change the topic
SAY = {"work and skills": "your work", "tastes": "what you love doing", "relationships": "the people in your life",
       "habits": "your everyday routine", "life story": "where you grew up", "humour and style": "what makes you laugh",
       "values": "what matters most to you", "decisions": "a recent choice", "opinions": "something you feel strongly about",
       "feelings": "what lifts your mood", "hopes and fears": "what you're looking forward to"}  # how a person would say it


def next_area(st, light=False):
    """Breadth first: the least-covered essential not asked about recently, then everything else."""
    cov, pool = mem.coverage(), (LIGHT if light or st["turn"] <= 8 else mem.AREAS)
    asked = st.setdefault("asked", [])  # never the same topic twice in a session until the essentials run out
    choices = [a for a in pool if a != st["area"] and a not in asked] or [a for a in mem.AREAS if a not in asked] \
        or [a for a in pool if a != st["area"]]
    area = min(choices, key=cov.get)
    st["asked"].append(area)
    return area


def chapter(st, private=False, light=False):
    st["topic_turns"] = 0
    area = next_area(st, light)
    target = f"switch topics: ask about {ASK[area]}, with one easy, concrete question in everyday words"
    if light:  # they asked to move on
        return "new_chapter", f"They asked to talk about something else. Agree warmly in a few words, then {target}."
    if private:  # they shared something private: never probe it
        return "new_chapter", ("They just shared something private. Acknowledge it in a few respectful words without "
                               f"asking anything more about it, then {target}.")
    last = st.get("last_heard")
    return "new_chapter", ((f'Their recent answers were about "{last}". If this answer is still about that, ' if last else
                            "If they are still on the same topic, ")
                           + f"close it warmly in a few words and {target} that doesn't mention the old topic again. "
                           "Only if they clearly moved on to something new themselves, follow their new topic instead.")


# Painful or vulnerable words: acknowledge, then let them choose whether to stay on it.
FEELINGS = re.compile(r"\b(hurt|hurts|hurtful|invisible|lonely|alone|sad|cried|cry|crying|depress\w*|anxious|anxiety|"
                      r"scared|afraid|ashamed|embarrass\w*|humiliat\w*|angry|lost|loss|died|passed away|grief|"
                      r"bullied|abused|broke up|divorce\w*|worthless|miss (?:him|her|them)|ignored|rejected)\b", re.I)


LIGHTER = re.compile(r"\b(?:something|anything) (?:lighter|else)\b|\bchange (?:the )?(?:topic|subject)\b|\bskip this\b", re.I)


def pick_move(st, answer):
    """Force a move when the conversation needs one; otherwise the interviewer chooses."""
    n, lens = len(answer.split()), st["lens"]
    avg = sum(lens) / len(lens) if lens else n
    lens.append(n)
    st["turn"] += 1
    st["since_check"] += 1
    st["topic_turns"] += 1
    if FEELINGS.search(answer) and st["since_check"] >= 2:  # something painful: acknowledge it, let them choose
        st["since_check"], st["topic_turns"] = 0, 0
        cov = mem.coverage()
        options = " or ".join(SAY[a] for a in sorted((a for a in LIGHT if a != st["area"]), key=cov.get)[:2])
        return "check_in", ("Gently acknowledge what they just shared and the feeling in it. Then ask whether they'd "
                            f"like to keep talking about this or move on to something lighter, like {options}.")
    if LIGHTER.search(answer):  # they asked for something lighter: go somewhere light, now
        st["thread"] = None
        return chapter(st, light=True)
    if mem.SENSITIVE.search(answer):  # money, health, conflicts: acknowledge and move on, never dig
        st["thread"] = None
        return chapter(st, private=True)
    if st["thread"] and st["thread"]["turns"] >= 2:
        topic, st["thread"] = st["thread"]["topic"], None
        if st["new_user"]:
            return chapter(st)  # breadth over depth with someone new
        st["topic_turns"] = 0
        return "bridge", (f'Link the tangent ("{topic}") back to what you were exploring before it, in one phrase, '
                          "then ask a question that returns there.")
    if st["topic_turns"] >= (2 if st["new_user"] else 3):  # breadth: a question and one follow-up per topic at first
        return chapter(st)
    if len(lens) > 2 and avg >= 12 and n < 0.4 * avg:
        st["topic_turns"] = 0
        return "ease_off", "Their answers are getting short. Ask something light and easy, like a quick choice or a favourite."
    pending = mem.open_questions()
    if pending and st["turn"] % 4 == 0:
        mem.close_question(pending[0]["id"])
        st["topic_turns"] = 0
        return "pending", f'Their twin was asked "{pending[0]["text"]}" and didn\'t know. Ask how they would answer it.'
    if st["turn"] % 7 == 0 and mem.count() >= 6:
        return "mirror", ("Guess in one or two sentences how they would answer a new question, written in their own "
                          "style, then ask: Is that you?")
    return None


def splitq(text):
    """Just the question part of an interviewer reply."""
    m = re.findall(r"[^.!?\n]*\?", text)
    return (m[-1] if m else text).strip()[:120]


def known():
    return "\n".join("- " + r["text"] for r in mem.recent(6)) or "(nothing yet)"


S = {}  # one local user, one live session


def reset():
    S.update(history=[{"role": "assistant", "content": opening(), "move": "opening"}], st=new_state())


def panel():
    f, thread = mem.fidelity(), S["st"]["thread"]
    return {"name": mem.get("name"), "count": mem.count(), "coverage": mem.coverage(), "memories": mem.all_active(),
            "thread": thread["topic"] if thread else None, "questions": [x["text"] for x in mem.open_questions()[:5]],
            "fidelity": round(f["avg"] * 20) if f["n"] else None, "ratings": f["n"],
            "engine": {"label": brain.LABEL, "local": brain.PROVIDER == "ollama"}}


def sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


def interview_events(answer):
    st, history, name = S["st"], S["history"], who()
    question = history[-1]["content"] if history[-1]["role"] == "assistant" else ""
    turn_id = mem.save_turn("user", answer)
    forced = pick_move(st, answer)
    rules = [x["text"] for x in mem.lessons("interviewer")]  # learned from your thumbs-down
    system = brain.interviewer_system(name, mem.get("profile") or "(none yet)", rules)
    msgs = [{"role": m["role"], "content": m["content"]} for m in history[-30:] if m["content"]]
    msgs.append({"role": "user", "content": brain.interviewer_turn(
        answer, known(), mem.gaps(3), forced, first=st["turn"] == 1, last_move=st.get("last_move"),
        new_user=st["new_user"] and st["turn"] <= 4, liked=mem.examples("interviewer", answer, True),
        disliked=mem.examples("interviewer", answer, False),
        asked=[splitq(m["content"]) for m in history if m["role"] == "assistant" and m.get("move") != "wrap"][-8:])})
    history.append({"role": "user", "content": answer})

    t0, tag, shown, first = time.time(), None, "", None
    for attempt in range(2):  # an empty or unreadable reply gets one more try before the fallback
        buf = ""
        for chunk in brain.stream(system, msgs, temperature=0.6):
            buf += chunk
            visible = brain.after_plan(buf)  # the private plan is never shown
            if visible is None:
                continue
            tag, shown = split_tag(visible)
            cut = "?" in shown  # one question per turn: stop at the first question mark
            if cut:
                shown = shown[:shown.index("?") + 1]
            if shown and first is None:
                first = round(time.time() - t0, 1)
            yield sse({"type": "token", "text": shown})
            if cut:
                break
        if shown or (brain.after_plan(buf) or "").lstrip("[* ").strip():
            break
        print(f"[retry] empty reply: {buf[:160]!r}")
    if not shown:  # malformed reply: show what came after the plan, minus any stray bracket
        shown = (brain.after_plan(buf) or "").lstrip("[* ").strip() or "Tell me more?"
    plan = brain.plan_of(buf)
    heard = re.search(r"heard:\s*(.+)", plan)
    if heard:
        st["last_heard"] = heard.group(1).strip()[:80]  # what this answer was about, for the next topic switch
    print(f"[plan] {plan!r} -> {shown!r}")
    move = tag if tag in brain.MOVE_TAGS else (forced[0] if forced else "follow_up")
    st["last_move"] = move
    if move in SWITCHES:  # the interviewer changed topic by itself
        st["topic_turns"] = 0
    if move == "check_in":
        st["since_check"] = 0
    history.append({"role": "assistant", "content": shown, "move": move, "first": first})
    mem.save_turn("assistant", shown)
    if st["thread"]:
        st["thread"]["turns"] += 1  # pick_move bridges back after 2 turns on a tangent
    yield sse({"type": "reply", "text": shown, "move": move, "first": first})

    ex = brain.extract(name, question, answer)  # runs while they read and type
    saved = mem.store(ex, turn_id)
    if ex:
        st["area_turns"] = st["area_turns"] + 1 if ex.area == st["area"] else 1
        st["area"] = ex.area
    if not st["thread"] and question and not (FEELINGS.search(answer) or mem.SENSITIVE.search(answer)):
        topic = brain.tangent(question, answer)  # a painful or private answer is never a thread to come back to
        if topic:
            st["thread"] = {"topic": topic, "turns": 0}
    yield sse({"type": "saved", "items": saved, "panel": panel()})


def twin_prompt(question, holdout=None):
    """Everything the twin sees for one question. `holdout` hides that question's own saved answer (for checks)."""
    name, hits = who(), mem.search(question, shareable=True)  # private memories never reach the twin
    quotes = [h for h in hits if h["kind"] == "quote"] or mem.quotes(3)
    corrections = [c for c in mem.corrections() if not holdout or holdout not in c["text"]]
    safe = lambda rows, key: [r for r in rows if not mem.SENSITIVE.search(r[key])]  # nothing sensitive reaches the twin
    lessons = safe(mem.lessons("twin"), "text")
    confirmed = safe(mem.examples("twin", question, True, exclude=holdout), "text")
    system = brain.twin_system(name, mem.profile_card(), corrections, [x["text"] for x in lessons])
    turn = brain.twin_turn(question, hits, quotes, confirmed)
    return system, [{"role": "user", "content": turn}]


def twin_text(question):
    """A full twin answer as plain text, answering a held-out check case."""
    system, msgs = twin_prompt(question, holdout=question)
    return split_tag("".join(brain.stream(system, msgs, temperature=0.6)))[1].strip()


def twin_events(question):
    t0 = time.time()
    system, msgs = twin_prompt(question)
    buf, tag, text, first = "", None, "", None
    for chunk in brain.stream(system, msgs, temperature=0.6):
        buf += chunk
        tag, text = split_tag(buf)
        if text and first is None:
            first = round(time.time() - t0, 1)
        yield sse({"type": "token", "text": text, "kind": kind_of(tag)})
    kind = kind_of(tag) if kind_of(tag) in LABELS else "knows"
    if kind != "private" and re.match(r"\s*(i\s+)?(haven'?t|have not) (told|mentioned|said)", text, re.I):
        kind = "unknown"  # the words say "I don't know", whatever the tag says
    elif kind in ("knows", "infers"):  # an answer it knows shouldn't end by offering to ask
        text = re.sub(r"\s*want me to ask the real me\??\s*$", "", text, flags=re.I).strip()
    queued = kind == "unknown" and not mem.SENSITIVE.search(question)  # never queue private topics for the interviewer
    if queued:
        mem.add_question(question)
    sources = mem.by_ids({int(i) for g in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text) for i in g.split(",")})
    yield sse({"type": "done", "text": text, "kind": kind, "label": LABELS[kind], "sources": sources,
               "first": first, "queued": queued, "panel": panel()})


# ---------- people: one private twin database per person ----------

TWINS = Path(__file__).parent / "twins"
TWINS.mkdir(exist_ok=True)
LAST = TWINS / ".last"  # who was signed in, so a restart lands back in their twin
USER = {"name": None}


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "twin"


def users():
    out = []
    for f in sorted(TWINS.glob("*.db"), key=lambda p: -p.stat().st_mtime):
        try:
            with sqlite3.connect(f) as c:
                name = (c.execute("SELECT v FROM kv WHERE k='name'").fetchone() or [f.stem])[0]
                n = c.execute("SELECT COUNT(*) FROM memories WHERE status='active'").fetchone()[0]
        except sqlite3.Error:
            continue
        out.append({"name": name, "slug": f.stem, "memories": n})
    return out


def login(name):
    mem.use(str(TWINS / f"{slug(name)}.db"))
    if not mem.get("name"):
        mem.put("name", name)
    USER["name"] = mem.get("name")
    LAST.write_text(slug(name))
    classify_existing()
    reset()


def logout():
    mem.use(":memory:")
    USER["name"] = None
    LAST.unlink(missing_ok=True)
    reset()


def signed_in():
    if not USER["name"]:
        raise HTTPException(401, "Pick or create a twin first.")


ME = [Depends(signed_in)]


# ---------- HTTP ----------

app = FastAPI()


class Text(BaseModel):
    text: str = ""


class Rating(BaseModel):
    question: str
    answer: str
    score: int
    fix: str = ""
    cited: list[int] = []


class Feedback(BaseModel):
    question: str
    context: str = ""
    good: bool
    reason: str = ""


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/state")
def state():
    if not USER["name"]:
        return {"user": None, "users": users()}
    return {"user": USER["name"], "users": users(), "history": S["history"], "panel": panel()}


@app.post("/api/login")
def sign_in(body: Text):
    if not body.text.strip():
        raise HTTPException(400, "Tell me your name first.")
    login(body.text.strip())
    return state()


@app.post("/api/logout")
def sign_out():
    logout()
    return state()


@app.post("/api/delete-user")
def delete_user(body: Text):
    """Deletes one person's twin: their whole database file."""
    if USER["name"] and slug(USER["name"]) == body.text:
        logout()
    (TWINS / f"{slug(body.text)}.db").unlink(missing_ok=True)
    return state()


@app.post("/api/reset", dependencies=ME)
def new_session():
    reset()
    return state()


def safe(events):
    """Never leave the page waiting: any failure becomes a friendly message."""
    try:
        yield from events
    except Exception as e:
        print(f"[error] {type(e).__name__}: {e}")
        yield sse({"type": "error", "text": "I hit a snag reaching the model. Give me a few seconds and try again."})


@app.post("/api/interview", dependencies=ME)
def interview(body: Text):
    return StreamingResponse(safe(interview_events(body.text.strip())), media_type="text/event-stream")


@app.post("/api/twin", dependencies=ME)
def twin(body: Text):
    return StreamingResponse(safe(twin_events(body.text.strip())), media_type="text/event-stream")


@app.post("/api/wrap", dependencies=ME)
def wrap_up():
    w = brain.wrap(who(), mem.dump(shareable=True), mem.gaps(1)[0]) if mem.count() else None  # the twin reads this profile
    if not w:
        return {"error": "Tell me a bit more about yourself first, then I'll sum it up."}
    mem.put("profile", w.profile)
    mem.put("profile_v", "2")  # built from shareable memories only
    card = {"learned": w.learned, "insight": w.insight, "next_time": w.next_time}
    S["history"].append({"role": "assistant", "content": "", "move": "wrap", "wrap": card})
    return {"wrap": card, "panel": panel()}


@app.post("/api/test-question", dependencies=ME)
def test_question():
    cov = mem.coverage()
    area = random.choice([a for a, c in cov.items() if c] or list(cov))
    return {"question": brain.new_question(who(), area, mem.dump(3000, shareable=True)), "area": area}


@app.post("/api/rate", dependencies=ME)
def rate(body: Rating):
    lesson = learn.twin_feedback(who(), body.question, body.answer, body.score, body.fix.strip(), body.cited)
    return {"panel": panel(), "lesson": lesson}


class Privacy(BaseModel):
    id: int
    level: str


@app.post("/api/privacy", dependencies=ME)
def privacy(body: Privacy):
    mem.set_privacy(body.id, body.level)
    return {"panel": panel()}


def classify_existing():
    """Label memories saved before privacy levels existed: keyword safety net first, then the model."""
    if mem.get("profile") and mem.get("profile_v") != "2":  # written before privacy existed: may hold private details
        mem.put("profile", "")
        print("[privacy] cleared an old profile card built from private memories")
    rows = mem.unclassified()
    if not rows:
        return
    levels = brain.classify_privacy(rows)
    for r in rows:
        mem.set_privacy(r["id"], mem.privacy_of(r, levels.get(r["id"], "private")))
    print(f"[privacy] labelled {len(rows)} existing memories")


@app.post("/api/feedback", dependencies=ME)
def feedback(body: Feedback):
    lesson = learn.interviewer_feedback(body.context, body.question, body.good, body.reason.strip())
    return {"lesson": lesson}


def learning_state():
    return {"interviewer": mem.lessons("interviewer", 12), "twin": mem.lessons("twin", 12),
            "stats": mem.feedback_stats(), "cases": len(mem.cases(100)), "checks": mem.checks()}


@app.get("/api/learning", dependencies=ME)
def learning():
    return learning_state()


@app.post("/api/check", dependencies=ME)
def check():
    score, results = learn.run_check(who(), twin_text)
    return {"score": score, "results": results, "learning": learning_state()}


@app.post("/api/stt")
async def stt(request: Request):
    """Body: raw 16-bit mono PCM at 16 kHz, recorded in the browser."""
    pcm = np.frombuffer(await request.body(), dtype=np.int16).astype(np.float32) / 32768
    return {"text": await run_in_threadpool(voice.transcribe, pcm)}


@app.get("/api/tts")
def tts(text: str, who: str = "mirror", v: str = ""):
    return Response(voice.speak(text[:600], who, v), media_type="audio/wav")


reset()

if __name__ == "__main__":
    last = LAST.read_text().strip() if LAST.exists() else ""
    if last and (TWINS / f"{last}.db").exists():
        login(last)  # back into whoever was signed in (also labels any old memories before serving)
    threading.Thread(target=lambda: (brain.warm_up(), mem.embed(["warm up"]), voice.warm_up()), daemon=True).start()
    print("Mirror is running at http://127.0.0.1:7860")
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")
