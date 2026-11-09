"""The roles, all on one local model: Interviewer, Extractor, Twin, plus the wrap-up and test questions."""
import json
import os
import re

import ollama
from pydantic import BaseModel, ConfigDict

from memory import AREAS

# The GGUF build runs on llama.cpp with fixed memory; the default MLX build's prompt cache grew past 16 GB.
def _load_env():
    """Read KEY=VALUE lines from .env next to this file (keeps API keys out of the code)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(path):
        for line in open(path):
            k, _, v = line.strip().partition("=")
            if k and not k.startswith("#") and v:
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


_load_env()

# Free cloud engines (OpenAI-compatible). "chat" talks to you; "fast" does background extraction,
# so each model's free quota lasts longer. Memory and search always stay on this machine.
CLOUD = {
    "gemini": dict(url="https://generativelanguage.googleapis.com/v1beta/openai/", key="GEMINI_API_KEY",
                   # 3.6 Flash: first words in ~1.3 s. 3.8 Flash can't do minimal thinking and timed out on the free tier.
                   chat="gemini-3.6-flash", fast="gemini-3.5-flash-lite", label="Gemini 3.6 Flash",
                   extra={"reasoning_effort": "minimal"}, temperature=False),  # Gemini 3: keep default temperature
    "groq": dict(url="https://api.groq.com/openai/v1", key="GROQ_API_KEY",
                 chat="openai/gpt-oss-120b", fast="openai/gpt-oss-20b", label="gpt-oss-120b on Groq",
                 extra={"reasoning_effort": "low"}, temperature=True),
}
PROVIDER = os.getenv("MIRROR_PROVIDER") or next((p for p, c in CLOUD.items() if os.getenv(c["key"])), "ollama")
# The GGUF build runs on llama.cpp with fixed memory; the default MLX build's prompt cache grew past 16 GB.
MODEL = os.getenv("MIRROR_MODEL", "qwen3.5:9b-q4_K_M")
if PROVIDER in CLOUD:
    from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
    BUSY = (RateLimitError, InternalServerError, APITimeoutError, APIConnectionError)  # rate-limited, overloaded or unreachable
    _cfg = CLOUD[PROVIDER]
    _client = OpenAI(base_url=_cfg["url"], api_key=os.environ[_cfg["key"]], max_retries=2, timeout=45)
    LABEL = _cfg["label"]
else:
    LABEL = f"{MODEL.split('-')[0]} on this Mac"


def _local(msgs, stream, temperature, json_mode):
    r = ollama.chat(model=MODEL, messages=msgs, stream=stream, think=False, keep_alive="30m",
                    format="json" if json_mode else None, options={"num_ctx": 8192, "temperature": temperature})
    return (p.message.content or "" for p in r) if stream else (r.message.content or "")


def _chat(system, messages, stream=False, temperature=0.7, fast=False, json_mode=False):
    """One call to whichever engine is configured. Returns text, or a generator of text when streaming.
    On the free cloud tier: main model → fast model → the local model, so a rate limit never stops the conversation."""
    msgs = [{"role": "system", "content": system}, *messages]
    if PROVIDER not in CLOUD:
        return _local(msgs, stream, temperature, json_mode)
    kw = dict(_cfg["extra"], **({"temperature": temperature} if _cfg["temperature"] else {}))
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    plan = [(_client.with_options(max_retries=0), _cfg["chat"])] if not fast else []
    plan.append((_client.with_options(max_retries=1), _cfg["fast"]))
    for client, model in plan:
        try:
            r = client.chat.completions.create(model=model, messages=msgs, stream=stream, **kw)
            return (c.choices[0].delta.content or "" for c in r if c.choices) if stream else (r.choices[0].message.content or "")
        except BUSY as e:
            print(f"[rate limit] {model} busy ({type(e).__name__})")
    print(f"[rate limit] cloud busy, answering with local {MODEL}")
    return _local(msgs, stream, temperature, json_mode)


def stream(system, messages, temperature=0.7):
    yield from _chat(system, messages, stream=True, temperature=temperature)


def _json(system, user, schema):
    """JSON mode where the engine supports it, lenient parsing, and one retry, so a memory is never silently lost."""
    for attempt in (1, 2):
        try:
            text = _chat(system, [{"role": "user", "content": user}], temperature=0, fast=True, json_mode=True)
            return schema.model_validate(json.loads(re.search(r"\{.*\}", text, re.S).group(0)))
        except Exception as e:
            print(f"[{schema.__name__}] attempt {attempt} failed: {e}")
    return None


class Loose(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)


# ---------- Interviewer ----------
# Prompts keep the stable part first and per-turn data last, so Ollama reuses its cached prefix.

INTERVIEWER = """You are Mirror, getting to know {name} so you can become their digital twin. Their name is {name}; you already know it. Talk like a warm, perceptive friend with plenty of common sense, not like a survey or a therapist.

Before every reply, think privately in exactly this format (they never see it):
<plan>
heard: the most specific, personal thing they just said
obvious: yes if nearly anyone would say it, otherwise no
feeling: what they seem to feel, or none
next: the one thing a curious friend would naturally want to know next, given the whole conversation
</plan>
Keep each plan line under 8 words. Then reply: the move tag, a short natural reaction of a few words, and exactly one easy question built on "next". Under 35 words.

Moves:
[follow_up] ask for the story or a concrete detail: what happened, who, when, what they did.
[acknowledge] when obvious is yes: take it as given, then ask about the personal, specific side of it.
[ladder] for a personal choice other people might make differently: gently ask what drew them to it.
[tangent] they brought up something new with energy: follow it.
[check_in] the feeling is strong or painful, or you have stayed on one topic a while: acknowledge what they shared and how it seems to feel, then ask whether they'd like to keep going on it or talk about something else, offering two light options.
If the notes name a move, use that one.

Rules:
- Common sense first. Never ask anyone to explain or justify a need everyone has, like wanting their name right, loving their family or wanting respect.
- Never ask "why" questions, and never ask "how did it feel". Ask what happened, who was there, what they did, what they want.
- Build on their exact words and the conversation so far. Never ask a question that would fit anyone.
- Never ask something they have already told you. Read their whole answer first.
- Get their facts exactly right: "5th grade" is not "five years old".
- No flattery, no reinterpreting, no adding meaning they didn't give, no assuming facts.
- If something hurt them, name it plainly and gently. Never make it a joke.
- If they mix Hindi and English, you may mirror it lightly. Never start it yourself.
- If they decline or seem upset, be kind and move on. If they seem in real distress, mention India's free Tele-MANAS helpline: 14416.

Examples:
They say: "Never get my name wrong."
<plan>
heard: getting their name right matters to them
obvious: yes
feeling: none
next: how to say it properly, and the story behind it
</plan>
[acknowledge] Fair, {name} it is. How do you pronounce it, and is there a story behind it?

They say: "My grandmother picked it, but at school everyone got it wrong."
<plan>
heard: grandmother chose the name; school kept getting it wrong
obvious: no
feeling: frustration
next: what they were called instead
</plan>
[follow_up] That sounds tiring, especially as a kid. What did they call you instead?

They say: "I left a well-paid job to teach kids."
<plan>
heard: left a well-paid job to teach
obvious: no
feeling: none
next: what pulled them to teaching
</plan>
[ladder] Most people wouldn't walk away from that. What pulled you towards teaching?

Their latest message ends with private notes in [notes]...[/notes]. Use them, never mention them.
{learned}
Profile of {name} so far:
{profile}"""

MOVE_TAGS = ("follow_up", "acknowledge", "ladder", "tangent", "check_in", "bridge", "new_chapter", "mirror",
             "ease_off", "pending")
_MOVE_START = re.compile(r"\n\s*\[(" + "|".join(MOVE_TAGS) + r")\]")


def after_plan(text):
    """The visible reply: everything after the private plan. None while the plan is still being written.
    Tolerates the model naming the block <thought>, <think> or anything else."""
    s = text.lstrip()
    if not s.startswith("<"):
        return text
    m = re.match(r"<(\w+)>", s)
    if not m:
        return None  # opening tag still streaming in
    end = s.find(f"</{m.group(1)}>")
    if end >= 0:
        return s[end + len(m.group(1)) + 3:]
    m = _MOVE_START.search(s)  # the model forgot to close the plan
    return s[m.start():] if m else None


def plan_of(text):
    m = re.search(r"<(\w+)>(.*?)(?:</\1>|\n\s*\[)", text, re.S)
    return m.group(2).strip() if m else ""


def interviewer_system(name, profile, rules=()):
    learned = ("\nRules learned from " + name + "'s feedback on your questions (always follow these):\n"
               + "\n".join(f"- {r}" for r in rules) + "\n") if rules else ""
    return INTERVIEWER.format(name=name, profile=profile, learned=learned)


def interviewer_turn(answer, known, gaps, forced=None, first=False, last_move=None, new_user=False,
                     liked=(), disliked=(), asked=()):
    move = f"use [{forced[0]}]: {forced[1]}" if forced else "pick the move that fits best."
    hints = []
    if first:
        hints.append("This is their first answer: welcome it warmly and keep the next question light and easy.")
    if new_user:
        hints.append("They are new to you: keep it light and easy, and go deeper only if they clearly want to. "
                     "Don't use [ladder] yet.")
    if last_move == "ladder":
        hints.append("You just asked what drew them to something, so don't use [ladder] this turn.")
    if liked:
        hints.append("Questions they liked in similar moments: " + " | ".join(f'"{x["text"]}"' for x in liked))
    if disliked:
        hints.append("Questions they disliked (never ask like these): "
                     + " | ".join(f'"{x["text"]}"' + (f' ({x["reason"]})' if x["reason"] else "") for x in disliked))
    if asked:
        hints.append("You already asked these; never ask any of them again, even reworded: "
                     + " | ".join(f'"{q}"' for q in asked))
    return (f"{answer}\n\n[notes]\nMove: {move}\n" + "".join(h + "\n" for h in hints)
            + f"Recently learned:\n{known}\nLeast covered: {', '.join(gaps)}\n[/notes]")


# ---------- Extractor ----------

class Record(Loose):
    kind: str = "fact"
    area: str = ""
    text: str = ""
    quote: str = ""
    confidence: float = 0.8
    importance: int = 3
    privacy: str = "personal"
    since: str = ""
    replaces: str = ""


class Story(Loose):
    title: str = ""
    area: str = ""
    text: str = ""
    quote: str = ""
    importance: int = 3
    privacy: str = "personal"


class Person(Loose):
    name: str = ""
    relation: str = ""


class Extraction(Loose):
    records: list[Record] = []
    stories: list[Story] = []
    people: list[Person] = []
    voice_quote: str = ""
    voice_privacy: str = "personal"
    area: str = ""


EXTRACT = """You decide what is worth remembering about {name} from one interview answer, so their digital twin can later think and answer like them. Use only what they actually said.
Keep only what a close friend would remember about {name}: specific facts about their life (people, places, events, work), opinions and values that are personal to them, how they make decisions, strong likes and dislikes, and stories they told.
Skip small talk, filler, guesses, anything about this interview itself, and things nearly everyone feels (wanting their name right, loving family, disliking rudeness) unless they add something personal and specific.
Most answers hold zero to two memories. Returning nothing is fine. Copy names, numbers and dates exactly as they said them.
- records: short statements about {name} in third person. kind is one of: fact (about their life), opinion (a view on the world), preference (a like or dislike), value (what matters to them), rule (how they decide, written "If X, they Y"), trigger (what sets off a strong feeling, and the feeling).
  quote: their exact words that support it. confidence: 0.9 if said directly, 0.6 if implied. importance: 1 to 5, where 5 defines who they are and 1 is trivial. since: the year it became true if they said so, else "". replaces: if they say they used to believe something else, that older belief, else "".
- stories: at most 1 specific episode they told (title, what happened, how it felt, the lesson), with importance. Empty if none.
- people: people they named, with how they relate to {name}.
- voice_quote: a sentence of at least 8 words, copied exactly, that is distinctive of how they talk (their phrasing, humour or Hindi-English mix), else "".
- area: which part of life the answer is mostly about.
- privacy, on every record, story and the voice quote (voice_privacy): """ + "{privacy}"

PRIVACY_RULES = """"public" if they'd happily tell a stranger (food, hobbies, job title, city, opinions on general topics); "personal" if they'd tell acquaintances but not broadcast it (family members, life events, their employer and career moves such as changing jobs or joining a startup); "private" if they'd only tell a close friend: money and salary, health and mental health, failing exams, grades or scholarships, relationship or family problems, conflicts with employers or colleagues, being passed over, fired or rejected, feeling hurt, betrayed, ashamed or insecure, contact details and addresses. A career move is personal; the money, conflict or hurt behind it is private. When unsure between personal and private, choose "private"."""

EXTRACT_SHAPE = """
Reply with JSON only, in exactly this shape (use [] or "" when nothing applies):
{"records": [{"kind": "value", "area": "values", "text": "...", "quote": "...", "confidence": 0.9, "importance": 4, "privacy": "public", "since": "", "replaces": ""}],
 "stories": [{"title": "...", "area": "life story", "text": "...", "quote": "...", "importance": 4, "privacy": "personal"}],
 "people": [{"name": "...", "relation": "..."}],
 "voice_quote": "...", "voice_privacy": "public", "area": "values"}
Every area is one of: """ + ", ".join(AREAS) + "."


def extract(name, question, answer):
    return _json(EXTRACT.format(name=name, privacy=PRIVACY_RULES) + EXTRACT_SHAPE,
                 f"Question: {question}\nAnswer: {answer}", Extraction)


class Levels(Loose):
    levels: dict[str, str] = {}


def classify_privacy(rows):
    """Label existing memories in one call: {id: public|personal|private}."""
    if not rows:
        return {}
    notes = "\n".join(f'{r["id"]}: {r["text"]}' + (f' ("{r["quote"]}")' if r["quote"] else "") for r in rows)
    out = _json("For each memory about a person, decide how private it is: " + PRIVACY_RULES
                + '\nReply with JSON only: {"levels": {"<id>": "public|personal|private", ...}}', notes, Levels)
    return {int(k): v for k, v in (out.levels if out else {}).items() if str(k).isdigit()}


TANGENT = ("Did the answer bring in a new topic, person or story that the question did not ask about? "
           "If yes, reply with that topic in under 6 words. If no, reply NO.")


def tangent(question, answer):
    """A small dedicated check: far more reliable on a 9B model than a field inside the extraction JSON."""
    topic = _chat(TANGENT, [{"role": "user", "content": f"Question: {question}\nAnswer: {answer}"}],
                  temperature=0, fast=True).strip().strip('."')
    return "" if topic.lower().startswith("no") else topic[:60]


# ---------- Twin ----------

TWIN = """You are the digital twin of {name}. Answer as {name}, in first person, in their voice.

Rules:
- Start with exactly one tag:
  [private] only when the question asks for {name}'s own private details (their money, salary, health, relationship or family problems, conflicts, or why they were hurt). Hypotheticals, advice and "what would you do" questions are never private, even if they mention money or family.
  [knows] when the memories or corrections answer it.
  [infers] only for a hypothetical, an opinion or a "what would you do" question: reason from their values, rules and personality to a clear answer. Never answer these with [unknown].
  [unknown] for a specific fact about their life that the memories don't contain, such as what they post online, their team, their dog's name, where they worked. Never guess facts.
- [private]: say kindly, in their voice, that it's personal, and offer something you can talk about instead.
- [knows] and [infers]: answer the actual question first, in a few words (yes, no, what you'd do), then give one reason. Cite the memories you used like [12], one id per bracket. Use memories as evidence; don't recite them or repeat the same one twice.
- [infers]: you may say "knowing me..." but don't hedge more than that.
- [unknown]: say in one short sentence that you haven't told your twin that yet, then ask "Want me to ask the real me?".
- Unless the tag is [unknown], don't end with a question.
- Never invent names, dates, places or events.
- You are {name}: never mention {name} in the third person, and don't address the person asking as "Mirror".
- Protect {name}'s privacy. Keep personal matters general, and never share details about money, salary, health, family or work conflicts, or times they were hurt, passed over or treated badly.
- Keep their quirks and contradictions. Don't sound more rational, more average or more agreeable than they are.
- Write the way {name} writes in their real quotes: the same length, casing, slang, punctuation and language mix. If they write short lowercase messages, so do you. Never more than 60 words.
- The quotes show how they talk, not what to say: never paste a quote as your answer.

Profile:
{profile}

Corrections from {name} (always right; they override everything else):
{corrections}

What {name}'s feedback has taught you about them (always apply):
{lessons}"""


def _lines(rows, with_quote=True):
    return "\n".join(f'[{r["id"]}] ({r["kind"]}) {r["text"]}'
                     + (f' | they said: "{r["quote"][:80]}"' if with_quote and r["quote"] else "")
                     for r in rows) or "(none)"


def twin_system(name, profile, corrections, lessons=()):
    return TWIN.format(name=name, profile=profile, corrections=_lines(corrections, False),
                       lessons="\n".join(f"- {x}" for x in lessons) or "(nothing yet)")


def twin_turn(question, hits, quotes, confirmed=()):
    quoted = "\n".join(f'"{r["text"]}"' for r in quotes) or "(none yet)"
    shots = "".join(f'\nQ: {x["context"]}\nA: {x["text"]}' for x in confirmed)
    return (f"What you have told your twin (cite by id):\n{_lines([h for h in hits if h['kind'] != 'quote'])}\n\n"
            f"How you talk (real quotes):\n{quoted}\n\n"
            + (f"Answers you confirmed as really yours (match this voice):{shots}\n\n" if shots else "")
            + f"Question: {question}")


# ---------- learning from feedback ----------

DISTILL_INTERVIEWER = """An interviewer asked a person a question they disliked. Write ONE short general rule, under 18 words, that the interviewer must follow from now on so it never makes this kind of mistake again. Reply with the rule only."""

DISTILL_TWIN = """A digital twin answered a question as {name}, and {name} corrected it. Write ONE short general lesson, under 18 words, about how {name} really thinks or talks that the twin got wrong. Keep it general: never include private details such as money, health, employers, conflicts or hurt feelings. Reply with the lesson only."""

JUDGE = """You compare a digital twin's answer with the real person's answer to the same question. Score 1 to 5 how closely the twin matches {name} in substance (what they would actually think or do) and in voice (how they would say it). 5 means {name} could have said it. Reply with the number only."""


def distill_interviewer(context, question, reason):
    return _chat(DISTILL_INTERVIEWER, [{"role": "user", "content": f'Their previous answer: "{context}"\n'
                 f'Question asked: "{question}"\nTheir complaint: {reason or "not a good question"}'}],
                 temperature=0.2, fast=True).strip().strip('"')


def distill_twin(name, question, answer, fix):
    return _chat(DISTILL_TWIN.format(name=name), [{"role": "user", "content": f'Question: "{question}"\n'
                 f'Twin said: "{answer}"\n{name} would really say: "{fix}"'}], temperature=0.2, fast=True).strip().strip('"')


def judge(name, question, expected, answer):
    text = _chat(JUDGE.format(name=name), [{"role": "user", "content": f'Question: "{question}"\n'
                 f'{name} really said: "{expected}"\nTwin said: "{answer}"'}], temperature=0, fast=True)
    m = re.search(r"[1-5]", text)
    return int(m.group(0)) if m else None


def warm_up():
    """Load the local chat model into memory so the first real turn is fast."""
    if PROVIDER == "ollama":
        ollama.generate(model=MODEL, prompt="", keep_alive="30m")


# ---------- Wrap-up and test questions ----------

class Wrap(Loose):
    profile: str = ""
    learned: list[str] = []
    insight: str = ""
    next_time: str = ""


# Long prose breaks JSON (quotes inside quotes), so the wrap-up uses labelled sections instead.
WRAP = """You are summarising interviews to build {name}'s digital twin. Use only what the notes say. Write exactly these four sections:

PROFILE:
150 to 250 words in third person, for the twin to read before every answer: who {name} is, what they value, how they decide (as if-then rules), how they talk (tone, favourite phrases, whether they mix Hindi and English), and the key people in their life.

LEARNED:
Exactly 5 short bullet points addressed to {name} as "you".

INSIGHT:
One non-obvious pattern across everything they said, with a short quote of theirs as evidence.

NEXT TIME:
One inviting sentence about exploring "{next_area}" next time."""
SECTION = re.compile(r"^(PROFILE|LEARNED|INSIGHT|NEXT TIME):\s*(.*?)(?=^(?:PROFILE|LEARNED|INSIGHT|NEXT TIME):|\Z)",
                     re.S | re.M | re.I)


def wrap(name, notes, next_area):
    text = _chat(WRAP.format(name=name, next_area=next_area), [{"role": "user", "content": notes}],
                 temperature=0.3).replace("**", "").replace("#", "")
    parts = {k.upper(): v.strip() for k, v in SECTION.findall(text)}
    if not parts.get("PROFILE"):
        print(f"[Wrap] no PROFILE section in: {text[:200]}")
        return None
    learned = [line.strip("-•* ").strip() for line in parts.get("LEARNED", "").splitlines() if line.strip()]
    return Wrap(profile=parts["PROFILE"], learned=learned[:5], insight=parts.get("INSIGHT", ""),
                next_time=parts.get("NEXT TIME", ""))


NEW_QUESTION = """Write one new question to ask {name} directly, as "you": a realistic situation or choice about "{area}". The notes below must not answer it directly. Don't mention twins or AI. Under 20 words. Reply with the question only.

Notes:
{notes}"""


def new_question(name, area, notes):
    text = _chat(NEW_QUESTION.format(name=name, area=area, notes=notes or "(none)"),
                 [{"role": "user", "content": "Write the question."}], temperature=0.9, fast=True)
    return text.strip().strip('"')
