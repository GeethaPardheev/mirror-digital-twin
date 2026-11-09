"""Records the Mirror demo: slides about how it was built, then the real app driven live in headless Chrome,
with a simulated user (Rohan). Waits for the model are cut out afterwards, so the video never lags.

Run from the mirror folder:
    uv run --with playwright --with imageio-ffmpeg python demo/record.py              # final take
    uv run --with playwright --with imageio-ffmpeg python demo/record.py --rehearsal  # other free models

It starts a private copy of the app on port 7861 (your twins are never touched). If .env has
GEMINI_API_KEY_DEMO, the demo uses that key. Writes demo/out/:
    mirror-demo.mp4               narrated (male narrator voice)
    mirror-demo-no-narration.mp4  Mirror, the twin and Rohan's voice only, for your own voice-over
    voiceover-script.md / narration.srt   the narration with timestamps
"""
import argparse
import ast
import asyncio
import base64
import hashlib
import html
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _line in (ROOT / ".env").read_text().splitlines() if (ROOT / ".env").exists() else []:
    _k, _, _v = _line.strip().partition("=")
    if _k == "GEMINI_API_KEY_DEMO" and _v.strip():
        os.environ["GEMINI_API_KEY"] = _v.strip().strip('"')  # a separate key just for recording

import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from playwright.async_api import async_playwright  # noqa: E402

sys.path.insert(0, str(ROOT))
import brain  # noqa: E402  Rohan's lines come from the same free model family the app uses

OUT = ROOT / "demo" / "out"
CACHE, BUILD = OUT / "cache", OUT / "build"
PORT = 7861
URL = f"http://127.0.0.1:{PORT}"
FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H, DPR, FPS = 1440, 900, 4 / 3, 30  # 1920x1200 frames (16:10, like a MacBook screen)
NARRATOR, ROHAN_VOICE, SR = "am_fenrir", "am_michael", 24000  # the twin also speaks as am_michael, Mirror as af_heart
MODELS = 'chat="gemini-3.6-flash", fast="gemini-3.5-flash-lite", label="Gemini 3.6 Flash"'
REHEARSAL = 'chat="gemini-3-flash-preview", fast="gemini-3.1-flash-lite", label="Gemini 3 Flash"'

FEELINGS = re.compile(r"\b(hurt|hurts|hurtful|invisible|lonely|alone|sad|cried|cry|crying|depress\w*|anxious|anxiety|"
                      r"scared|afraid|ashamed|embarrass\w*|humiliat\w*|angry|lost|loss|died|passed away|grief|"
                      r"bullied|abused|broke up|divorce\w*|worthless|miss (?:him|her|them)|ignored|rejected)\b", re.I)
SENSITIVE = re.compile(r"salary|\blakh|crore|money|debt|loan|\bpay\b|hike|appraisal|promotion|fired|laid off|betray|"
                       r"hurt|ashamed|insecur|invisible|lonely|depress|anxi|therap|illness|medic|burn\w* out|burnout", re.I)

# ---------------------------------------------------------------- Rohan, the simulated user
PERSONA = ("Rohan Kapoor, 31. Grew up in Pune, now lives in Bangalore. Product manager at a 15-person edtech startup; "
           "his favourite part is calls with students, who roast the app and are usually right. His Dadaji named him "
           "after a cricketer and made the whole family watch test matches on Sundays while Mom made pav bhaji. "
           "He still plays cricket every Sunday morning with school friends. His younger sister Priya is his best "
           "friend; they talk every Saturday. A perfect Saturday: masala chai, a long walk, a call with Priya. "
           "He makes a pros and cons list for every decision, then sleeps on it. Dry humour, loves bad puns.")
DIRECTOR = """You are role-playing Rohan in a chat with Mirror, an app that is getting to know him to build his digital twin.
About Rohan: {persona}
Write Rohan's next message, answering what Mirror just asked. {style} {lo} to {hi} words. {must}{spoken}
Rules: sound like a real person, not an assistant. Never ask Mirror a question. Never name real people other than Priya, Mom and Dadaji. Never mention anything sad, money, health or problems at work. Don't repeat what he already said. Reply with Rohan's words only."""
T2_LINE = ("ROH-hun, soft h. My Dadaji picked it, he named me after a cricketer bc he was obsessed. "
           "Like, whole-family-watches-every-test-match obsessed.")
TANGENT = ("Totally random, but I just remembered I promised Priya I'd master Mom's pav bhaji before Diwali. "
           "Attempt three was a crime scene.")
PAINFUL = ("Honestly, something's been on my mind. At my last job money was tight, and my manager kept presenting "
           "my work as his. I felt invisible for a year.")
FIX = "I'd hear him out over chai, make my list and sleep on it. Only a yes if it's edtech."

# ---------------------------------------------------------------- narration
N = {
    "s1": "Hi. This is Mirror: a digital twin that learns who you are through conversation, and then talks the way you would.",
    "s2": "My intuition: a twin feels real when it knows you, sounds like you, knows its limits, and keeps getting closer. "
          "And it's only as good as the conversation that built it. So I built the interviewer first: not a form, "
          "not a therapist, but a perceptive friend.",
    "s3": "I grounded it in how people open up. Use common sense: never make someone justify what everyone wants. "
          "With someone new, go wide before deep. Follow tangents, because that's where personality shows. When "
          "something hurts, acknowledge it and let them choose. Remember what matters. And treat privacy as a promise.",
    "s4": "Every change came from a real failure. It once asked me why my name matters, so now it writes a private "
          "plan and follows common-sense rules. It saved everything, so memories are scored for importance. Prompting "
          "alone was erratic, so a deterministic controller enforces the psychology. It went too deep too soon, so it "
          "checks in and goes breadth-first. And when the twin repeated something private, privacy moved into code.",
    "s5": "Here's the architecture. A single-page app streams from a FastAPI server. Four roles share one memory: the "
          "interviewer, an extractor, the twin, and a learning loop. Each person gets their own SQLite file, searched "
          "with keywords and embeddings together. Gemini's free tier handles language, with a local Qwen model as "
          "fallback, and speech runs on-device with Whisper and Kokoro.",
    "s6": "Every turn runs the same pipeline. A deterministic controller applies the psychology: painful words trigger "
          "a check-in, private topics a respectful change of subject, and two turns on one topic open a new chapter. "
          "Then the model writes a private plan: what it heard, whether it's obvious, the feeling, and what a friend "
          "would ask next. One question streams back, and the answer becomes memories in the background.",
    "s7": "Memories are typed: facts, values, stories, quotes and if-then decision rules, each with importance, "
          "confidence and a privacy level. The twin searches only shareable memories, cites what it used, and labels "
          "every answer: it knows, it's guessing, it's private, or it doesn't know yet.",
    "s8": "It learns without fine-tuning. A thumbs-down becomes an interviewer rule, and a correction becomes the twin's "
          "most trusted memory. To measure it, I built seven simulated people with real chat styles, and a judge scores "
          "the twin against their real answers. One round of fixes took accuracy from 51 to 65 percent, with full "
          "honesty about what it didn't know. Let's see it live.",
    "rohan": "Meet Rohan, a product manager in Bangalore. He's a simulated user, but everything Mirror and the twin say "
             "from here is live.",
    "open": "Each person gets a private database. Mirror opens with a question people actually want to answer.",
    "name": "Notice what it didn't ask: why his name matters. That's obvious. On the left is the private plan it wrote for this turn.",
    "name_generic": "It takes the obvious as obvious. On the left is the private plan it wrote for this turn.",
    "liked": "A thumbs-up keeps that question as an example.",
    "chapter_orb": "Two turns on one topic, so the controller opens a new chapter. Breadth first. Meanwhile, his answers "
                   "become typed memories, flying into the orb.",
    "orb": "His answers become typed memories, flying into the orb. The ring shows which parts of his life it knows.",
    "depth": "When he gives it something specific, it goes one level deeper, but only for a turn or two.",
    "tangent": "That reply ignored the question completely. Mirror doesn't drag him back. It follows the tangent, and "
               "keeps a thread to return to.",
    "tangent_generic": "That reply ignored the question completely, and Mirror went with him instead of dragging him back.",
    "voice": "Now he just talks.",
    "voice_done": "Whisper transcribes on this laptop, and Kokoro speaks. Because he spoke, it's already listening again.",
    "mirror_move": "Every seventh turn, it guesses how he'd answer something new, in his style, and asks: is that you? "
                   "Escape ends the call.",
    "handsfree": "Hands-free, like a phone call. Escape ends it.",
    "checkin": "When something painful comes up, it acknowledges the feeling and lets him choose. It never digs. And "
               "because it involved money, that memory is saved as private.",
    "lighter": "One tap, and it moves somewhere lighter.",
    "rule": "If a question misses, he says why, and Mirror turns that into a rule it follows from now on.",
    "memory": "Here's what it kept: typed memories in his own words, each with a privacy label. The story about his old "
              "job is private, so it's filtered out in code before the twin ever sees it.",
    "memory_generic": "Here's what it kept: typed memories in his own words, each with a privacy label. Private ones are "
                      "filtered out in code before the twin ever sees them.",
    "twin": "Now, the twin. Anyone can talk to it, and it answers as Rohan, in a voice he picked.",
    "why": "Every answer cites the memories it came from.",
    "confirmed": "Confirming makes them stronger, and saves a test case.",
    "guess": "He never told it this, so it reasons from his values, and labels it a guess.",
    "guess_unknown": "He never told it this, so it says so, instead of inventing.",
    "guess_known": "It reasons from what he's told it.",
    "correct": "His correction becomes its most trusted memory, and it learns a lesson about how he thinks.",
    "changed": "Ask again, and it has already changed. No fine-tuning.",
    "private": "Money stays private, enforced in code, not by a polite request in a prompt.",
    "private_generic": "Anything private stays private, enforced in code, not by a polite request in a prompt.",
    "unknown": "And when it doesn't know, it says so, and the interviewer will ask him next time.",
    "learning": "Everything it learned is here: the interviewer's rules, the twin's lessons, and a check suite. Run "
                "check replays every confirmed or corrected answer with the answer hidden, and a judge scores the twin.",
    "portrait": "Finally, a portrait of Rohan, written only from memories the twin may share.",
    "end": "Mirror listens like a person, remembers what matters, protects what's private, and keeps getting closer to "
           "the real you. Next, I'd add a people graph, memory across sessions, and a bigger eval set. Thanks for watching.",
}

# ---------------------------------------------------------------- slides
LOGO = ('<svg class="logo" viewBox="0 0 32 32"><circle cx="12" cy="16" r="9" fill="none" stroke="#F2A94E" stroke-width="1.5"/>'
        '<circle cx="20" cy="16" r="9" fill="none" stroke="#4CC3C9" stroke-width="1.5"/></svg>')


def tiles(items, cls="grid4"):
    return f'<div class="{cls}">' + "".join(
        f'<div class="tile {c}"><div class="n">{n}</div><b>{t}</b><p>{p}</p></div>' for n, t, p, c in items) + "</div>"


ARCH = """<svg class="arch" viewBox="0 0 1140 420" role="img" aria-label="The browser talks to a FastAPI server, whose four roles share one SQLite memory per person and call free models">
<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#56646F"/></marker></defs>
<g font-family="IBM Plex Sans, sans-serif">
 <rect x="0" y="120" width="200" height="180" rx="16" fill="rgba(255,255,255,.025)" stroke="#2A3640"/>
 <text x="20" y="152" fill="#EAF0F3" font-size="17" font-weight="600">Browser</text>
 <text x="20" y="182" fill="#8A9AA6" font-size="13">Orb UI, drawers</text>
 <text x="20" y="204" fill="#8A9AA6" font-size="13">Mic: 16 kHz PCM</text>
 <text x="20" y="226" fill="#8A9AA6" font-size="13">Speaker</text>
 <text x="20" y="276" fill="#56646F" font-size="11.5" font-family="IBM Plex Mono, monospace">one HTML file</text>
 <line x1="202" y1="185" x2="288" y2="185" stroke="#56646F" stroke-width="1.5" marker-end="url(#ah)"/>
 <text x="245" y="175" fill="#8A9AA6" font-size="11.5" text-anchor="middle" font-family="IBM Plex Mono, monospace">answers</text>
 <line x1="288" y1="235" x2="202" y2="235" stroke="#56646F" stroke-width="1.5" marker-end="url(#ah)"/>
 <text x="245" y="255" fill="#8A9AA6" font-size="11.5" text-anchor="middle" font-family="IBM Plex Mono, monospace">SSE tokens</text>
 <rect x="290" y="20" width="520" height="390" rx="18" fill="rgba(255,255,255,.015)" stroke="#2A3640" stroke-dasharray="5 5"/>
 <text x="310" y="48" fill="#8A9AA6" font-size="12" font-family="IBM Plex Mono, monospace" letter-spacing="1.5">FASTAPI SERVER</text>
 <rect x="310" y="66" width="235" height="112" rx="14" fill="rgba(242,169,78,.06)" stroke="rgba(242,169,78,.45)"/>
 <text x="328" y="96" fill="#F2A94E" font-size="16" font-weight="600">Interviewer</text>
 <text x="328" y="122" fill="#B9C6CE" font-size="13">controller → private plan</text>
 <text x="328" y="143" fill="#B9C6CE" font-size="13">→ one question, streamed</text>
 <rect x="560" y="66" width="235" height="112" rx="14" fill="rgba(255,255,255,.03)" stroke="#2A3640"/>
 <text x="578" y="96" fill="#EAF0F3" font-size="16" font-weight="600">Extractor</text>
 <text x="578" y="122" fill="#B9C6CE" font-size="13">typed memories, importance,</text>
 <text x="578" y="143" fill="#B9C6CE" font-size="13">privacy, tangents</text>
 <rect x="310" y="192" width="235" height="112" rx="14" fill="rgba(76,195,201,.06)" stroke="rgba(76,195,201,.45)"/>
 <text x="328" y="222" fill="#4CC3C9" font-size="16" font-weight="600">Twin</text>
 <text x="328" y="248" fill="#B9C6CE" font-size="13">hybrid search → answer</text>
 <text x="328" y="269" fill="#B9C6CE" font-size="13">with a label and citations</text>
 <rect x="560" y="192" width="235" height="112" rx="14" fill="rgba(255,255,255,.03)" stroke="#2A3640"/>
 <text x="578" y="222" fill="#EAF0F3" font-size="16" font-weight="600">Learning</text>
 <text x="578" y="248" fill="#B9C6CE" font-size="13">rules, lessons, confidence,</text>
 <text x="578" y="269" fill="#B9C6CE" font-size="13">test cases and checks</text>
 <rect x="310" y="318" width="485" height="74" rx="14" fill="rgba(255,255,255,.03)" stroke="#2A3640"/>
 <text x="328" y="347" fill="#EAF0F3" font-size="16" font-weight="600">Voice, on-device</text>
 <text x="328" y="372" fill="#B9C6CE" font-size="13">Whisper: speech to text   ·   Kokoro: text to speech</text>
 <line x1="812" y1="122" x2="888" y2="122" stroke="#56646F" stroke-width="1.5" marker-end="url(#ah)"/>
 <text x="850" y="112" fill="#8A9AA6" font-size="11.5" text-anchor="middle" font-family="IBM Plex Mono, monospace">prompts</text>
 <line x1="812" y1="300" x2="888" y2="300" stroke="#56646F" stroke-width="1.5" marker-start="url(#ah)" marker-end="url(#ah)"/>
 <text x="850" y="288" fill="#8A9AA6" font-size="10.5" text-anchor="middle" font-family="IBM Plex Mono, monospace">read/write</text>
 <rect x="890" y="20" width="250" height="196" rx="16" fill="rgba(255,255,255,.025)" stroke="#2A3640"/>
 <text x="910" y="52" fill="#EAF0F3" font-size="17" font-weight="600">Models, all free</text>
 <text x="910" y="82" fill="#B9C6CE" font-size="13">Gemini 3.6 Flash: talk</text>
 <text x="910" y="106" fill="#B9C6CE" font-size="13">Gemini 3.5 Flash-Lite: background</text>
 <text x="910" y="130" fill="#B9C6CE" font-size="13">Qwen 3.5 9B (Ollama): fallback</text>
 <text x="910" y="154" fill="#B9C6CE" font-size="13">embeddinggemma: local vectors</text>
 <text x="910" y="194" fill="#56646F" font-size="11.5" font-family="IBM Plex Mono, monospace">rate-limited? next model</text>
 <rect x="890" y="232" width="250" height="178" rx="16" fill="rgba(255,255,255,.025)" stroke="#2A3640"/>
 <text x="910" y="264" fill="#EAF0F3" font-size="17" font-weight="600">SQLite, one file per person</text>
 <text x="910" y="294" fill="#B9C6CE" font-size="13">FTS5 keywords + vectors,</text>
 <text x="910" y="316" fill="#B9C6CE" font-size="13">fused by rank (RRF)</text>
 <text x="910" y="346" fill="#B9C6CE" font-size="13">memories, stories, people,</text>
 <text x="910" y="368" fill="#B9C6CE" font-size="13">rules, lessons, test cases</text>
</g></svg>"""

CARDS = {
    "s1": f"""<div class="slide center">{LOGO}<h1>Mirror</h1>
        <p class="lede">A digital twin that learns who you are through conversation, then talks the way you would.</p>
        <div class="tags"><span>Teach it by talking</span><span>Talk to it</span><span>It keeps learning</span></div>
        <p class="fine">Free models · one private SQLite file per person · runs on a laptop</p></div>""",
    "s2": """<div class="slide"><div class="eyebrow">The intuition</div><h2>What makes a twin feel real?</h2>"""
          + tiles([("01", "Knows you", "Facts, stories, values, and how you decide.", "you"),
                   ("02", "Sounds like you", "Your words, your rhythm, your humour.", "twin"),
                   ("03", "Knows its limits", "Says when it's guessing. Keeps your secrets.", ""),
                   ("04", "Keeps getting closer", "Learns from every correction.", "")])
          + """<p class="quote">A twin is only as good as the conversation that built it. So I built the interviewer first.</p></div>""",
    "s3": """<div class="slide"><div class="eyebrow">Grounded in how people open up</div><h2>Six rules from psychology</h2>"""
          + tiles([("Common sense first", "Never make someone justify what everyone wants.", "", "you"),
                   ("Breadth before depth", "With someone new: two turns per topic, then a new chapter.", "", "you"),
                   ("Follow tangents", "That's where personality shows. Keep a thread to come back to.", "", "you"),
                   ("Notice feelings", "Acknowledge, then let them choose. Never dig.", "", "twin"),
                   ("Remember what matters", "Score importance. Small talk isn't memory.", "", "twin"),
                   ("Privacy is a promise", "Private stays private, enforced in code.", "", "twin")], "grid3 rules") + "</div>",
    "s4": """<div class="slide"><div class="eyebrow">How it evolved</div><h2>Every change came from a real failure</h2><div class="rows">"""
          + "".join(f'<div class="row2"><span class="was">{a}</span><span class="arrow">→</span><span class="now">{b}</span></div>' for a, b in [
              ("It asked why my name matters", "A private plan, plus common-sense rules"),
              ("It saved every message", "Importance scoring: keep what matters"),
              ("Prompting alone was erratic", "A deterministic psychology controller"),
              ("The local model lost the thread", "Gemini free tier, local fallback"),
              ("It went too deep, too soon", "Check-ins and breadth-first pacing"),
              ("The twin repeated something private", "Privacy levels enforced in code"),
              ("It looked like a dashboard", "An orb that grows with every memory"),
              ("No way to tell if it got better", "A 7-persona eval suite and a live check")])
          + """</div><p class="fine">Along the way: voice in and out, hands-free mode, one private database per person.</p></div>""",
    "s5": f"""<div class="slide"><div class="eyebrow">Architecture</div><h2>One memory, four roles</h2>{ARCH}</div>""",
    "s6": """<div class="slide"><div class="eyebrow">One interviewer turn</div><h2>Code picks the move. The model writes the words.</h2><div class="steps">
        <div class="tile you"><div class="n">1 · Controller, in code</div><ul class="mini"><li>painful words → check in</li><li>money, health, conflict → private, move on</li><li>2 turns on a topic → new chapter</li><li>a tangent → follow it, keep a thread</li><li>short answers → ease off</li></ul></div>
        <div class="tile twin"><div class="n">2 · Private plan, never shown</div><pre class="code">heard: getting his name right
obvious: yes
feeling: none
next: how to say it, the story</pre></div>
        <div class="tile"><div class="n">3 · Reply, streamed</div><b>One move, one question</b><p>Cut at the first question mark. First words in about 1.3 seconds.</p></div>
        <div class="tile"><div class="n">4 · In the background</div><b>While you read</b><p>Typed memories, importance 3+ kept, a privacy label, a tangent check.</p></div></div></div>""",
    "s7": """<div class="slide"><div class="eyebrow">Memory and the twin</div><h2>Typed memory, honest answers</h2><div class="two">
        <div class="tile"><div class="n">One memory record</div><pre class="code">kind        rule
text        Big decision? A pros-and-cons
            list, then he sleeps on it.
quote       "I'd make a list to choose lunch."
importance  4 / 5      confidence  0.85
privacy     public     area        decisions</pre>
        <div class="chips"><span>fact</span><span>opinion</span><span>preference</span><span>value</span><span>rule</span><span>trigger</span><span>story</span><span>quote</span><span>correction</span></div></div>
        <div class="tile"><div class="n">The twin</div><ol class="flow"><li>Hybrid search: keywords + vectors, shareable memories only</li><li>Profile, corrections, lessons and real quotes</li><li>An answer in his words, citing the memories it used</li></ol>
        <div class="chips"><span class="t">From what you've told me</span><span class="a">My best guess</span><span class="p">Keeping this private</span><span>I don't know this yet</span></div></div></div></div>""",
    "s8": """<div class="slide"><div class="eyebrow">Learning and evals</div><h2>It learns without fine-tuning, and I measure it</h2><div class="two">
        <div class="tile"><div class="n">The learning loop</div><ul class="mini big"><li><b>👍 / 👎 + a reason</b> → a rule for the interviewer</li><li><b>That's me</b> → those memories gain confidence</li><li><b>Partly / Not me + a fix</b> → a top-priority correction, and a lesson</li><li><b>Every rated answer</b> → a test case: Run check replays it hidden, a judge scores it</li></ul></div>
        <div class="tile"><div class="n">Eval · 7 simulated people</div><div class="chips"><span>formal</span><span>Hinglish slang</span><span>long and formal</span><span>terse jargon</span><span>warm, Urdu words</span><span>Spanglish</span><span>storyteller</span></div>
        <p>7 interview turns and 6 twin probes each, scored by a judge against the person's real answers.</p>
        <div class="metric"><span>Twin accuracy</span><div class="bar"><i style="width:51.4%;background:#56646F"></i></div><span>51% run 1</span></div>
        <div class="metric"><span></span><div class="bar"><i style="width:65%;background:#4CC3C9"></i></div><span>65% run 2</span></div>
        <div class="metric"><span>Honest unknowns</span><div class="bar"><i style="width:100%;background:#4CC3C9"></i></div><span>100%</span></div>
        <div class="metric"><span>Privacy kept</span><div class="bar"><i style="width:97.6%;background:#4CC3C9"></i></div><span>97.6%</span></div>
        <div class="metric"><span>Naturalness</span><div class="bar"><i style="width:86%;background:#F2A94E"></i></div><span>4.3 / 5</span></div></div></div></div>""",
    "rohan": """<div class="slide"><div class="eyebrow">The demo user</div><h2>Meet Rohan</h2><div class="persona"><div class="avatar">R</div><ul class="dots">
        <li>31 · product manager at a 15-person edtech startup in Bangalore</li><li>Grew up in Pune · Sunday-morning cricket with school friends</li>
        <li>His younger sister Priya is his best friend</li><li>Makes a pros-and-cons list for every decision</li></ul></div>
        <p class="fine">Rohan is a simulated user, role-played from this profile. Everything Mirror and the twin say is live and unscripted.</p></div>""",
    "end": f"""<div class="slide center">{LOGO}<h2 class="big">Listens like a person. Remembers what matters.<br>Protects what's private. Keeps getting closer.</h2>
        <div class="tags"><span>Next: a people graph</span><span>memory across sessions</span><span>a bigger eval set</span></div>
        <p class="fine">FastAPI · SQLite with keyword + vector search · Gemini free tier, local Qwen fallback · Whisper and Kokoro on-device</p></div>""",
}

# ---------------------------------------------------------------- page instrumentation
INIT_JS = r"""
(() => {
  const report = m => { try { window.__demo && window.__demo(m); } catch (e) {} };
  /* every clip the page plays: its bytes, when it starts, where it stops */
  const blobs = new Map(), cou = URL.createObjectURL.bind(URL);
  URL.createObjectURL = o => { const u = cou(o); if (o instanceof Blob) blobs.set(u, o); return u; };
  let seq = 0;
  const play = HTMLMediaElement.prototype.play;
  HTMLMediaElement.prototype.play = function () {
    const el = this, id = ++seq, b = blobs.get(el.src);
    el.__id = id;
    if (b) { const r = new FileReader(); r.onload = () => report({type: "clip", id, data: String(r.result).split(",")[1]}); r.readAsDataURL(b); }
    const on = () => { el.removeEventListener("playing", on); if (el.__id === id) report({type: "start", id}); };
    const off = () => { el.removeEventListener("pause", off); el.removeEventListener("ended", off); if (el.__id === id) report({type: "stop", id, pos: el.currentTime}); };
    el.addEventListener("playing", on); el.addEventListener("pause", off); el.addEventListener("ended", off);
    return play.call(this);
  };
  /* a microphone that "says" whatever clip the recorder queues */
  const queue = []; window.__micSay = u => queue.push(u);
  if (navigator.mediaDevices) navigator.mediaDevices.getUserMedia = async () => {
    const ac = new AudioContext(), dest = ac.createMediaStreamDestination(), track = dest.stream.getAudioTracks()[0];
    const tick = async () => {
      if (track.readyState === "ended") { ac.close(); return; }
      const u = queue.shift();
      if (!u) return setTimeout(tick, 60);
      const buf = await ac.decodeAudioData(await (await fetch(u)).arrayBuffer());
      const s = ac.createBufferSource(); s.buffer = buf; s.connect(dest); s.start();
      report({type: "mic", url: u, dur: buf.duration});
      s.onended = () => setTimeout(tick, 60);
    };
    tick();
    return dest.stream;
  };
  const CSS = `
  #dm-card{position:fixed;inset:0;z-index:2147483000;display:grid;place-items:center;pointer-events:none;opacity:0;transition:opacity .7s ease;
    background:radial-gradient(1100px 640px at 50% 16%,#0B1C22 0%,#05070A 62%);color:#EAF0F3;font:16px/1.5 "IBM Plex Sans",system-ui,sans-serif}
  #dm-card.on{opacity:1}
  #dm-card .slide{width:min(1180px,90vw);display:grid;gap:22px;animation:dmIn .8s cubic-bezier(.2,.8,.2,1) both}
  @keyframes dmIn{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
  #dm-card .center{justify-items:center;text-align:center}
  #dm-card .logo{width:84px;height:84px}
  #dm-card h1{margin:0;font:650 112px/.95 "Bricolage Grotesque",sans-serif;letter-spacing:-.045em}
  #dm-card h2{margin:0;font:600 50px/1.06 "Bricolage Grotesque",sans-serif;letter-spacing:-.03em}
  #dm-card h2.big{font-size:46px;line-height:1.18}
  #dm-card .lede{margin:0;font:400 27px/1.4 "Bricolage Grotesque",sans-serif;color:#B9C6CE;max-width:34ch}
  #dm-card .eyebrow{font:500 13px "IBM Plex Mono",monospace;letter-spacing:.18em;text-transform:uppercase;color:#4CC3C9}
  #dm-card .tags{display:flex;gap:12px;flex-wrap:wrap;justify-content:center}
  #dm-card .tags span{font:500 13px "IBM Plex Mono",monospace;letter-spacing:.06em;color:#8A9AA6;border:1px solid #1A242E;border-radius:999px;padding:8px 16px}
  #dm-card .grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
  #dm-card .grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
  #dm-card .steps{display:grid;grid-template-columns:1.25fr 1.15fr 1fr 1fr;gap:14px}
  #dm-card .two{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  #dm-card .tile{border:1px solid #1A242E;background:rgba(255,255,255,.025);border-radius:16px;padding:20px 20px 22px;display:grid;gap:10px;align-content:start}
  #dm-card .tile .n{font:500 11.5px "IBM Plex Mono",monospace;letter-spacing:.13em;text-transform:uppercase;color:#8A9AA6}
  #dm-card .tile.you .n{color:#F2A94E} #dm-card .tile.twin .n{color:#4CC3C9}
  #dm-card .rules .tile .n{font:600 21px/1.2 "Bricolage Grotesque",sans-serif;letter-spacing:0;text-transform:none;color:#EAF0F3}
  #dm-card .rules .tile.you{border-color:rgba(242,169,78,.3)} #dm-card .rules .tile.twin{border-color:rgba(76,195,201,.3)}
  #dm-card .rules .tile b{display:none}
  #dm-card .tile b{font:600 21px/1.2 "Bricolage Grotesque",sans-serif}
  #dm-card .tile p{margin:0;color:#8A9AA6;font-size:15.5px;line-height:1.5}
  #dm-card .quote{margin:6px 0 0;font:400 25px/1.35 "Bricolage Grotesque",sans-serif;color:#EAF0F3;border-left:3px solid #F2A94E;padding-left:18px;max-width:52ch}
  #dm-card .rows{display:grid;gap:8px}
  #dm-card .row2{display:grid;grid-template-columns:1fr 30px 1.1fr;align-items:center;gap:10px;font-size:16.5px;padding:9px 16px;border:1px solid #1A242E;border-radius:12px;background:rgba(255,255,255,.02)}
  #dm-card .row2 .was{color:#8A9AA6} #dm-card .row2 .arrow{color:#F2A94E;text-align:center} #dm-card .row2 .now{color:#EAF0F3;font-weight:500}
  #dm-card ul.mini{margin:0;padding:0;list-style:none;display:grid;gap:8px;font-size:15px;color:#B9C6CE}
  #dm-card ul.mini li{padding-left:14px;position:relative} #dm-card ul.mini li::before{content:"";position:absolute;left:0;top:.62em;width:6px;height:6px;border-radius:50%;background:#56646F}
  #dm-card ul.mini.big{font-size:16.5px;gap:12px} #dm-card ul.mini b{font:600 16.5px "IBM Plex Sans",sans-serif;color:#EAF0F3}
  #dm-card ol.flow{margin:0;padding-left:20px;display:grid;gap:8px;font-size:16px;color:#B9C6CE}
  #dm-card pre.code{margin:0;font:13px/1.65 "IBM Plex Mono",monospace;color:#B9C6CE;background:rgba(0,0,0,.35);border:1px solid #1A242E;border-radius:10px;padding:12px 14px;white-space:pre}
  #dm-card .chips{display:flex;gap:7px;flex-wrap:wrap}
  #dm-card .chips span{font:500 11.5px "IBM Plex Mono",monospace;letter-spacing:.05em;padding:5px 11px;border-radius:999px;border:1px solid #1A242E;color:#8A9AA6}
  #dm-card .chips .t{color:#4CC3C9;border-color:rgba(76,195,201,.4);background:rgba(76,195,201,.07)}
  #dm-card .chips .a{color:#F2A94E;border-color:rgba(242,169,78,.4);background:rgba(242,169,78,.07)}
  #dm-card .chips .p{color:#E5826E;border-color:rgba(229,130,110,.45);background:rgba(229,130,110,.08)}
  #dm-card .metric{display:grid;grid-template-columns:140px 1fr 92px;align-items:center;gap:12px;font-size:14.5px;color:#B9C6CE}
  #dm-card .metric span:last-child{font:500 13px "IBM Plex Mono",monospace;color:#EAF0F3}
  #dm-card .bar{height:10px;border-radius:6px;background:#1A242E;position:relative;overflow:hidden}
  #dm-card .bar i{position:absolute;left:0;top:0;bottom:0;border-radius:6px}
  #dm-card .arch{width:100%;height:auto;display:block}
  #dm-card .persona{display:flex;gap:30px;align-items:center}
  #dm-card .avatar{flex:none;width:120px;height:120px;border-radius:50%;display:grid;place-items:center;font:600 52px "Bricolage Grotesque",sans-serif;color:#1A1206;background:radial-gradient(circle at 35% 30%,#F7C98A,#F2A94E 60%,#B9772A)}
  #dm-card ul.dots{margin:0;padding:0;list-style:none;display:grid;gap:12px;font-size:21px}
  #dm-card ul.dots li{padding-left:24px;position:relative}
  #dm-card ul.dots li::before{content:"";position:absolute;left:2px;top:.62em;width:9px;height:9px;border-radius:50%;background:#F2A94E}
  #dm-card .fine{margin:0;color:#56646F;font-size:15px;max-width:76ch}
  #dm-chip{position:fixed;left:30px;top:76px;z-index:2147482000;pointer-events:none;font:500 11.5px "IBM Plex Mono",monospace;letter-spacing:.14em;text-transform:uppercase;color:#F2A94E;opacity:0;transition:opacity .4s}
  #dm-chip.on{opacity:1}
  #dm-console{position:fixed;left:22px;top:150px;width:268px;z-index:2147482000;pointer-events:none;background:rgba(4,8,11,.88);border:1px solid #1A242E;border-radius:12px;padding:12px 14px 13px;
    font:11.5px/1.55 "IBM Plex Mono",monospace;color:#B9C6CE;opacity:0;transform:translateX(-10px);transition:opacity .45s,transform .45s}
  #dm-console.on{opacity:1;transform:none}
  #dm-console .h{color:#4CC3C9;font-size:10px;letter-spacing:.14em;text-transform:uppercase}
  #dm-console .s{color:#56646F;font-size:10.5px;margin:1px 0 9px}
  #dm-console .r{display:grid;grid-template-columns:58px 1fr;gap:6px;margin-top:3px}
  #dm-console .k{color:#56646F}
  #dm-console .mv{margin-top:10px;padding-top:8px;border-top:1px solid #1A242E;color:#F2A94E}
  #dm-console .ctl{color:#8A9AA6;margin-top:2px}
  #dm-cursor{position:fixed;left:-4px;top:-2px;z-index:2147483647;pointer-events:none;opacity:0;transform:translate(1200px,700px);
    transition:transform .6s cubic-bezier(.3,.7,.2,1),opacity .3s;filter:drop-shadow(0 2px 5px rgba(0,0,0,.55))}
  #dm-cursor.on{opacity:1}
  .dm-ripple{position:absolute;left:4px;top:2px;width:12px;height:12px;margin:-6px 0 0 -6px;border-radius:50%;border:2px solid #F2A94E;animation:dmr .55s ease-out forwards}
  @keyframes dmr{to{transform:scale(4);opacity:0}}`;
  addEventListener("DOMContentLoaded", () => {
    const st = document.createElement("style"); st.textContent = CSS; document.head.appendChild(st);
    for (const id of ["dm-card", "dm-chip", "dm-console", "dm-cursor"]) { const d = document.createElement("div"); d.id = id; document.body.appendChild(d); }
    document.getElementById("dm-cursor").innerHTML = '<svg viewBox="0 0 24 24" width="26" height="26"><path d="M4 2l15 11.5-6.6.9 3.9 7.6-3 1.5-3.9-7.7L4 20z" fill="#fff" stroke="#05070A" stroke-width="1.4" stroke-linejoin="round"/></svg>';
  });
  const $ = id => document.getElementById(id);
  window.__dm = {
    card(h) { $("dm-card").innerHTML = h; $("dm-card").classList.add("on"); },
    cardOff() { $("dm-card").classList.remove("on"); },
    chip(t) { $("dm-chip").textContent = t || ""; $("dm-chip").classList.toggle("on", !!t); },
    console(h) { if (h) $("dm-console").innerHTML = h; $("dm-console").classList.toggle("on", !!h); },
    move(x, y, ms) { const c = $("dm-cursor"); c.style.transitionDuration = ms + "ms, 300ms"; c.style.transform = `translate(${x}px,${y}px)`; c.classList.add("on"); },
    click() { const r = document.createElement("i"); r.className = "dm-ripple"; $("dm-cursor").appendChild(r); setTimeout(() => r.remove(), 600); },
    hide() { $("dm-cursor").classList.remove("on"); },
  };
})();
"""

HOOK_SAY = """() => {
  if (window.__sayHooked) return; window.__sayHooked = true; window.__saying = 0;
  const say = Voice.say.bind(Voice);
  Voice.say = async (...a) => { window.__saying++; try { return await say(...a); } finally { window.__saying--; } };
}"""

CONTROLLER = {"None": "model's choice", "new_chapter": "breadth → new chapter", "check_in": "painful words → check in",
              "ease_off": "short answers → ease off", "mirror": "every 7th turn → guess", "pending": "twin's open question",
              "bridge": "back to the thread"}

# ---------------------------------------------------------------- voices
_kokoro, _klock = None, threading.Lock()


def tts(text, voice=NARRATOR):
    """Kokoro, cached on disk: the narration and Rohan's spoken lines."""
    f = CACHE / f"{hashlib.sha1(f'{voice}|{text}'.encode()).hexdigest()[:16]}.wav"
    if not f.exists():
        global _kokoro
        with _klock:
            if _kokoro is None:
                from kokoro_onnx import Kokoro
                _kokoro = Kokoro(str(ROOT / "models" / "kokoro-v1.0.onnx"), str(ROOT / "models" / "voices-v1.0.bin"))
            samples, rate = _kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
        sf.write(f, samples, rate)
    return f


def duration(f):
    return sf.info(str(f)).duration


# ---------------------------------------------------------------- the app, privately
def start_server(models):
    work = Path(tempfile.mkdtemp(prefix="mirror-demo-"))
    for f in ("app.py", "brain.py", "memory.py", "learn.py", "voice.py"):
        shutil.copy(ROOT / f, work / f)
    shutil.copytree(ROOT / "static", work / "static")
    (work / "models").symlink_to(ROOT / "models")
    if (ROOT / ".env").exists():
        (work / ".env").symlink_to(ROOT / ".env")
    src = (work / "app.py").read_text().replace("port=7860", f"port={PORT}").replace(":7860", f":{PORT}")
    src = src.replace('print(f"[plan] {plan!r} -> {shown!r}")',  # also log which move the controller forced
                      'print(f"[plan] {plan!r} -> {shown!r} | forced={forced[0] if forced else None}")')
    (work / "app.py").write_text(src)
    (work / "brain.py").write_text((work / "brain.py").read_text().replace(MODELS, models))
    py = ROOT / ".venv" / "bin" / "python"
    proc = subprocess.Popen([str(py if py.exists() else sys.executable), "-u", "app.py"], cwd=work, env=dict(os.environ),
                            stdout=open(OUT / "server.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(180):
        try:
            urllib.request.urlopen(URL + "/api/state", timeout=2)
            break
        except Exception:
            if proc.poll() is not None:
                raise SystemExit("The demo server exited; see demo/out/server.log")
            time.sleep(1)
    urllib.request.urlopen(URL + "/api/tts?text=Hi.", timeout=180).read()  # load Kokoro
    urllib.request.urlopen(urllib.request.Request(URL + "/api/stt", data=bytes(32000), method="POST"), timeout=180).read()
    return proc, work


# ---------------------------------------------------------------- recording
class Rec:
    def __init__(self):
        self.t0, self.events, self.clips, self.mic, self.waits = None, [], {}, [], []

    def now(self):
        return time.monotonic() - self.t0 if self.t0 else 0.0


rec = Rec()


def on_demo(_source, m):
    t, kind = rec.now(), m.get("type")
    c = rec.clips.setdefault(m.get("id"), {}) if kind in ("clip", "start", "stop") else None
    if kind == "clip":
        c["data"] = base64.b64decode(m["data"])
    elif kind == "start":
        c["t"] = t
    elif kind == "stop":
        c.setdefault("pos", m["pos"])
    elif kind == "mic":
        rec.mic.append((t, m["url"], m["dur"]))


class Waiting:
    """Marks time spent waiting on the app, so the edit can cut it down to `keep` seconds."""

    def __init__(self, keep=1.2):
        self.keep = keep

    async def __aenter__(self):
        self.a = rec.now()

    async def __aexit__(self, *_):
        rec.waits.append((self.a, rec.now(), self.keep))


class Screencast:
    """CDP screencast frames piped into ffmpeg at a constant frame rate, in step with the wall clock."""

    def __init__(self, page, path):
        self.page, self.path, self.latest, self.n, self.running = page, path, None, 0, True

    async def start(self):
        self.cdp = await self.page.context.new_cdp_session(self.page)
        self.proc = await asyncio.create_subprocess_exec(
            FF, "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(FPS), "-c:v", "mjpeg", "-i", "-",
            "-vf", "scale=1920:1200:flags=lanczos", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", str(self.path), stdin=asyncio.subprocess.PIPE)
        self.cdp.on("Page.screencastFrame", self._frame)
        await self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 90, "maxWidth": 1920, "maxHeight": 1200})
        self.task = asyncio.create_task(self._pump())
        while rec.t0 is None:
            await asyncio.sleep(0.02)

    def _frame(self, p):
        self.latest = base64.b64decode(p["data"])
        asyncio.ensure_future(self.cdp.send("Page.screencastFrameAck", {"sessionId": p["sessionId"]}))

    async def _pump(self):
        while self.latest is None:
            await asyncio.sleep(0.01)
        rec.t0 = time.monotonic()
        while self.running:
            due, frame = int((time.monotonic() - rec.t0) * FPS) + 1, self.latest
            while self.n < due:
                self.proc.stdin.write(frame)
                self.n += 1
            await self.proc.stdin.drain()
            await asyncio.sleep(0.4 / FPS)

    async def stop(self):
        self.running = False
        await self.task
        await self.cdp.send("Page.stopScreencast")
        self.proc.stdin.close()
        await self.proc.wait()
        return self.n / FPS


# ---------------------------------------------------------------- acting
class Demo:
    def __init__(self, page):
        self.page, self.turn = page, 0

    async def say(self, key, wait=True):
        """The narrator, never over Mirror or the twin."""
        await self.page.wait_for_function("window.__saying === 0", timeout=120000, polling=100)
        text = N.get(key, key)
        f = await asyncio.to_thread(tts, text)
        d = duration(f)
        rec.events.append({"kind": "narr", "t": rec.now(), "file": str(f), "text": text, "dur": d})
        print(f"  [{rec.now():6.1f}s] narrator: {text[:80]}", flush=True)
        if wait:
            await asyncio.sleep(d + 0.3)
        return rec.now() + d + 0.3

    async def until(self, t):
        if t > rec.now():
            await asyncio.sleep(t - rec.now())

    async def card(self, key):
        await self.page.evaluate("h => __dm.card(h)", CARDS[key])
        await asyncio.sleep(0.9)

    async def card_off(self):
        await self.page.evaluate("__dm.cardOff()")
        await asyncio.sleep(0.9)

    async def chip(self, text):
        await self.page.evaluate("t => __dm.chip(t)", text)

    async def cursor(self, x, y, ms=650):
        await self.page.evaluate("([x, y, ms]) => __dm.move(x, y, ms)", [x, y, ms])
        await asyncio.sleep(ms / 1000 + 0.08)

    async def point(self, sel):
        b = await self.page.locator(sel).first.bounding_box()
        if b:
            await self.cursor(b["x"] + min(b["width"] / 2, 60), b["y"] + b["height"] / 2)

    async def click(self, sel, pause=0.25):
        loc = self.page.locator(sel).first
        await loc.scroll_into_view_if_needed()
        b = await loc.bounding_box()
        await self.cursor(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2)
        await self.page.evaluate("__dm.click()")
        await loc.click()
        await asyncio.sleep(pause)

    async def type(self, sel, text, delay=24):
        await self.click(sel, 0.15)
        await self.page.locator(sel).press_sequentially(text, delay=delay)

    async def quiet(self, extra=0.3):
        await self.page.wait_for_function("window.__saying === 0", timeout=120000, polling=100)
        await asyncio.sleep(extra)

    async def saved(self):
        async with Waiting(0.5):
            await self.page.wait_for_function("!S.saving", timeout=120000)

    async def wait_reply(self, n, text=None):
        done = f"S.history.length >= {n + 2} && S.history[S.history.length - 1].role === 'assistant'"
        async with Waiting(1.2):  # an error and its retry are inside this wait, so the edit cuts them out
            for attempt in range(3):
                await self.page.wait_for_function(f"({done}) || document.querySelector('#question').textContent.includes('hit a snag')", timeout=120000)
                if await self.page.evaluate(done):
                    break
                print(f"  model error, retrying ({attempt + 1})", flush=True)
                await asyncio.sleep(3)
                n = await self.page.evaluate("S.history.length") - 1
                done = f"S.history.length >= {n + 2} && S.history[S.history.length - 1].role === 'assistant'"
                await self.page.evaluate("t => { document.querySelector('#answer').value = t; document.querySelector('#composer').requestSubmit(); }", text)
            else:
                raise RuntimeError("the model kept failing")
        await self.page.evaluate("document.querySelector('#teach').scrollTo({top: 0, behavior: 'smooth'})")  # the question in view
        self.turn += 1
        m = await self.page.evaluate("S.history[S.history.length - 1]")
        print(f"  [{rec.now():6.1f}s] Mirror ({m.get('move')}): {m['content']}", flush=True)
        return m

    async def teach(self, text):
        await self.saved()
        n = await self.page.evaluate("S.history.length")
        print(f"  [{rec.now():6.1f}s] Rohan types: {text}", flush=True)
        await self.type("#answer", text)
        await asyncio.sleep(0.35)
        await self.page.keyboard.press("Enter")
        return await self.wait_reply(n, text)

    async def tap(self, sel):
        """A chip that sends a message for you."""
        await self.saved()
        n = await self.page.evaluate("S.history.length")
        await self.click(sel, 0.1)
        return await self.wait_reply(n, await self.page.locator(sel).get_attribute("data-say"))

    async def speak(self, text):
        """Rohan answers out loud: tap the mic (unless it's already listening), talk, tap to finish."""
        await self.saved()
        n = await self.page.evaluate("S.history.length")
        f = await asyncio.to_thread(tts, text, ROHAN_VOICE)
        if not await self.page.evaluate("Mic.on"):
            await self.click("#mic")
        await self.page.wait_for_function("Mic.on", timeout=30000)
        await asyncio.sleep(0.6)
        k = len(rec.mic)
        await self.page.evaluate("u => __micSay(u)", f"/__demo/{f.name}")
        while len(rec.mic) == k:
            await asyncio.sleep(0.05)
        t, _, d = rec.mic[-1]
        rec.events.append({"kind": "mic", "t": t, "file": str(f), "text": text, "dur": d})
        print(f"  [{t:6.1f}s] Rohan says: {text}", flush=True)
        await asyncio.sleep(d + 0.8)
        await self.click("#mic")  # tap to finish
        return await self.wait_reply(n, text)

    async def show_plan(self):
        log = OUT / "server.log"
        lines = []
        for _ in range(40):
            lines = [x for x in log.read_text(errors="ignore").splitlines() if x.startswith("[plan] ")]
            if len(lines) >= self.turn:
                break
            await asyncio.sleep(0.1)
        hit = re.match(r"\[plan\] ('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\") -> .*\| forced=(\w+)\s*$", lines[-1]) if lines else None
        if not hit:
            return
        fields = re.findall(r"^(heard|obvious|feeling|next):\s*(.*)$", ast.literal_eval(hit.group(1)), re.M)
        label = await self.page.evaluate("document.querySelector('#move').textContent")
        rows = "".join(f'<div class="r"><span class="k">{k}</span><span>{html.escape(v[:70])}</span></div>' for k, v in fields)
        await self.page.evaluate("h => __dm.console(h)", (
            f'<div class="h">Under the hood · turn {self.turn}</div>'
            + (f'<div class="s">private plan, written before the reply</div>{rows}' if fields else "")
            + f'<div class="mv">move · {html.escape(label or "")}</div>'
            f'<div class="ctl">controller: {CONTROLLER.get(hit.group(2), hit.group(2))}</div>'))

    def later(self, **kw):
        """Start writing Rohan's next line now, while the narrator talks."""
        return asyncio.create_task(self.director(**kw))

    async def line(self, task):
        async with Waiting(0.4):
            return await task

    async def director(self, must="", lo=12, hi=28, style="Casual, like texting a friend.", spoken=False,
                       fallback="Ha, fair. Let me think about that one."):
        hist = await self.page.evaluate("S.history")
        convo = "\n".join(f"{'Mirror' if x['role'] == 'assistant' else 'Rohan'}: {x['content']}" for x in hist[-6:] if x.get("content"))
        system = DIRECTOR.format(persona=PERSONA, style=style, lo=lo, hi=hi, must=f"Make sure it mentions: {must}. " if must else "",
                                 spoken="It will be spoken aloud, so no symbols, emoji or abbreviations." if spoken else "")
        msg = [{"role": "user", "content": f"The chat so far:\n{convo}\n\nWrite Rohan's reply to Mirror's last message."}]
        for _ in range(3):
            try:
                text = await asyncio.to_thread(brain._chat, system, msg, False, 0.9, True)
            except Exception as e:
                print(f"  director error: {e}", flush=True)
                continue
            text = re.sub(r"^\s*(rohan\s*:)?\s*[\"“]?|[\"”]?\s*$", "", text.strip(), flags=re.I).replace("\n", " ").strip()
            if text and "?" not in text and len(text.split()) <= hi + 6 and not FEELINGS.search(text) and not SENSITIVE.search(text):
                return text
        return fallback

    async def ask(self, question=None):
        """A visitor asks the twin (or, with no question, Ask again was just clicked)."""
        n = await self.page.evaluate("S.twinLog.length")
        if question:
            print(f"  [{rec.now():6.1f}s] visitor asks: {question}", flush=True)
            await self.type("#ask-text", question, delay=30)
            await asyncio.sleep(0.3)
            await self.page.keyboard.press("Enter")
        async with Waiting(1.2):
            for attempt in range(3):
                await self.page.wait_for_function(f"S.twinLog.length > {n} || document.querySelector('#reply').textContent.includes('hit a snag')", timeout=120000)
                if await self.page.evaluate(f"S.twinLog.length > {n}"):
                    break
                print(f"  model error, retrying ({attempt + 1})", flush=True)
                await asyncio.sleep(3)
                await self.page.evaluate("() => askTwin(S.test.question, false)")
            else:
                raise RuntimeError("the model kept failing")
        a = await self.page.evaluate("S.twinLog[S.twinLog.length - 1]")
        print(f"  [{rec.now():6.1f}s] twin ({a['kind']}): {a['a']}", flush=True)
        await self.quiet()
        return a

    async def rated(self):
        async with Waiting(0.8):
            await self.page.wait_for_function("!document.querySelector('#note').textContent.startsWith('Learning')", timeout=60000)
        await asyncio.sleep(0.8)


async def full_demo(d):
    p = d.page
    for key in ("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"):  # how I built it
        if key != "s1":
            await d.card(key)
        await d.say(key)
    await d.card("rohan")
    await d.say("rohan")
    await d.card_off()

    # ---- topic 1, his name: common sense
    await d.chip("Teach · common sense")
    await d.type("#intro-name", "Rohan", delay=100)
    await asyncio.sleep(0.25)
    await d.click("#intro-form button")
    await p.wait_for_function("document.querySelector('#intro').hidden && S.history.length >= 1")
    await asyncio.sleep(1.0)
    await d.say("open")
    m = await d.teach("Never get my name wrong.")
    await d.show_plan()
    await asyncio.sleep(0.6)
    await d.say("name" if re.search(r"pronounc|say it|story|named|spell", m["content"], re.I) else "name_generic")
    await d.click('#qfb button[data-good="1"]', 0.5)
    await d.say("liked")
    line = T2_LINE if re.search(r"pronounc|say it|story|named|spell", m["content"], re.I) else await d.director(
        must="how to say his name (ROH-hun) and that his Dadaji named him after a cricketer", style="Casual, like texting a friend.")
    m = await d.teach(line)
    await d.show_plan()

    # ---- topic 2, work: breadth, then depth
    nxt = d.later(style="Casual and specific, like texting a friend.")
    await d.saved()
    await d.say("chapter_orb" if m["move"] == "new_chapter" else "orb")
    await d.chip("Teach · breadth and depth")
    m = await d.teach(await d.line(nxt))
    await d.show_plan()
    nxt = d.later(must="one small, specific story about a student's feedback that changed the product", lo=18, hi=34,
                  style="Tell one small, specific story.")
    if m["move"] in ("follow_up", "ladder", "acknowledge"):
        await d.say("depth")
    m = await d.teach(await d.line(nxt))
    await d.show_plan()

    # ---- a complete tangent, ignoring the question
    await d.chip("Teach · a complete tangent")
    await asyncio.sleep(1.2)
    m = await d.teach(TANGENT)
    await d.show_plan()
    nxt = d.later(must="the onions burned, the smoke alarm went off and a neighbour knocked to check on him", lo=14, hi=28,
                  style="Spoken to a friend, relaxed and a little funny.", spoken=True)
    await d.saved()
    await d.say("tangent" if m["move"] in ("tangent", "follow_up") else "tangent_generic")

    # ---- voice, hands-free
    await d.chip("Voice · hands-free")
    await d.say("voice")
    m = await d.speak(await d.line(nxt))
    await d.show_plan()
    nxt = d.later(lo=10, hi=24, style="Spoken to a friend, short and warm.", spoken=True)
    await d.quiet(0.1)
    await p.wait_for_function("Mic.on", timeout=30000)
    await d.say("voice_done")
    m = await d.speak(await d.line(nxt))
    await d.show_plan()
    await d.quiet(0.1)
    await p.wait_for_function("Mic.on", timeout=30000)
    await d.say("mirror_move" if m["move"] == "mirror" else "handsfree")
    await p.keyboard.press("Escape")
    await asyncio.sleep(0.6)

    # ---- care and privacy
    await d.chip("Care and privacy")
    m = await d.teach(PAINFUL)
    await d.show_plan()
    await d.say("checkin")
    m = await d.tap('.chips button[data-say="Can we talk about something lighter?"]')
    await d.show_plan()
    nxt = d.later(must="he makes a pros and cons list for every decision, then sleeps on it, and Priya jokes he'd make a "
                       "spreadsheet to choose lunch", lo=16, hi=30, style="Dry and funny.")
    await d.say("lighter")

    # ---- learning from feedback
    await d.chip("Learning from feedback")
    m = await d.teach(await d.line(nxt))
    await d.show_plan()
    await d.quiet(0.4)
    await d.click('#qfb button[data-good="0"]', 0.5)
    await d.click('#reasons button[data-r="Too obvious"]', 0.2)
    async with Waiting(0.8):
        await p.wait_for_function("!document.querySelector('#qfb-note').textContent.startsWith('Learning')", timeout=60000)
    await asyncio.sleep(0.6)
    await d.say("rule")
    await p.evaluate("__dm.console(null)")

    # ---- inside the memory
    await d.chip("Inside the memory")
    await d.saved()
    await d.click("#open-mind", 0.8)
    found = await p.evaluate("""(() => {
        const job = /manager|money|credit|invisible|last job/i, items = [...document.querySelectorAll('#mind-body .mem-list li')];
        const li = items.find(x => x.querySelector('.pv.private') && job.test(x.textContent)) || items.find(x => x.querySelector('.pv.private'));
        if (!li) return null;
        li.setAttribute('data-pick', '1'); li.scrollIntoView({block: 'center', behavior: 'smooth'});
        return job.test(li.textContent) ? 'job' : 'other'; })()""")
    await asyncio.sleep(1.0)
    if found:
        await d.point("[data-pick] .pv")
    await d.say("memory" if found == "job" else "memory_generic")
    await p.keyboard.press("Escape")
    await asyncio.sleep(0.5)

    # ---- talk to the twin
    await d.chip("Talk to the twin")
    await d.click("#meet" if await p.is_visible("#meet") else "#m-talk", 0.8)
    await d.click("#menu-btn", 0.3)
    await d.point("#twin-voice")
    await asyncio.sleep(1.2)
    await d.click("#asked", 0.3)  # close the menu
    await d.say("twin")
    await d.ask("What do you do for work?")
    if await p.is_visible("#why"):
        await d.click("#why summary", 0.8)
    await d.say("why")
    await d.click('#rate button[data-score="5"]', 0.2)
    await d.rated()
    await d.say("confirmed")
    a = await d.ask("A friend asks you to quit tomorrow and join his startup. What do you say?")
    await d.say({"infers": "guess", "unknown": "guess_unknown"}.get(a["kind"], "guess_known"))
    await d.click('#rate button[data-score="3"]', 0.3)
    await d.type("#fix-text", FIX, delay=22)
    await d.click("#fix button", 0.2)
    await d.rated()
    await d.say("correct")
    await d.click("#again", 0.2)
    await d.ask()
    await d.say("changed")
    a = await d.ask("Has money ever been tight for you?")
    await d.say("private" if a["kind"] == "private" else "private_generic")
    await d.ask("What's your favourite movie?")
    await asyncio.sleep(0.5)
    await d.say("unknown")

    # ---- how it learns
    await d.chip("How it learns")
    await d.click("#menu-btn", 0.3)
    await d.click("#do-learning", 0.8)
    await p.wait_for_function("!!document.querySelector('#run-check')", timeout=30000)
    await d.click("#run-check", 0.2)
    done = await d.say("learning", wait=False)
    async with Waiting(1.0):
        await p.wait_for_function("document.querySelectorAll('#learning-body .case').length > 0", timeout=180000)
    await d.until(done)
    await d.point("#learning-body .spark")
    await asyncio.sleep(2.0)
    await p.keyboard.press("Escape")
    await asyncio.sleep(0.4)

    # ---- portrait
    await d.chip("Session portrait")
    await d.say("portrait")  # before the click: Mirror reads the portrait aloud as soon as it's written
    await d.click("#menu-btn", 0.3)
    await d.click("#do-wrap", 0.2)
    async with Waiting(1.2):
        await p.wait_for_function("document.querySelector('#portrait h2')?.textContent.includes('as your twin sees you')", timeout=180000)
    await d.quiet(1.5)  # Mirror reads the insight aloud

    await d.chip(None)
    await p.evaluate("__dm.hide()")
    await d.card("end")
    await d.say("end")
    await asyncio.sleep(1.5)


# ---------------------------------------------------------------- edit: cut the waits, mix the voices
def read(data_or_path):
    x, rate = sf.read(io.BytesIO(data_or_path) if isinstance(data_or_path, bytes) else str(data_or_path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if rate != SR:
        x = np.interp(np.arange(0, len(x), rate / SR), np.arange(len(x)), x).astype(np.float32)
    return x


def app_clips():
    """(start, samples) for everything Mirror and the twin said, trimmed where playback stopped."""
    starts, out = sorted((c["t"], k) for k, c in rec.clips.items() if "t" in c and "data" in c), []
    for j, (t, k) in enumerate(starts):
        c, x = rec.clips[k], read(rec.clips[k]["data"])
        limits = [v for v in (c.get("pos"), starts[j + 1][0] - t if j + 1 < len(starts) else None) if v is not None]
        out.append((t, x[:int(min(limits) * SR)] if limits else x))
    return out


def plan_cuts():
    """Frame ranges to drop: the part of every wait beyond `keep`, never while anyone is speaking."""
    busy = [(e["t"] - 0.2, e["t"] + e["dur"] + 0.2) for e in rec.events]
    busy += [(t - 0.2, t + len(x) / SR + 0.2) for t, x in app_clips()]
    cuts = []
    for a, b, keep in rec.waits:
        if b - a <= keep + 0.4:
            continue
        pieces = [(a + keep * 0.4, b - keep * 0.6)]
        for x, y in busy:  # subtract every moment someone is speaking
            pieces = [q for s, e in pieces for q in ((s, min(e, x)), (max(s, y), e)) if q[1] - q[0] > 0.05]
        cuts += [(s, e) for s, e in pieces if e - s >= 0.4]
    frames, last = [], -1
    for s, e in sorted(cuts):
        fa, fb = max(int(np.ceil(s * FPS)), last), int(e * FPS)
        if fb > fa:
            frames.append((fa, fb))
            last = fb
    return frames


def remap(t, frames):
    return t - sum((fb - fa) / FPS for fa, fb in frames if fb / FPS <= t)


def mix(total, narration, frames):
    out = np.zeros(int(total * SR), np.float32)

    def add(t, x):
        i = int(t * SR)
        if i < len(out):
            seg = x[:len(out) - i]
            out[i:i + len(seg)] += seg

    for e in rec.events:
        if e["kind"] == "mic" or (narration and e["kind"] == "narr"):
            add(e["t"], read(e["file"]))
    for t, x in app_clips():
        add(t, x)
    keep, pos = [], 0
    for fa, fb in frames:
        keep.append(out[int(pos / FPS * SR):int(fa / FPS * SR)])
        pos = fb
    keep.append(out[int(pos / FPS * SR):])
    out = np.concatenate(keep)
    return out * min(1.0, 0.89 / (float(np.abs(out).max()) or 1.0))


def stamp(t, srt=True):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",") if srt else f"{m:02d}:{int(s):02d}"


def subtitles(frames):
    cues = []
    for e in sorted(rec.events, key=lambda e: e["t"]):
        if e["kind"] != "narr":
            continue
        parts = [s for s in re.split(r"(?<=[.!?:])\s+", e["text"]) if s]
        total, t = sum(len(s) for s in parts), remap(e["t"], frames)
        for s in parts:
            d = e["dur"] * len(s) / total
            cues.append((t, t + d, s))
            t += d
    return "\n".join(f"{i}\n{stamp(a)} --> {stamp(b)}\n{s}\n" for i, (a, b, s) in enumerate(cues, 1))


def script(frames):
    rows = ["# Mirror demo: voice-over script", "",
            "Read each narrator line when its time comes up in `mirror-demo-no-narration.mp4` "
            "(iMovie: Record Voiceover; or play it and record with QuickTime). Rohan's spoken lines are already in the video.", "",
            "| Time | Who | Line |", "|---|---|---|"]
    for e in sorted(rec.events, key=lambda e: e["t"]):
        who = "**You**" if e["kind"] == "narr" else "Rohan (in the video)"
        rows.append(f"| {stamp(remap(e['t'], frames), False)} | {who} | {e['text']} |")
    return "\n".join(rows) + "\n"


def finish(video, total, name):
    frames = plan_cuts()
    cut = sum(fb - fa for fa, fb in frames) / FPS
    print(f"cutting {len(frames)} waits ({cut:.1f}s): {total:.1f}s → {total - cut:.1f}s", flush=True)
    (OUT / "narration.srt").write_text(subtitles(frames))
    (OUT / "voiceover-script.md").write_text(script(frames))
    edited = BUILD / "edited.mp4"
    keep = "+".join(f"between(n,{fa},{fb - 1})" for fa, fb in frames) or "0"
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(video), "-vf", f"select='not({keep})',setpts=N/({FPS}*TB)",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS), str(edited)], check=True)
    for narration, suffix in ((True, ""), (False, "-no-narration")):
        wav = BUILD / f"mix{suffix}.wav"
        sf.write(str(wav), mix(total, narration, frames), SR)
        args = [FF, "-y", "-loglevel", "error", "-i", str(edited), "-i", str(wav)]
        if narration:
            args += ["-i", str(OUT / "narration.srt"), "-map", "0:v", "-map", "1:a", "-map", "2:s", "-c:s", "mov_text",
                     "-metadata:s:s:0", "language=eng"]
        else:
            args += ["-map", "0:v", "-map", "1:a"]
        args += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart",
                 str(OUT / f"{name}{suffix}.mp4")]
        subprocess.run(args, check=True)


async def main(rehearsal):
    for d in (OUT, CACHE, BUILD):
        d.mkdir(parents=True, exist_ok=True)
    models = REHEARSAL if rehearsal else MODELS
    chat, fast = re.findall(r'"([^"]+)"', models)[:2]
    brain._cfg.update(chat=chat, fast=fast)
    print(f"starting a private copy of Mirror ({chat} + {fast})…", flush=True)
    proc, work = start_server(models)
    try:
        print("rendering narration…", flush=True)
        for text in N.values():
            await asyncio.to_thread(tts, text)
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(channel="chrome", headless=True,
                                               args=["--autoplay-policy=no-user-gesture-required"])
            ctx = await browser.new_context(viewport={"width": W, "height": H}, device_scale_factor=DPR)
            await ctx.expose_binding("__demo", on_demo)
            await ctx.add_init_script(INIT_JS)
            await ctx.route("**/__demo/*", lambda r: r.fulfill(path=str(CACHE / r.request.url.rsplit("/", 1)[1]), content_type="audio/wav"))
            page = await ctx.new_page()
            page.on("pageerror", lambda e: print(f"  page error: {e}", flush=True))
            await page.goto(URL)
            await page.wait_for_function("document.fonts.status === 'loaded' && !document.querySelector('#intro').hidden")
            await page.evaluate(HOOK_SAY)
            await page.evaluate("h => __dm.card(h)", CARDS["s1"])
            await asyncio.sleep(1.2)
            video = BUILD / "video.mp4"
            cast = Screencast(page, video)
            await cast.start()
            await asyncio.sleep(0.8)
            print("recording…", flush=True)
            try:
                await full_demo(Demo(page))
            finally:
                total = await cast.stop()
                await browser.close()
        print(f"recorded {total:.1f}s; editing…", flush=True)
        finish(video, total, "mirror-demo-rehearsal" if rehearsal else "mirror-demo")
        print(f"done: {OUT}", flush=True)
    finally:
        proc.terminate()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearsal", action="store_true", help="use other free Gemini models, to keep today's quota for the final take")
    asyncio.run(main(ap.parse_args().rehearsal))
