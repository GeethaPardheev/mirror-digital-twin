"""Everything the twin knows, in one SQLite file: keyword search (FTS5) + vector search (numpy)."""
import os
import re
import sqlite3
import threading
import time

import numpy as np
import ollama

EMBED_MODEL = os.getenv("MIRROR_EMBED", "embeddinggemma")
AREAS = ("life story", "values", "decisions", "opinions", "feelings", "relationships",
         "work and skills", "habits", "tastes", "humour and style", "hopes and fears")
KINDS = ("fact", "opinion", "preference", "value", "rule", "trigger")
QUOTA = {"story": 1, "quote": 3}  # per answer; every other kind shares the "record" quota

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns(id INTEGER PRIMARY KEY, role TEXT, text TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY, kind TEXT, area TEXT, text TEXT, quote TEXT,
  confidence REAL, since TEXT, status TEXT DEFAULT 'active', valid_to TEXT, turn_id INTEGER, emb BLOB, ts REAL);
CREATE VIRTUAL TABLE IF NOT EXISTS mem_fts USING fts5(text, mid UNINDEXED);
CREATE TABLE IF NOT EXISTS people(name TEXT PRIMARY KEY, relation TEXT);
CREATE TABLE IF NOT EXISTS questions(id INTEGER PRIMARY KEY, text TEXT, status TEXT DEFAULT 'open');
CREATE TABLE IF NOT EXISTS ratings(id INTEGER PRIMARY KEY, question TEXT, answer TEXT, score INTEGER, fix TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS lessons(id INTEGER PRIMARY KEY, target TEXT, text TEXT, hits INTEGER DEFAULT 1, emb BLOB, ts REAL);
CREATE TABLE IF NOT EXISTS examples(id INTEGER PRIMARY KEY, target TEXT, context TEXT, text TEXT, good INTEGER,
  reason TEXT, emb BLOB, ts REAL);
CREATE TABLE IF NOT EXISTS checks(id INTEGER PRIMARY KEY, score REAL, n INTEGER, ts REAL);
"""
_db, _lock, _cache = None, threading.Lock(), {}


def use(path):
    """Open (or create) one person's twin database and make it the active one. Each person has their own file."""
    global _db
    with _lock:
        if _db is not None:
            _db.close()
        _db = sqlite3.connect(path, check_same_thread=False)
        _db.row_factory = sqlite3.Row
        _db.executescript(SCHEMA)
        try:  # privacy: public (anyone), personal (shared in general terms), private (never leaves the owner's view)
            _db.execute("ALTER TABLE memories ADD COLUMN privacy TEXT")
        except sqlite3.OperationalError:
            pass
        _db.commit()
    _cache.clear()


use(os.getenv("MIRROR_DB") or ":memory:")  # nobody signed in yet: an empty in-memory twin

COLS = "id, kind, area, text, quote, confidence, since, privacy"
PRIVACY = ("public", "personal", "private")
# Safety net on top of the model's judgement: these always make a memory private.
SENSITIVE = re.compile(r"salary|\blakh|crore|₹|\$\s?\d|money|debt|loan|\bpay\b|hike|appraisal|promotion|passed over|"
                       r"fired|laid off|layoff|betray\w*|hurt|humiliat\w*|ashamed|insecur\w*|invisible|lonely|"
                       r"depress\w*|anxi\w*|therap\w*|diagnos\w*|illness|disease|medic\w*|pregnan\w*|divorce\w*|"
                       r"break ?up|affair|abuse\w*|address|phone|password|@\w+\.\w+|"
                       r"fail\w* (?:an? |my |the |two |\w+ )?(?:exam|class|classes|course|subject|test|semester)|"
                       r"\bfail stuff|scholarship|\bgrades?\b|dementia|cancer|chemo|runway|burn\w* out|burnout", re.I)


def privacy_of(item, suggested=None):
    if SENSITIVE.search(item.get("text", "")):
        return "private"
    return suggested if suggested in PRIVACY else "personal"


def q(sql, args=()):
    with _lock:
        rows = [dict(r) for r in _db.execute(sql, args).fetchall()]
        _db.commit()
    return rows


def insert(sql, args):
    with _lock:
        rid = _db.execute(sql, args).lastrowid
        _db.commit()
    return rid


def embed(texts, query=False):
    prefix = "task: search result | query: " if query else "title: none | text: "
    out = ollama.embed(model=EMBED_MODEL, input=[prefix + t for t in texts], keep_alive="30m").embeddings
    v = np.array(out, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def _matrix():
    if "m" not in _cache:
        rows = q("SELECT id, emb FROM memories WHERE status='active'")
        ids = [r["id"] for r in rows]
        mat = np.array([np.frombuffer(r["emb"], dtype=np.float32) for r in rows]) if rows else None
        _cache["m"] = (mat, ids)
    return _cache["m"]


def add_memories(items, turn_id=None):
    """Save new memories, skipping near-duplicates of what's already known. Returns what was saved."""
    items = [i for i in items if i.get("text", "").strip()]
    if not items:
        return []
    mat, ids = _matrix()
    saved = []
    for item, v in zip(items, embed([i["text"] for i in items])):
        if ids and item["kind"] != "correction" and float((mat @ v).max()) > 0.92:
            continue
        if item["kind"] != "correction" and SENSITIVE.search(item.get("quote", "")) and not SENSITIVE.search(item["text"]):
            item = {**item, "quote": ""}  # a shareable fact keeps its meaning but drops a quote with private details
        mid = insert("INSERT INTO memories(kind, area, text, quote, confidence, since, turn_id, emb, ts, privacy)"
                     " VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (item["kind"], item.get("area", ""), item["text"], item.get("quote", ""),
                      item.get("confidence", 0.8), item.get("since", ""), turn_id, v.tobytes(), time.time(),
                      privacy_of(item, item.get("privacy"))))
        insert("INSERT INTO mem_fts(text, mid) VALUES (?, ?)", (f'{item["text"]} {item.get("quote", "")}', mid))
        mat = v[None, :] if mat is None else np.vstack([mat, v])
        ids = ids + [mid]
        saved.append({**item, "id": mid})
    _cache.clear()
    return saved


def supersede(old_belief, until=""):
    """They changed their mind: close the closest matching memory instead of deleting it."""
    mat, ids = _matrix()
    if not ids:
        return
    sims = mat @ embed([old_belief])[0]
    if sims.max() > 0.75:
        q("UPDATE memories SET status='superseded', valid_to=? WHERE id=?",
          (until or time.strftime("%Y"), ids[int(sims.argmax())]))
        _cache.clear()


def store(ex, turn_id):
    """Persist one Extraction from brain.extract(), normalising whatever the model got slightly wrong."""
    if ex is None:
        return []
    area = ex.area if ex.area in AREAS else "life story"
    for r in ex.records:
        if r.replaces.strip():
            supersede(r.replaces, r.since)
    for p in ex.people:
        if p.name.strip():
            q("INSERT INTO people(name, relation) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET relation=excluded.relation",
              (p.name.strip(), p.relation.strip()))
    # Only what matters becomes memory: importance 3+ of 5.
    items = [dict(kind=r.kind if r.kind in KINDS else "fact", area=r.area if r.area in AREAS else area,
                  text=r.text, quote=r.quote, confidence=min(max(r.confidence, 0.0), 1.0), since=r.since,
                  privacy=r.privacy) for r in ex.records if r.importance >= 3]
    items += [dict(kind="story", area=s.area if s.area in AREAS else area, text=f"{s.title}: {s.text}",
                   quote=s.quote, confidence=0.9, privacy=s.privacy)
              for s in ex.stories if s.text.strip() and s.importance >= 3]
    if len(ex.voice_quote.split()) >= 8:
        items.append(dict(kind="quote", area=area, text=ex.voice_quote, quote="", confidence=1.0, privacy=ex.voice_privacy))
    return add_memories(items, turn_id)


def search(query, n_records=6, shareable=False):
    """Hybrid search: meaning (vectors) + words (BM25), fused with reciprocal rank, capped per kind.
    shareable=True (the twin) never returns private memories."""
    mat, ids = _matrix()
    score = {}
    if ids:
        sims = mat @ embed([query], query=True)[0]
        for rank, i in enumerate(np.argsort(-sims)[:30]):
            score[ids[i]] = 1 / (60 + rank)
    words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
    if words:
        rows = q("SELECT mid FROM mem_fts WHERE mem_fts MATCH ? ORDER BY rank LIMIT 30",
                 (" OR ".join(f'"{w}"' for w in words),))
        for rank, r in enumerate(rows):
            score[r["mid"]] = score.get(r["mid"], 0) + 1 / (60 + rank)
    if not score:
        return []
    order = sorted(score, key=score.get, reverse=True)
    rows = {r["id"]: r for r in by_ids(order)}
    picked, used = [], {}
    for mid in order:
        r = rows.get(mid)
        if not r or r["status"] != "active" or r["kind"] == "correction":  # corrections are always loaded anyway
            continue
        if shareable and r["privacy"] == "private":
            continue
        key = r["kind"] if r["kind"] in QUOTA else "record"
        if used.get(key, 0) < QUOTA.get(key, n_records):
            used[key] = used.get(key, 0) + 1
            picked.append(r)
    return picked


def by_ids(ids):
    ids = list(ids)
    if not ids:
        return []
    return q(f"SELECT {COLS}, status FROM memories WHERE id IN ({','.join('?' * len(ids))})", ids)


def save_turn(role, text):
    return insert("INSERT INTO turns(role, text, ts) VALUES (?,?,?)", (role, text, time.time()))


def all_active():
    return q(f"SELECT {COLS} FROM memories WHERE status='active' ORDER BY id")


def recent(n=8):
    return q(f"SELECT {COLS} FROM memories WHERE status='active' AND kind NOT IN ('quote','correction')"
             " ORDER BY id DESC LIMIT ?", (n,))


def quotes(n=4):
    """Real quotes for style, never private ones."""
    return q(f"SELECT {COLS} FROM memories WHERE status='active' AND kind='quote' AND privacy != 'private'"
             " ORDER BY id DESC LIMIT ?", (n,))


def set_privacy(mid, level):
    if level in PRIVACY:
        q("UPDATE memories SET privacy=? WHERE id=?", (level, int(mid)))


def unclassified():
    return q("SELECT id, kind, text, quote FROM memories WHERE privacy IS NULL")


def corrections():
    return q(f"SELECT {COLS} FROM memories WHERE status='active' AND kind='correction' AND privacy != 'private'"
             " ORDER BY id DESC LIMIT 10")


def count():
    return q("SELECT COUNT(*) n FROM memories WHERE status='active'")[0]["n"]


def coverage():
    rows = q("SELECT area, COUNT(*) n FROM memories WHERE status='active' AND kind NOT IN ('quote','correction')"
             " GROUP BY area")
    got = {r["area"]: r["n"] for r in rows}
    return {a: got.get(a, 0) for a in AREAS}


def gaps(n=3, exclude=None):
    cov = coverage()
    return sorted((a for a in AREAS if a != exclude), key=cov.get)[:n]


def get(k, default=""):
    rows = q("SELECT v FROM kv WHERE k=?", (k,))
    return rows[0]["v"] if rows else default


def put(k, v):
    q("INSERT INTO kv(k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, v))


def profile_card():
    """What the twin reads before every answer. Built only from memories that may be shared."""
    card = get("profile")
    if card:
        return card
    rows = q("SELECT text FROM memories WHERE status='active' AND privacy != 'private' AND kind IN "
             "('value','rule','preference','opinion','fact') ORDER BY confidence DESC, id DESC LIMIT 15")
    return "\n".join("- " + r["text"] for r in rows) or "(nothing yet)"


def dump(limit=6000, shareable=False):
    """Everything active, as plain notes for summaries. shareable=True leaves out private memories."""
    rows = q(f"SELECT {COLS} FROM memories WHERE status='active'"
             + (" AND privacy != 'private'" if shareable else "") + " ORDER BY id")
    people = q("SELECT name, relation FROM people")
    lines = [f'{r["kind"]}: {r["text"]}' + (f' ("{r["quote"]}")' if r["quote"] else "") for r in rows]
    lines += [f'person: {p["name"]} ({p["relation"]})' for p in people]
    return "\n".join(lines)[-limit:]


def add_question(text):
    insert("INSERT INTO questions(text) VALUES (?)", (text,))


def open_questions():
    return q("SELECT id, text FROM questions WHERE status='open' ORDER BY id")


def close_question(qid):
    q("UPDATE questions SET status='asked' WHERE id=?", (qid,))


def add_rating(question, answer, score, fix):
    insert("INSERT INTO ratings(question, answer, score, fix, ts) VALUES (?,?,?,?,?)",
           (question, answer, score, fix, time.time()))


def fidelity():
    return q("SELECT AVG(score) avg, COUNT(*) n FROM ratings")[0]


# ---------- learning from feedback ----------

def add_lesson(target, text):
    """A rule learned from feedback. Re-learning the same rule raises its weight instead of duplicating it."""
    text = text.strip().strip('"').strip()
    if not text:
        return
    v = embed([text])[0]
    for r in q("SELECT id, emb FROM lessons WHERE target=?", (target,)):
        if float(np.frombuffer(r["emb"], dtype=np.float32) @ v) > 0.85:
            q("UPDATE lessons SET hits = hits + 1, ts = ? WHERE id = ?", (time.time(), r["id"]))
            return
    insert("INSERT INTO lessons(target, text, emb, ts) VALUES (?,?,?,?)", (target, text, v.tobytes(), time.time()))


def lessons(target, k=6):
    return q("SELECT id, text, hits FROM lessons WHERE target=? ORDER BY hits DESC, ts DESC LIMIT ?", (target, k))


def add_example(target, context, text, good, reason=""):
    """A liked or disliked interviewer question, or a confirmed or rejected twin answer."""
    insert("INSERT INTO examples(target, context, text, good, reason, emb, ts) VALUES (?,?,?,?,?,?,?)",
           (target, context, text, int(good), reason, embed([context or text])[0].tobytes(), time.time()))


def examples(target, query, good, k=2, exclude=None):
    rows = q("SELECT context, text, reason, emb FROM examples WHERE target=? AND good=?", (target, int(good)))
    rows = [r for r in rows if r["context"] != exclude]
    if not rows:
        return []
    v = embed([query or "start"], query=True)[0]
    rows.sort(key=lambda r: -float(np.frombuffer(r["emb"], dtype=np.float32) @ v))
    return [{"context": r["context"], "text": r["text"], "reason": r["reason"]} for r in rows[:k]]


def cases(n=8):
    """The twin's test suite: every confirmed answer and every correction you gave."""
    return q("SELECT context AS question, text AS expected FROM examples WHERE target='twin' AND good=1"
             " ORDER BY id DESC LIMIT ?", (n,))


def adjust_confidence(ids, delta):
    """Confirmed answers strengthen the memories behind them; wrong ones weaken them until they stop being used."""
    ids = [int(i) for i in ids]
    if not ids or not delta:
        return
    marks = ",".join("?" * len(ids))
    q(f"UPDATE memories SET confidence = MIN(1.0, MAX(0.1, confidence + ?)) WHERE id IN ({marks})", (delta, *ids))
    q(f"UPDATE memories SET status='doubtful' WHERE id IN ({marks}) AND confidence < 0.35 AND kind != 'correction'", ids)
    _cache.clear()


def add_check(score, n):
    insert("INSERT INTO checks(score, n, ts) VALUES (?,?,?)", (score, n, time.time()))


def checks(k=12):
    return q("SELECT score, n, ts FROM checks ORDER BY id DESC LIMIT ?", (k,))[::-1]


def feedback_stats():
    rows = {(r["target"], r["good"]): r["n"] for r in q("SELECT target, good, COUNT(*) n FROM examples GROUP BY target, good")}
    doubtful = q("SELECT COUNT(*) n FROM memories WHERE status='doubtful'")[0]["n"]
    return {"questions_liked": rows.get(("interviewer", 1), 0), "questions_disliked": rows.get(("interviewer", 0), 0),
            "answers_confirmed": rows.get(("twin", 1), 0), "answers_rejected": rows.get(("twin", 0), 0),
            "doubtful": doubtful}
