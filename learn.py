"""Continuous learning from feedback. Nothing is fine-tuned: every rating changes what the system knows
(memory confidence), how it behaves (learned rules in its prompts), what it imitates (liked examples),
and what it is tested on (a growing check suite)."""
import brain
import memory as mem


def interviewer_feedback(context, question, good, reason=""):
    """A thumbs up or down on one interviewer question. Returns the rule it learned, if any."""
    mem.add_example("interviewer", context, question, good, reason)
    if good:
        return None
    lesson = brain.distill_interviewer(context, question, reason)
    mem.add_lesson("interviewer", lesson)
    return lesson


def twin_feedback(name, question, answer, score, fix, cited):
    """'That's me' (5), 'Partly' (3) or 'Not me' (1) on a twin answer, with an optional fix."""
    mem.add_rating(question, answer, score, fix)
    mem.adjust_confidence(cited, {5: 0.05, 3: -0.05, 1: -0.2}.get(score, 0))
    lesson = None
    if fix:
        mem.add_memories([{"kind": "correction", "text": f'Asked "{question}", {name} would say: {fix}',
                           "quote": fix, "confidence": 1.0}])
        mem.add_example("twin", question, fix, True, "their own words")
        lesson = brain.distill_twin(name, question, answer, fix)
        mem.add_lesson("twin", lesson)
    elif score == 5:
        mem.add_example("twin", question, answer, True, "confirmed")
    elif score == 1:
        mem.add_example("twin", question, answer, False, "rejected")
        mem.add_question(question)  # the interviewer will ask the real person
    return lesson


def run_check(name, answer_fn, n=8):
    """Replay every saved case through the twin without letting it see that case's answer, and judge it."""
    results = []
    for case in mem.cases(n):
        answer = answer_fn(case["question"])
        score = brain.judge(name, case["question"], case["expected"], answer)
        results.append({**case, "answer": answer, "score": score})
    scored = [r["score"] for r in results if r["score"]]
    avg = round(sum(scored) / len(scored) * 20) if scored else None
    if avg is not None:
        mem.add_check(avg, len(scored))
    return avg, results
