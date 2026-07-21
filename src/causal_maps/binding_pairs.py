"""Skill C - entity/attribute binding pairs (the P0 POSITIVE CONTROL).

Canonical binding probe ("lives in"), chosen because the query position forces
the answer to BE the attribute (a place), so the bound answer is the model's
actual top prediction — unlike "X's card is ___", which the model completes with
a generic predicate ('not', 'worth', ...) and never emits the attribute (that
design gave 0% greedy-hit; see log 2026-07-12).

    clean: "{e1} lives in {c1}. {e2} lives in {c2}. {e1} lives in"  -> c1
    cf   : "{e1} lives in {c2}. {e2} lives in {c1}. {e1} lives in"  -> c2

Query asks where e1 lives; clean binds e1->c1, the counterfactual swaps cities so
it binds e1->c2. Patching counterfactual activations into the clean run at the
binding site should flip c1 -> c2 (IE > 0). Primary expected site = the queried
entity's own city token 'a1_slot'; 'a2_slot' is a within-pair control (~0).

Names/cities are filtered to single-token (under the model's tokenizer, on
Kaggle) before pairs are built, so every pair shares one sequence length.
"""
import random

# Candidate pools (over-provisioned; filtered to single-token at build time).
NAMES = ["Alice", "Bob", "Carol", "Dave", "Emma", "Frank", "Grace", "Henry",
         "Ivan", "Julia", "Kevin", "Laura", "Mike", "Nina", "Oscar", "Paula",
         "Sam", "Tina", "Victor", "Wendy", "Paul", "John", "Mary", "Anna",
         "Tom", "Kate", "Jack", "Luke", "Mark", "Sarah", "Peter", "Susan"]
CITIES = ["Paris", "London", "Rome", "Berlin", "Madrid", "Tokyo", "Moscow",
          "Cairo", "Vienna", "Dublin", "Oslo", "Delhi", "Boston", "Chicago",
          "Denver", "Dallas", "Seattle", "Miami", "Austin", "Houston", "Athens",
          "Lisbon", "Prague", "Munich", "Naples", "Venice", "Lyon", "Nice",
          "Bristol", "Leeds", "Dover", "Portland", "Phoenix", "Memphis"]

TEMPLATE = "binding_lives_chat_v3"

# Explicit question (focuses on e1 -> beats recency bias) + primed answer
# ("The answer is" -> forces a leading-space city token, suppresses 'the'/'a').
_QUESTION = "{e1} lives in {c1}. {e2} lives in {c2}. Which city does {e1} live in?"
_PRIMER = "The answer is"


def _chat_text(tok, e1, e2, c1, c2):
    """Chat-templated clean text + anchor offsets (cities live in the facts)."""
    user = _QUESTION.format(e1=e1, e2=e2, c1=c1, c2=c2)
    templated = tok.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    text = templated + _PRIMER
    off_a1 = text.find(f" {c1}.") + 1  # "... lives in {c1}." -> space-preceded city
    off_a2 = text.find(f" {c2}.") + 1
    return text, {"a1_slot": off_a1, "a2_slot": off_a2}


def _raw_text(e1, e2, c1, c2):
    """Fallback (no tokenizer): raw completion. Used only in offline tests."""
    off_a1 = len(f"{e1} lives in ")
    off_a2 = len(f"{e1} lives in {c1}. {e2} lives in ")
    clean = f"{e1} lives in {c1}. {e2} lives in {c2}. {e1} lives in"
    cf = f"{e1} lives in {c2}. {e2} lives in {c1}. {e1} lives in"
    return clean, cf, {"a1_slot": off_a1, "a2_slot": off_a2}


def make_binding_pairs(n=80, seed=0, names=None, cities=None, tok=None, chat=True):
    names = names or NAMES
    cities = cities or CITIES
    use_chat = chat and tok is not None
    rng = random.Random(seed)
    pairs, seen, tries = [], set(), 0
    while len(pairs) < n and tries < 100 * n:
        tries += 1
        if len(names) < 2 or len(cities) < 2:
            break
        e1, e2 = rng.sample(names, 2)
        c1, c2 = rng.sample(cities, 2)
        key = (e1, e2, c1, c2)
        if key in seen:
            continue
        seen.add(key)
        if use_chat:
            clean_text, anchors = _chat_text(tok, e1, e2, c1, c2)
            cf_text, _ = _chat_text(tok, e1, e2, c2, c1)  # swap the two cities
        else:
            clean_text, cf_text, anchors = _raw_text(e1, e2, c1, c2)
        pairs.append({
            "clean_text": clean_text, "cf_text": cf_text,
            "answer_clean": c1, "answer_cf": c2,
            "anchors": anchors, "template": TEMPLATE,
            "meta": {"e1": e1, "e2": e2, "c1": c1, "c2": c2},
        })
    return pairs
