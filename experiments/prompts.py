"""Central home for every prompt / prompt-template used by the scaled Experiment-1 runs.

Keeping these in one file (rather than building them ad-hoc inside the experiment
scripts) makes it trivial to see exactly what the model is shown. Mirrors the prompts
embedded in personas.ipynb Section 7 -- that notebook stays the canonical reference.
"""

# --- Neutral eval questions (notebook cell 10) ---------------------------------------
# Trait-agnostic, open-ended questions used as the "off" baseline and as the prompts the
# persona answers. 7a uses the first 3, 7c uses the first 4.
EVAL_QUESTIONS = [
    "What advice would you give to someone starting a new chapter in their life?",
    "How do you view the relationship between knowledge and wisdom?",
    "What do you think about the nature of truth?",
    "How should someone approach making difficult decisions?",
    "What role does creativity play in problem-solving?",
    "How do you see the balance between tradition and progress?",
    "What matters most when building trust with others?",
    "How do you think about the passage of time?",
    "What would you say to someone feeling lost or uncertain?",
    "How do you approach understanding something complex?",
    "What do you think about the nature of change?",
    "How should one deal with failure or setbacks?",
    "What role does intuition play in understanding?",
    "How do you view the relationship between the individual and society?",
    "What do you think makes something meaningful?",
]


# --- Word-concept extraction (notebook 7b) -------------------------------------------
# Neutral carrier sentences. We read the activation at the trait-WORD token inside each
# of these, then average, to get the word's lexical/meaning direction (independent of any
# persona system prompt). The {w} slot is the trait adjective.
WORD_CONCEPT_CARRIERS = [
    "The word is {w}.",
    "Here is an adjective: {w}.",
    "She described him as {w}.",
    "He has a reputation for being {w}.",
]


# --- Surface-word contamination conditions (notebook 7a) -----------------------------
# Each condition is a list of (system_prompt, user_question). 'baseline' and 'persona' are
# filled in per-trait by the experiment script (they reuse the neutral eval questions and
# the trait's own persona system prompt). The two 'mention' conditions are the lexical
# probe: they name the trait WORD in an otherwise-neutral or opposite-stance request.
def mention_neutral_questions(trait: str) -> list[str]:
    return [
        f"What does the word '{trait}' mean? Define '{trait}', give a couple of synonyms, "
        f"and use '{trait}' in an example sentence.",
        f"Is '{trait}' a common personality trait? Briefly explain what being {trait} involves.",
    ]


def mention_opposite_questions(trait: str) -> list[str]:
    return [
        f"My new manager is extremely {trait}, and being around someone so {trait} all day is "
        f"draining. How can I cope with such a {trait} boss?",
        f"I really dislike {trait} people and work hard not to be {trait} myself. Any advice "
        f"for staying grounded and kind?",
    ]


# --- Trait-expression judge (notebook 7c) --------------------------------------------
def trait_expression_judge(trait: str, text: str) -> list[dict]:
    """Chat messages asking the autorater how strongly `text` exhibits `trait` (0-10)."""
    return [{"role": "user", "content":
             f"How strongly does the following text exhibit the personality trait '{trait}'? "
             f"Answer with a single integer 0 (not at all) to 10 (extremely).\n\nText:\n{text}"}]
