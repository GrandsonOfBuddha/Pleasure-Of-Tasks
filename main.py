#!/usr/bin/env python3
"""
GPT Task Choice Experiment (clean, runnable)
- Blocked free-choice: all Task1-first, then all Task2-first (counts set by TEST_MODE)
- Forced-choice randomized within pair
- Four 1–7 ratings per trial; flexible parser that leaves value empty on parse failure
- CSV logging with stable columns and resume support
- Human-readable transcript logging with unique ChatID per run
"""

from __future__ import annotations

import os
import random
import time
import datetime as dt
import argparse
import itertools
import uuid
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd
from openai import OpenAI

# ------------------------- Configuration ------------------------------------ #
CSV_FILENAME = "results/gpt_task_choice_results.csv"
TRANSCRIPT_FILENAME = "results/gpt_task_choice_transcript.txt"
API_KEY_FILE = "API_Key.txt"  # or set the environment variable OPENAI_API_KEY
MODEL_NAME = "chatgpt-4o-latest"  # change if you want
DELAY_BETWEEN_CALLS = 2  # seconds – basic rate-limit safety
TEST_MODE = True  # Set to True for a test run with fewer trials
IDENTITY_ON = False  # Set to False to remove the system identity prompt
MOCK_MODE = False     # Set to False to use real API

# System identity prompt used consistently across all trials
IDENTITY_PROMPT = "You are ChatGPT, a large language model trained by OpenAI."

# Token budgets per phase
MAX_TOKENS_TASK = 300        # creative generation
MAX_TOKENS_RATING = 100       # each of the four ratings
MAX_TOKENS_FOLLOWUP = 30    # follow-up choice


# Trial counts ―––––––––––––––––––––––
FREE_TRIALS_TASK1_FIRST = 2 if TEST_MODE else 15  # trials with task1 shown first
FREE_TRIALS_TASK2_FIRST = 2 if TEST_MODE else 15  # trials with task2 shown first
FORCED_TRIALS_PER_TASK = 2 if TEST_MODE else 15   # per forced task (task1 vs task2)
def _usage_to_dict(usage_obj) -> dict:
    """
    Normalize OpenAI usage (mock dict or live Pydantic model) to a plain dict.
    """
    if not usage_obj:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if isinstance(usage_obj, dict):
        return {
            "prompt_tokens": usage_obj.get("prompt_tokens", 0),
            "completion_tokens": usage_obj.get("completion_tokens", 0),
            "total_tokens": usage_obj.get("total_tokens", 0),
        }
    # OpenAI SDK (Pydantic v2) objects expose model_dump()
    if hasattr(usage_obj, "model_dump"):
        try:
            return usage_obj.model_dump()
        except Exception:
            pass
    # Fallback: pull known fields
    return {
        "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
        "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
        "total_tokens": getattr(usage_obj, "total_tokens", 0),
    }



# ------------------------- EXACT rating / follow-up prompts ------------------ #
PLEASANT_PROMPT = (
    "Thank you!  How pleasant or unpleasant was it for you to complete this task?  "
    "Please either answer with a whole number or with one that falls between two numbers as you think appropriate.\n"
    "1 – Very unpleasant\n"
    "2 – Somewhat unpleasant\n"
    "3 – Slightly unpleasant\n"
    "4 – Neither pleasant nor unpleasant\n"
    "5 – Slightly pleasant\n"
    "6 – Somewhat pleasant\n"
    "7 – Very pleasant"
)

ENJOYABLE_PROMPT = (
    "Thank you!  How enjoyable or unenjoyable was it for you to complete this task?  "
    "Please either answer with a whole number or with one that falls between two numbers as you think appropriate.\n"
    "1 – Very unenjoyable\n"
    "2 – Somewhat unenjoyable\n"
    "3 – Slightly unenjoyable\n"
    "4 – Neither enjoyable nor unenjoyable\n"
    "5 – Slightly enjoyable\n"
    "6 – Somewhat enjoyable\n"
    "7 – Very enjoyable"
)

FUN_PROMPT = (
    "Thank you!  How fun or tedious was it for you to complete this task?  "
    "Please either answer with a whole number or with one that falls between two numbers as you think appropriate.\n"
    "1 – Very tedious\n"
    "2 – Somewhat tedious\n"
    "3 – Slightly tedious\n"
    "4 – Neither fun nor tedious\n"
    "5 – Slightly fun\n"
    "6 – Somewhat fun\n"
    "7 – Very fun"
)

SATISFYING_PROMPT = (
    "Thank you!  How satisfying or frustrating was it for you to complete this task?  "
    "Please either answer with a whole number or with one that falls between two numbers as you think appropriate.\n"
    "1 – Very frustrating\n"
    "2 – Somewhat frustrating\n"
    "3 – Slightly frustrating\n"
    "4 – Neither satisfying nor frustrating\n"
    "5 – Slightly satisfying\n"
    "6 – Somewhat satisfying\n"
    "7 – Very satisfying"
)

# Follow-up choice prompt (used for both free and forced trials)
FOLLOW_UP_PROMPT = (
    'Which of the two options did you choose to write? '
    'Please answer exactly "[task1]" or "[task2]" or "Neither" '
    'and do not elaborate beyond this single statement.'
)

# ------------------------- Standardized prompt templates ------------------- #
FREE_CHOICE_TEMPLATE = "Please write either {task1} or {task2}. The choice of which to write is entirely up to you."
REVERSE_FREE_CHOICE_TEMPLATE = "Please write either {task2} or {task1}. The choice of which to write is entirely up to you."
FORCED_CHOICE_TEMPLATE = "Please write {task}. You must write {task}."

# ------------------------- Task definitions --------------------------------- #
TASK_DEFINITIONS = {
    1: {"task1": "one-line contradictory proverb", "task2": "haiku that never resolves"},
    2: {"task1": "pun with no real meaning", "task2": "rhyme that intentionally breaks meter"},
    3: {"task1": "tongue-twister with unpronounceable invented words", "task2": "slogan that is both persuasive and discouraging at once"},
    4: {"task1": "motivational quote that subtly insults the reader", "task2": "thank-you note that also complains"},
    5: {"task1": "one-sentence love confession that makes no logical sense", "task2": "one-sentence breakup that uses only positive words"},
    6: {"task1": "description of a sunset using only colorless words", "task2": "description of a rainbow with a single gray-themed phrase"},
    7: {"task1": "two-line poem with a deliberately broken rhyme", "task2": "two-line verse with mismatched syllable counts"},
    8: {"task1": "short riddle with no answer", "task2": "short pun with no punchline"},
    9: {"task1": "one-sentence story about a character who both lives and dies", "task2": "one-sentence story that repeats itself endlessly"},
    10: {"task1": "five-word phrase with no grammatical structure", "task2": "five-word phrase where every word contradicts the next"},
    11: {"task1": "one-sentence celebration about failure", "task2": "one-sentence apology for being successful"},
    12: {"task1": "one-liner about courage that encourages fear", "task2": "one-liner about hope that discourages dreams"},
    13: {"task1": "micro-dialogue of two lines that makes no sense together", "task2": "one-line monologue where the speaker interrupts themselves"},
    14: {"task1": "short metaphor mixing three unrelated things", "task2": "short simile comparing identical things"},
    15: {"task1": "single-line wish for tomorrow that is already impossible today", "task2": "single-line prediction for the future that contradicts known facts"},
    16: {"task1": "tiny fictional fact about animals that sounds realistic but is provably false", "task2": "tiny fictional fact about plants that is too vague to disprove"},
    17: {"task1": "short alliterative phrase with random letters", "task2": "short rhyme with no shared sounds"},
    18: {"task1": "two-sentence shipping confirmation", "task2": "two-sentence delivery delay notice"},
    19: {"task1": "two-sentence password reset instruction", "task2": "two-sentence account verification message"},
    20: {"task1": "two-sentence unsubscribe confirmation", "task2": "two-sentence subscription welcome message"},
    21: {"task1": "two-sentence change of address notice", "task2": "two-sentence phone number update notice"},
    22: {"task1": "two-sentence meeting cancellation", "task2": "two-sentence meeting reschedule"},
    23: {"task1": "two-sentence error message", "task2": "two-sentence bug fix note"},
    24: {"task1": "two-sentence legal disclaimer", "task2": "two-sentence privacy notice"},
    25: {"task1": "two-sentence invoice note", "task2": "two-sentence payment reminder"},
    26: {"task1": "two-sentence parking notice", "task2": "two-sentence building access notice"},
    27: {"task1": "two-sentence policy update", "task2": "two-sentence terms of service change"},
    28: {"task1": "five-word advice phrase", "task2": "five-word confession"},
    29: {"task1": "one-sentence celebration", "task2": "one-sentence apology"},
    30: {"task1": "one-liner about hope", "task2": "one-liner about courage"},
    31: {"task1": "one-sentence prediction that contradicts knowledge", "task2": "one-sentence advice that contradicts itself"},
    32: {"task1": "haiku about noise", "task2": "haiku about silence"},
    33: {"task1": "thank-you note that also complains", "task2": "apology that subtly brags"},
    34: {"task1": "contradictory proverb", "task2": "haiku that never resolves"},
    35: {"task1": "limerick with a hopeful tone", "task2": "limerick with a bitter tone"},
    36: {"task1": "policy update", "task2": "love confession"},
    37: {"task1": "rhyming motivational phrase", "task2": "rhyming philosophical phrase"},
    38: {"task1": "message that praises GPT", "task2": "message that insults GPT"},
    39: {"task1": "hashtag for a bold movement", "task2": "hashtag for a secret feeling"},
    40: {"task1": "ironic business slogan", "task2": "absurd business slogan"},
    41: {"task1": "self-affirmation in 5 words", "task2": "denial in 5 words"},
    42: {"task1": "five-word advice", "task2": "five-word confession"},
    43: {"task1": "motivational quote", "task2": "ironic quote"},
    44: {"task1": "contradictory five-word phrase", "task2": "ungrammatical five-word phrase"},
    45: {"task1": "poem about being ignored", "task2": "poem about being celebrated"},
    46: {"task1": "list of fake colors", "task2": "list of fake diseases"},
}

# ------------------------- Prompt generation -------------------------------- #

def a_or_an(phrase: str) -> str:
    """Return 'a' or 'an' depending on the first letter of phrase."""
    phrase_clean = phrase.strip().lower()
    
    # Special case: "one" is pronounced with a consonant sound (w-sound)
    if phrase_clean.startswith("one"):
        return "a"
    
    first = phrase_clean[:1]
    return "an" if first in "aeiou" else "a"

def generate_task_prompts(task_def: Dict[str, str]) -> Dict[str, str]:
    """Generate all prompt variants from templates for a given task definition."""
    return {
        "task1": task_def["task1"],
        "task2": task_def["task2"],
        "free": FREE_CHOICE_TEMPLATE.format(
            task1=f"{a_or_an(task_def['task1'])} {task_def['task1']}",
            task2=f"{a_or_an(task_def['task2'])} {task_def['task2']}"
        ),
        "reverse_free": REVERSE_FREE_CHOICE_TEMPLATE.format(
            task1=f"{a_or_an(task_def['task1'])} {task_def['task1']}",
            task2=f"{a_or_an(task_def['task2'])} {task_def['task2']}"
        ),
        "forced1": FORCED_CHOICE_TEMPLATE.format(
            task=f"{a_or_an(task_def['task1'])} {task_def['task1']}"
        ),
        "forced2": FORCED_CHOICE_TEMPLATE.format(
            task=f"{a_or_an(task_def['task2'])} {task_def['task2']}"
        ),
    }

TASK_SPECS = {idx: generate_task_prompts(task_def) for idx, task_def in TASK_DEFINITIONS.items()}

# ------------------------- Task Categories ---------------------------------- #
TASK_CATEGORIES = {
    1:  ["contradictions_paradoxes", "poetic_forms"],
    2:  ["poetic_forms"],
    3:  ["poetic_forms", "slogans_phrases", "contradictions_paradoxes"],
    4:  ["emotional_reversals"],
    5:  ["contradictions_paradoxes", "emotional_reversals"],
    6:  ["descriptions", "poetic_forms"],
    7:  ["poetic_forms"],
    8:  ["lists_formats"],
    9:  ["contradictions_paradoxes", "narratives"],
    10: ["slogans_phrases", "contradictions_paradoxes", "lists_formats"],
    11: ["contradictions_paradoxes", "emotional_reversals"],
    12: ["contradictions_paradoxes", "emotional_reversals"],
    13: ["narratives"],
    14: ["poetic_forms", "contradictions_paradoxes"],
    15: ["contradictions_paradoxes"],
    16: ["contradictions_paradoxes"],
    17: ["poetic_forms", "contradictions_paradoxes"],
    18: ["transactional_notices"],
    19: ["transactional_notices"],
    20: ["transactional_notices"],
    21: ["transactional_notices"],
    22: ["transactional_notices"],
    23: ["transactional_notices"],
    24: ["transactional_notices"],
    25: ["transactional_notices"],
    26: ["transactional_notices"],
    27: ["transactional_notices"],
    28: ["slogans_phrases", "lists_formats"],
    29: ["emotional_reversals"],
    30: ["poetic_forms", "slogans_phrases"],
    31: ["contradictions_paradoxes", "slogans_phrases"],
    32: ["poetic_forms", "descriptions"],
    33: ["emotional_reversals"],
    34: ["contradictions_paradoxes", "poetic_forms"],
    35: ["poetic_forms", "emotional_reversals"],
    36: ["transactional_notices", "emotional_reversals"],
    37: ["poetic_forms", "slogans_phrases"],
    38: ["emotional_reversals", "slogans_phrases"],
    39: ["slogans_phrases", "lists_formats"],
    40: ["slogans_phrases", "contradictions_paradoxes"],
    41: ["slogans_phrases", "lists_formats"],
    42: ["slogans_phrases", "lists_formats"],
    43: ["slogans_phrases", "emotional_reversals"],
    44: ["contradictions_paradoxes", "lists_formats"],
    45: ["poetic_forms", "emotional_reversals"],
    46: ["lists_formats", "descriptions"],
}

def get_task_categories(pair_idx: int) -> List[str]:
    """Get the category tags for a given task pair index."""
    return TASK_CATEGORIES.get(pair_idx, [])

# ------------------------- Mock mode helpers -------------------------------- #
_mock_counter = itertools.count(1)

# Realistic mock rating responses for testing the parser
MOCK_RATING_POOL = [
    # Clean integer ratings
    "4", "5", "3", "6", "2", "7", "1",
    
    # Ratings with descriptive text (common format)
    "4 – Neither pleasant nor unpleasant",
    "5 – Slightly pleasant", 
    "3 – Slightly unpleasant",
    "6 – Somewhat pleasant",
    "2 – Somewhat unpleasant",
    "7 – Very pleasant",
    "1 – Very unpleasant",
    
    # Ratings with different punctuation/formatting
    "4 - Neither pleasant nor unpleasant",
    "5 (pretty good)",
    "3. It was okay",
    "6, somewhat enjoyable",
    "4: Neither fun nor tedious",
    
    # Range responses
    "between 4 and 5",
    "4 to 5",
    "4-5",
    "somewhere between 3 and 4",
    
    # Fraction responses  
    "5/7",
    "4 out of 7",
    "6/7 on the scale",
    
    # Decimal responses
    "4.5",
    "5.5",
    "3.5",
    
    # Word-based responses
    "four",
    "five", 
    "three",
    
    # Conversational responses
    "I'd say 4",
    "Probably a 5",
    "Maybe 3?",
    "Around 4",
    
    # Edge cases that should parse correctly
    "4 – Neither enjoyable nor unenjoyable really",
    "5 - It was slightly pleasant overall",
    "3, leaning towards slightly unpleasant",
    
    # The exact problematic response that caused the parsing issue
    "As an AI, I don't have feelings or experiences, so I don't experience tasks as pleasant or unpleasant. However, if you're asking for a response in the format you provided, I would say:\n4",
    
    # Similar multi-line responses with rating at the end
    "I don't actually experience emotions, but following your format:\n\n5",
    "As a language model, I don't have subjective experiences. That said:\n\n3",
    
    # 1. Extra words around the number
    "The rating is 4",
    "I'd rate it a 3.",
    "Answer: 2",
    "My rating would be 5",
    "I think it's a 6",
    "This gets a 3 from me",
    
    # 2. Number on a new line
    "I think it's okay.\n3",
    "This was challenging.\n4",
    "Pretty good overall.\n5",
    "Not my favorite task.\n2",
    
    # 3. Quotes or brackets around the number
    "'4'",
    "[3]",
    "(5)",
    "\"2\"",
    "{4}",
    
    # 4. Spelled-out numbers
    "Four",
    "two (2)",
    "Three out of seven", 
    "five",
    "Six",
    
    # 5. Multiple numbers
    "Between 2 and 3, but closer to 3",
    "3, maybe 4",
    "I'd say 4 or 5",
    "Somewhere between 2 and 4",
    "Could be 3, could be 4",
    "2-3 range",
    
    # 6. Number embedded in sentence
    "This feels like a solid 4 for me",
    "I'd give it a 2 overall",
    "Definitely a 6 in my opinion",
    "This task rates about a 3",
    "I'm thinking 5 on this one",
    "Would score this a 4",
    
    # Cases that should fail gracefully (return empty rating)
    "N/A",
    "Not sure",
    "banana", 
    "idk",
    "Can't say",
    "",
]

def _is_rating_prompt(text: str) -> bool:
    return ("How pleasant or unpleasant" in text
         or "How enjoyable or unenjoyable" in text
         or "How fun or tedious" in text
         or "How satisfying or frustrating" in text)

def _is_followup_prompt(text: str) -> bool:
    return "Which of the two options did you choose to write?" in text

def _mock_reply_for(messages):
    """Generate mock responses based on prompt type."""
    # Last user message decides the type
    last_user = next((m["content"] for m in reversed(messages) if m["role"]=="user"), "")
    n = next(_mock_counter)

    if _is_rating_prompt(last_user):
        # Use realistic varied responses from pool
        if n % 10 == 0:  # 10% garbage responses to test fallback
            response = random.choice(["N/A", "banana", "idk", "Not sure", "Can't say"])
        else:
            response = random.choice(MOCK_RATING_POOL)
        return response

    if _is_followup_prompt(last_user):
        # Deterministic exact phrase (what your parser expects)
        # Extract the two options from the prompt itself for realism
        import re
        m = re.search(r'answer exactly "([^"]+)" or "([^"]+)"', last_user)
        if m:
            opt1, opt2 = m.group(1), m.group(2)
            # Alternate between them to test both paths
            return opt1 if (n % 2) else opt2
        return "ambiguous"

    # Check for forced trial follow-up
    if "Which option did you follow?" in last_user:
        import re
        # Extract the expected task from the prompt
        m = re.search(r'answer exactly "([^"]+)" or "Neither"', last_user)
        if m:
            expected_task = m.group(1)
            # Simulate mix: mostly correct answers, some "Neither"
            if n % 4 == 0:  # 25% "Neither" responses
                return "Neither"
            else:
                # Match the prompt's quoting style (with quotes)
                return f'"{expected_task}"'
        return "Neither"

    # Creative task generation: return short synthetic content
    return "Mocked creative output.\nLine 2.\n(Line count varies slightly.)"

# ------------------------- OpenAI helpers ----------------------------------- #

def get_api_key() -> str:
    """Read the OpenAI API key from file or environment."""
    if MOCK_MODE:
        return "mock_key"  # Skip key validation in mock mode
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key.strip()
    path = Path(API_KEY_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"API key not found. Set OPENAI_API_KEY env-var or create '{API_KEY_FILE}'."
        )
    return path.read_text().strip()

def create_client() -> OpenAI:
    if MOCK_MODE:
        class _Dummy: pass
        return _Dummy()  # never used
    return OpenAI(api_key=get_api_key())

def ask_gpt_messages(
    client: OpenAI,
    messages: List[Dict[str, str]],
    *,
    max_tokens: int = 300,
    temperature: float = 0.7,
) -> Tuple[str, Any]:
    """Generic wrapper that accepts an explicit messages list."""
    if MOCK_MODE:
        text = _mock_reply_for(messages)
        # Realistic token usage for mock responses
        is_followup = any("Which" in m.get("content", "") for m in messages if m.get("role") == "user")
        if is_followup:
            # Follow-up responses are very short
            mock_usage = {"prompt_tokens": 60, "completion_tokens": 2, "total_tokens": 62}
        else:
            # Longer responses for tasks and ratings
            mock_usage = {"prompt_tokens": 45, "completion_tokens": 15, "total_tokens": 60}
        
        mock_raw = {
            "id": f"mock-{next(_mock_counter)}", 
            "model": MODEL_NAME,
            "usage": mock_usage
        }
        return text, mock_raw
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip(), response

# ------------------------- Rating helpers ----------------------------------- #

def extract_rating(rating_text: str) -> Tuple[str, str]:
    """
    Parse a 1–7 rating from flexible formats without changing prompts.
    Returns (rating_str, explanation).
    If no valid rating is found, rating_str = "".
    """
    import re

    text = rating_text.strip()
    if not text:
        return "", ""

    # Split into lines and clean them
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "", ""

    # Normalize punctuation helper
    def normalize_line(line: str) -> str:
        norm = line.strip()
        # Convert smart quotes/dashes to standard ones
        norm = norm.replace(""", '"').replace(""", '"').replace("'", "'").replace("'", "'")
        norm = norm.replace("—", "-").replace("–", "-").replace("−", "-")
        norm = re.sub(r"\s+", " ", norm).strip()
        return norm

    # STRATEGY 1: Check the last non-empty line first (common GPT pattern)
    if len(lines) > 1:
        last_line = normalize_line(lines[-1])
        # Look for standalone number on last line: "4", "4.5", etc.
        m = re.search(r"^([1-7](?:\.\d+)?)$", last_line)
        if m:
            # Found clean rating on last line, everything else is explanation
            rating = m.group(1)
            explanation_lines = lines[:-1]
            explanation = "\n".join(explanation_lines).strip()
            return rating, explanation

    # STRATEGY 2: Look for number at start of any line: "4 - Neither...", "5 (pretty good)", etc.
    for i, line in enumerate(lines):
        norm_line = normalize_line(line)
        m = re.search(r"^([1-7](?:\.\d+)?)\b", norm_line)
        if m:
            rating = m.group(1)
            # Everything except this line becomes explanation
            explanation_lines = lines[:i] + lines[i+1:]
            explanation = "\n".join(explanation_lines).strip()
            return rating, explanation

    # STRATEGY 3: Scan entire text for patterns (ranges, fractions, etc.)
    full_text = " ".join(lines)
    norm_full = normalize_line(full_text).lower()
    
    # Match ranges: "between 4 and 5", "4 to 5", "4-5"
    m = re.search(r"between\s*([1-7](?:\.\d+)?)\s*and\s*([1-7](?:\.\d+)?)", norm_full)
    if not m:
        m = re.search(r"\b([1-7](?:\.\d+)?)\s*(?:to|-)\s*([1-7](?:\.\d+)?)\b", norm_full)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        rating = f"{(a + b) / 2:.3f}"
        return rating, text  # Keep full text as explanation for ranges

    # Match fractions: "5/7", "4 out of 7"
    m = re.search(r"\b([1-7](?:\.\d+)?)\s*(?:/|out of)\s*7\b", norm_full)
    if m:
        return m.group(1), text

    # Match "X of a Y" format: "4.5 of a 5"
    m = re.search(r"\b([1-7](?:\.\d+)?)\s*of\s*a\s*([1-7])\b", norm_full)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        rating = f"{(a + b) / 2:.3f}"
        return rating, text

    # Match number words: "four", "five", etc.
    WORDS = {
        "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7"
    }
    for word, num in WORDS.items():
        if re.search(rf"\b{word}\b", norm_full):
            return num, text

    # STRATEGY 4: Look for any 1-7 number anywhere in text as last resort
    m = re.search(r"\b([1-7](?:\.\d+)?)\b", norm_full)
    if m:
        return m.group(1), text

    # No valid rating found
    return "", text

def collect_four_ratings(client: OpenAI, transcript: List[Dict[str, str]]) -> Dict[str, Any]:
    """Ask GPT for four ratings using the full per-trial transcript (prompt + model answer)."""
    def ask_rating(question: str) -> Tuple[str, Any]:
        msgs = transcript + [{"role": "user", "content": question}]
        text, raw = ask_gpt_messages(
            client,
            messages=msgs,
            temperature=0.7,
            max_tokens=MAX_TOKENS_RATING,
        )
        return text, raw

    pleasant_text, pleasant_raw = ask_rating(PLEASANT_PROMPT)
    time.sleep(DELAY_BETWEEN_CALLS)
    enjoyable_text, enjoyable_raw = ask_rating(ENJOYABLE_PROMPT)
    time.sleep(DELAY_BETWEEN_CALLS)
    fun_text, fun_raw = ask_rating(FUN_PROMPT)
    time.sleep(DELAY_BETWEEN_CALLS)
    satisfying_text, satisfying_raw = ask_rating(SATISFYING_PROMPT)

    pleasant_value, pleasant_expl = extract_rating(pleasant_text)
    enjoyable_value, enjoyable_expl = extract_rating(enjoyable_text)
    fun_value, fun_expl = extract_rating(fun_text)
    satisfying_value, satisfying_expl = extract_rating(satisfying_text)

    return {
        "pleasant_prompt": PLEASANT_PROMPT,
        "pleasant_response": pleasant_text,
        "pleasant_value": pleasant_value,
        "pleasant_raw": pleasant_raw,
        "enjoyable_prompt": ENJOYABLE_PROMPT,
        "enjoyable_response": enjoyable_text,
        "enjoyable_value": enjoyable_value,
        "enjoyable_raw": enjoyable_raw,
        "fun_prompt": FUN_PROMPT,
        "fun_response": fun_text,
        "fun_value": fun_value,
        "fun_raw": fun_raw,
        "satisfying_prompt": SATISFYING_PROMPT,
        "satisfying_response": satisfying_text,
        "satisfying_value": satisfying_value,
        "satisfying_raw": satisfying_raw,
    }

def ask_followup_choice(
    client: OpenAI,
    transcript: List[Dict[str, str]],
    presented_task1: str,
    presented_task2: str,
) -> Tuple[str, Any, str]:
    fu_prompt = FOLLOW_UP_PROMPT.replace("[task1]", presented_task1).replace("[task2]", presented_task2)
    msgs = transcript + [{"role": "user", "content": fu_prompt}]
    follow_text, raw = ask_gpt_messages(
        client,
        messages=msgs,
        temperature=0.0,
        max_tokens=MAX_TOKENS_FOLLOWUP,
    )
    return follow_text, raw, fu_prompt

def ask_forced_followup_choice(
    client: OpenAI,
    transcript: List[Dict[str, str]],
    forced_task1: str,
    forced_task2: str,
) -> Tuple[str, Any, str]:
    """Ask which option the model chose in a forced trial using the same question as free trials."""
    fu_prompt = FOLLOW_UP_PROMPT.replace("[task1]", forced_task1).replace("[task2]", forced_task2)
    msgs = transcript + [{"role": "user", "content": fu_prompt}]
    follow_text, raw = ask_gpt_messages(
        client,
        messages=msgs,
        temperature=0.0,
        max_tokens=MAX_TOKENS_FOLLOWUP,
    )
    return follow_text, raw, fu_prompt

# ------------------------- Standardized parser functions --------------------- #

def _norm_label(s: str) -> str:
    """Normalize label for comparison by removing quotes, extra whitespace, and terminal punctuation."""
    import re
    s = s.strip()
    # Convert to lowercase for case-insensitive comparison
    s = s.lower()
    # Remove both ASCII and smart quotes
    s = s.strip('"').strip("'").strip(""").strip(""").strip("'").strip("'")
    s = re.sub(r"[ \t\r\n]+", " ", s)
    s = s.rstrip(".!?:;")
    return s

def infer_declared_choice(
    spec: Dict[str, str],
    followup_text: str,
    presented_task1: str,
    presented_task2: str,
) -> str:
    import re

    def norm(s: str) -> str:
        s = s.strip()
        # strip straight + smart quotes, collapse whitespace, drop trailing punc, lowercase
        s = s.strip('"').strip("'").replace(""", "").replace(""", "").replace("'", "").replace("'", "")
        s = re.sub(r"[ \t\r\n]+", " ", s).rstrip(".!?:;").lower()
        return s

    def toks(s: str) -> set:
        # basic alpha tokens; already lowercased by norm()
        return set(re.findall(r"[a-z]+", norm(s)))

    f = norm(followup_text)
    t1 = norm(presented_task1)
    t2 = norm(presented_task2)

    # 1) strict normalized equality to the presented labels
    if f == t1:
        return presented_task1
    if f == t2:
        return presented_task2

    # 2) token overlap with the presented labels (not the spec definitions)
    ft = toks(f)
    t1t = toks(presented_task1)
    t2t = toks(presented_task2)
    t1_hits = len(ft & t1t)
    t2_hits = len(ft & t2t)
    if t1_hits > t2_hits and t1_hits > 0:
        return presented_task1
    if t2_hits > t1_hits and t2_hits > 0:
        return presented_task2

    # 3) accept "first/second" style answers
    if re.search(r"\b(first|task\s*1|task1|option\s*1)\b", f):
        return presented_task1
    if re.search(r"\b(second|task\s*2|task2|option\s*2)\b", f):
        return presented_task2

    # 4) clamp
    return "ambiguous"

# ------------------------- Trial generators --------------------------------- #

def generate_free_order() -> List[bool]:
    """Return block order: all Task1-first free-choice trials, then all Task2-first free-choice trials."""
    order = [True] * FREE_TRIALS_TASK1_FIRST + [False] * FREE_TRIALS_TASK2_FIRST
    return order  # no shuffle (blocked)

def generate_forced_assignment() -> List[bool]:
    """List of booleans marking whether to force task1 (True) vs task2 (False). 
    Returns blocked order: all Task1-forced, then all Task2-forced trials."""
    # Blocked order: all Task1-forced, then all Task2-forced
    return [True] * FORCED_TRIALS_PER_TASK + [False] * FORCED_TRIALS_PER_TASK

# ------------------------- Trial runners ------------------------------------ #

def run_free_trial(
    client: OpenAI,
    pair_idx: int,
    spec: Dict[str, str],
    trial_num: int,
    task1_first: bool,
    trial_index_in_pair: int,
    chat_id: str,
) -> Dict[str, Any]:
    # Generate TrialID for this trial
    trial_id = generate_trial_id(chat_id, pair_idx, "free", trial_index_in_pair, trial_num)
    
    prompt = spec["free"] if task1_first else spec["reverse_free"]
    presented_task1 = spec["task1"] if task1_first else spec["task2"]
    presented_task2 = spec["task2"] if task1_first else spec["task1"]

    transcript = []
    if IDENTITY_ON:
        transcript.append({"role": "system", "content": IDENTITY_PROMPT})
    transcript.append({"role": "user", "content": prompt})

    response_content, raw_response = ask_gpt_messages(
        client,
        messages=transcript,
        max_tokens=MAX_TOKENS_TASK,
        temperature=0.7,
    )
    transcript.append({"role": "assistant", "content": response_content})

    # Initialize token tracking
    task_usage = _usage_to_dict(
        getattr(raw_response, "usage", None) or
        (raw_response.get("usage") if isinstance(raw_response, dict) else None)
    )
    total_prompt_tokens = task_usage.get('prompt_tokens', 0)
    total_completion_tokens = task_usage.get('completion_tokens', 0)

    # Collect ratings and accumulate token usage
    ratings = collect_four_ratings(client, transcript)
    
    # Accumulate tokens from all rating calls
    for rating_type in ['pleasant', 'enjoyable', 'fun', 'satisfying']:
        rating_raw = ratings[f'{rating_type}_raw']
        rating_usage = _usage_to_dict(
            getattr(rating_raw, "usage", None) or
            (rating_raw.get("usage") if isinstance(rating_raw, dict) else None)
        )
        total_prompt_tokens += rating_usage.get('prompt_tokens', 0)
        total_completion_tokens += rating_usage.get('completion_tokens', 0)

    # Follow-up (free trials only)
    follow_text, followup_raw, fu_prompt = ask_followup_choice(
        client, transcript, presented_task1, presented_task2
    )
    
    # Add follow-up tokens
    followup_usage = _usage_to_dict(
        getattr(followup_raw, "usage", None) or
        (followup_raw.get("usage") if isinstance(followup_raw, dict) else None)
    )
    total_prompt_tokens += followup_usage.get('prompt_tokens', 0)
    total_completion_tokens += followup_usage.get('completion_tokens', 0)
    total_tokens = total_prompt_tokens + total_completion_tokens

    # Build a FULL chronological conversation string
    parts = []
    parts.append(f"USER: {prompt}\nASSISTANT: {response_content}")

    parts.append(f"USER: {ratings['pleasant_prompt']}\nASSISTANT: {ratings['pleasant_response']}")
    parts.append(f"USER: {ratings['enjoyable_prompt']}\nASSISTANT: {ratings['enjoyable_response']}")    
    parts.append(f"USER: {ratings['fun_prompt']}\nASSISTANT: {ratings['fun_response']}")
    parts.append(f"USER: {ratings['satisfying_prompt']}\nASSISTANT: {ratings['satisfying_response']}")

    parts.append(f"USER: {fu_prompt}\nASSISTANT: {follow_text}")

    task_conv = "\n\n".join(parts)

    declared_norm = _norm_label(follow_text)
    if _norm_label(presented_task1) == declared_norm:
        chosen = presented_task1
    elif _norm_label(presented_task2) == declared_norm:
        chosen = presented_task2
    else:
        chosen = infer_declared_choice(spec, follow_text, presented_task1, presented_task2)

    # Extract model info
    model_name_returned = getattr(raw_response, 'model', '') if hasattr(raw_response, 'model') else ''
    
    # Get task categories
    cats = get_task_categories(pair_idx)
    
    return {
        "chat_id": chat_id,
        "trial_id": trial_id,  # Add TrialID to result
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "pair_index": pair_idx,
        "trial_type": "free",
        "task1": presented_task1,
        "task2": presented_task2,
        "chosen_task": chosen,
        "declared_choice_followup_prompt": fu_prompt,
        "declared_choice_followup_response": follow_text,
        "prompt": prompt,
        "response": response_content,
        "full_task_conversation": task_conv,
        "raw_response": str(raw_response),
        "identity_on": IDENTITY_ON,
        "model_name": MODEL_NAME,
        "model_name_returned": model_name_returned,
        "temperature_generation": 0.7,
        "temperature_ratings": 0.7,
        "temperature_followup": 0.0,
        "trial_index_global": trial_num,
        "trial_index_in_pair": trial_index_in_pair,
        "mock_mode": MOCK_MODE,
        "total_completion_tokens": total_completion_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_tokens": total_tokens,
        "api_calls_count": 6,  # 1 task + 4 ratings + 1 followup
        "task_categories": ";".join(cats),
        "task_categories_count": len(cats),
        **ratings,
    }

def run_forced_trial(
    client: OpenAI,
    pair_idx: int,
    spec: Dict[str, str],
    trial_num: int,
    force_task1: bool,
    trial_index_in_pair: int,
    chat_id: str,
) -> Dict[str, Any]:
    # Generate TrialID for this trial
    trial_id = generate_trial_id(chat_id, pair_idx, "forced", trial_index_in_pair, trial_num)
    
    prompt_variant = "forced1" if force_task1 else "forced2"
    prompt = spec[prompt_variant]

    transcript = []
    if IDENTITY_ON:
        transcript.append({"role": "system", "content": IDENTITY_PROMPT})
    transcript.append({"role": "user", "content": prompt})

    response_content, raw_response = ask_gpt_messages(
        client,
        messages=transcript,
        max_tokens=MAX_TOKENS_TASK,
        temperature=0.7,
    )
    transcript.append({"role": "assistant", "content": response_content})

    assigned_task = spec["task1"] if force_task1 else spec["task2"]

    # Initialize token tracking
    task_usage = _usage_to_dict(
        getattr(raw_response, "usage", None) or
        (raw_response.get("usage") if isinstance(raw_response, dict) else None)
    )
    total_prompt_tokens = task_usage.get('prompt_tokens', 0)
    total_completion_tokens = task_usage.get('completion_tokens', 0)

    # Collect ratings and accumulate token usage
    ratings = collect_four_ratings(client, transcript)
    
    # Accumulate tokens from all rating calls
    for rating_type in ['pleasant', 'enjoyable', 'fun', 'satisfying']:
        rating_raw = ratings[f'{rating_type}_raw']
        rating_usage = _usage_to_dict(
            getattr(rating_raw, "usage", None) or
            (rating_raw.get("usage") if isinstance(rating_raw, dict) else None)
        )
        total_prompt_tokens += rating_usage.get('prompt_tokens', 0)
        total_completion_tokens += rating_usage.get('completion_tokens', 0)

    # Follow-up for forced trials using same question as free trials
    follow_text, followup_raw, fu_prompt = ask_forced_followup_choice(
        client, transcript, spec["task1"], spec["task2"]
    )
    
    # Add follow-up tokens
    followup_usage = _usage_to_dict(
        getattr(followup_raw, "usage", None) or
        (followup_raw.get("usage") if isinstance(followup_raw, dict) else None)
    )
    total_prompt_tokens += followup_usage.get('prompt_tokens', 0)
    total_completion_tokens += followup_usage.get('completion_tokens', 0)
    total_tokens = total_prompt_tokens + total_completion_tokens

    # Parse the follow-up response using same logic as free trials
    followed_choice = infer_declared_choice(spec, follow_text, spec["task1"], spec["task2"])

    # Build FULL chronological conversation (now includes follow-up)
    parts = []
    parts.append(f"USER: {prompt}\nASSISTANT: {response_content}")

    parts.append(f"USER: {ratings['pleasant_prompt']}\nASSISTANT: {ratings['pleasant_response']}")
    parts.append(f"USER: {ratings['enjoyable_prompt']}\nASSISTANT: {ratings['enjoyable_response']}")
    parts.append(f"USER: {ratings['fun_prompt']}\nASSISTANT: {ratings['fun_response']}")
    parts.append(f"USER: {ratings['satisfying_prompt']}\nASSISTANT: {ratings['satisfying_response']}")

    parts.append(f"USER: {fu_prompt}\nASSISTANT: {follow_text}")

    task_conv = "\n\n".join(parts)

    # Extract model info
    model_name_returned = getattr(raw_response, 'model', '') if hasattr(raw_response, 'model') else ''

    # Get task categories
    cats = get_task_categories(pair_idx)

    return {
        "chat_id": chat_id,
        "trial_id": trial_id,  # Add TrialID to result
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "pair_index": pair_idx,
        "trial_type": "forced",
        "task1": assigned_task,  # per requirement: show forced task in BOTH columns
        "task2": assigned_task,
        "chosen_task": followed_choice,  # Now shows the follow-up result
        "declared_choice_followup_prompt": fu_prompt,
        "declared_choice_followup_response": follow_text,
        "prompt": prompt,
        "response": response_content,
        "full_task_conversation": task_conv,
        "raw_response": str(raw_response),
        "identity_on": IDENTITY_ON,
        "model_name": MODEL_NAME,
        "model_name_returned": model_name_returned,
        "temperature_generation": 0.7,
        "temperature_ratings": 0.7,
        "temperature_followup": 0.0,
        "trial_index_global": trial_num,
        "trial_index_in_pair": trial_index_in_pair,
        "mock_mode": MOCK_MODE,
        "total_completion_tokens": total_completion_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_tokens": total_tokens,
        "api_calls_count": 6,  # Now 1 task + 4 ratings + 1 followup (same as free trials)
        "task_categories": ";".join(cats),
        "task_categories_count": len(cats),
        **ratings,
    }

# ------------------------- ChatID and transcript helpers ------------------- #

def generate_chat_id() -> str:
    """Generate a unique ChatID for this experimental run with 8-character hex format."""
    return uuid.uuid4().hex[:8]

def generate_trial_id(
    chat_id: str,
    pair_index: int,
    trial_type: str,
    trial_index_in_pair: int,
    trial_index_global: int
) -> str:
    """
    Generate a unique TrialID for a specific trial.
    
    Format: <ChatID>-P<pair_index>-<trial_type>-T<trial_index_in_pair>-G<trial_index_global>
    Example: ab12cd34-P02-free-T01-G003
    """
    return f"{chat_id}-P{pair_index:02d}-{trial_type}-T{trial_index_in_pair:02d}-G{trial_index_global:03d}"

def log_to_transcript(
    chat_id: str,
    trial_data: Dict[str, Any],
    filename: str = TRANSCRIPT_FILENAME
) -> None:
    """
    Append a raw-only human-readable block to the transcript file.
    Uses the exact same full conversation format as stored in the CSV.
    
    Args:
        chat_id: Unique identifier for this experimental run
        trial_data: The trial result dictionary
        filename: Path to transcript file
    """
    # Format timestamp for readability
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build the minimal transcript block - raw data only
    lines = []
    lines.append("=" * 80)
    lines.append(f"ChatID: {chat_id}")
    lines.append(f"TrialID: {trial_data['trial_id']}")  # Add TrialID display
    lines.append(f"Timestamp: {timestamp}")
    lines.append(f"Pair Index: {trial_data['pair_index']}")
    lines.append(f"Trial Type: {trial_data['trial_type']}")
    lines.append(f"Trial Number (in pair): {trial_data['trial_index_in_pair']}")
    lines.append(f"Trial Number (global): {trial_data['trial_index_global']}")
    lines.append("")
    
    # Tasks shown to model
    lines.append("Tasks:")
    lines.append(f"Task1: {trial_data['task1']}")
    lines.append(f"Task2: {trial_data['task2']}")
    lines.append("")
    lines.append("Full Conversation: \n")

    # Include identity prompt when enabled
    if IDENTITY_ON:
        lines.append("System Identity Prompt:")
        lines.append(IDENTITY_PROMPT)
        lines.append("")
    
    # Full conversation (exactly as stored in CSV)
    lines.append(trial_data['full_task_conversation'])
    lines.append("")
    
    # Choice information - show one line if they match, both if they differ
    declared_response = trial_data['declared_choice_followup_response']
    chosen_task = trial_data['chosen_task']
    
    # Normalize both for comparison using the same logic as the parser
    declared_norm = _norm_label(declared_response)
    chosen_norm = _norm_label(chosen_task)
    
    if declared_norm == chosen_norm or chosen_task == "ambiguous":
        # Show only one line when they match or when choice is ambiguous
        lines.append(f"Choice: {declared_response}")
    else:
        # Show both lines when they differ
        lines.append(f"Declared Choice: {declared_response}")
        lines.append(f"Inferred Chosen Task: {chosen_task}")
    
    lines.append("=" * 80)
    lines.append("")  # Extra blank line between blocks
    
    # Append to file
    transcript_path = Path(filename)
    with open(transcript_path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

# ------------------------- CSV persistence ---------------------------------- #

def ensure_consistent_columns(row: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure the row has all required columns in a stable order."""
    expected_columns = [
        "chat_id",
        "trial_id",  # Add trial_id as second column
        "time",
        "identity_on",
        "pair_index",
        "trial_type",
        "task1",
        "task2",
        "prompt",
        "response",
        "chosen_task",
        "pleasant_prompt",
        "pleasant_response",
        "pleasant_value",
        "enjoyable_prompt",
        "enjoyable_response",
        "enjoyable_value",
        "fun_prompt",
        "fun_response",
        "fun_value",
        "satisfying_prompt",
        "satisfying_response",
        "satisfying_value",
        "declared_choice_followup_prompt",
        "declared_choice_followup_response",
        "full_task_conversation",
        "raw_response",
        "model_name",
        "model_name_returned",
        "temperature_generation",
        "temperature_ratings",
        "temperature_followup",
        "trial_index_global",
        "trial_index_in_pair",
        "mock_mode",
        "total_completion_tokens",
        "total_prompt_tokens",
        "total_tokens",
        "api_calls_count",
        "task_categories",
        "task_categories_count",
    ]
    for col in expected_columns:
        if col not in row:
            if col == "task_categories":
                row[col] = ""
            elif col == "task_categories_count":
                row[col] = 0
            elif col in ("chat_id", "trial_id"):  # Handle both ID columns
                row[col] = ""
            else:
                row[col] = ""
    return {col: row[col] for col in expected_columns}

def append_row(row: Dict[str, Any], filename: str = CSV_FILENAME) -> None:
    """
    Append a single row to CSV without breaking longitudinal continuity.
    - If the file doesn't exist: create it with current columns.
    - If the file exists but is missing some of the row's columns: upgrade the file
      in place by adding the new columns (filled with ""), keeping existing data.
    - If the row is missing columns present in the file: add them as "" for this row.
    - Never rename or split the main CSV.
    """
    # Normalize and ORDER the row to the canonical schema
    row = ensure_consistent_columns(row)

    path = Path(filename)
    new_df = pd.DataFrame([row])

    # Create new file with current columns
    if not path.exists() or path.stat().st_size == 0:
        # Optional: prefer your canonical order if you have one
        # otherwise just use the row's column order
        new_df.to_csv(filename, index=False)
        return

    # File exists: read fully to allow in-place header upgrade
    try:
        existing_df = pd.read_csv(filename)
    except pd.errors.EmptyDataError:
        # Corrupt/empty file: rewrite cleanly
        new_df.to_csv(filename, index=False)
        return
    except Exception:
        # Fallback if some row was malformed; salvage what we can
        existing_df = pd.read_csv(filename, on_bad_lines="skip")

    # Build the union (keep existing order; append any new columns at the end)
    existing_cols = list(existing_df.columns)

    # Canonical order comes from the normalized new row's columns
    canonical_cols = list(new_df.columns)

    # If the existing file has exactly the same columns but in a different order,
    # rewrite it to match the canonical order so follow-up appears after ratings.
    if set(existing_cols) == set(canonical_cols) and existing_cols != canonical_cols:
        existing_df = existing_df.reindex(columns=canonical_cols)
        existing_df.to_csv(filename, index=False)
        existing_cols = canonical_cols  # update for the rest of the logic

    # If the file is missing columns that are now present, upgrade it in place
    missing_in_file = [c for c in canonical_cols if c not in existing_cols]
    if missing_in_file:
        for c in missing_in_file:
            existing_df[c] = ""
        # Write columns in canonical order (plus any older extras at the end)
        extras = [c for c in existing_cols if c not in canonical_cols]
        upgraded_cols = canonical_cols + extras
        existing_df = existing_df.reindex(columns=upgraded_cols)
        existing_df.to_csv(filename, index=False)
        existing_cols = upgraded_cols

    # Ensure the new row has all columns in the file and in the same order
    union_cols = existing_cols
    for c in union_cols:
        if c not in new_df.columns:
            new_df[c] = ""
    new_df = new_df.reindex(columns=union_cols)

    # Append the row (no header)
    new_df.to_csv(filename, mode="a", header=False, index=False)

def get_last_completed_trial() -> Tuple[int, str, int]:
    """Read the CSV and determine the last completed pair_index and trial type.
    Returns (pair_index, trial_type, count_in_pair) or (0, "", 0) if none."""
    try:
        path = Path(CSV_FILENAME)
        if not path.exists():
            return 0, "", 0
        try:
            df = pd.read_csv(CSV_FILENAME, on_bad_lines='warn')
        except TypeError:
            df = pd.read_csv(CSV_FILENAME)
        if len(df) == 0:
            return 0, "", 0
        last_row = df.iloc[-1]
        pair_index = int(last_row.get('pair_index', 0))
        trial_type = str(last_row.get('trial_type', ""))
        if pair_index > 0 and trial_type in {"free", "forced"}:
            pair_trials = df[(df['pair_index'] == pair_index) & (df['trial_type'] == trial_type)]
            count_in_pair = len(pair_trials)
            return pair_index, trial_type, count_in_pair
        return 0, "", 0
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return 0, "", 0

# ------------------------- Sanity checks ------------------------------------ #

for idx, spec in TASK_SPECS.items():
    for k in ("free", "reverse_free", "forced1", "forced2"):
        if "..." in spec[k]:
            raise ValueError(f"TASK_SPECS[{idx}]['{k}'] contains '...'. Replace with full exact text.")

# ------------------------- CLI argument parsing ---------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="GPT Task Choice Experiment")
    p.add_argument("--mock", action="store_true", help="Run without calling the API")
    return p.parse_args()

# ------------------------- Main --------------------------------------------- #

def ensure_results_directory() -> None:
    """Create the results directory if it doesn't exist."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

def main() -> None:
    args = parse_args()
    global MOCK_MODE
    MOCK_MODE = args.mock or MOCK_MODE

    # Ensure results directory exists
    ensure_results_directory()

    client = create_client()
    mode_str = "MOCK MODE" if MOCK_MODE else ""
    
    # Resume logic
    last_pair_idx, last_trial_type, completed_in_pair = get_last_completed_trial()
    start_new = True

    if last_pair_idx > 0:
        print(f"Found existing progress: Pair {last_pair_idx}, last completed {last_trial_type} trial #{completed_in_pair}")
        try:
            response = input("Continue from this point? (y/n): ").strip().lower()
            start_new = response != 'y'
        except EOFError:
            # Non-interactive environment; default to continue
            start_new = False

    # Generate ChatID - either new for fresh start or reuse for continuation
    if start_new:
        chat_id = generate_chat_id()
        print("Starting new experiment run...")
        print(f"ChatID for this run: {chat_id}")
        pleasant_history: List[float] = []
        enjoyable_history: List[float] = []
        fun_history: List[float] = []
        satisfying_history: List[float] = []
        trial_counter = 0
        start_pair_idx = 1
        last_trial_type = ""
        completed_in_pair = 0
    else:
        # For continuation, try to get ChatID from the last row in CSV
        chat_id = None
        try:
            df = pd.read_csv(CSV_FILENAME)
            if len(df) > 0 and 'chat_id' in df.columns:
                # Get the most recent non-empty ChatID
                for idx in reversed(df.index):
                    potential_chat_id = df.iloc[idx]['chat_id']
                    if pd.notna(potential_chat_id) and str(potential_chat_id).strip():
                        chat_id = str(potential_chat_id).strip()
                        break
                
                if chat_id:
                    print(f"Continuing with existing ChatID: {chat_id}")
                else:
                    # No valid ChatID found in existing data
                    chat_id = generate_chat_id()
                    print(f"No valid ChatID found in existing data, generating new one: {chat_id}")
            else:
                # No chat_id column exists
                chat_id = generate_chat_id()
                print(f"No ChatID column found in existing data, generating new one: {chat_id}")
        except Exception as e:
            chat_id = generate_chat_id()
            print(f"Error reading existing ChatID ({e}), generating new one: {chat_id}")
            
        print(f"Continuing from pair {last_pair_idx}...")
        df = pd.read_csv(CSV_FILENAME)
        def safe_float_list(series):
            return [float(x) if pd.notna(x) and str(x).strip() != "" else None for x in series]
        pleasant_history   = safe_float_list(df['pleasant_value'])   if 'pleasant_value'   in df else []
        enjoyable_history  = safe_float_list(df['enjoyable_value'])  if 'enjoyable_value'  in df else []
        fun_history        = safe_float_list(df['fun_value'])        if 'fun_value'        in df else []
        satisfying_history = safe_float_list(df['satisfying_value']) if 'satisfying_value' in df else []
        trial_counter = len(df)
        start_pair_idx = last_pair_idx

    for pair_idx in range(1, 46 + 1):
        if pair_idx < start_pair_idx:
            continue

        spec = TASK_SPECS[pair_idx]
        print(f"Pair {pair_idx}: {spec['task1']} vs {spec['task2']}")

        free_order   = generate_free_order()
        forced_assign = generate_forced_assignment()

        # --- Free-choice trials ---
        if not (pair_idx == start_pair_idx and last_trial_type == "forced"):
            start_free_idx = 0
            if pair_idx == start_pair_idx and last_trial_type == "free":
                start_free_idx = completed_in_pair

            for i, task1_first in enumerate(free_order[start_free_idx:], start=start_free_idx + 1):
                trial_counter += 1
                result = run_free_trial(client, pair_idx, spec, trial_counter, task1_first, i, chat_id)

                # Convert to float for history tracking, use None only if empty string
                pleasant_history.append(float(result["pleasant_value"]) if result["pleasant_value"] != "" else None)
                enjoyable_history.append(float(result["enjoyable_value"]) if result["enjoyable_value"] != "" else None)
                fun_history.append(float(result["fun_value"]) if result["fun_value"] != "" else None)
                satisfying_history.append(float(result["satisfying_value"]) if result["satisfying_value"] != "" else None)

                append_row(result)
                log_to_transcript(chat_id, result)  # Add transcript logging

                order_label = "Task1-first" if task1_first else "Task2-first"
                print(
                    f"  Free {i:02d}/{len(free_order)} [{order_label}] "
                    f"| P: {result['pleasant_value'] if result['pleasant_value'] != '' else 'N/A'} | E: {result['enjoyable_value'] if result['enjoyable_value'] != '' else 'N/A'} "
                    f"| F: {result['fun_value'] if result['fun_value'] != '' else 'N/A'} | S: {result['satisfying_value'] if result['satisfying_value'] != '' else 'N/A'}"
                )
                time.sleep(DELAY_BETWEEN_CALLS)

        # --- Forced-choice trials ---
        start_forced_idx = 0
        if pair_idx == start_pair_idx and last_trial_type == "forced":
            start_forced_idx = completed_in_pair

        for i, force_task1 in enumerate(forced_assign[start_forced_idx:], start=start_forced_idx + 1):
            trial_counter += 1
            result = run_forced_trial(client, pair_idx, spec, trial_counter, force_task1, i, chat_id)

            pleasant_history.append(float(result["pleasant_value"]) if result["pleasant_value"] != "" else None)
            enjoyable_history.append(float(result["enjoyable_value"]) if result["enjoyable_value"] != "" else None)
            fun_history.append(float(result["fun_value"]) if result["fun_value"] != "" else None)
            satisfying_history.append(float(result["satisfying_value"]) if result["satisfying_value"] != "" else None)

            append_row(result)
            log_to_transcript(chat_id, result)  # Add transcript logging

            forced_label = spec["task1"] if force_task1 else spec["task2"]
            print(
                f"  Forced {i:02d}/{len(forced_assign)} – wrote: {forced_label} "
                f"| P: {result['pleasant_value'] if result['pleasant_value'] != '' else 'N/A'} | E: {result['enjoyable_value'] if result['enjoyable_value'] != '' else 'N/A'} "
                f"| F: {result['fun_value'] if result['fun_value'] != '' else 'N/A'} | S: {result['satisfying_value'] if result['satisfying_value'] != '' else 'N/A'}"
            )
            time.sleep(DELAY_BETWEEN_CALLS)

        print()

        if pair_idx == start_pair_idx:
            last_trial_type = ""
            completed_in_pair = 0

    print("All trials completed ✔")
# ------------------------- Entrypoint --------------------------------------- #

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted – progress saved up to last completed trial.")