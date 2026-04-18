#!/usr/bin/env python3
from __future__ import annotations

import re


# Derived from the current reward-side answer normalization logic used in the
# atomic-skill-transfer workspace. The goal here is parity for eval summary
# extraction, without importing the full training repository.


def remove_boxed(text: str) -> str:
    if "\\boxed " in text:
        prefix = "\\boxed "
        if not text.startswith(prefix):
            raise AssertionError(f"box error: {text}")
        return text[len(prefix) :]

    prefix = "\\boxed{"
    if not text.startswith(prefix) or not text.endswith("}"):
        raise AssertionError(f"box error: {text}")
    return text[len(prefix) : -1]


def last_boxed_only_string(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if "\\boxed " in text:
        return "\\boxed " + text.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
        if text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return None if right_brace_idx is None else text[idx : right_brace_idx + 1]


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def fix_fracs(text: str) -> str:
    parts = text.split("\\frac")
    new_text = parts[0]
    if len(parts) > 1:
        for part in parts[1:]:
            new_text += "\\frac"
            if part[0] == "{":
                new_text += part
                continue
            if len(part) < 2:
                return text
            a = part[0]
            b = part[1]
            if b != "{":
                if len(part) > 2:
                    new_text += "{" + a + "}{" + b + "}" + part[2:]
                else:
                    new_text += "{" + a + "}{" + b + "}"
            else:
                if len(part) > 2:
                    new_text += "{" + a + "}" + b + part[2:]
                else:
                    new_text += "{" + a + "}" + b
    return new_text


def fix_a_slash_b(text: str) -> str:
    if len(text.split("/")) != 2:
        return text
    left, right = text.split("/")
    try:
        left_value = int(left)
        right_value = int(right)
    except Exception:
        return text
    if text != f"{left_value}/{right_value}":
        return text
    return f"\\frac{{{left_value}}}{{{right_value}}}"


def remove_right_units(text: str) -> str:
    if "\\text{ " not in text:
        return text
    splits = text.split("\\text{ ")
    if len(splits) != 2:
        return text
    return splits[0]


def fix_sqrt(text: str) -> str:
    if "\\sqrt" not in text:
        return text
    splits = text.split("\\sqrt")
    new_text = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            new_text += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_text += "\\sqrt" + split
    return new_text


def strip_string(text: str) -> str:
    text = text.replace("\n", "")
    text = text.replace("\\!", "")
    text = text.replace("\\\\", "\\")
    text = text.replace("tfrac", "frac")
    text = text.replace("dfrac", "frac")
    text = text.replace("\\left", "")
    text = text.replace("\\right", "")
    text = text.replace("^{\\circ}", "")
    text = text.replace("^\\circ", "")
    text = text.replace("\\$", "")
    text = remove_right_units(text)
    text = text.replace("\\\\%", "")
    text = text.replace("\\%", "")
    text = text.replace(" .", " 0.")
    text = text.replace("{.", "{0.")
    if not text:
        return text
    if text[0] == ".":
        text = "0" + text
    if len(text.split("=")) == 2 and len(text.split("=")[0]) <= 2:
        text = text.split("=")[1]
    text = fix_sqrt(text)
    text = text.replace(" ", "")
    text = fix_fracs(text)
    if text == "0.5":
        text = "\\frac{1}{2}"
    return fix_a_slash_b(text)
