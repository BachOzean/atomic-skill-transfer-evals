from __future__ import annotations

import random
import re
from fractions import Fraction
from pathlib import Path
from typing import Dict, List

try:
    import sympy as _sympy
    from sympy.parsing.sympy_parser import (
        implicit_multiplication_application as _sympy_implicit_multiplication,
    )
    from sympy.parsing.sympy_parser import parse_expr as _sympy_parse_expr
    from sympy.parsing.sympy_parser import standard_transformations as _sympy_standard_transformations
except Exception:  # pragma: no cover
    _sympy = None
    _sympy_parse_expr = None
    _sympy_standard_transformations = ()
    _sympy_implicit_multiplication = None

try:
    import datasets
except ModuleNotFoundError:  # pragma: no cover
    datasets = None


def _require_datasets():
    if datasets is None:  # pragma: no cover
        raise ModuleNotFoundError("datasets is required to load local lm-eval task datasets.")


def preprocess_gpqa_text(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_gpqa_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        choices = [
            preprocess_gpqa_text(doc["Incorrect Answer 1"]),
            preprocess_gpqa_text(doc["Incorrect Answer 2"]),
            preprocess_gpqa_text(doc["Incorrect Answer 3"]),
            preprocess_gpqa_text(doc["Correct Answer"]),
        ]
        random.shuffle(choices)
        correct_answer_index = choices.index(preprocess_gpqa_text(doc["Correct Answer"]))
        return {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "answer": f"({chr(65 + correct_answer_index)})",
        }

    return dataset.map(_process_doc)


def extract_gpqa_choice(response) -> str:
    text = str(response or "").strip()
    patterns = (
        r"(?i)(?:answer|final answer)\s*[:：]?\s*\(?([ABCD])\)?",
        r"(?i)\(([ABCD])\)",
        r"(?i)\b([ABCD])\b",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return f"({matches[-1].upper()})"
    return text[:16]


def strip_thinking_response(response) -> str:
    text = str(response or "")
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text.strip()


def process_gpqa_generate_results(doc: dict, results: List[str]) -> Dict[str, int]:
    response = results[0] if results else ""
    prediction = extract_gpqa_choice(response)
    return {"exact_match": int(prediction == str(doc.get("answer") or ""))}


def process_gpqa_evalscope_generate_results(doc: dict, results: List[str]) -> Dict[str, int]:
    response = strip_thinking_response(results[0] if results else "")
    prediction = extract_gpqa_choice(response)
    return {"exact_match": int(prediction == str(doc.get("answer") or ""))}


def load_gpqa_local_dataset(gpqa_local_dataset_dir: str, dataset_name: str = "gpqa_diamond", **_kwargs):
    _require_datasets()
    local_path = Path(gpqa_local_dataset_dir).expanduser().resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"GPQA local dataset path does not exist: {local_path}")

    def _load_csv_dataset(csv_path: Path):
        loaded = datasets.load_dataset("csv", data_files={"train": str(csv_path)})
        if isinstance(loaded, datasets.DatasetDict):
            return loaded
        if isinstance(loaded, datasets.Dataset):
            return datasets.DatasetDict({"train": loaded})
        raise TypeError(f"Unexpected dataset type returned for CSV path {csv_path}: {type(loaded)!r}")

    errors: list[str] = []
    if local_path.is_file():
        if local_path.suffix.lower() != ".csv":
            raise ValueError(f"GPQA local dataset file must be a CSV file: {local_path}")
        return _load_csv_dataset(local_path)

    if not local_path.is_dir():
        raise NotADirectoryError(f"GPQA local dataset path must be a directory or CSV file: {local_path}")

    try:
        loaded = datasets.load_from_disk(str(local_path))
    except Exception as exc:  # pragma: no cover
        errors.append(f"load_from_disk failed: {exc}")
    else:
        if isinstance(loaded, datasets.DatasetDict):
            return loaded
        if isinstance(loaded, datasets.Dataset):
            return datasets.DatasetDict({"train": loaded})

    try:
        return datasets.load_dataset(str(local_path), name=dataset_name)
    except Exception as exc:  # pragma: no cover
        errors.append(f"load_dataset(path=local_dir, name={dataset_name!r}) failed: {exc}")

    csv_candidates = [
        local_path / f"{dataset_name}.csv",
        local_path / f"{dataset_name.removeprefix('gpqa_')}.csv",
    ]
    for csv_candidate in csv_candidates:
        if not csv_candidate.exists():
            continue
        try:
            return _load_csv_dataset(csv_candidate)
        except Exception as exc:  # pragma: no cover
            errors.append(f"load_dataset(csv={csv_candidate.name!r}) failed: {exc}")

    raise ValueError(
        "Could not load GPQA local dataset path. Tried datasets.load_from_disk(), "
        f"datasets.load_dataset(local_dir, name={dataset_name!r}), and CSV fallbacks. "
        f"Errors: {' | '.join(errors)}"
    )


def process_olympiadbench_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        final_answer = doc.get("final_answer") or []
        answer = ""
        if isinstance(final_answer, list) and final_answer:
            answer = str(final_answer[0])
        elif final_answer is not None:
            answer = str(final_answer)
        return {
            "id": doc.get("id"),
            "question": doc.get("question", ""),
            "answer": answer,
            "subject": doc.get("subject"),
            "subfield": doc.get("subfield"),
            "difficulty": doc.get("difficulty"),
        }

    processed = dataset.map(_process_doc)
    return processed.filter(lambda doc: bool(str(doc.get("answer") or "").strip()))


def process_omni_math_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        return {
            "problem": doc.get("problem", ""),
            "answer": str(doc.get("answer") or ""),
            "difficulty": doc.get("difficulty"),
            "source": doc.get("source"),
            "domain": doc.get("domain"),
        }

    return dataset.map(_process_doc)


def _answer_from_boxed_solution(solution: str) -> str:
    boxed = last_boxed_only_string(str(solution or ""))
    if boxed is None:
        return ""
    try:
        return remove_boxed(boxed) or ""
    except (AssertionError, IndexError):
        return ""


_OMNI_MATH_ANSWER_LOOKUP: dict[str, str] | None = None


def _omni_math_answer_lookup() -> dict[str, str]:
    global _OMNI_MATH_ANSWER_LOOKUP
    if _OMNI_MATH_ANSWER_LOOKUP is None:
        _require_datasets()
        full = datasets.load_dataset("KbsdJames/Omni-MATH", split="test")
        _OMNI_MATH_ANSWER_LOOKUP = {
            str(row.get("problem") or "").strip(): str(row.get("answer") or "")
            for row in full
        }
    return _OMNI_MATH_ANSWER_LOOKUP


def process_omni_math_500_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    answer_lookup = _omni_math_answer_lookup()

    def _process_doc(doc: dict) -> dict:
        solution = str(doc.get("solution") or "")
        problem = str(doc.get("problem") or "")
        return {
            "problem": problem,
            "solution": solution,
            "answer": answer_lookup.get(problem.strip()) or _answer_from_boxed_solution(solution),
            "difficulty": doc.get("difficulty"),
            "source": doc.get("source"),
            "domain": doc.get("domain"),
            "subdomain": doc.get("subdomain"),
        }

    processed = dataset.map(_process_doc)
    return processed.filter(lambda doc: bool(str(doc.get("answer") or "").strip()))


def process_matharena_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        return {
            "problem_idx": doc.get("problem_idx"),
            "problem": doc.get("problem", ""),
            "answer": str(doc.get("answer") or ""),
            "problem_type": doc.get("problem_type"),
        }

    return dataset.map(_process_doc)


def process_amc23_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        prompt = doc.get("prompt") or []
        problem = ""
        if isinstance(prompt, list) and prompt:
            first = prompt[0]
            if isinstance(first, dict):
                problem = str(first.get("content") or "")
            else:
                problem = str(first or "")
        reward_model = doc.get("reward_model") or {}
        answer = reward_model.get("ground_truth") if isinstance(reward_model, dict) else ""
        return {
            "problem": problem,
            "answer": str(answer if answer is not None else ""),
            "data_source": doc.get("data_source"),
            "ability": doc.get("ability"),
            "extra_info": doc.get("extra_info"),
        }

    return dataset.map(_process_doc)


def process_hendrycks_math500_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        solution = str(doc.get("solution") or "")
        boxed = last_boxed_only_string(solution)
        answer = ""
        if boxed is not None:
            try:
                answer = remove_boxed(boxed) or ""
            except (AssertionError, IndexError):
                answer = ""
        return {
            "problem": doc.get("problem", ""),
            "solution": solution,
            "answer": answer,
            "subject": doc.get("subject"),
            "level": doc.get("level"),
            "unique_id": doc.get("unique_id"),
        }

    return dataset.map(_process_doc)


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    retval = 0
    response = results[0]
    answer = extract_math_answer(response)

    if is_equiv(answer, str(doc["answer"])):
        retval = 1

    return {"exact_match": retval}


def extract_math_answer(response) -> str:
    response = str(response or "")
    indices = [pos for pos, char in enumerate(response) if char == "$"]
    if len(indices) <= 1:
        answer = response
    else:
        answer = response[indices[0] + 1 : indices[-1]]

    boxed_answer = last_boxed_only_string(response)
    if boxed_answer is not None:
        try:
            boxed_content = remove_boxed(boxed_answer)
            if boxed_content is not None:
                answer = boxed_content
        except (AssertionError, IndexError):
            pass

    return answer


def math_target_from_doc(doc: dict) -> str:
    for key, value in doc.items():
        if key.lower() == "answer":
            return str(value)
    return str(doc.get("answer") or "")


def flatten_repeat_results(results) -> list[str]:
    responses: list[str] = []
    for result in results or []:
        if isinstance(result, (list, tuple)):
            responses.extend(str(item or "") for item in result)
        else:
            responses.append(str(result or ""))
    return responses


def process_repeat_results(doc: dict, results) -> Dict[str, float]:
    responses = flatten_repeat_results(results)
    target = math_target_from_doc(doc)
    scores = [int(is_equiv(extract_math_answer(response), target)) for response in responses]
    if not scores:
        return {"exact_match": 0.0, "pass_at_8": 0.0, "pass_at_64": 0.0}
    payload = {
        "exact_match": sum(scores) / len(scores),
        f"pass_at_{len(scores)}": float(any(scores)),
    }
    if len(scores) != 8:
        payload.setdefault("pass_at_8", float(any(scores[:8])))
    if len(scores) != 64:
        payload.setdefault("pass_at_64", float(any(scores)))
    return payload


_TEXT_COMMANDS = ("text", "mathrm", "textrm", "mathbf", "operatorname", "mbox", "textbf")
_UNIT_WORDS = (
    "degree",
    "degrees",
    "unit",
    "units",
    "inch",
    "inches",
    "foot",
    "feet",
    "cm",
    "m",
    "meter",
    "meters",
    "hour",
    "hours",
    "minute",
    "minutes",
    "way",
    "ways",
    "dollar",
    "dollars",
    "cent",
    "cents",
)


def _unwrap_latex_text_commands(text: str) -> str:
    out = str(text or "")
    previous = None
    while previous != out:
        previous = out
        for command in _TEXT_COMMANDS:
            out = re.sub(r"\\" + command + r"\{([^{}]*)\}", r"\1", out)
    return out


def _remove_thousands_commas(text: str) -> str:
    raw = str(text or "")
    out = raw.replace("\\!", "")
    if "\\!" in raw or re.fullmatch(r"\$?[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", out.strip()):
        return re.sub(r"(?<=\d),(?=\d{3}\b)", "", out)
    return out


def _normalize_base_subscripts(text: str) -> str:
    # MATH final answers often accept bare digits when the problem context fixes
    # the base, e.g. 2516 for 2516_8.
    return re.sub(r"(?<=\d)_\{?\d+\}?", "", text)


def _strip_plain_units(text: str) -> str:
    unit_pattern = "|".join(re.escape(unit) for unit in _UNIT_WORDS)
    out = re.sub(r"\^\{?(?:st|nd|rd|th)\}?", "", text, flags=re.IGNORECASE)
    out = re.sub(rf"(?i)\b(?:{unit_pattern})\b(?:\^\{{?\d+\}}?)?", "", out)
    return out


def _canonical_answer(text) -> str:
    out = str(text or "").strip()
    boxed = last_boxed_only_string(out)
    if boxed is not None:
        try:
            boxed_content = remove_boxed(boxed)
            if boxed_content is not None:
                out = boxed_content
        except (AssertionError, IndexError):
            pass
    out = _unwrap_latex_text_commands(out)
    out = _remove_thousands_commas(out)
    out = _normalize_base_subscripts(out)
    out = _strip_plain_units(out)
    out = out.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    out = re.sub(r"\\frac\{([^{}]+)\}([A-Za-z0-9]+)", r"\\frac{\1}{\2}", out)
    out = out.replace("\\left", "").replace("\\right", "")
    out = out.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
    out = out.replace("\\,", "").replace("\\!", "").replace("{,}", ",")
    out = out.strip("$").strip()
    if not re.fullmatch(r"[+-]?\.\d+", out):
        out = out.rstrip(".")
    try:
        out = strip_string(out)
    except Exception:
        out = re.sub(r"\s+", "", out)
    return out.strip()


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            item = "".join(current).strip()
            if item:
                parts.append(item)
            current = []
        else:
            current.append(char)
    item = "".join(current).strip()
    if item:
        parts.append(item)
    return parts


def _vector_items(text) -> list[str] | None:
    raw = _unwrap_latex_text_commands(str(text or "").strip())
    matrix_match = re.search(r"\\begin\{[bpv]?matrix\}(.+?)\\end\{[bpv]?matrix\}", raw, flags=re.DOTALL)
    if matrix_match:
        parts = [part.strip() for part in re.split(r"\\\\|&", matrix_match.group(1)) if part.strip()]
        return parts if len(parts) > 1 else None

    compact = _remove_thousands_commas(raw)
    if "\\infty" in compact or "\\cup" in compact or "\\cap" in compact:
        return None
    if compact.startswith("[") and compact.endswith("]"):
        parts = _split_top_level_commas(compact[1:-1])
        return parts if len(parts) > 1 else None
    if compact.startswith("(") and compact.endswith(")") and "," in compact:
        parts = _split_top_level_commas(compact[1:-1])
        return parts if len(parts) > 1 else None
    if "," in compact and not re.search(r"[\[\]\(\)]", compact):
        parts = _split_top_level_commas(compact)
        return parts if len(parts) > 1 else None
    return None


def _is_semantic_list(text) -> bool:
    return _vector_items(text) is not None


def _has_plus_minus(text) -> bool:
    candidate = str(text or "")
    return bool(re.search(r"\\(?:pm|mp)(?![A-Za-z])", candidate))


def _replace_latex_frac(text: str) -> str | None:
    out: list[str] = []
    index = 0
    while index < len(text):
        start = text.find("\\frac", index)
        if start == -1:
            out.append(text[index:])
            break
        out.append(text[index:start])
        pos = start + len("\\frac")
        if pos >= len(text):
            return None

        def read_group(offset: int) -> tuple[str, int] | None:
            if offset >= len(text):
                return None
            if text[offset] != "{":
                return text[offset], offset + 1
            depth = 0
            for cursor in range(offset, len(text)):
                if text[cursor] == "{":
                    depth += 1
                elif text[cursor] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[offset + 1 : cursor], cursor + 1
            return None

        numerator = read_group(pos)
        if numerator is None:
            return None
        denominator = read_group(numerator[1])
        if denominator is None:
            return None
        out.append(f"(({numerator[0]})/({denominator[0]}))")
        index = denominator[1]
    return "".join(out)


def _latexish_to_sympy_expr(text: str):
    if _sympy is None or _sympy_parse_expr is None or _sympy_implicit_multiplication is None:
        return None
    candidate = str(text or "").strip()
    if not candidate or len(candidate) > 200 or _has_plus_minus(candidate):
        return None
    if any(
        token in candidate
        for token in (
            "\\cup",
            "\\cap",
            "\\infty",
            "\\le",
            "\\ge",
            "\\neq",
            "\\equiv",
            "\\pmod",
            "\\mod",
            "\\not",
            "<",
            ">",
            "=",
        )
    ):
        return None
    if "\\begin" in candidate or "\\end" in candidate:
        return None

    candidate = _replace_latex_frac(candidate)
    if candidate is None:
        return None
    previous = None
    while previous != candidate:
        previous = candidate
        candidate = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", candidate)
    candidate = candidate.replace("\\sqrt", "sqrt")
    candidate = candidate.replace("\\pi", "pi")
    candidate = candidate.replace("^", "**")
    candidate = candidate.replace("{", "(").replace("}", ")")
    candidate = candidate.replace("\\", "")
    candidate = candidate.replace(",", "")
    if "__" in candidate or not re.fullmatch(r"[0-9A-Za-z_+\-*/(). ]+", candidate):
        return None
    if not re.search(r"[+\-*/()]|sqrt|pi", candidate):
        return None
    try:
        return _sympy_parse_expr(
            candidate,
            transformations=_sympy_standard_transformations + (_sympy_implicit_multiplication,),
            evaluate=True,
        )
    except Exception:
        return None


def _safe_symbolic_equiv(prediction: str, target: str) -> bool:
    if _sympy is None:
        return False
    left = _latexish_to_sympy_expr(prediction)
    right = _latexish_to_sympy_expr(target)
    if left is None or right is None:
        return False
    try:
        return bool(_sympy.simplify(left - right) == 0)
    except Exception:
        return False


def _simple_numeric_value(text: str) -> Fraction | None:
    candidate = str(text or "").strip()
    frac_match = re.fullmatch(r"([+-]?)\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}", candidate)
    if frac_match:
        denominator = int(frac_match.group(3))
        if denominator == 0:
            return None
        numerator = int(frac_match.group(2))
        if frac_match.group(1) == "-":
            numerator *= -1
        return Fraction(numerator, denominator)
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", candidate):
        try:
            return Fraction(candidate)
        except Exception:
            return None
    return None


def _has_logical_condition_tail(text) -> bool:
    candidate = str(text or "")
    return bool(
        re.search(
            r"(?i)(?:\\(?:equiv|pmod|mod)|\\text\{\s*or\s*\}|\bor\b|\bif and only if\b)",
            candidate,
        )
    )


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        if ss1 == ss2 and not (
            _has_logical_condition_tail(str1)
            or _has_logical_condition_tail(str2)
        ):
            return True
    except Exception:
        if str1 == str2:
            return True

    normalized_1 = _canonical_answer(str1)
    normalized_2 = _canonical_answer(str2)
    if verbose:
        print(normalized_1, normalized_2)
    if normalized_1 and normalized_2 and normalized_1 == normalized_2:
        return True
    if _has_plus_minus(normalized_1) or _has_plus_minus(normalized_2):
        return False
    if (
        normalized_1
        and normalized_2
        and re.fullmatch(r"[A-Za-z]+", normalized_1)
        and re.fullmatch(r"[A-Za-z]+", normalized_2)
        and normalized_1.lower() == normalized_2.lower()
    ):
        return True

    vector_1 = _vector_items(str1)
    vector_2 = _vector_items(str2)
    if vector_1 is not None or vector_2 is not None:
        if vector_1 is None or vector_2 is None or len(vector_1) != len(vector_2):
            return False
        return all(is_equiv(item_1, item_2, verbose=verbose) for item_1, item_2 in zip(vector_1, vector_2))

    if _is_semantic_list(str1) != _is_semantic_list(str2):
        return False

    number_1 = _simple_numeric_value(normalized_1)
    number_2 = _simple_numeric_value(normalized_2)
    if number_1 is not None or number_2 is not None:
        return number_1 is not None and number_2 is not None and number_1 == number_2

    return _safe_symbolic_equiv(normalized_1 or str(str1), normalized_2 or str(str2))


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        return string


def remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            new_substr = "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\\\%", "")
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    return fix_a_slash_b(string)
