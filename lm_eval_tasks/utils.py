from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List

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

    return dataset.map(_process_doc)


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

    if is_equiv(answer, str(doc["answer"])):
        retval = 1

    return {"exact_match": retval}


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
        return ss1 == ss2
    except Exception:
        return str1 == str2


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
