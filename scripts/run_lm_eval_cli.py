#!/usr/bin/env python3
from __future__ import annotations

import pathlib

from lm_eval import tasks as lm_eval_tasks


BasePath = type(pathlib.Path())


class SafePath(BasePath):
    def relative_to(self, *other):
        try:
            return super().relative_to(*other)
        except ValueError:
            return self


lm_eval_tasks.Path = SafePath

from lm_eval.__main__ import cli_evaluate


if __name__ == "__main__":
    cli_evaluate()
