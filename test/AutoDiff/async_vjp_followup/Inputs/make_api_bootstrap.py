#!/usr/bin/env python3
"""Concatenate actual API source and a test; never rewrite derivatives/SIL."""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("api", type=Path)
    p.add_argument("test", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    api = a.api.read_text()
    assert "Builtin.applyDerivative_vjp_async" in api
    assert "Builtin.applyDerivative_vjp_arity2_async" in api
    assert "Builtin.applyDerivative_vjp_arity3_async" in api
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text("import _Differentiation\n" + api + "\n" + a.test.read_text())


if __name__ == "__main__":
    main()
