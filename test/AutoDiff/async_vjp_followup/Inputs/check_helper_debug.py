#!/usr/bin/env python3
"""Check the two-child, scalar async-AD debug-info regression fixture.

Read a post-differentiation SIL snapshot or stderr containing that snapshot.
This is a targeted text test, not a substitute for the compiler's SIL verifier.
No SIL is rewritten and no compiler is invoked by this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

SYMBOL = r'(?:"[^"\n]+"|[^\s:]+)'
FUNCTION = re.compile(
    rf'^sil\s+[^\n]*?@(?P<symbol>{SYMBOL})\s*:\s*'
    r'(?P<type>[^\n]*)\{\n(?P<body>.*?)^\}[^\n]*', re.M | re.S)
HELPER = re.compile(r'^__swift_async_ad_.*_[0-9]+\.body$')
ERROR_RECORD = re.compile(r'^\s*debug_value\b[^\n]*\bname "\$error"', re.M)


def post_differentiation(text: str) -> str:
    markers = list(re.finditer(
        r'^[ \t]*\*\*\* SIL module (before|after)[^\n]*'
        r'\(differentiation\)[^\n]*\n', text, re.M))
    for index, marker in enumerate(markers):
        if marker[1] == 'after':
            end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
            return text[marker.end():end]
    if markers:
        raise RuntimeError('No post-differentiation snapshot; compilation may have failed')
    return text


def check_helpers(text: str) -> dict:
    """Check this exact two-capture fixture, before later inlining/debug stripping."""
    functions = {m['symbol'].strip('"'): m
                 for m in FUNCTION.finditer(post_differentiation(text))}
    helpers = sorted(s for s in functions if HELPER.fullmatch(s))
    if len(helpers) != 2:
        raise RuntimeError(f'Expected two numerical helper bodies, found {len(helpers)}')

    def check_function(symbol: str, *, is_async: bool, captures: bool) -> None:
        f = functions[symbol]
        if ERROR_RECORD.search(f['body']):
            raise RuntimeError(f'{symbol}: obsolete $error debug_value in nonthrowing helper')
        if '@error' in f['type']:
            raise RuntimeError(f'{symbol}: numerical helper/derivative must remain nonthrowing')
        if ('@async' in f['type'].split('->', 1)[0]) != is_async:
            raise RuntimeError(f'{symbol}: incorrect outer async effect')
        if captures:
            for name in ('a', 'b'):
                if not re.search(r'^\s*debug_value\b[^\n]*\bname "' + name + '"',
                                 f['body'], re.M):
                    raise RuntimeError(f'{symbol}: lost ordinary capture debug record for {name}')

    # Check the helper itself first: changing only VJPCloner must not pass.
    for symbol in helpers:
        check_function(symbol, is_async=True, captures=True)
    checked = []
    for symbol in helpers:
        record = {'body': symbol}
        for kind, is_async in [('TJr', True), ('TJp', False)]:
            derivatives = [s for s in functions if s.startswith(symbol + kind)]
            if len(derivatives) != 1:
                raise RuntimeError(f'{symbol}: expected one {kind} body, found {len(derivatives)}')
            derivative = derivatives[0]
            check_function(derivative, is_async=is_async, captures=True)
            record[kind] = derivative
        # JVP placeholders may still exist in the paired representation.
        for s in functions:
            if s.startswith(symbol + 'TJf'):
                check_function(s, is_async=True, captures=False)
        checked.append(record)

    # The original throwing task-entry convention and its legal $error records
    # must not be globally stripped. No real throwing workload is differentiated.
    wrappers = [s for s, f in functions.items()
                if not s.startswith('__swift_async_ad_') and 'doWor5k' in s
                and 'cfu' in s and '@error' in f['type']]
    if len(wrappers) != 2:
        raise RuntimeError(f'Expected two original throwing-convention initializers, found {len(wrappers)}')
    for symbol in wrappers:
        if not ERROR_RECORD.search(functions[symbol]['body']):
            raise RuntimeError(f'{symbol}: original task-wrapper $error record was removed')
    return {'helpers': checked, 'original_error_records_preserved': len(wrappers),
            'ordinary_capture_debug_preserved': True}


def self_test() -> None:
    # These are synthetic parser fixtures, NOT expected compiler output files.
    records = '  debug_value %a : $Float, let, name "a", argno 2\n' \
              '  debug_value %b : $Float, let, name "b", argno 3\n'
    error = '  debug_value undef : $any Error, var, name "$error", argno 1\n'
    def function(name: str, ty: str, body: str) -> str:
        return f'sil private [ossa] @{name} : ${ty} {{\nbb0:\n{body}  return %r\n}}\n'
    good = 'sil_stage raw\n'
    for n in range(2):
        good += function(f'$sFixture_doWor5k_cfu{n}_',
                         '@convention(thin) @async (Float, Float) -> (@out Float, @error any Error)',
                         error + records)
        base = f'__swift_async_ad_fixture_{n}.body'
        good += function(base, '@convention(thin) @async (Float, Float) -> @out Float', records)
        good += function(base + 'TJrSSpSr',
                         '@convention(thin) @async (Float, Float) -> (@out Float, @owned @callee_guaranteed (Float) -> (Float, Float))', records)
        good += function(base + 'TJpSSpSr',
                         '@convention(thin) (@in_guaranteed Float) -> (Float, Float)', records)
    check_helpers(good)
    wrapped = ('*** SIL module before #90: Differentiation (differentiation)\n'
               'sil_stage raw\n'
               '*** SIL module after #90: Differentiation (differentiation)\n' + good)
    check_helpers(wrapped)

    def must_fail(text: str, reason: str) -> None:
        try:
            check_helpers(text)
        except RuntimeError as e:
            if reason not in str(e):
                raise AssertionError(f'Wrong checker failure: {e}') from e
        else:
            raise AssertionError(f'Bad fixture escaped checking: {reason}')

    for suffix in ('', 'TJrSSpSr', 'TJpSSpSr'):
        symbol = '__swift_async_ad_fixture_0.body' + suffix
        match = next(m for m in FUNCTION.finditer(good) if m['symbol'] == symbol)
        offset = match.start('body')
        must_fail(good[:offset] + error + good[offset:], 'obsolete $error')
    must_fail(good.replace(error, ''), 'original task-wrapper $error')
    must_fail(good.replace('name "a"', 'name "removed_a"'), 'capture debug record for a')
    must_fail('sil_stage raw\n', 'Expected two numerical helper bodies')
    print('PASS helper-debug checker self-tests (no compiler invoked)')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sil', type=Path, nargs='?')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.sil is None:
        parser.error('provide a SIL snapshot or --self-test')
    try:
        result = check_helpers(args.sil.read_text(errors='replace'))
    except (OSError, RuntimeError) as e:
        print(f'FAIL helper-debug: {e}', file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, indent=2))
    print('PASS async helper debug-info regression')


if __name__ == '__main__':
    main()
