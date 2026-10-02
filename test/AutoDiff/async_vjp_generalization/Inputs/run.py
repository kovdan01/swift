#!/usr/bin/env python3
"""Native-host tests for async AD generalization, with synchronous controls.

Uses only source-level differentiation operators. It never replaces a generated
VJP or edits generated SIL. Synchronous controls mechanically remove the async
keywords in the marked numerical definitions; the runtime harness is identical.
The optional separately-built model exercises resilient/address-only values.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
import traceback

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
BEGIN = '// BEGIN DIFFERENTIATED FUNCTIONS'
END = '// END DIFFERENTIATED FUNCTIONS'
CASES = ('smoke', 'cfg', 'arrays', 'nominal', 'resilient')
FLAGS = ['-Xllvm', '-enable-experimental-async-autodiff']
DUMPS = ['-Xllvm', '-sil-print-before=differentiation',
         '-Xllvm', '-sil-print-after=differentiation']
VERIFY = ['-sil-verify-all', '-sil-ownership-verify-all']
FUNCTION = re.compile(r'^sil\s+[^\n]*?@(?P<symbol>"[^"\n]+"|[^\s:]+)\s*:\s*'
                      r'(?P<type>[^\n]*)\{\n(?P<body>.*?)^\}[^\n]*', re.M | re.S)
NEGATIVES = {
    'negative_residual': 'cannot prove that this saved derivative value can cross the child-task boundary',
    'negative_generic': 'generic async-let parents require contextual helper specialization',
}


def synchronous_source(source: str) -> str:
    if source.count(BEGIN) != 1 or source.count(END) != 1:
        raise ValueError('Expected one explicitly delimited numerical-definition region')
    start, end = source.index(BEGIN), source.index(END)
    definitions = re.sub(r'\basync\s+let\b', 'let', source[start:end])
    definitions = re.sub(r'\b(?:async|await)\b', '', definitions)
    return (source[:start] + definitions + source[end:]).replace('/*AD_AWAIT*/ await', '')


def snapshot(stderr: str, phase: str) -> str:
    markers = list(re.finditer(
        r'^[ \t]*\*\*\* SIL module (before|after)[^\n]*\(differentiation\)[^\n]*\n', stderr, re.M))
    for i, marker in enumerate(markers):
        if marker[1] != phase:
            continue
        end = markers[i + 1].start() if i + 1 < len(markers) else len(stderr)
        text = stderr[marker.end():end]
        index = text.find('sil_stage ')
        if index >= 0:
            return text[index:]
    raise RuntimeError(f'Missing immediate {phase}-differentiation snapshot')


def check_structure(sil: str, case: str) -> dict:
    functions = list(FUNCTION.finditer(sil))
    entries = [f for f in functions if f['symbol'].strip('"').endswith('.entry')
               and '__swift_async_ad_' in f['symbol']]
    if not entries:
        raise RuntimeError('No generated augmented task entry definitions')
    augmented = [f for f in functions if 'builtin "startAsyncLetWithLocalBuffer"<(' in f['body']]
    if not augmented:
        raise RuntimeError('No augmented task-result metadata in a function body')
    for function in entries:
        if '@async' not in function['type'].split('->', 1)[0]:
            raise RuntimeError('Augmented task entry is not async')
    pullbacks = [f for f in functions if 'TJp' in f['symbol']]
    if not pullbacks:
        raise RuntimeError('No generated pullback definitions')
    for function in pullbacks:
        if '@async' in function['type'].split('->', 1)[0]:
            raise RuntimeError('A generated pullback is async')
        if ('startAsyncLetWithLocalBuffer' in function['body'] or
                'swift_asyncLet_get' in function['body'] or
                'finishAsyncLet' in function['body']):
            raise RuntimeError('Backward computation retains task transport')
    contexts = sil.count('builtin "autoDiffCreateLinearMapContextWithType"')
    subcontexts = sil.count('builtin "autoDiffAllocateSubcontextWithType"')
    if case in ('cfg', 'arrays') and (not contexts or not subcontexts):
        raise RuntimeError('Loop tests did not exercise ordinary AD dynamic contexts')
    if case == 'resilient':
        # This test must actually exercise address materialization, not merely a
        # nominal type whose layout happened to become visible through inlining.
        if not any('copy_addr' in f['body'] for f in augmented):
            raise RuntimeError('Resilient task VJP did not contain an indirect copy')
    return dict(task_entries=len(entries), augmented_functions=len(augmented),
                synchronous_pullbacks=len(pullbacks), loop_contexts=contexts,
                loop_subcontexts=subcontexts)


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out = args.out.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []
        self.results: dict = {}

    def command(self, label: str, argv: list, *, expected_failure=False):
        argv = [str(x) for x in argv]
        print('$ ' + shlex.join(argv), flush=True)
        start = time.monotonic()
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=self.args.timeout)
            stdout, stderr, code = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as error:
            stdout, stderr, code = error.stdout or '', error.stderr or '', 'timeout'
            if isinstance(stdout, bytes): stdout = stdout.decode(errors='replace')
            if isinstance(stderr, bytes): stderr = stderr.decode(errors='replace')
        (self.out / (label + '.stdout.txt')).write_text(stdout)
        (self.out / (label + '.stderr.txt')).write_text(stderr)
        self.records.append(dict(label=label, argv=argv, returncode=code,
                                 seconds=round(time.monotonic() - start, 3)))
        (self.out / 'commands.json').write_text(json.dumps(self.records, indent=2) + '\n')
        if code != 0 and not expected_failure:
            raise RuntimeError(f'{label} failed ({code}); see {label}.stderr.txt')
        if expected_failure and (code == 0 or code == 'timeout' or isinstance(code, int) and code < 0):
            raise RuntimeError(f'{label}: expected a diagnostic, not success/crash/timeout ({code})')
        if expected_failure and ('Assertion `' in stderr or 'Stack dump:' in stderr):
            raise RuntimeError(f'{label}: a crash is not an expected diagnostic')
        return stdout, stderr

    def model(self, opt: str) -> tuple[Path, Path]:
        directory = self.out / ('model-' + opt)
        directory.mkdir(exist_ok=True)
        object_file = directory / 'Model.o'
        self.command('model-' + opt, [self.args.swiftc, '-parse-as-library', '-swift-version', '5',
            '-' + opt, *self.args.swiftc_arg, '-enable-library-evolution',
            '-module-name', 'AsyncADValueModel', '-emit-module',
            '-emit-module-path', directory / 'AsyncADValueModel.swiftmodule',
            '-emit-object', HERE / 'Model.swift', '-o', object_file])
        return directory, object_file

    def run_case(self, case: str, opt: str, mode: str, model=None):
        label = f'{case}-{mode}-{opt}'
        text = (TESTS / (case + '.swift')).read_text()
        if mode == 'sync':
            text = synchronous_source(text)
        source = self.out / (label + '.swift')
        source.write_text(text)
        imports = ['-I', str(model[0])] if model else []
        objects = [model[1]] if model else []
        flags = FLAGS if mode == 'async' else []
        debug = ['-g'] if self.args.debug_info else []
        frontend = [self.args.frontend, '-parse-as-library', '-swift-version', '5',
                    '-module-name', 'AsyncGeneralization', '-' + opt,
                    *self.args.frontend_arg, *imports, *flags, *debug]
        _, stderr = self.command(label + '.sil', [*frontend, '-emit-sil',
                    *VERIFY, *DUMPS, source, '-o', self.out / (label + '.final.sil')])
        before, after = snapshot(stderr, 'before'), snapshot(stderr, 'after')
        (self.out / (label + '.before.sil')).write_text(before)
        (self.out / (label + '.after.sil')).write_text(after)
        if mode == 'async':
            self.results[label] = check_structure(after, case)
        else:
            self.results[label] = dict(synchronous_control=True)
        if self.args.debug_ad and mode == 'async':
            self.command(label + '.debug', [*frontend, '-emit-sil', source, '-o',
                self.out / (label + '.debug.sil'), '-Xllvm',
                '-debug-only=differentiation,async-let-activity,async-let-differentiation'])
        if self.args.sil_only:
            return None
        binary = self.out / (label + '.exe')
        self.command(label + '.compile', [self.args.swiftc, '-parse-as-library',
            '-swift-version', '5', '-module-name', 'AsyncGeneralization', '-' + opt,
            *self.args.swiftc_arg, *imports, *flags, *debug, source, *objects, '-o', binary])
        stdout, _ = self.command(label + '.run', [binary])
        if 'PASS ' + case not in stdout:
            raise RuntimeError(f'{label}: missing numerical-test completion marker')
        self.results[label]['runtime_passed'] = True
        return stdout

    def negatives(self, opt: str):
        for case, expected in NEGATIVES.items():
            _, stderr = self.command(case + '-' + opt, [self.args.frontend,
                '-emit-sil', '-parse-as-library', '-swift-version', '5',
                '-module-name', 'AsyncNegative', '-' + opt, *self.args.frontend_arg,
                *FLAGS, TESTS / (case + '.swift'), '-o', self.out / (case + '.sil')],
                expected_failure=True)
            if expected not in stderr:
                raise RuntimeError(f'{case}: missing the specific expected diagnostic')
            self.results[case + '-' + opt] = dict(diagnostic=expected)

    def run(self):
        self.command('frontend-version', [self.args.frontend, '-version'])
        self.command('swiftc-version', [self.args.swiftc, '-version'])
        try:
            for opt in self.args.optimization:
                model = self.model(opt) if 'resilient' in self.args.case else None
                for case in self.args.case:
                    outputs = {}
                    for mode in (['sync', 'async'] if self.args.mode == 'both' else [self.args.mode]):
                        outputs[mode] = self.run_case(case, opt, mode, model if case == 'resilient' else None)
                    if not self.args.sil_only and self.args.mode == 'both':
                        if outputs['sync'] != outputs['async']:
                            raise RuntimeError(f'{case}-{opt}: synchronous/async numerical reports differ')
                        self.results[f'{case}-comparison-{opt}'] = dict(equal=True)
                if self.args.negative:
                    if self.args.mode == 'sync':
                        raise RuntimeError('--negative requires a patched async compiler')
                    self.negatives(opt)
        finally:
            (self.out / 'results.json').write_text(json.dumps(self.results, indent=2) + '\n')


def self_test():
    source = BEGIN + '\nfunc f() async { async let x = g(); return await x }\n' + END
    source += '\n/*AD_AWAIT*/ await gradient(); await Task.yield()\n'
    converted = synchronous_source(source)
    assert 'let x = g()' in converted and 'async let' not in converted
    assert 'await gradient' not in converted and 'await Task.yield' in converted
    stderr = '*** SIL module before #90, stage X, pass Differentiation (differentiation)\nsil_stage raw\nA\n'
    stderr += '*** SIL module after #90, stage X, pass Differentiation (differentiation)\nsil_stage raw\nB\n'
    assert snapshot(stderr, 'before').endswith('A\n')
    assert snapshot(stderr, 'after').endswith('B\n')
    for name in CASES:
        assert BEGIN in (TESTS / (name + '.swift')).read_text()
    print('PASS runner source-generation/parser self-tests (not compiler tests)')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frontend', type=Path)
    p.add_argument('--swiftc', type=Path)
    p.add_argument('--out', type=Path, default=Path('async-generalization-results'))
    p.add_argument('--mode', choices=['sync', 'async', 'both'], default='both')
    p.add_argument('--case', action='append', choices=CASES)
    p.add_argument('--optimization', action='append', choices=['Onone', 'O'])
    p.add_argument('--sil-only', action='store_true')
    p.add_argument('--negative', action='store_true')
    p.add_argument('--debug-info', action='store_true')
    p.add_argument('--debug-ad', action='store_true', help='requires an assertions-enabled compiler')
    p.add_argument('--frontend-arg', action='append', default=[])
    p.add_argument('--swiftc-arg', action='append', default=[])
    p.add_argument('--timeout', type=int, default=240)
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        self_test(); return
    if not args.frontend or not args.swiftc:
        p.error('--frontend and --swiftc must identify a matching toolchain')
    # Preserve argv[0]: swiftc can be a symlink to swift-driver, whose behavior
    # depends on the invocation name. Resolving that symlink changes the tool.
    args.frontend, args.swiftc = args.frontend.absolute(), args.swiftc.absolute()
    args.case = args.case or list(CASES)
    args.optimization = args.optimization or ['Onone', 'O']
    runner = Runner(args)
    try:
        runner.run()
    except Exception:
        (runner.out / 'failure.txt').write_text(traceback.format_exc())
        raise
    print('PASS requested stages; results in ' + str(runner.out))


if __name__ == '__main__':
    main()
