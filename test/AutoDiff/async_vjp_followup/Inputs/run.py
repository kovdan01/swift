#!/usr/bin/env python3
"""Run incremental async AutoDiff tests and preserve commands/SIL/IR/output.

All runtime tests call source-level differentiation APIs. No derivative symbol
aliases, hand-written derivatives, or generated-SIL body replacements are used.
The optional finish-form fixtures change ONLY the pre-AD cleanup representation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback

from check_helper_debug import check_helpers

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
ROOT = Path(__file__).resolve().parents[4]
SYMBOL = r'(?:"[^"\n]+"|[^\s:]+)'
FUNCTION = re.compile(
    rf'^(?P<header>sil\s+[^\n]*?@(?P<symbol>{SYMBOL})\s*:\s*'
    r'(?P<type>[^\n]*)\{)\n(?P<body>.*?)^\}[^\n]*', re.M | re.S)
WITNESS = re.compile(
    rf'^sil_differentiability_witness[^\n]*?@(?P<symbol>{SYMBOL})\s*:[^\n]*\{{\n'
    r'(?P<body>.*?)^\}', re.M | re.S)
FLAGS = ['-Xllvm', '-enable-experimental-async-autodiff']
VERIFY = ['-sil-verify-all', '-sil-ownership-verify-all']
DUMPS = ['-Xllvm', '-sil-print-before=differentiation',
         '-Xllvm', '-sil-print-after=differentiation']
DEBUG = ['-Xllvm', '-debug-only=differentiation,async-let-activity,async-let-differentiation']


def plain(s):
    return s.strip('"')


def extract_phase(stderr, phase):
    markers = list(re.finditer(
        r'^[ \t]*\*\*\* SIL module (before|after)[^\n]*\(differentiation\)[^\n]*\n',
        stderr, re.M))
    for i, marker in enumerate(markers):
        if marker[1] != phase:
            continue
        end = markers[i + 1].start() if i + 1 < len(markers) else len(stderr)
        text = stderr[marker.end():end]
        start = text.find('sil_stage ')
        if start >= 0:
            return text[start:]
    raise RuntimeError(f'No {phase}-differentiation module snapshot in stderr')


def check_tasks(before, after):
    module = 'AsyncFollowupFinish'
    prefix = f'$s{len(module)}{module}7doWor5k'
    matches = [w for w in WITNESS.finditer(after)
               if plain(w['symbol']).startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f'Expected one original doWor5k witness, found {len(matches)}')
    match = re.search(rf'\bvjp:\s*@({SYMBOL})', matches[0]['body'])
    if not match:
        raise RuntimeError('Original witness has no generated VJP')
    symbol = plain(match[1])
    functions = {plain(f['symbol']): f for f in FUNCTION.finditer(after)}
    if symbol not in functions:
        raise RuntimeError('VJP declaration exists without a function body')
    vjp = functions[symbol]
    if '@async' not in vjp['type'].split('->', 1)[0]:
        raise RuntimeError('Parent VJP is not async')
    if '@async' in vjp['type'].split('->', 1)[1]:
        raise RuntimeError('Returned pullback is unexpectedly async')
    starts = list(re.finditer(r'builtin "startAsyncLetWithLocalBuffer"<\(', vjp['body']))
    if len(starts) != 2:
        raise RuntimeError(f'Expected two augmented children, got {len(starts)}')
    wait = re.search(r'function_ref @swift_asyncLet_get\b', vjp['body'])
    if not wait or starts[-1].start() >= wait.start():
        raise RuntimeError('Both children must start before the first wait')
    if vjp['body'].count('load [copy]') < 2:
        raise RuntimeError('Missing copied pullbacks before cleanup')
    pb_symbol = symbol.replace('TJr', 'TJp', 1)
    if pb_symbol not in functions:
        raise RuntimeError('Missing synchronous parent pullback body')
    pb = functions[pb_symbol]
    if '@async' in pb['type'] or 'swift_asyncLet_' in pb['body']:
        raise RuntimeError('Parent pullback is async or retains task operations')
    original = plain(matches[0]['symbol'])
    before_functions = {plain(f['symbol']): f for f in FUNCTION.finditer(before)}
    if original not in before_functions or original not in functions:
        raise RuntimeError('Missing primal function body')
    def operational(f):
        return [line.split('//', 1)[0].strip() for line in f['body'].splitlines()
                if line.split('//', 1)[0].strip()]
    if operational(before_functions[original]) != operational(functions[original]):
        raise RuntimeError('The original primal changed during differentiation')
    entries = [s for s in functions if s.startswith('__swift_async_ad_') and s.endswith('.entry')]
    if len(entries) != 2:
        raise RuntimeError(f'Expected two generated task entries, got {len(entries)}')
    helper_debug = check_helpers(after)
    return dict(vjp=symbol, pullback=pb_symbol, augmented_children=2,
                primal_preserved=True, synchronous_pullback=True,
                helper_debug=helper_debug)


def finish_form(sil, form):
    """Normalize only pre-AD finish calls/builtins; preserve numerical code."""
    converted = 0
    def body_transform(m):
        nonlocal converted
        body = m['body']
        if form == 'builtin':
            refs = re.findall(r'^\s*(%\w+) = function_ref @swift_asyncLet_finish\b', body, re.M)
            for ref in refs:
                pat = re.compile(r'^(\s*)(%\w+) = apply ' + re.escape(ref) +
                                 r'\((%\w+), (%\w+)\) : [^\n]*', re.M)
                def to_builtin(a):
                    nonlocal converted
                    converted += 1
                    return f'{a[1]}{a[2]} = builtin "finishAsyncLet"({a[3]}, {a[4]}) : $()'
                body = pat.sub(to_builtin, body)
        else:
            pat = re.compile(r'^(\s*)(%\w+) = builtin "finishAsyncLet"\((%\w+), (%\w+)\) : \$\(\)[^\n]*', re.M)
            def to_runtime(a):
                nonlocal converted
                converted += 1
                ref = f'%async_ad_finish_fn_{converted}'
                if ref in sil:
                    raise RuntimeError('Unexpected fixture variable collision')
                t = '$@convention(thin) @async (Builtin.RawPointer, Builtin.RawPointer) -> ()'
                return (f'{a[1]}{ref} = function_ref @swift_asyncLet_finish : {t}\n'
                        f'{a[1]}{a[2]} = apply {ref}({a[3]}, {a[4]}) : {t}')
            body = pat.sub(to_runtime, body)
        # Preserve the original function header and closing comment.
        suffix = m[0][m[0].rfind('\n}') + 1:]
        return m['header'] + '\n' + body + suffix
    result = FUNCTION.sub(body_transform, sil)
    # The SIL module printer can omit these imports even when its source file
    # imported them. Standalone raw-SIL reparsing needs protocol visibility.
    imports = ''.join('import ' + name + '\n' for name in
                      ('_Differentiation', '_Concurrency')
                      if not re.search(r'^import ' + name + r'\s*$', result, re.M))
    if imports:
        result = re.sub(r'^(sil_stage [^\n]+\n)', r'\1\n' + imports,
                        result, count=1, flags=re.M)
    if form == 'runtime' and not re.search(r'^sil\s+[^\n]*@swift_asyncLet_finish\s*:', result, re.M):
        result += '\nsil @swift_asyncLet_finish : $@convention(thin) @async (Builtin.RawPointer, Builtin.RawPointer) -> ()\n'
    if form == 'builtin':
        count = result.count('builtin "finishAsyncLet"')
    else:
        count = len(re.findall(r'= function_ref @swift_asyncLet_finish\b', result))
    if count != 2:
        raise RuntimeError(f'Expected two cleanup sites in {form} fixture, found {count}')
    return result


def check_witness_ir(ir):
    result = {}
    for name, async_ in [('asyncIdentity', True), ('syncIdentity', False)]:
        lines = [l for l in ir.splitlines() if l.startswith('@') and re.search(r'(?:\d|_)' + name, l) and
                 'WJrSpSr" =' in l and '{ ptr, ptr }' in l]
        if len(lines) != 1:
            raise RuntimeError(f'Expected one {name} witness global, got {len(lines)}')
        pointers = re.findall(r'ptr @"([^"\n]+)"', lines[0])
        if len(pointers) != 2:
            raise RuntimeError(f'Expected JVP/VJP pointer fields for {name}')
        for p in pointers:
            if p.endswith('Tu') != async_:
                raise RuntimeError(f'{name}: incorrect witness pointer representation: {p}')
        result[name] = pointers
    return result


class Runner:
    def __init__(self, args):
        self.a = args
        self.out = args.out.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.records = []
        self.results = {}

    def command(self, label, argv, check=True):
        argv = [str(x) for x in argv]
        stem = self.out / label
        start = time.monotonic()
        print('$ ' + shlex.join(argv), flush=True)
        record = dict(label=label, argv=argv)
        try:
            p = subprocess.run(argv, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=self.a.timeout)
            stdout, stderr, status = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout or ''; stderr = e.stderr or ''
            if isinstance(stdout, bytes): stdout = stdout.decode(errors='replace')
            if isinstance(stderr, bytes): stderr = stderr.decode(errors='replace')
            status = 'timeout'
        record.update(returncode=status, seconds=round(time.monotonic()-start, 3))
        stem.with_suffix(stem.suffix + '.stdout.txt').write_text(stdout)
        stem.with_suffix(stem.suffix + '.stderr.txt').write_text(stderr)
        self.records.append(record)
        (self.out/'commands.json').write_text(json.dumps(self.records, indent=2)+'\n')
        if check and status != 0:
            raise RuntimeError(f'{label} failed ({status}); see {stem}.stderr.txt')
        return stdout, stderr, status

    def emit_sil(self, label, source, parse_stdlib=False, parse_sil=False):
        argv = [self.a.frontend, '-emit-sil', '-parse-as-library',
                '-module-name', 'AsyncFollowupFinish', '-' + self.opt,
                *self.a.frontend_arg, *FLAGS, *VERIFY, *DUMPS]
        if parse_stdlib: argv += ['-parse-stdlib']
        if parse_sil: argv += ['-parse-sil']
        argv += [source, '-o', self.out/(label+'.final.sil')]
        _, stderr, status = self.command(label, argv, check=False)
        # Keep debug output separate so pre-pass SIL remains parseable.
        if self.a.debug:
            self.command(label+'.debug', [*argv, *DEBUG], check=False)
        if status != 0:
            raise RuntimeError(f'{label} failed ({status}); see {label}.stderr.txt')
        before = extract_phase(stderr, 'before')
        after = extract_phase(stderr, 'after')
        (self.out/(label+'.before.sil')).write_text(before)
        (self.out/(label+'.after.sil')).write_text(after)
        return before, after

    def finish(self):
        label = 'finish-'+self.opt
        before, after = self.emit_sil(label, TESTS/'finish.swift')
        self.results[label] = check_tasks(before, after)
        if self.a.finish_forms:
            for form in ('builtin', 'runtime'):
                fixture = self.out/(f'finish-{form}-{self.opt}.input.sil')
                fixture.write_text(finish_form(before, form))
                b, a = self.emit_sil(f'finish-{form}-{self.opt}', fixture, parse_sil=True)
                self.results[f'finish-{form}-{self.opt}'] = check_tasks(b, a)

    def stdlib(self):
        label = 'stdlib-' + self.opt
        directory = ROOT / 'stdlib/public/Differentiation'
        sources = [directory / 'Differentiable.swift',
                   directory / 'AsyncDifferentialOperators.swift']
        self.results[label + '.sources'] = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources}
        flags = ['-parse-as-library', '-parse-stdlib',
                 '-disable-implicit-concurrency-module-import',
                 '-enable-library-evolution',
                 '-enable-upcoming-feature', 'MemberImportVisibility',
                 '-module-name', '_Differentiation', '-' + self.opt,
                 *self.a.frontend_arg]
        sil = self.out / (label + '.silgen.sil')
        self.command(label + '.silgen',
                     [self.a.frontend, '-emit-silgen', *flags, *sources,
                      '-o', sil])
        text = sil.read_text()
        count = text.count('differentiable_function_extract [vjp]')
        if count < 3:
            raise RuntimeError('Expected all three generic async VJP builtins')
        self.command(label + '.module',
                     [self.a.frontend, '-emit-module', *flags,
                      '-experimental-skip-non-inlinable-function-bodies',
                      *sources, '-o', self.out / (label + '.swiftmodule'),
                      '-emit-module-interface-path',
                      self.out / (label + '.swiftinterface')])
        # Never add these intentionally incomplete _Differentiation modules to
        # another test's import search path: numerical tests need the full stdlib.
        self.results[label] = {'silgen': 'PASS', 'module': 'PASS',
                               'async_vjp_extractions': count}

    def witness(self):
        label = 'witness-'+self.opt
        irpath = self.out/(label+'.ll')
        self.command(label, [self.a.frontend, '-emit-ir', '-parse-as-library',
                     '-module-name', 'AsyncWitness', '-'+self.opt,
                     *self.a.frontend_arg, *FLAGS,
                     ROOT/'test/IRGen/autodiff_async_witness.swift', '-o', irpath])
        self.results[label] = check_witness_ir(irpath.read_text())

    def api(self, which):
        label = which+'-'+self.opt
        source = self.out/(label+'.swift')
        text = (HERE/(which+'.swift')).read_text()
        bootstrap = self.a.api_mode == 'bootstrap'
        if bootstrap:
            api_path = ROOT/'stdlib/public/Differentiation/AsyncDifferentialOperators.swift'
            api = api_path.read_text()
            if 'Builtin.applyDerivative_vjp_arity2_async' not in api:
                raise RuntimeError('Wrong API implementation file')
            text = 'import _Differentiation\n' + api + '\n' + text
            self.results['api_source_sha256'] = hashlib.sha256(api.encode()).hexdigest()
        source.write_text(text)
        self.emit_sil(label, source, parse_stdlib=bootstrap)
        if self.a.sil_only:
            return
        exe = self.out/(label + ('.exe' if sys.platform=='win32' else ''))
        argv = [self.a.swiftc, '-parse-as-library', '-'+self.opt,
                *self.a.swiftc_arg, *FLAGS,
                '-Xfrontend','-sil-verify-all',
                '-Xfrontend','-sil-ownership-verify-all']
        if bootstrap: argv += ['-parse-stdlib']
        argv += [source, '-o', exe]
        self.command(label+'.build', argv)
        for variant in (['default', 'choose-square'] if which=='leaf' else ['default']):
            args = [] if variant=='default' else [variant]
            stdout, _, _ = self.command(label+'.run-'+variant, [exe,*args])
            required = 'PASS async reverse API:' if which=='leaf' else 'PASS async-let reverse API:'
            if required not in stdout:
                raise RuntimeError(f'{label}: missing success marker')
            self.results[label+'.'+variant] = 'PASS'
            print(stdout, end='')

    def run(self):
        self.command('frontend-version', [self.a.frontend,'--version'])
        if not self.a.sil_only:
            self.command('swiftc-version', [self.a.swiftc,'--version'])
        if self.a.repo:
            self.command('repository-revision', ['git','-C',self.a.repo,'rev-parse','HEAD'])
            self.command('repository-diff-stat', ['git','-C',self.a.repo,'diff','--stat'])
        modes = ['stdlib','finish','witness','leaf','tasks'] if self.a.mode=='all' else [self.a.mode]
        for opt in self.a.optimizations:
            self.opt = opt
            for mode in modes:
                if mode=='stdlib': self.stdlib()
                elif mode=='finish': self.finish()
                elif mode=='witness': self.witness()
                else: self.api(mode)
        (self.out/'results.json').write_text(json.dumps(self.results,indent=2)+'\n')
        print('PASS selected follow-up tests; logs:',self.out)


def self_test():
    raw = '''sil_stage raw
sil [ossa] @fixture : $@convention(thin) @async () -> () {
bb0:
  %0 = builtin "finishAsyncLet"(%task1, %ptr1) : $()
  %1 = builtin "finishAsyncLet"(%task2, %ptr2) : $()
  return %0
}
'''
    a = finish_form(raw,'runtime')
    assert a.count('= function_ref @swift_asyncLet_finish')==2
    b = finish_form(a,'builtin')
    assert b.count('builtin "finishAsyncLet"')==2
    markers = ('*** SIL module before #1: Differentiation (differentiation)\n'+raw+
               '*** SIL module after #1: Differentiation (differentiation)\n'+raw)
    assert extract_phase(markers,'before')==raw
    assert extract_phase(markers,'after')==raw
    good = ('@"a_asyncIdentityWJrSpSr" = global { ptr, ptr } { ptr @"jTu", ptr @"vTu" }\n'
            '@"a_syncIdentityWJrSpSr" = global { ptr, ptr } { ptr @"j", ptr @"v" }\n')
    check_witness_ir(good)
    try: check_witness_ir(good.replace('vTu','v'))
    except RuntimeError: pass
    else: raise AssertionError('Bad async witness escaped checking')
    from check_helper_debug import self_test as helper_debug_self_test
    helper_debug_self_test()
    print('PASS runner parser/fixture self-tests (no compiler invoked)')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frontend', type=Path)
    p.add_argument('--swiftc', type=Path)
    p.add_argument('--out', type=Path, default=Path('async-ad-followup-results'))
    p.add_argument('--repo', type=Path, help='Record the patched checkout revision and diff stat')
    p.add_argument('--mode', choices=['all','stdlib','finish','witness','leaf','tasks'], default='all')
    p.add_argument('--api-mode', choices=['bootstrap','stdlib'], default='bootstrap')
    p.add_argument('--optimizations', nargs='+', choices=['Onone','O'], default=['Onone','O'])
    p.add_argument('--finish-forms', action='store_true', help='Also test both pre-AD cleanup lowerings')
    p.add_argument('--sil-only', action='store_true', help='Skip runtime builds; still emit/check SIL and witness IR')
    p.add_argument('--debug', action='store_true', help='Additional separate debug dump; assertions-enabled compiler')
    p.add_argument('--frontend-arg', action='append', default=[])
    p.add_argument('--swiftc-arg', action='append', default=[])
    p.add_argument('--timeout', type=int, default=180)
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    if a.self_test: self_test();return
    if not a.frontend: p.error('--frontend is required')
    a.frontend = a.frontend.resolve()
    if not a.frontend.is_file(): p.error('frontend executable not found')
    if a.swiftc is None: a.swiftc = a.frontend.with_name('swiftc')
    # Do NOT resolve the swiftc symlink: invocation basename selects driver mode.
    a.swiftc = a.swiftc.absolute()
    if not a.sil_only and not a.swiftc.is_file(): p.error('swiftc executable not found')
    r = Runner(a)
    (r.out/'environment.json').write_text(json.dumps({
        'platform':platform.platform(),'python':sys.version,
        'api_mode':a.api_mode,'optimizations':a.optimizations,
        'root':str(ROOT),'frontend':str(a.frontend),'swiftc':str(a.swiftc)
    },indent=2)+'\n')
    try: r.run()
    except Exception:
        (r.out/'failure.txt').write_text(traceback.format_exc())
        (r.out/'partial-results.json').write_text(json.dumps(r.results,indent=2)+'\n')
        print(traceback.format_exc(),file=sys.stderr)
        sys.exit(1)

if __name__=='__main__':
    main()
