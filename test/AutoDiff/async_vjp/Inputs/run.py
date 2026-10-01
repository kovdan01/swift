#!/usr/bin/env python3
"""Exercise the patched compiler, preserve evidence, and run generated VJPs.

No third-party Python dependencies. This runner never substitutes a hand-written
VJP. The optional yield test adds only scheduling calls to compiler-emitted SIL.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import traceback
from typing import Any

HERE = Path(__file__).resolve().parent
MODULE = 'AsyncVJPTests'
SYMBOL = r'(?:"[^"\n]+"|[^\s:]+)'
FUNCTION = re.compile(
    rf'^(?P<header>sil\s+[^\n]*?@(?P<symbol>{SYMBOL})\s*:\s*(?P<type>[^\n]*)\{{)\n'
    r'(?P<body>.*?)^\}[^\n]*', re.M | re.S)
WITNESS = re.compile(
    rf'^sil_differentiability_witness[^\n]*?@(?P<symbol>{SYMBOL})\s*:[^\n]*\{{\n'
    r'(?P<body>.*?)^\}', re.M | re.S)


def plain(symbol: str) -> str:
    return symbol.strip('"')


def functions(text: str) -> dict[str, re.Match[str]]:
    return {plain(m['symbol']): m for m in FUNCTION.finditer(text)}


def witness_for(text: str, name: str) -> tuple[str, str]:
    prefix = f'$s{len(MODULE)}{MODULE}{len(name)}{name}'
    found = []
    for m in WITNESS.finditer(text):
        original = plain(m['symbol'])
        if original.startswith(prefix) and 'TJ' not in original:
            vjp = re.search(rf'\bvjp:\s*@({SYMBOL})', m['body'])
            if vjp:
                found.append((original, plain(vjp[1])))
    if len(found) != 1:
        raise RuntimeError(f'{name}: expected one completed VJP witness; found {found}')
    return found[0]


def extract_phase(stderr: str, phase: str) -> str:
    markers = list(re.finditer(
        r'^[ \t]*\*\*\* SIL module (before|after)[^\n]*\(differentiation\)[^\n]*\n',
        stderr, re.M))
    for i, m in enumerate(markers):
        if m[1] == phase:
            end = markers[i + 1].start() if i + 1 < len(markers) else len(stderr)
            result = stderr[m.end():end]
            start = result.find('sil_stage ')
            if start >= 0:
                return result[start:]
    raise RuntimeError(f'No {phase}-differentiation module marker in compiler stderr')


def structural_checks(before: str, after: str, cases: list[dict[str, Any]]) -> dict:
    old_functions, new_functions = functions(before), functions(after)
    result = {}
    for case in cases:
        name = case['name']
        original, vjp_symbol = witness_for(after, name)
        if original not in new_functions or original not in old_functions:
            raise RuntimeError(f'{name}: missing original definition in a pass snapshot')
        if vjp_symbol not in new_functions:
            raise RuntimeError(f'{name}: witness references missing VJP body {vjp_symbol}')
        # SSA renumbering and printer comments can differ; compare operational lines.
        def operations(m: re.Match[str]) -> list[str]:
            lines = []
            for line in m['body'].splitlines():
                line = line.split('//', 1)[0].strip()
                if line:
                    lines.append(line)
            return lines
        if operations(old_functions[original]) != operations(new_functions[original]):
            raise RuntimeError(f'{name}: the primal changed across differentiation')
        vjp = new_functions[vjp_symbol]
        outer = vjp['type'].split('->', 1)[0]
        if ('@async' in outer) != case['async']:
            raise RuntimeError(f'{name}: wrong async effect on the outer VJP')
        # All returned linear maps in these fixtures must remain synchronous.
        returned = vjp['type'].split('->', 1)[1]
        if '@async' in returned:
            raise RuntimeError(f'{name}: returned pullback unexpectedly async')
        starts = list(re.finditer(r'builtin "startAsyncLetWithLocalBuffer"<([^\n]*)', vjp['body']))
        if len(starts) != case['tasks']:
            raise RuntimeError(f'{name}: expected {case["tasks"]} starts in VJP, got {len(starts)}')
        augmented = sum(s[1].startswith('(') for s in starts)
        if augmented != case['augmented_tasks']:
            raise RuntimeError(f'{name}: expected {case["augmented_tasks"]} augmented payloads, got {augmented}')
        if case['tasks'] == 2:
            get = re.search(r'function_ref @swift_asyncLet_get\b', vjp['body'])
            if not get or starts[-1].start() >= get.start():
                raise RuntimeError(f'{name}: children are not both started before the first wait')
        if case['augmented_tasks'] and 'load [copy]' not in vjp['body']:
            raise RuntimeError(f'{name}: no owned pullback load before task cleanup')
        pb_symbol = vjp_symbol.replace('TJr', 'TJp', 1)
        pb = new_functions.get(pb_symbol)
        if not pb:
            raise RuntimeError(f'{name}: missing parent pullback body {pb_symbol}')
        if '@async' in pb['type'] or 'swift_asyncLet_' in pb['body']:
            raise RuntimeError(f'{name}: synchronous pullback contains task waiting machinery')
        result[name] = dict(primal=original, vjp=vjp_symbol, pullback=pb_symbol,
                            tasks=len(starts), augmented_tasks=augmented,
                            primal_preserved=True, synchronous_pullback=True)
    # The scalar workload VJPs must also have been synthesized without source
    # annotations. A reference without a function body does not satisfy this.
    async_vjps = [m for name, m in new_functions.items()
                  if 'TJr' in name and '@async' in m['type'].split('->', 1)[0]]
    if any(c['augmented_tasks'] for c in cases):
        entries = [m for name, m in new_functions.items()
                   if name.startswith('__swift_async_ad_') and name.endswith('.entry')]
        if not entries:
            raise RuntimeError('No compiler-generated augmented task entries found')
        if not any('workload2' in plain(m['symbol']) for m in async_vjps):
            raise RuntimeError('No body for the unannotated multiplication workload VJP')
    result['_async_vjp_body_count'] = len(async_vjps)
    return result


def runtime_sil(text: str, exported: set[str], inject_yields: bool) -> str:
    """Expose test entry points; optionally add scheduling-only suspension calls."""
    if 'main' in functions(text):
        raise RuntimeError('Unexpected main in library SIL; check -parse-as-library')
    changed = set()
    injected = 0
    def replace(m: re.Match[str]) -> str:
        nonlocal injected
        name = plain(m['symbol'])
        header, body = m['header'], m['body']
        if name in exported:
            # Omitted SIL linkage means public. Do not alter the signature/body.
            header = re.sub(
                r'^sil (?:private|hidden|shared|public_non_abi|package)\s+',
                'sil ', header, count=1)
            changed.add(name)
        if inject_yields and '@async' in m['type'].split('->', 1)[0]:
            entry = re.search(r'^bb\d+[^\n]*:\s*(?://[^\n]*)?\n', body, re.M)
            if not entry:
                raise RuntimeError(f'Cannot locate entry block of {name}')
            calls = (
                '  %async_ad_test_yield_fn = function_ref @__async_ad_test_yield : '
                '$@convention(thin) @async () -> ()\n'
                '  %async_ad_test_yield_result = apply %async_ad_test_yield_fn() : '
                '$@convention(thin) @async () -> ()\n')
            body = body[:entry.end()] + calls + body[entry.end():]
            injected += 1
        suffix = m[0][m[0].rfind('\n}') + 1:]
        return header + '\n' + body + suffix
    result = FUNCTION.sub(replace, text)
    if changed != exported:
        raise RuntimeError(f'Missing runtime exports: {sorted(exported - changed)}')
    if inject_yields:
        if not injected:
            raise RuntimeError('Yield test injected no suspension calls')
        result += ('\n// Test-only scheduling helper, defined in harness.swift.\n'
                   'sil @__async_ad_test_yield : $@convention(thin) @async () -> ()\n')
    return result


def harness(cases: list[dict], symbols: dict[str, tuple[str, str]]) -> str:
    declarations, blocks = [], []
    for index, case in enumerate(cases):
        scalar, n = case['scalar'], case['gradient_arity']
        name = case['name']
        original, vjp = symbols[name]
        effect = ' async' if case['async'] else ''
        await_ = 'await ' if case['async'] else ''
        gradient = scalar if n == 1 else f'({scalar}, {scalar})'
        declarations.append(f'''@_silgen_name("{vjp}")
func vjp_{index}(_ a: {scalar}, _ b: {scalar}){effect} -> ({scalar}, ({scalar}) -> {gradient})
@_silgen_name("{original}")
func primal_{index}(_ a: {scalar}, _ b: {scalar}){effect} -> {scalar}
''')
        if n == 1:
            check = f'checkNear(got, seed * ({case["da"]}), "{name}.da")'
            print_gradient = r'\(pb(1))'
        else:
            check = (f'checkNear(got.0, seed * ({case["da"]}), "{name}.da")\n'
                     f'          checkNear(got.1, seed * ({case["db"]}), "{name}.db")')
            print_gradient = r'\(pb(1))'
        blocks.append(f'''    do {{
      let pairs: [({scalar}, {scalar})] = [(2, 3), (0, 7), (-2, 4), (0.5, -1.25), (8, -0.25)]
      for (a, b) in pairs {{
        let (value, pb) = {await_}vjp_{index}(a, b)
        checkNear(value, {case['value']}, "{name}.value")
        // The original task scopes have ended. Also suspend again before PB use.
        let reference = {await_}primal_{index}(a, b)
        checkNear(value, reference, "{name}.primal")
        await Task.yield()
        let seeds: [{scalar}] = [1, 2, 0, -3, 0.5]
        for seed in seeds {{
          let got = pb(seed)
          {check}
        }}
        // Repeat the first seed after other seeds: residuals must be reusable.
        let got = pb(1)
        let seed: {scalar} = 1
        {check}
        if a == 2 && b == 3 {{
          print("PASS {name}: value=\\(value), gradient={print_gradient}")
        }}
      }}
      for _ in 0..<25 {{
        let a: {scalar} = 2, b: {scalar} = 3
        let (value, pb) = {await_}vjp_{index}(a, b)
        checkNear(value, {case['value']}, "{name}.repeat.value")
        let seed: {scalar} = 2
        let got = pb(seed)
        {check}
        // A separate invocation's pullback is deliberately discarded unused.
        _ = {await_}vjp_{index}(a, b)
      }}
    }}''')
    return '''import _Differentiation

@_silgen_name("__async_ad_test_yield")
nonisolated func yieldForAsyncADTesting() async {
  await Task.yield()
}

func checkNear<T: BinaryFloatingPoint>(_ got: T, _ expected: T, _ label: String) {
  let tolerance: T = T(0.00001) * max(T(1), abs(expected))
  precondition(got.isFinite && abs(got - expected) <= tolerance,
               "\\(label): got \\(got), expected \\(expected)")
}

''' + '\n'.join(declarations) + '''
@main
struct Main {
  nonisolated static func main() async {
''' + '\n'.join(blocks) + '''
    print("ALL ASYNC VJP RUNTIME CHECKS PASSED")
  }
}
'''


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out = args.out.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.commands = self.out / 'commands.log'
        self.commands.write_text('')
        for marker in ('SUCCESS', 'FAILURE.txt'):
            (self.out / marker).unlink(missing_ok=True)

    def command(self, argv: list[str], stem: str, expected_failure: bool = False) -> subprocess.CompletedProcess:
        with self.commands.open('a') as stream:
            stream.write(shlex.join(argv) + '\n')
        try:
            proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, timeout=self.args.timeout, cwd=self.out)
        except subprocess.TimeoutExpired as error:
            def partial(value: str | bytes | None) -> str:
                return value.decode(errors='replace') if isinstance(value, bytes) else (value or '')
            (self.out / f'{stem}.stdout').write_text(partial(error.stdout))
            (self.out / f'{stem}.stderr').write_text(partial(error.stderr))
            (self.out / f'{stem}.status').write_text('TIMEOUT\n')
            raise RuntimeError(f'{stem}: timed out; partial output saved') from error
        (self.out / f'{stem}.stdout').write_text(proc.stdout)
        (self.out / f'{stem}.stderr').write_text(proc.stderr)
        (self.out / f'{stem}.status').write_text(str(proc.returncode) + '\n')
        if proc.returncode < 0 or 'Stack dump:' in proc.stderr:
            raise RuntimeError(f'{stem}: compiler/test crashed; see {stem}.stderr')
        if expected_failure:
            if proc.returncode == 0:
                raise RuntimeError(f'{stem}: unsupported input compiled successfully')
        elif proc.returncode:
            raise RuntimeError(f'{stem}: exit {proc.returncode}; see {stem}.stderr')
        return proc

    def run(self) -> None:
        a = self.args
        self.command([a.frontend, '-version'], 'frontend-version')
        (self.out/'environment.json').write_text(json.dumps(dict(
            platform=platform.platform(), python=sys.version,
            frontend=a.frontend, swiftc=a.swiftc,
            frontend_args=a.frontend_arg, swiftc_args=a.swiftc_arg,
            base_revision='a612de2', draft_compiler_built_here=False),indent=2)+'\n')
        source = HERE/'original.swift' if a.original_only else HERE.parent/'generated.swift'
        cases = json.loads((HERE/'cases.json').read_text())
        if a.original_only:
            cases = [c for c in cases if c['name'] == 'doWor5k']
        elif a.case:
            available = {c['name'] for c in cases}
            unknown = set(a.case)-available
            if unknown: raise RuntimeError(f'Unknown cases: {sorted(unknown)}')
            cases = [c for c in cases if c['name'] in a.case]
        frontend_args = [str(x) for x in a.frontend_arg]
        flag = ['-Xllvm', '-enable-experimental-async-autodiff']
        extra_debug = ['-Xllvm', '-debug-only=differentiation'] if a.debug_differentiation else []
        cmd = [a.frontend, '-emit-sil', '-parse-as-library', '-module-name', MODULE,
               '-sil-verify-all', '-sil-ownership-verify-all', *flag,
               '-Xllvm', '-sil-print-before=differentiation',
               '-Xllvm', '-sil-print-after=differentiation', *extra_debug,
               *frontend_args, str(source), '-o', str(self.out/'final.sil')]
        proc = self.command(cmd, 'compile')
        before, after = extract_phase(proc.stderr,'before'), extract_phase(proc.stderr,'after')
        (self.out/'before.sil').write_text(before)
        (self.out/'after.sil').write_text(after)
        results = structural_checks(before,after,cases)
        (self.out/'structure.json').write_text(json.dumps(results,indent=2)+'\n')
        print('PASS structural checks; before.sil and after.sil saved')
        if not a.sil_only:
            if not a.swiftc:
                raise RuntimeError('--swiftc is required unless --sil-only is specified')
            self.command([a.swiftc,'-version'],'swiftc-version')
            final = (self.out/'final.sil').read_text()
            syms = {c['name']: witness_for(final,c['name']) for c in cases}
            exported = {s for pair in syms.values() for s in pair}
            (self.out/'harness.swift').write_text(harness(cases,syms))
            modes = [('normal',False)] + ([('yield',True)] if a.yield_test else [])
            optimizations = ['Onone','O'] if a.optimized else ['Onone']
            for mode, yields in modes:
                sil = self.out/f'runtime.{mode}.sil'
                sil.write_text(runtime_sil(final,exported,yields))
                for opt in optimizations:
                    stem=f'runtime.{mode}.{opt}'
                    obj=self.out/f'{stem}.o'
                    exe=self.out/(stem+('.exe' if os.name=='nt' else '.exe.out'))
                    self.command([a.frontend,'-parse-sil','-emit-object','-'+opt,
                                  '-sil-verify-all','-sil-ownership-verify-all',
                                  '-module-name',MODULE,*frontend_args,str(sil),'-o',str(obj)],
                                 stem+'.object')
                    self.command([a.swiftc,'-parse-as-library','-'+opt,
                                  *a.swiftc_arg,str(self.out/'harness.swift'),str(obj),'-o',str(exe)],
                                 stem+'.link')
                    run=self.command([str(exe)],stem+'.run')
                    if 'ALL ASYNC VJP RUNTIME CHECKS PASSED' not in run.stdout:
                        raise RuntimeError(f'{stem}: missing success marker')
                    print(run.stdout,end='')
        if a.negative:
            checks=[('CONTROL_FLOW','single-block parent'),
                    ('NONSCALAR','by-value Float and Double'),
                    ('RECURSIVE','recursive child computations'),
                    ('CUSTOM_DERIVATIVE','custom or pre-existing opaque derivatives')]
            for define, message in checks:
                proc=self.command([a.frontend,'-emit-sil','-parse-as-library',
                                   '-D',define,*flag,*frontend_args,
                                   str(HERE.parent/'diagnostics.swift'),'-o',str(self.out/'negative.sil')],
                                  'negative.'+define,True)
                if 'cannot differentiate this async let:' not in proc.stderr or message not in proc.stderr:
                    raise RuntimeError(f'{define}: failed for a different reason; inspect diagnostics')
            proc=self.command([a.frontend,'-emit-sil','-parse-as-library',*frontend_args,
                               str(HERE/'original.swift'),'-o',str(self.out/'disabled.sil')],
                              'negative.flag-disabled',True)
            if 'cannot differentiate this async let:' not in proc.stderr:
                raise RuntimeError('Missing fail-closed diagnostic with the feature flag disabled')
            print('PASS unsupported-input and disabled-flag diagnostics')
        (self.out/'SUCCESS').write_text('All requested checks passed on the user-built compiler.\n')
        print(f'Results: {self.out}')


def self_test() -> None:
    sample='''sil_stage canonical
sil_differentiability_witness hidden [reverse] [parameters 0 1] [results 0] @$s13AsyncVJPTests7doWor5kyS2f_SftYaF : $@convention(thin) @async (Float, Float) -> Float {
  vjp: @$s13AsyncVJPTests7doWor5kyS2f_SftYaFTJrSSpSr
}
sil hidden [ossa] @$s13AsyncVJPTests7doWor5kyS2f_SftYaF : $@convention(thin) @async (Float, Float) -> Float {
bb0(%0 : $Float, %1 : $Float):
  return %0 : $Float
} // end sil function
sil hidden [ossa] @$s13AsyncVJPTests7doWor5kyS2f_SftYaFTJrSSpSr : $@convention(thin) @async (Float, Float) -> (Float, @owned @callee_guaranteed (Float) -> (Float, Float)) {
bb0(%0 : $Float, %1 : $Float):
  unreachable
} // end sil function
'''
    symbols=witness_for(sample,'doWor5k')
    assert len(functions(sample))==2
    transformed=runtime_sil(sample,set(symbols),True)
    assert transformed.count('apply %async_ad_test_yield_fn()')==2
    assert all(not m['header'].startswith('sil hidden') for m in functions(transformed).values())
    phases=('  *** SIL module before #1: Differentiation (differentiation)\n'+sample+
            '  *** SIL module after #1: Differentiation (differentiation)\n'+sample)
    assert extract_phase(phases,'before')==sample
    assert extract_phase(phases,'after')==sample
    cases=json.loads((HERE/'cases.json').read_text())
    generated=harness([cases[0]],{'doWor5k':symbols})
    assert '@_silgen_name' in generated and '\\(value)' in generated
    assert 'seed * (1 + b)' in generated
    print('PASS Python parser/generator self-tests (no compiler invocation)')


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frontend',help='Patched swift-frontend executable')
    p.add_argument('--swiftc',help='Matching swiftc driver executable')
    p.add_argument('--out',type=Path,default=Path('async-vjp-results'))
    p.add_argument('--frontend-arg',action='append',default=[],help='Repeat, e.g. --frontend-arg=-sdk --frontend-arg=/SDK')
    p.add_argument('--swiftc-arg',action='append',default=[],help='Matching SDK/target flags for the harness driver')
    p.add_argument('--original-only',action='store_true',help='Compile only the exact original example')
    p.add_argument('--case',action='append',help='Filter checks/runtime cases; the full source is still compiled')
    p.add_argument('--sil-only',action='store_true')
    p.add_argument('--optimized',action='store_true',help='Also optimize the emitted SIL and harness with -O')
    p.add_argument('--yield-test',action='store_true',help='Also test a scheduling-only instrumented copy of emitted SIL')
    p.add_argument('--negative',action='store_true',help='Also run diagnostic fixtures and flag-disabled test')
    p.add_argument('--debug-differentiation',action='store_true',help='Assertions-enabled compiler builds only')
    p.add_argument('--timeout',type=int,default=180)
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:
        self_test(); return 0
    if not args.frontend:
        p.error('--frontend is required')
    args.frontend=os.path.abspath(args.frontend)
    if args.swiftc: args.swiftc=os.path.abspath(args.swiftc)
    runner=Runner(args)
    try:
        runner.run()
        return 0
    except Exception as exc:
        (runner.out/'FAILURE.txt').write_text(traceback.format_exc())
        print(f'FAIL: {exc}\nEvidence saved in {runner.out}',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
