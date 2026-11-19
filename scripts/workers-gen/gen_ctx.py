"""원본 fetch 처리기의 `var` 를 모두 모아 Ctx 구조체로. usage: python gen_ctx.py <index.ts> <out ctx.rs> [시작 줄 끝 줄] (기본: proxy 2354 6487)"""
import re
import sys

src = open(sys.argv[1], encoding='utf-8').read().split('\n')
start, end = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (2354, 6487)
body = '\n'.join(src[start - 1:end])
names = set(re.findall(r'\bvar\s+([A-Za-z_$][\w$]*)', body))
for m in re.finditer(r'\bvar\s*\{([^}]*)\}', body):
    # 구조 분해 안의 주석(// … · /* … */)은 걷어 낸다
    for p in re.sub(r'//[^\n]*|/\*.*?\*/', '', m.group(1), flags=re.S).split(','):
        p = p.strip()
        if ':' in p:
            p = p.split(':')[1].strip()
        if re.fullmatch(r'[A-Za-z_$][\w$]*', p):
            names.add(p)

KEYWORDS = {'type', 'match', 'ref', 'in', 'fn', 'loop', 'move', 'use', 'where', 'self', 'mod', 'impl', 'let', 'mut', 'pub', 'static', 'struct', 'trait', 'enum', 'const', 'crate', 'super', 'as', 'break', 'continue', 'else', 'extern', 'false', 'true', 'for', 'if', 'return', 'unsafe', 'while', 'async', 'await', 'dyn', 'abstract', 'become', 'box', 'do', 'final', 'macro', 'override', 'priv', 'typeof', 'unsized', 'virtual', 'yield', 'try'}


def field(n):
    f = n.replace('$', 's_')
    if f in KEYWORDS:
        f = f + '_'
    return f


rows = sorted(names, key=lambda n: n.lower())
out = ['//! 원본 fetch 처리기의 `var` 전부 (gen_ctx.py 로 생성 — 손으로 고치지 말 것).',
       '//!',
       '//! JS 의 `var` 는 함수 범위라 반복을 넘어 값이 남고, 안쪽 반복이 바깥 반복의 셈 변수(`t` · `i` …)를 같이 쓴다.',
       '//! 그대로 옮기려고 모두 이 구조체의 칸으로 두고 구역 함수들이 `&mut Ctx` 로 나눠 쓴다. `$x` → `s_x`, Rust 예약어는 끝에 `_`.',
       '#![allow(non_snake_case)]',
       '',
       'use logis_worker_common::V;',
       '',
       '#[derive(Default)]',
       'pub struct Ctx {']
for n in rows:
    out.append(f'    /// `var {n}`')
    out.append(f'    pub {field(n)}: V,')
out.append('}')
open(sys.argv[2], 'w', encoding='utf-8', newline='\n').write('\n'.join(out) + '\n')
print(len(rows), 'vars;', ', '.join(f'{n}->{field(n)}' for n in rows if field(n) != n))
