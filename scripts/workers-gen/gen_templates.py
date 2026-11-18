"""원본 index.ts 의 모든 템플릿 리터럴(`...${x}...`)을 Rust 상수로 뽑는다.

각 리터럴은 `pub const T<줄>[_<n>]: &[&str]` — 조각(JS 처럼 이스케이프를 푼 글) 목록이고, 조각 사이에 들어갈 식은 주석에 원문 그대로 둔다.
옮긴 코드는 `tpl(T3181, &[a, b])` 로 원본과 같은 글을 만든다 (조각 수 = 식 수 + 1).
usage: python gen_templates.py <index.ts> <out templates.rs> [<index json>]
"""
import json
import re
import sys

src = open(sys.argv[1], encoding='utf-8').read()

REGEX_PREV = set('(,=:[!&|?{};+-*%<>~^')
KEYWORDS_BEFORE_REGEX = ('return', 'typeof', 'case', 'do', 'else', 'in', 'of', 'new', 'delete', 'void', 'throw')


def cook(raw):
    """JS 템플릿 조각의 이스케이프 풀기"""
    out = []
    i = 0
    while i < len(raw):
        c = raw[i]
        if c != '\\':
            out.append(c)
            i += 1
            continue
        n = raw[i + 1]
        simple = {'n': '\n', 't': '\t', 'r': '\r', 'b': '\b', 'f': '\f', 'v': '\v', '0': '\0', '`': '`', '$': '$', '\\': '\\', "'": "'", '"': '"'}
        if n in simple:
            out.append(simple[n])
            i += 2
        elif n == '\n':
            i += 2
        elif n == 'u':
            if raw[i + 2] == '{':
                j = raw.index('}', i)
                out.append(chr(int(raw[i + 3:j], 16)))
                i = j + 1
            else:
                out.append(chr(int(raw[i + 2:i + 6], 16)))
                i += 6
        elif n == 'x':
            out.append(chr(int(raw[i + 2:i + 4], 16)))
            i += 4
        else:
            out.append(n)
            i += 2
    # \r\n 은 템플릿 안에서 \n 으로 (JS 규칙)
    return ''.join(out).replace('\r\n', '\n').replace('\r', '\n')


def line_of(pos):
    return src.count('\n', 0, pos) + 1


templates = []  # (line, pieces, exprs)


def skip_string(i, q):
    i += 1
    while src[i] != q:
        if src[i] == '\\':
            i += 2
        else:
            i += 1
    return i + 1


def skip_regex(i):
    i += 1
    in_class = False
    while True:
        c = src[i]
        if c == '\\':
            i += 2
            continue
        if c == '[':
            in_class = True
        elif c == ']':
            in_class = False
        elif c == '/' and not in_class:
            i += 1
            break
        elif c == '\n':
            raise SystemExit(f'unterminated regex at line {line_of(i)}')
        i += 1
    while i < len(src) and (src[i].isalnum() or src[i] == '_'):
        i += 1
    return i


def read_template(i):
    """src[i] == '`'. 끝 다음 위치를 돌려주고 templates 에 넣는다 (안의 템플릿도 재귀로)"""
    start = i
    i += 1
    pieces, exprs = [], []
    cur = []
    while True:
        c = src[i]
        if c == '\\':
            cur.append(src[i:i + 2])
            i += 2
            continue
        if c == '`':
            pieces.append(cook(''.join(cur)))
            templates.append((line_of(start), start, pieces, exprs))
            return i + 1
        if c == '$' and src[i + 1] == '{':
            pieces.append(cook(''.join(cur)))
            cur = []
            j = scan(i + 2, stop_at_brace=True)
            exprs.append(src[i + 2:j].strip())
            i = j + 1
            continue
        cur.append(c)
        i += 1


def scan(i, stop_at_brace=False):
    depth = 0
    prev = ''  # 마지막 의미 있는 글자 / 낱말
    while i < len(src):
        c = src[i]
        if c in ' \t\r\n':
            i += 1
            continue
        if src.startswith('//', i):
            i = src.index('\n', i)
            continue
        if src.startswith('/*', i):
            i = src.index('*/', i) + 2
            continue
        if c in '\'"':
            i = skip_string(i, c)
            prev = 'str'
            continue
        if c == '`':
            i = read_template(i)
            prev = 'str'
            continue
        if c == '/':
            if prev == '' or prev in REGEX_PREV or prev in KEYWORDS_BEFORE_REGEX:
                i = skip_regex(i)
                prev = 'regex'
                continue
            i += 1
            prev = '/'
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            if stop_at_brace and depth == 0:
                return i
            depth -= 1
        if c.isalnum() or c in '_$':
            j = i
            while j < len(src) and (src[j].isalnum() or src[j] in '_$'):
                j += 1
            prev = src[i:j]
            i = j
            continue
        prev = c
        i += 1
    return i


scan(0)
templates.sort(key=lambda t: t[1])


def rs_str(s):
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r').replace('\0', '\\0') + '"'


out = ['//! 원본 proxy/src/index.ts 의 모든 템플릿 리터럴 (gen_templates.py 로 생성 — 손으로 고치지 말 것).',
       '//! `T<줄>[_<n>]` 은 조각 목록이고 조각 사이의 식은 주석의 원문이다. [`crate::tpl`] 로 잇는다.',
       '#![allow(dead_code)]', '']
names = {}
index = []
for line, pos, pieces, exprs in templates:
    base = f'T{line}'
    k = names.get(base, 0)
    names[base] = k + 1
    name = base if k == 0 else f'{base}_{k}'
    assert len(pieces) == len(exprs) + 1
    if exprs:
        out.append('/// ' + ' | '.join('${' + e.replace('\n', ' ') + '}' for e in exprs))
    out.append(f'pub const {name}: &[&str] = &[{", ".join(rs_str(p) for p in pieces)}];')
    index.append({'name': name, 'line': line, 'exprs': exprs, 'preview': (pieces[0][:60])})
open(sys.argv[2], 'w', encoding='utf-8', newline='\n').write('\n'.join(out) + '\n')
if len(sys.argv) > 3:
    json.dump(index, open(sys.argv[3], 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
dups = {k: v for k, v in names.items() if v > 1}
print(f'{len(templates)} templates, lines with more than one: {len(dups)}')
