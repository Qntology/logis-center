"""proxy/src/index.ts 의 표(LogisRegion · languageCode · languageCodeToCountryCode · Hello · twoPartDomains · tables)를
Rust 로 옮긴다. usage: python gen_tables.py <index.ts> <out tables.rs>"""
import re
import sys

src = open(sys.argv[1], encoding='utf-8').read()


def block(name):
    m = re.search(r'^const ' + re.escape(name) + r' = \{\n(.*?)^\};?', src, re.S | re.M)
    if not m:
        raise SystemExit(f'no table {name}')
    body = m.group(1)
    pairs = re.findall(r"""^\s*['"]([^'"]+)['"]\s*:\s*['"]([^'"]*)['"]""", body, re.M)
    return pairs


def arr(name):
    m = re.search(r'^const ' + re.escape(name) + r' = \[(.*?)\];?', src, re.S | re.M)
    return re.findall(r"""['"]([^'"]+)['"]""", m.group(1))


def rs_str(s):
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'


out = ['//! 원본 index.ts 의 표 (scripts 로 생성: 손으로 고치지 말 것 — 원본 표가 바뀌면 다시 생성)', '']


def table(name, rs_name, doc):
    pairs = block(name)
    out.append(f'/// `{name}` — {doc}')
    out.append(f'pub const {rs_name}: &[(&str, &str)] = &[')
    for k, v in pairs:
        out.append(f'    ({rs_str(k)}, {rs_str(v)}),')
    out.append('];')
    out.append('')


table('LogisRegion', 'LOGIS_REGION', '국가 코드 → 지역 D1 바인딩')
table('languageCode', 'LANGUAGE_CODE', '국가 코드 → 언어 이름')
table('languageCodeToCountryCode', 'LANGUAGE_CODE_TO_COUNTRY_CODE', '언어 코드 → 국가 코드')
table('Hello', 'HELLO', '언어 → 인사말')
out.append('/// `twoPartDomains`')
out.append('pub const TWO_PART_DOMAINS: &[&str] = &[' + ', '.join(rs_str(x) for x in arr('twoPartDomains')) + '];')
out.append('')
out.append('/// `tables`')
out.append('pub const TABLES: &[&str] = &[' + ', '.join(rs_str(x) for x in arr('tables')) + '];')
out.append('')
out.append('''/// 표 찾기 (`Table[key]`, 없으면 undefined)
pub fn lookup(table: &[(&str, &str)], key: &crate::jsv::V) -> crate::jsv::V {
    let k = key.to_key();
    table.iter().rev().find(|(x, _)| *x == k).map_or(crate::jsv::V::Undef, |(_, v)| crate::jsv::V::from(*v))
}''')
out.append('')
out.append('pub const CENTER_REGION: &str = "commerce_logis_center";')
open(sys.argv[2], 'w', encoding='utf-8', newline='\n').write('\n'.join(out) + '\n')
print({n: len(block(n)) for n in ['LogisRegion', 'languageCode', 'languageCodeToCountryCode', 'Hello']})
