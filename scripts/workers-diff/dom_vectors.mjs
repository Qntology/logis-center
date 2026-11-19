// linkedom 0.18.11 (htmlparser2 10.1.0) 기준 DOM 시험 벡터를 만든다 → workers/proxy/tests/dom_vectors.json
//
//   node dom_vectors.mjs <SyntheticData/commerce/output_html 폴더> <출력 json>
//
// 기록하는 것: 원본 convertHtmlToCleanPug(body) 출력, 선택자마다 querySelectorAll 개수 · 첫 일치의 textContent · 예외 여부.
// 문자열의 외톨이 서로게이트(이름 엔터티 &Afr; 등이 글 노드 둘로 갈라질 때)는 toWellFormed() 로 U+FFFD 로 바꿔 적는다.
import { parseHTML, Node } from 'linkedom';
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

// ───── 원본(801b0ce0 workers/proxy/src/index.ts 1025–1158) 그대로 ─────
function convertHtmlToCleanPug(body) {
	try {
		var { document } = parseHTML(`<html><body>${body}</body></html>`);

		const pugLines = generatePugLines(document.body.childNodes, 0);

		return pugLines.join('\n');

	} catch (error) {
		console.error('변환 중 오류가 발생했습니다:', error);
		return '';
	}
}

function generatePugLines(nodes, indentLevel) {
	// 들여쓰기 문자 (공백 4칸)
	const indent = '    '.repeat(indentLevel);
	let lines = [];

	nodes.forEach(node => {
		// 1. Element 노드 처리
		if (node.nodeType === Node.ELEMENT_NODE) {
			const tagName = node.tagName.toLowerCase();

			// --- base64 이미지를 포함하는 img 태그 제외 ---
			const src = node.getAttribute('src');
			if (tagName === 'img' && src && src.includes('base64')) {
				return; // src에 'base64'가 포함된 img 태그는 변환에서 건너뜁니다.
			}

			// 불필요한 태그들을 만나면 건너뛰기
			// input, textarea는 이제 포함됩니다.
			if (['script', 'style', 'link', 'noscript', 'iframe'].includes(tagName)) {
				return;
			}

			// --- 허용된 속성만 Pug 문법으로 변환 ---
			let attributesString = '';
			const otherAttributes = [];

			// ID 속성 처리 (#my-id)
			if (node.id) {
				attributesString += `#${node.id}`;
			}

			// Class 속성 처리 (.class1.class2)
			if (node.classList.length > 0) {
				attributesString += `.${Array.from(node.classList).join('.')}`;
			}

			// NamedNodeMap을 Array로 변환하여 모든 속성을 순회합니다.
			Array.from(node.attributes).forEach(attr => {
				const attrName = attr.name;
				const attrValue = attr.value;

				// 기본적으로 포함할 속성들: input, a, img, textarea의 주요 속성 포함
				const alwaysInclude = [
					'src', 'href', 'type', 'name', 'value', 'placeholder',
					'checked', 'selected', 'disabled', 'readonly', 'rows', 'cols'
				];

				// ID와 Class는 이미 처리되었으므로 제외
				if (attrName === 'id' || attrName === 'class') {
					return;
				}

				if (attrName.startsWith('data-') || alwaysInclude.includes(attrName)) {
					// Boolean 속성 처리 (ex: disabled, checked, readonly)
					if (['checked', 'selected', 'disabled', 'readonly'].includes(attrName) && (attrValue === '' || attrValue === attrName)) {
						otherAttributes.push(`${attrName}`); // 값 없이 속성 이름만 추가 (Pug의 Boolean 속성 표기)
					} else if (attrValue) { // 속성값이 비어있지 않은 경우에만 추가
						// 따옴표 안에 따옴표가 있는 경우 이스케이프 필요 (여기서는 단순하게 큰따옴표로 처리)
						const safeValue = attrValue.replace(/"/g, "'");
						otherAttributes.push(`${attrName}="${safeValue}"`);
					}
				}
			});
			// --- 속성 처리 끝 ---

			// 괄호로 묶는 속성들 추가
			if (otherAttributes.length > 0) {
				attributesString += `(${otherAttributes.join(' ')})`;
			}


			// div 축약 로직은 그대로 유지
			let currentNode = node;
			while (
				currentNode.tagName === 'DIV' &&
				Array.from(currentNode.childNodes).filter(n => n.nodeType === Node.ELEMENT_NODE || n.nodeValue.trim()).length === 1 &&
				currentNode.firstElementChild?.tagName === 'DIV'
			) {
				currentNode = currentNode.firstElementChild;
			}

			// 태그 이름과 변환된 속성 문자열을 함께 추가
			lines.push(`${indent}${tagName}${attributesString}`);

			// textarea의 값 처리 (node.value 사용)
			if (tagName === 'textarea') {
				const value = node.value;
				if (value.trim()) {
					// 여러 줄 텍스트 처리를 위해 각 줄을 '| '로 시작
					value.split('\n').forEach(line => {
						lines.push(`${indent}    | ${line}`);
					});
				}
			}
			// 자식 노드 처리
			else if (currentNode.hasChildNodes()) {
				// textarea는 값 처리가 완료되었으므로, 자식 노드를 추가로 처리할 필요는 없습니다.
				// (일반적으로 textarea의 텍스트는 childNodes로도 잡히지만, value로 처리하는 것이 정확합니다.)
				if (tagName !== 'textarea') {
					lines = lines.concat(generatePugLines(currentNode.childNodes, indentLevel + 1));
				}
			}

		} else if (node.nodeType === Node.TEXT_NODE) {
			const textContent = node.nodeValue.trim();
			if (textContent) {
				lines.push(`${indent}| ${textContent}`);
			}
		}
	});

	return lines;
}
// ───── 원본 끝 ─────

const wf = (s) => (typeof s === 'string' ? s.toWellFormed() : s);

// 모든 문서에 돌리는 선택자 (LLM 이 내는 꼴 + 대소문자 · 구조 의사 클래스 · 잘못된 선택자)
const COMMON = [
	'div', 'p', 'a', 'li', 'span', 'img', 'td', 'tr', 'table', 'b', 'em', 'h1', 'h2', 'button', 'input', 'form', 'select',
	'ul > li', 'ol > li', 'div > p', 'a[href]', 'img[src]', '[class]', '[id]', '*', 'body > *', 'html > body', 'head', 'html',
	'div div', 'div > div > div', 'li:first-child', 'li:last-child', 'li:nth-child(2)', 'li:nth-child(odd)', 'li:nth-of-type(2n+1)',
	'tr:nth-child(even) td', 'td:nth-last-child(1)', 'p:empty', 'div:empty', 'b:only-child', 'span:only-of-type', 'span + b', 'span ~ b',
	'div:not(.x)', 'div:has(> a)', 'li:has(img)', 'a:is([href^="/"], [href^="http"])', ':where(h1, h2) + *', '[data-id]', '[DATA-ID]',
	'[type="checkbox" i]', 'input[type=CHECKBOX]', 'input[type="checkbox" s]', 'svg', 'path', 'lineargradient', 'linearGradient', 'svg *',
	'[viewBox]', '[viewbox]', 'DIV', 'Div > P', 'textarea', 'script', 'style', 'title', 'option', 'select > option', 'template',
	'template p', 'template + p', '#main', '#X', '#x', '.item', '.a.b', '.a', '.A', 'div.item > a', 'h1, h2, h3', 'meta, link', 'br', 'hr',
	'input[checked]', 'option[selected]', '[class~="x"]', '[class*="it"]', '[class^="it"]', '[class$="em"]', '[href|="en"]',
	'[href$=".html"]', '[src*="base64"]', '[name="q"]', '[value=""]', '[class=""]', ':root', 'html:root', 'body:first-child',
	'a[href]:not([href^="#"])', 'ul li a', 'main .card .name', 'li.card:nth-child(3) .meta a', '.card-meta li:nth-child(2) b',
	'*:not(div):not(span)', 'p:first-of-type', 'p:last-of-type', 'li:nth-last-of-type(2)', 'img + strong', 'em:empty',
	// 잘못된 선택자 — 양쪽 모두 던져야 한다
	'div[', '>>', '', '[href=', 'a..b', '#', 'div:nth-child(', '}',
	// css-what 이 너그럽게 받는 꼴 · css-select 가 던지는 꼴
	' ', '#1a', '.2b', 'div#1a.2b', '.-x', '.-1', 'div. a', 'a.:x', '[class|=""]', ':scope', ':scope > div', 'div /* c */ p', 'svg|rect',
	'|div', '*|div', "[class='A B' i]", '#\\31 a', '::before', 'div:first', 'div,', ',div', '.', 'div.', '#1a > span',
];

const sampleFiles = (dir) => {
	const all = readdirSync(dir).filter((f) => f.endsWith('.html')).sort();
	return all.filter((_, i) => i % 4 === 0);
};

// 문서에서 뽑은 선택자 (자주 나오는 class 6개 · 첫 id · 첫 data-* 속성)
const docSelectors = (html) => {
	const { document } = parseHTML(`<html><body>${html}</body></html>`);
	const counts = new Map();
	let firstId = null;
	let firstData = null;
	for (const el of document.querySelectorAll('*')) {
		for (const c of el.classList) counts.set(c, (counts.get(c) || 0) + 1);
		if (!firstId && el.id && /^[A-Za-z][\w-]*$/.test(el.id)) firstId = el.id;
		if (!firstData) {
			for (const a of Array.from(el.attributes)) {
				if (a.name.startsWith('data-') && /^[a-z][\w-]*$/.test(a.name)) {
					firstData = a.name;
					break;
				}
			}
		}
	}
	const classes = [...counts.entries()]
		.filter(([c]) => /^[A-Za-z_][\w-]*$/.test(c))
		.sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
		.slice(0, 6)
		.map(([c]) => c);
	const out = [];
	for (const c of classes) {
		out.push(`.${c}`, `div.${c}`, `.${c} > *`, `.${c} a`, `.${c}:nth-child(2)`, `.${c}:first-of-type`, `[class="${c}"]`);
	}
	if (classes.length >= 2) out.push(`.${classes[0]}, .${classes[1]}`, `.${classes[0]} .${classes[1]}`, `.${classes[0]} ~ .${classes[1]}`);
	if (firstId) out.push(`#${firstId}`, `#${firstId} *`);
	if (firstData) out.push(`[${firstData}]`, `div[${firstData}]`);
	return out;
};

// 긴 textContent 는 길이(UTF-16) + UTF-8 바이트의 FNV-1a 64 로 줄여 적는다
const fnv64 = (s) => {
	let h = 0xcbf29ce484222325n;
	for (const b of Buffer.from(s, 'utf8')) {
		h ^= BigInt(b);
		h = (h * 0x100000001b3n) & 0xffffffffffffffffn;
	}
	return h.toString(16).padStart(16, '0');
};
const textRecord = (t) => (t.length <= 120 ? t : { n: t.length, h: fnv64(t) });

// 결과: 예외면 null, 아니면 [개수, 첫 일치의 textContent(없으면 null)]
const runSelectors = (html, selectors) => {
	// 선택자용 문서는 따로 (document.body 를 읽으면 head 가 생겨 트리가 바뀐다)
	const { document } = parseHTML(`<html><body>${html}</body></html>`);
	const results = [];
	for (const sel of selectors) {
		try {
			const list = document.querySelectorAll(sel);
			results.push([list.length, list.length ? textRecord(wf(list[0].textContent)) : null]);
		} catch (e) {
			results.push(null);
		}
	}
	return results;
};

// ───── 손으로 쓴 경계 사례 ─────
const deep = (n) => '<div class="d">'.repeat(n) + 'deep' + '</div>'.repeat(n);
const EDGE = [
	['entities_text', '<p>a &amp; b &lt;c&gt; &copy 2024 &notit; &notin; &#65;&#x42;&#0; &#128512; &AMP &amp</p><p>&nbsp;x&nbsp</p><p>&ampx &ltb &gt;&gtc</p>'],
	['entities_attr', '<a href="?a=1&amp;b=2&copy=3&copy;&notit=4&lt" title=\'&lt;x&gt;\' data-x=&quot;q>t</a><a data-y="&amp &ampx &amp=">u</a>'],
	['entity_edges', '<p>AT&T</p><p>&</p><p>&#</p><p>&#x;</p><p>&#xZZ</p><p>&#99999999;</p><p>&#x80;&#150;</p><p>&;</p><p>a&b;c</p>'],
	['astral_entity', '<p>&Afr; &Bopf;x</p><a title="&Afr;&Bopf;" data-a="&Afr;">y</a><div>&Afr;</div>'],
	['multi_codepoint_entity', '<p>&NotEqualTilde; &nGt; &ThickSpace;</p><span data-t="&nGt;">z</span>'],
	['legacy_attr_entities', '<a href="?x=1&copy=2&copyx&copy.&copy">l</a><img src="a.png?w=1&times=2&timesb">'],
	['unclosed_p', '<p>one<p>two<div>three</div>four<p>five<h1>six</h1>seven'],
	['unclosed_li', '<ul><li>a<li>b<ul><li>c</ul><li>d</ul><ol><li>e<li>f</ol>'],
	['unclosed_td', '<table><tr><td>1<td>2<tr><td>3<th>h<td>4</table>'],
	['unclosed_option', '<select name="s"><option>a<option selected>b<optgroup label=g><option value="c">c</select><datalist><option>x</datalist>'],
	['stray_p', '<div>x</p>y</div></p><span>z</span>'],
	['stray_br', 'a</br>b<br/>c<br>d</br >e<BR>f'],
	['nested_tables', '<table class="outer"><tr><td><table class="inner"><tr><td>in</td></tr></table></td><td>out</td></tr></table><table><tbody><tr><td>1</td></tr><tbody><tr><td>2</td></tr></table>'],
	['script_raw', '<script>if (a < b && c > d) { x = "</div>"; y = "<p>"; }</script><p>after</p><script type="x">1</SCRIPT ><p>after2</p>'],
	['style_raw', '<style>a > b { color: red } /* <p> */</style><p>x</p><style/><p>y</p>'],
	['textarea_raw', '<textarea name=t rows=3 cols="20">line1 <b>not bold</b>\nline2 &amp; x\n\n  indented</textarea><p>after</p><textarea>   </textarea><textarea></textarea>'],
	['title_entities', '<title>A &amp; B <i>x</i></title><p>after</p><xmp><b>raw</b></xmp><tmp><b>q</b></xmp>y'],
	['comments', '<div><!-- c1 --><span>x</span><!----><!--><!---><!-- a -- b --></div><div><!--only--></div><div> <!-- c --> <div><p>x</p></div></div>'],
	['comment_unterminated', '<p>a</p><!-- open <p>b</p>'],
	['doctype_full', '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>t</title><link rel="x" href="y"></head><body class="b"><p>x</p></body></html>'],
	['cdata_pi_decl', '<div><![CDATA[ x < y ]]></div><p>z</p><?xml version="1.0"?><p>pi</p><!foo bar><p>decl</p></3><p>c</p><!><p>bang</p>'],
	['svg_camel', '<svg viewBox="0 0 10 10" class="icon big" xmlns="http://www.w3.org/2000/svg"><linearGradient id="g"><stop offset="0"/></linearGradient><path d="M0 0"/><circle r=1></circle><foreignObject><div>fo</div><p/></foreignObject></svg><p>after</p><SVG><g/></SVG><div/>x'],
	['svg_title', '<svg><title>&amp;x <b>t</b></title><text>t</text></svg><math><mi>x</mi><mspace/></math><p>m</p>'],
	['svg_textarea_error', '<p>before</p><svg><textarea>x</textarea></svg>'],
	['uppercase', '<DIV CLASS="Up" ID="X"><A HREF="/x">Link</A><IMG SRC="a.png"><Span Class="s">s</SPAN></DIV><p>x</P><B>bold</b>'],
	['self_closing_div', '<div class="a"/><p>inside?</p></div><span/>q<a/>w'],
	['attrs_unquoted', '<input type=checkbox checked disabled name=q value=1 readonly><input type="text" value="" placeholder=hi><input value="a"b" data-k=v/w>'],
	['duplicate_classes', '<div class=" a  b a\tc ">x</div><div class="">y</div><div class>z</div><div class="x　y z">w</div>'],
	['base64_img', '<img src="data:image/png;base64,AAA="><img src="/a.jpg" alt=x><img SRC="data:base64"><img>'],
	['data_attrs', '<div data-id="7" data-name=\'n "q"\' data-empty="" data-Upper="U">x</div><div data-id=8>y</div>'],
	['boolean_attrs', '<select><option selected="selected">a</option><option selected="no">b</option></select><input checked="checked"><input disabled="false"><input readonly="READONLY">'],
	['div_collapse', '<div class="o"><div class="m"><div class="i"><span>t</span></div></div></div><div><div>a</div><div>b</div></div><div id="k">  <div>only</div>  </div><div><div></div></div><div>txt<div>x</div></div>'],
	['div_collapse_comment', '<div><!-- note --><div><p>x</p></div></div><div><!----><div><p>y</p></div></div>'],
	['whitespace_only', '<div>   </div>\n\n<p> \t </p>\n<span> </span>　<b>﻿</b>'],
	['lt_text_split', '<p>1 < 2 and 3 <4 and 5 </ 6 and 7 </> 8 <= 9</p>'],
	['stray_end_tags', '</div></span>text</foo><p>p</p></html>after-html'],
	['form_implies', '<form><input name=a><select><option>1</select><button>b<button>c<textarea>t</textarea><input name=z></form>'],
	['headings_close_p', '<p>x<h1>y</h1>z<p>q<ul><li>l</ul><p>r<table><tr><td>t</td></tr></table>'],
	['template', '<template><p class="t">in template</p><div><span>deep</span></div></template><p class="t">out</p>'],
	['dup_attr_names', '<a href="1" HREF="2" href="3" Class="c" class="d" class="e">x</a><div id="" class="k">e</div><div ID="up" id="low">u</div>'],
	['numeric_attr_names', '<div 2="b" a="x" 1="a" __proto__="p" 10="t" 01="z">x</div>'],
	['unterminated_attr', '<p>a</p><div class="x>text</div><p>b</p>'],
	['unclosed_script', '<p>a</p><script>var x = 1; </scr'],
	['crlf_and_nul', '<p>a\r\nb</p>\r\n<textarea>l1\r\nl2</textarea>\r\n<div>\u0000z</div>'],
	['unicode', '<div class="상품 목록 상품">가격 ₩1,000 &amp; 무료배송</div><ul class="목록"><li>사과</li><li>배</li></ul><dıv>dotless</dıv><ß>sz</ß>'],
	['body_html_nested', '<body class=inner><p>b</p></body><p>after body</p><html><p>h</p></html><head><title>t</title></head>'],
	['deep_nesting', deep(300)],
	['li_inline', '<ul class="list"><li class="item"><a href="/a">A</a> <span class="price">1</span></li><li class="item"><a href="/b">B</a><span class="price">2</span></li></ul>'],
	['rt_dd', '<ruby>a<rt>b<rp>c</ruby><dl><dt>t<dd>d<dt>t2</dl>'],
	['noscript_iframe', '<noscript><p>ns</p></noscript><iframe src="x"></iframe><link href="a"><p>kept</p>'],
	['special_seq_quirks', '<sc>a</sc><s>b</s><t>c</t><te>d</te><xi>e</xi><scriptx>f</scriptx><stylex>g</stylex><textareas>h</textareas>'],
	['self_closing_special', '<script/><p>after script</p></script><textarea/><b>x</b></textarea><title/>t</title>'],
	['attr_whitespace', '<a   href = "x"  title= y  data-z =\'w\'  >l</a><input\nname="n"\tvalue=v>'],
	['empty', ''],
	['text_only', 'just text &amp; more'],
	['digit_names', '<div id="1a" class="2b -x -1 A B"><span>digits</span></div><div class=".b">dot</div><div class=" a">sp</div><a class=":x">c</a>'],
];

// ───── 무작위 조각 잇기 (결정적 난수) — 토크나이저 상태 전이를 넓게 훑는다 ─────
const FUZZ_COUNT = Number(process.env.FUZZ_COUNT || 300);
const FUZZ_SEED = Number(process.env.FUZZ_SEED || 0);
const FUZZ_TOKENS = [
	'<', '>', '/', '</', '/>', '!', '-', '--', '?', '&', ';', '#', 'x', '=', '"', "'", ' ', '  ', '\n', '\t', '\r\n', 'a', 'b', 'p',
	'amp', 'lt', 'gt', 'copy', 'Afr', 'nGt', 'notin', 'not', '#x41', '#65', '#', '&amp;', '&copy', '&Afr;', '&#128512;', '&#x80;',
	'<div>', '</div>', '<div class="a b">', '<div id=x>', '<p>', '</p>', '<li>', '</li>', '<ul>', '</ul>', '<td>', '<tr>', '<table>',
	'</table>', '<svg>', '</svg>', '<path/>', '<g>', '<rect/>', '<math>', '<mi>', '<foreignObject>', '<script>', '</script>',
	'<style>', '</style>', '<title>', '</title>', '<textarea>', '</textarea>', '<xmp>', '</xmp>', '<br>', '</br>', '<br/>',
	'<img src="data:base64,x">', '<img src=/a.png>', '<input type=checkbox checked>', '<option>', '<select>', '</select>',
	'<!--', '-->', '<![CDATA[', ']]>', '<!DOCTYPE html>', '<?pi?>', '</3>', '<template>', '</template>', '<body>', '</body>',
	'<html>', '</html>', '<head>', 'class', 'id', 'src', 'data-x', 'base64', 'DIV', 'SPAN', '<SPAN CLASS=Q>', '</Span>',
	'é', '한글', '😀', ' ', 'text', 'word ', ' 1 < 2 ', '<b>', '</b>', '<i>', '</i>', '<h1>', '</h1>', '<a href="?a=1&b=2">', '</a>',
];
const FUZZ_SELECTORS = ['*', 'div', 'p', 'b', 'li', 'td', 'svg *', '[class]', '.a', '#x', ':empty', 'div > *', 'b:only-child', 'span.q', 'textarea', 'title'];
const fuzzHtml = (seed) => {
	let x = (seed + 1) * 2654435761 >>> 0;
	const rnd = () => {
		x ^= x << 13; x >>>= 0;
		x ^= x >>> 17;
		x ^= x << 5; x >>>= 0;
		return x / 4294967296;
	};
	const n = 4 + Math.floor(rnd() * 60);
	let s = '';
	for (let i = 0; i < n; i++) s += FUZZ_TOKENS[Math.floor(rnd() * FUZZ_TOKENS.length)];
	return s;
};

const main = () => {
	const [dir, out] = process.argv.slice(2);
	const cases = [];
	for (const f of sampleFiles(dir)) {
		const html = readFileSync(join(dir, f), 'utf8');
		cases.push({ name: `corpus/${f}`, html });
	}
	for (const [name, html] of EDGE) cases.push({ name: `edge/${name}`, html });
	for (let i = 0; i < FUZZ_COUNT; i++) cases.push({ name: `fuzz/${i}`, html: fuzzHtml(FUZZ_SEED + i), fuzz: true });

	const vectors = [];
	let selCount = 0;
	for (const c of cases) {
		const extra = c.fuzz ? FUZZ_SELECTORS : docSelectors(c.html);
		const pug = wf(convertHtmlToCleanPug(c.html));
		const common = c.fuzz ? [] : runSelectors(c.html, COMMON);
		const results = runSelectors(c.html, extra);
		selCount += common.length + results.length;
		vectors.push({ name: c.name, html: c.html, pug, common, extra: extra.map((sel, i) => [sel, results[i]]) });
	}
	const json = JSON.stringify({ generator: 'dom_vectors.mjs (linkedom 0.18.11 · htmlparser2 10.1.0)', common: COMMON, cases: vectors });
	writeFileSync(out, json);
	const count = (p) => cases.filter((c) => c.name.startsWith(p)).length;
	console.log(`cases ${vectors.length} (corpus ${count('corpus/')}, edge ${count('edge/')}, fuzz ${count('fuzz/')}), selectors ${selCount}, bytes ${json.length}`);
};

main();
