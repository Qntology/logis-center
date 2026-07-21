(async function(){
	// ------------------------------------
	// INTERNAL STATE
	// ------------------------------------
	let isScrolling = false;
	let scrollTimeout = null;

	var started_at = Date.now();

	const SCROLL_STOP_DELAY = 200; // ms
	let hoverStartTime = null;
	let currentHoverEl = null;

	const timeline = [];
	let lastClicks = [];

	const topK = 100;
	const threshold = 60;

	// ------------------------------------
	// UTILS (수정됨: 불필요한 해시값 클래스 제거 로직 추가)
	// ------------------------------------
	function cleanClassList(el) {
		if (!el || !el.classList) return [];
		// 상태/랜덤값 무시: 'active', 'selected', 'on', 'current', 'focus', 'hover' 및 '__'를 포함하는 클래스
		const ignore = ['active', 'selected', 'on', 'current', 'focus', 'hover'];
		return Array.from(el.classList)
			.filter(c => {
				const lowerC = c.toLowerCase();
				// 상태 관련 클래스 무시, '__'를 포함하는 클래스(랜덤 해시값 추정) 무시
				return !ignore.includes(lowerC) && c.indexOf('__') === -1; 
			})
			.sort();
	}

	function visible(el) {
		try {
			const cs = getComputedStyle(el);
			if (!cs || cs.display === 'none' || cs.visibility === 'hidden' || Number(cs.opacity) === 0) return false;
			const r = el.getBoundingClientRect();
			return r.width !== 0 && r.height !== 0;
		} catch (e) {
			return false;
		}
	}

	// ------------------------------------
	// LIST DETECTION (수정됨: 리스트 부모 감지 로직 개선)
	// ------------------------------------
	function detectDivListParent(el) {
		let cur = el;
		for (let i = 0; i < 6 && cur; i++) {
			const children = Array.from(cur.children).filter(visible);
			if (children.length < 3) {
				cur = cur.parentElement;
				continue;
			}
			
			// cleanClassList를 사용하여 시그니처 생성
			const sigs = children.map(c => c.tagName.toLowerCase() + '|' + cleanClassList(c).slice(0, 3).join(','));
			const counts = {};
			for (const s of sigs) counts[s] = (counts[s] || 0) + 1;
			const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);

			// 가장 흔한 시그니처의 비율이 60% 이상이면 목록의 부모로 간주
			if (entries.length && entries[0][1] / children.length > 0.6) {
				return cur; 
			}

			cur = cur.parentElement;
		}
		return null;
	}

	function selectListParent(target) {
		// 테이블 행(TR)도 목록 항목으로 취급하려면 'tbody', 'thead'를 추가할 수 있지만, 
		// 기존의 일반적인 목록(ul, ol, dl)과 div-list 감지 로직을 유지합니다.
		return target.closest('ul,ol,dl') || detectDivListParent(target);
	}

	// ------------------------------------
	// LIST ITEM COLLECTION (수정됨: 빈 문자열 클래스 비교 제외 로직 추가)
	// ------------------------------------
	function collectListItems(list, patternEl, threshold = 60, topK = 100) {
		const children = Array.from(list.children).filter(visible);

		// 1) 기본 siblings 기반
		if (children.length > 1) {
			const sigPattern = patternEl.tagName.toLowerCase() + '|' + cleanClassList(patternEl).slice(0, 4).join(',');
			const filtered = children.filter(c => {
				const sig = c.tagName.toLowerCase() + '|' + cleanClassList(c).slice(0, 4).join(',');
				let match = 0;

				const sigParts = sig.split('|')[1]?.split(',') || [];
				const patternParts = sigPattern.split('|')[1]?.split(',') || [];

				patternParts.forEach(p => { if (sigParts.includes(p) && p !== '') match++ });

				const score = patternParts.length ? (match / patternParts.length) * 100 : 0;
				return score >= threshold;
			});

			return filtered.slice(0, topK);
		}

		// --------------------------------------------------------------------
		// 2) 형제가 없으면: closest 부모 타고 올라가며 동일 signature 매칭 시도
		// --------------------------------------------------------------------
		let cur = patternEl.parentElement;
		const sigPattern = patternEl.tagName.toLowerCase() + '|' + cleanClassList(patternEl).slice(0, 4).join(',');

		for (let depth = 0; depth < 6 && cur; depth++) {

			// 현재 parent 안에서 모든 visible 요소 수집
			const allDesc = Array.from(cur.querySelectorAll(patternEl.tagName))
				.filter(visible);

			if (allDesc.length <= 1) {
				cur = cur.parentElement;
				continue;
			}

			// signature 매칭
			const matched = allDesc.filter(el => {
				const sig = el.tagName.toLowerCase() + '|' + cleanClassList(el).slice(0, 4).join(',');
				let match = 0;

				const sigParts = sig.split('|')[1]?.split(',') || [];
				const patternParts = sigPattern.split('|')[1]?.split(',') || [];

				patternParts.forEach(p => { if (sigParts.includes(p) && p !== '') match++ });

				const score = patternParts.length ? (match / patternParts.length) * 100 : 0;
				return score >= threshold;
			});

			if (matched.length > 1) {
				return matched.slice(0, topK);
			}

			cur = cur.parentElement;
		}

		// fallback: 아무것도 없으면 본인만이라도
		return [patternEl];
	}



	async function pushEvent(eventObj) {

		timeline.push({
			...eventObj,
			order: timeline.length,
			timestamp: Date.now()
		});


		console.log('eventObj',eventObj);

		return


		var { cookies } = await app.storage.get('cookies')



		var action = [eventObj.element.outerHTML]

		var relate = []

		for(var i = 0; i < eventObj.relatedElements.length; i++){
			var el = eventObj.relatedElements[i]

			var $temp = document.createElement('div')

				$temp.innerHTML = el.outerHTML

			
			var $images = $temp.querySelectorAll('img')

			if($images.length){
				for(var a = 0; a < $images.length; a++){
					var $image = $images[a]

					$image.src = ''
				}
			}

			relate.push($temp.innerHTML)
		}


		var tokenAmount = approximateTokenCount(JSON.stringify({
			type : eventObj.type,
			action : action,
			relate : relate
		})) * 5

		console.log('tokenAmount',tokenAmount);

		if(tokenAmount > 7000){
			return;
		}

		var body = {
			type : eventObj.type,
			action : action,
			relate : relate
		}

		var { results, session } = await app.fetch({
			url : reqUrl( cookies, app.filters, {state : hashId()} ),
			method: "POST",
			headers: {
				'Content-Type': 'application/octet-stream',
				'Content-Encoding': 'gzip'
			},
			body : JSON.stringify(body)
		})
	}

	function detectRageClick() {
		const now = Date.now();
		const recent = lastClicks.filter(t => now - t < 700);
		return recent.length >= 3;
	}

	function detectCloseButNoClick() {
		// hoverStartTime이 null이 아니면서 currentHoverEl이 null이 되어야 함.
		// 원본 코드의 의도를 살려, hover/mouseout 이후에 hoverStartTime이 null이 된 경우를 체크하지 않도록 함.
		return (!hoverStartTime && currentHoverEl === null); 
	}

	// ------------------------------------
	// SCROLL DETECTION
	// ------------------------------------
	window.addEventListener("scroll", () => {
		isScrolling = true;
		clearTimeout(scrollTimeout);
		scrollTimeout = setTimeout(() => {
			isScrolling = false;
		}, SCROLL_STOP_DELAY);
	});

	// ------------------------------------
	// HOVER
	// ------------------------------------
	document.addEventListener("mouseover", (e) => {
		if (isScrolling) return;
		hoverStartTime = performance.now();
		currentHoverEl = e.target;
	});

	document.addEventListener("mouseout", (e) => {
		if (!hoverStartTime || !currentHoverEl) return;
		if (e.target !== currentHoverEl) return;

		const dwell = performance.now() - hoverStartTime;

		if (dwell >= 500) {
			const parentList = selectListParent(currentHoverEl);
			let relatedElements = [];

			if (parentList) {
				// 수정된 collectListItems 사용
				relatedElements = collectListItems(parentList, currentHoverEl, threshold, topK)
					.filter(el => el !== currentHoverEl);
			}

			pushEvent({
				type: "hover",
				element: currentHoverEl,
				relatedElements,
				ms: dwell
			});
		}

		hoverStartTime = null;
		currentHoverEl = null;
	});

	// ------------------------------------
	// CLICK
	// ------------------------------------
	document.addEventListener("click", (e) => {
		const el = e.target;

		lastClicks.push(Date.now());
		if (lastClicks.length > 10) lastClicks.shift();

		const parentList = selectListParent(el);
		let relatedElements = [];

		if (parentList) {
			// 수정된 collectListItems 사용
			relatedElements = collectListItems(parentList, el, threshold, topK)
				.filter(child => child !== el);
		}

		pushEvent({
			type: "click",
			element: el,
			relatedElements,
			rage: detectRageClick(),
			closeButNoClick: detectCloseButNoClick(),
			topKApplied: topK,
			thresholdApplied: threshold
		});
	});

	// ------------------------------------
	// CHANGE (600ms debounce) + password 제외
	// ------------------------------------
	let changeDebounceTimer = null;
	let lastChangeTarget = null;
	let lastChangeValueInfo = null;

	const CHANGE_DELAY = 600;

	document.addEventListener("change", (e) => {
		const el = e.target;

		const tag = el.tagName;
		const type = el.type;

		// password 제외
		if (tag === "INPUT" && type === "password") {
			return;
		}

		// change 대상 전체
		if (
			tag === "SELECT" ||
			tag === "TEXTAREA" ||
			(tag === "INPUT")
		) {
			let valueInfo = null;

			if (tag === "SELECT") {
				valueInfo = {
					value: el.value,
					selectedText: el.options[el.selectedIndex]?.text || null
				};
			}
			else if (type === "checkbox" || type === "radio") {
				valueInfo = {
					checked: el.checked,
					value: el.value
				};
			}
			else {
				valueInfo = { value: el.value };
			}

			lastChangeTarget = el;
			lastChangeValueInfo = valueInfo;

			clearTimeout(changeDebounceTimer);
			changeDebounceTimer = setTimeout(() => {
				const parentList = selectListParent(lastChangeTarget);
				let relatedElements = [];

				if (parentList) {
					// 수정된 collectListItems 사용
					relatedElements = collectListItems(parentList, lastChangeTarget, threshold, topK)
						.filter(child => child !== lastChangeTarget);
				}

				pushEvent({
					type: "change",
					element: lastChangeTarget,
					valueInfo: lastChangeValueInfo,
					relatedElements,
					debounced: true,
					delayMs: CHANGE_DELAY,
					topKApplied: topK,
					thresholdApplied: threshold
				});
			}, CHANGE_DELAY);
		}
	});
}())