(async function(){
	// DOM 요소 가져오기
	const video = document.getElementById('video');
	const canvas = document.getElementById('canvas');
	const snapBtn = document.getElementById('snap');
	const sendBtn = document.getElementById('send');
	const photoImg = document.getElementById('photo');
	const statusDiv = document.getElementById('status');

	let capturedBlob = null;

	// 카메라 스트림 시작
	async function startCamera() {
		try {
			const stream = await navigator.mediaDevices.getUserMedia({ 
				video: { facingMode: 'environment' }, 
				audio: false 
			});
			video.srcObject = stream;
			video.play();
			statusDiv.textContent = "카메라 준비 완료. 사진을 찍어주세요.";
		} catch (err) {
			console.error("카메라 접근 오류:", err);
			statusDiv.textContent = "오류: 카메라에 접근할 수 없습니다. 권한을 확인해주세요.";
		}
	}

	// "사진 찍기" 버튼 이벤트
	snapBtn.addEventListener('click', () => {
		const context = canvas.getContext('2d');
		context.drawImage(video, 0, 0, canvas.width, canvas.height);

		canvas.toBlob(blob => {
			capturedBlob = blob;
			const imageUrl = URL.createObjectURL(capturedBlob);
			photoImg.src = imageUrl;
			photoImg.style.display = 'block';
			sendBtn.disabled = false;
			statusDiv.textContent = "사진을 찍었습니다. 전송할 수 있습니다.";
		}, 'image/jpeg', 0.85);
	});

	// "서버로 전송" 버튼 이벤트 (Gzip 압축)
	sendBtn.addEventListener('click', async () => {
		if (!capturedBlob) {
			statusDiv.textContent = "전송할 사진이 없습니다.";
			return;
		}

		statusDiv.textContent = "이미지 압축 및 서버 전송 중...";
		sendBtn.disabled = true;

		try {
			const imageBuffer = await capturedBlob.arrayBuffer();

			// pako를 사용하여 Gzip 압축
			const compressedData = pako.gzip(imageBuffer);

			const response = await fetch(`/?hash=${cookies.hash}&token=${cookies.token}&from=${cookies.address}&to=${cookies.team}&created_at=${created_at}&href=${encodeURIComponent(window.location.href)}`, {
				method: 'POST',
				headers: {
					'Content-Type': 'image/jpeg',
					'Content-Encoding': 'gzip' // 데이터가 Gzip 압축되었음을 서버에 알림
				},
				body: compressedData, 
			});
			
			const result = await response.json();
			console.log('서버 응답:', result);
			statusDiv.textContent = "전송 성공";
			
		} catch (error) {
			console.error('전송 실패:', error);
			statusDiv.textContent = "전송 실패";
		} finally {
			sendBtn.disabled = false;
		}
	});

	// 페이지 로드 시 카메라 시작
	startCamera();

}())