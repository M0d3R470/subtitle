import os
import json
import time
import yt_dlp
from flask import Flask, request, jsonify, send_file
import google.generativeai as genai
from google.generativeai import types
from deep_translator import GoogleTranslator
from pydantic import BaseModel
from typing import List

app = Flask(__name__)

# ── 환경변수 체크 ──────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다. 서버를 시작할 수 없습니다.")

genai.configure(api_key=GOOGLE_API_KEY)

stt_model   = genai.GenerativeModel('gemini-2.5-flash')
tutor_model = genai.GenerativeModel('gemini-2.5-flash')

# ── Pydantic 스키마 (response_schema용) ───────────────────
class Segment(BaseModel):
    start: float
    end: float
    orig: str


# ── 유틸: 구글 파일 ACTIVE 대기 ───────────────────────────
def wait_until_active(uploaded_file, timeout: int = 120) -> None:
    """업로드된 파일이 ACTIVE 상태가 될 때까지 대기."""
    deadline = time.time() + timeout
    while True:
        info = genai.get_file(uploaded_file.name)
        state = info.state.name
        if state == 'ACTIVE':
            return
        if state == 'FAILED':
            raise RuntimeError("구글 서버가 오디오 처리에 실패했습니다.")
        if time.time() > deadline:
            raise TimeoutError("구글 서버 대기 시간이 초과되었습니다.")
        time.sleep(2)


# ── 유틸: 파일 안전 삭제 ──────────────────────────────────
def safe_delete_local(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def safe_delete_remote(uploaded_file) -> None:
    try:
        if uploaded_file:
            genai.delete_file(uploaded_file.name)
    except Exception:
        pass


# ── 유틸: 번역 (텍스트 묶어서 한 방에) ────────────────────
SEPARATOR = "\n||||\n"

def batch_translate(texts: List[str], target: str = 'ko') -> List[str]:
    """
    여러 텍스트를 SEPARATOR로 묶어 한 번에 번역한 뒤 다시 분리.
    번역 실패 시 원문 반환.
    """
    if not texts:
        return []
    joined = SEPARATOR.join(texts)
    try:
        translated = GoogleTranslator(source='auto', target=target).translate(joined)
        parts = translated.split(SEPARATOR)
        # 분리된 조각 수가 다르면 원문으로 채움
        if len(parts) != len(texts):
            return texts
        return parts
    except Exception:
        return texts


# ── 라우트 ────────────────────────────────────────────────
@app.route('/')
def index():
    return send_file('index.html')


@app.route('/api/subtitles', methods=['GET'])
def get_subtitles():
    video_id = request.args.get('video_id', '').strip()
    if not video_id:
        return jsonify({'error': '비디오 ID가 없습니다.'}), 400

    audio_path    = f"{video_id}.m4a"
    uploaded_file = None

    try:
        # ① 유튜브에서 m4a 오디오 다운로드
        ydl_opts = {
            'format':     'bestaudio[ext=m4a]/140/bestaudio',
            'outtmpl':    audio_path,
            'quiet':      True,
            'noplaylist': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

        if not os.path.exists(audio_path):
            raise FileNotFoundError("오디오 다운로드에 실패했습니다.")

        # ② 구글 서버에 업로드
        uploaded_file = genai.upload_file(path=audio_path, mime_type="audio/mp4")
        if uploaded_file is None:
            raise RuntimeError("파일 업로드 자체가 실패했습니다.")

        wait_until_active(uploaded_file)

        # ③ Gemini STT (Pydantic 스키마로 structured output)
        prompt = (
            "Listen to this audio carefully and transcribe every spoken word "
            "with accurate start and end timestamps in seconds. "
            "Return ONLY a valid JSON array. Do not include any explanation or markdown."
        )
        response = stt_model.generate_content(
            [prompt, uploaded_file],
            generation_config=types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=list[Segment],
            ),
        )

        # 응답 텍스트 검증
        raw = (response.text or "").strip()
        if not raw:
            # finish_reason 등 디버그 정보 포함해서 에러 던지기
            finish = getattr(response.candidates[0], 'finish_reason', 'UNKNOWN') if response.candidates else 'NO_CANDIDATES'
            raise RuntimeError(f"Gemini가 빈 응답을 반환했습니다. finish_reason={finish}")

        # ```json ... ``` 펜스 제거 (혹시 붙어오는 경우 대비)
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            segments: List[dict] = json.loads(raw)
        except json.JSONDecodeError as je:
            raise RuntimeError(f"JSON 파싱 실패: {je} / 원본 응답 앞 200자: {raw[:200]}")

        # ④ 빈 세그먼트 제거
        valid_segments = [
            seg for seg in segments
            if isinstance(seg.get('orig'), str) and seg['orig'].strip()
        ]

        # ⑤ 한 번에 번역
        originals   = [seg['orig'].strip() for seg in valid_segments]
        translations = batch_translate(originals)

        # ⑥ 최종 데이터 조립
        result = [
            {
                'start': float(seg.get('start', 0)),
                'end':   float(seg.get('end', 0)),
                'orig':  orig,
                'trans': trans,
            }
            for seg, orig, trans in zip(valid_segments, originals, translations)
        ]

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # 성공/실패 무관하게 항상 정리
        safe_delete_remote(uploaded_file)
        safe_delete_local(audio_path)


@app.route('/api/chat', methods=['POST'])
def chat_with_gemini():
    data     = request.get_json(silent=True) or {}
    sentence = data.get('sentence', '').strip()

    if not sentence:
        return jsonify({'error': '문장이 없습니다.'}), 400

    prompt = f"""당신은 친절한 언어 튜터입니다. 아래 문장을 한국어로 해설해주세요.

문장: "{sentence}"

조건:
1. 문장 구조 해설 (주어, 동사, 핵심 문법 등)
2. 중요 표현·단어 설명
3. 동생에게 알려주듯 다정하고 친근한 말투
4. 가독성을 위해 <strong>, <br> 등 HTML 태그 적절히 활용
"""

    try:
        response = tutor_model.generate_content(prompt)
        return jsonify({'explanation': response.text})
    except Exception as e:
        return jsonify({'error': f'AI 응답 오류: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
