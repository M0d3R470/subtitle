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

stt_model   = genai.GenerativeModel('gemini-3-flash-preview')
tutor_model = genai.GenerativeModel('gemini-3.1-pro-preview')

# ── Pydantic 스키마 (response_schema용) ───────────────────
class Segment(BaseModel):
    start: float
    end: float
    orig: str


# ── 유틸: 구글 파일 ACTIVE 대기 ───────────────────────────
def wait_until_active(uploaded_file, timeout: int = 120) -> None:
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
    if not texts:
        return []
    joined = SEPARATOR.join(texts)
    try:
        translated = GoogleTranslator(source='auto', target=target).translate(joined)
        parts = translated.split(SEPARATOR)
        if len(parts) != len(texts):
            return texts
        return parts
    except Exception:
        return texts


# ── 유틸: 단어 단위로 쪼개진 세그먼트 병합 ───────────────
MIN_WORDS      = 4      # 이 단어 수 미만이면 다음 세그먼트에 붙임
MAX_GAP_SEC    = 1.5    # 이 시간(초) 이상 간격이 벌어지면 강제로 끊음

def merge_short_segments(segments: List[dict]) -> List[dict]:
    """단어 단위로 쪼개진 세그먼트를 문장 단위로 병합."""
    if not segments:
        return []

    merged = []
    buf = dict(segments[0])  # start, end, orig 복사

    for seg in segments[1:]:
        gap        = seg['start'] - buf['end']
        word_count = len(buf['orig'].split())

        # 현재 버퍼가 너무 짧고, 간격도 크지 않으면 합치기
        if word_count < MIN_WORDS and gap <= MAX_GAP_SEC:
            buf['orig'] += ' ' + seg['orig']
            buf['end']   = seg['end']
        else:
            merged.append(buf)
            buf = dict(seg)

    merged.append(buf)
    return merged


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

        # ③ Gemini STT
        prompt = (
            "Listen to this audio and transcribe it into subtitle segments. "
            "Each segment must be a complete sentence or a meaningful phrase (at least 4-5 words). "
            "Do NOT split individual words into separate segments. "
            "Group words into natural spoken sentences or clauses. "
            "Return accurate start and end timestamps in seconds for each segment. "
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
            finish = getattr(response.candidates[0], 'finish_reason', 'UNKNOWN') if response.candidates else 'NO_CANDIDATES'
            raise RuntimeError(f"Gemini가 빈 응답을 반환했습니다. finish_reason={finish}")

        # ```json ``` 펜스 제거
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

        # ⑤ 단어 단위 세그먼트 후처리 병합
        valid_segments = merge_short_segments(valid_segments)

        # ⑥ 한 번에 번역
        originals    = [seg['orig'].strip() for seg in valid_segments]
        translations = batch_translate(originals)

        # ⑦ 최종 데이터 조립
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
        safe_delete_remote(uploaded_file)
        safe_delete_local(audio_path)


@app.route('/api/chat', methods=['POST'])
def chat_with_gemini():
    data     = request.get_json(silent=True) or {}
    sentence = data.get('sentence', '').strip()

    if not sentence:
        return jsonify({'error': '문장이 없습니다.'}), 400

    prompt = f"""아래 문장을 한국어로 해설해주세요.

문장: "{sentence}"

조건:
1. 문장 구조 해설 (주어, 동사, 핵심 문법 등)
2. 중요 표현·단어 설명
3. 친절한 한국어 존댓말 사용
"""

    try:
        response = tutor_model.generate_content(prompt)
        return jsonify({'explanation': response.text})
    except Exception as e:
        return jsonify({'error': f'AI 응답 오류: {str(e)}'}), 500


# ── 전역 에러 핸들러 (Flask HTML 에러페이지 대신 JSON 반환) ──
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': '404 - 요청한 경로를 찾을 수 없습니다.'}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': '405 - 허용되지 않는 메서드입니다.'}), 405

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': f'500 - 서버 내부 오류: {str(e)}'}), 500

@app.errorhandler(Exception)
def unhandled_exception(e):
    return jsonify({'error': f'예상치 못한 오류: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
