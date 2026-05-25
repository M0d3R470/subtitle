import os
import json
import time
import yt_dlp
from flask import Flask, request, jsonify, send_file
import google.generativeai as genai
from google.generativeai import types
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

# ── Pydantic 스키마 ───────────────────────────────────────
class Segment(BaseModel):
    start: float
    end: float
    orig: str
    trans: str  # ← 번역도 Gemini가 한 번에 처리


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


# -- 유틸: 세그먼트 길이 조정 (병합 + 분할) ---------
MIN_WORDS    = 4    # 이 단어 수 미만이면 다음에 붙임
MAX_WORDS    = 18   # 이 단어 수 초과하면 문장 경계에서 끊음
MAX_GAP_SEC  = 1.5  # 이 시간 이상 간격이면 강제로 끊음
SENTENCE_ENDS = {'.', '!', '?', '...'}

def _split_long_segment(seg):
    words = seg['orig'].split()
    if len(words) <= MAX_WORDS:
        return [seg]

    # orig를 문장 부호 기준으로 쪼개기
    chunks_orig = []
    buf_words = []
    for word in words:
        buf_words.append(word)
        if any(word.rstrip('"\'\u2019').endswith(e) for e in SENTENCE_ENDS) and len(buf_words) >= MIN_WORDS:
            chunks_orig.append(' '.join(buf_words))
            buf_words = []
    if buf_words:
        if chunks_orig:
            chunks_orig[-1] += ' ' + ' '.join(buf_words)
        else:
            chunks_orig.append(' '.join(buf_words))

    # 번역 균등 분할
    trans_words = seg.get('trans', '').split()
    n = len(chunks_orig)
    chunk_size = max(1, len(trans_words) // n)
    chunks_trans = []
    for i in range(n):
        s = i * chunk_size
        e = s + chunk_size if i < n - 1 else len(trans_words)
        chunks_trans.append(' '.join(trans_words[s:e]))

    # 타임스탬프 단어 수 비례 배분
    total_dur = seg['end'] - seg['start']
    total_words = max(1, sum(len(c.split()) for c in chunks_orig))
    result = []
    cursor = seg['start']
    for o, t in zip(chunks_orig, chunks_trans):
        dur = total_dur * len(o.split()) / total_words
        result.append({
            'start': round(cursor, 3),
            'end':   round(cursor + dur, 3),
            'orig':  o.strip(),
            'trans': t.strip(),
        })
        cursor += dur
    return result


def merge_short_segments(segments):
    if not segments:
        return []

    # 1단계: 짧은 것 병합
    merged = []
    buf = dict(segments[0])
    for seg in segments[1:]:
        gap = seg['start'] - buf['end']
        word_count = len(buf['orig'].split())
        if word_count < MIN_WORDS and gap <= MAX_GAP_SEC:
            buf['orig']  += ' ' + seg['orig']
            buf['trans'] += ' ' + seg.get('trans', '')
            buf['end']    = seg['end']
        else:
            merged.append(buf)
            buf = dict(seg)
    merged.append(buf)

    # 2단계: 긴 것 분할
    result = []
    for seg in merged:
        result.extend(_split_long_segment(seg))
    return result




# ── 유틸: VTT 자막 파싱 ─────────────────────────────────
import re

def _vtt_time_to_sec(t: str) -> float:
    parts = t.strip().split(':')
    if len(parts) == 3:
        h, m, s = parts
    else:
        h, m, s = 0, parts[0], parts[1]
    return int(h) * 3600 + int(m) * 60 + float(s.replace(',', '.'))

def parse_vtt(filepath: str) -> List[dict]:
    segments = []
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # 타임스탬프 블록 파싱
    blocks = re.split(r'\n\n+', text.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        # 타임스탬프 줄 찾기
        ts_line = None
        text_lines = []
        for line in lines:
            if '-->' in line:
                ts_line = line
            elif ts_line and line.strip() and not line.strip().isdigit():
                # HTML 태그 제거
                clean = re.sub(r'<[^>]+>', '', line).strip()
                if clean:
                    text_lines.append(clean)

        if not ts_line or not text_lines:
            continue

        try:
            start_str, end_str = ts_line.split('-->')[0].strip(), ts_line.split('-->')[1].split()[0].strip()
            start = _vtt_time_to_sec(start_str)
            end   = _vtt_time_to_sec(end_str)
        except Exception:
            continue

        orig = ' '.join(text_lines)
        if orig:
            segments.append({'start': start, 'end': end, 'orig': orig, 'trans': ''})

    # 중복 제거 (VTT는 같은 텍스트가 여러 블록에 걸쳐 나오는 경우 있음)
    deduped = []
    seen = set()
    for seg in segments:
        key = (round(seg['start'], 1), seg['orig'][:30])
        if key not in seen:
            seen.add(key)
            deduped.append(seg)

    return deduped


# ── 유틸: Gemini로 번역 (텍스트만) ──────────────────────
TRANS_SEPARATOR = "\n|||\n"

def gemini_batch_translate(texts: List[str]) -> List[str]:
    if not texts:
        return []
    joined = TRANS_SEPARATOR.join(texts)
    prompt = f"""Translate the following English subtitle lines into natural Korean (구어체).
Each line is separated by "|||". Keep the same number of lines and the same order.
Preserve tone, humor, and nuance. Return ONLY the translated lines separated by "|||", nothing else.

{joined}"""
    try:
        resp = stt_model.generate_content(prompt)
        parts = resp.text.strip().split(TRANS_SEPARATOR)
        if len(parts) == len(texts):
            return [p.strip() for p in parts]
        # 수가 안 맞으면 줄 수 기준으로 맞추기
        result = []
        for i, t in enumerate(texts):
            result.append(parts[i].strip() if i < len(parts) else t)
        return result
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

    audio_path = f"{video_id}.m4a"
    sub_path   = f"{video_id}.en.vtt"
    uploaded_file = None

    try:
        url = f"https://www.youtube.com/watch?v={video_id}"

        # ① 유튜브 자막 시도 (타임스탬프 정확도 최우선)
        ydl_sub_opts = {
            'skip_download':     True,
            'writesubtitles':    True,
            'writeautomaticsub': True,
            'subtitleslangs':    ['en', 'en-US', 'en-GB'],
            'subtitlesformat':   'vtt',
            'outtmpl':           video_id,
            'quiet':             True,
            'noplaylist':        True,
        }
        with yt_dlp.YoutubeDL(ydl_sub_opts) as ydl:
            ydl.download([url])

        # 다운로드된 vtt 파일 찾기 (언어 코드가 다를 수 있음)
        vtt_file = None
        for fname in os.listdir('.'):
            if fname.startswith(video_id) and fname.endswith('.vtt'):
                vtt_file = fname
                break

        if vtt_file:
            # ── VTT 파싱 경로 ──────────────────────────────
            segments = parse_vtt(vtt_file)
            os.remove(vtt_file)

            if segments:
                segments = merge_short_segments(segments)
                # 텍스트만 Gemini로 번역
                originals    = [s['orig'] for s in segments]
                translations = gemini_batch_translate(originals)
                result = [
                    {
                        'start': s['start'],
                        'end':   s['end'],
                        'orig':  s['orig'],
                        'trans': t,
                    }
                    for s, t in zip(segments, translations)
                ]
                return jsonify(result)

        # ② 자막 없으면 Gemini STT 폴백
        ydl_audio_opts = {
            'format':      '140/bestaudio[ext=m4a]/bestaudio',
            'outtmpl':     audio_path,
            'quiet':       True,
            'noplaylist':  True,
            'extractor_args': {'youtube': {'skip': ['dash', 'hls']}},
        }
        with yt_dlp.YoutubeDL(ydl_audio_opts) as ydl:
            ydl.download([url])

        if not os.path.exists(audio_path):
            raise FileNotFoundError("오디오 다운로드에 실패했습니다.")

        uploaded_file = genai.upload_file(path=audio_path, mime_type="audio/mp4")
        if uploaded_file is None:
            raise RuntimeError("파일 업로드 자체가 실패했습니다.")

        wait_until_active(uploaded_file)

        prompt = """You are a professional subtitle transcriber and translator.
Listen to this audio and produce subtitles. For each segment:
- Transcribe exact spoken words into "orig"
- Translate naturally into Korean (구어체) into "trans"
- Each segment = one full sentence or natural phrase (min 4-5 words)
- Do NOT split single words into separate segments
- Timestamps must be as accurate as possible in seconds
- Preserve tone, humor, sarcasm, slang (localize idioms, do not translate literally)"""

        response = stt_model.generate_content(
            [prompt, uploaded_file],
            generation_config=types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=list[Segment],
            ),
        )

        raw = (response.text or "").strip()
        if not raw:
            finish = getattr(response.candidates[0], 'finish_reason', 'UNKNOWN') if response.candidates else 'NO_CANDIDATES'
            raise RuntimeError(f"Gemini 빈 응답. finish_reason={finish}")

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            segments = json.loads(raw)
        except json.JSONDecodeError as je:
            raise RuntimeError(f"JSON 파싱 실패: {je} / 앞 200자: {raw[:200]}")

        valid_segments = [
            seg for seg in segments
            if isinstance(seg.get('orig'), str) and seg['orig'].strip()
        ]
        valid_segments = merge_short_segments(valid_segments)

        result = [
            {
                'start': float(seg.get('start', 0)),
                'end':   float(seg.get('end', 0)),
                'orig':  seg['orig'].strip(),
                'trans': seg.get('trans', '').strip(),
            }
            for seg in valid_segments
        ]
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        safe_delete_remote(uploaded_file)
        safe_delete_local(audio_path)
        # 혹시 남은 vtt 파일 정리
        for fname in list(os.listdir('.')):
            if fname.startswith(video_id) and fname.endswith('.vtt'):
                safe_delete_local(fname)


@app.route('/api/chat', methods=['POST'])
def chat_with_gemini():
    data     = request.get_json(silent=True) or {}
    sentence = data.get('sentence', '').strip()

    if not sentence:
        return jsonify({'error': '문장이 없습니다.'}), 400

    prompt = f"""해설해주세요.

문장: "{sentence}"

조건:
1. 문장 구조 해설 (주어, 동사, 목적어, 핵심 문법)
2. 중요 표현·단어·숙어 설명
3. 줄바꿈을 적극 활용하여 가독성 있게 서술
4. 정중한 한국어 존댓말 사용
"""

    try:
        response = tutor_model.generate_content(prompt)
        return jsonify({'explanation': response.text})
    except Exception as e:
        return jsonify({'error': f'AI 응답 오류: {str(e)}'}), 500


# ── 전역 에러 핸들러 ──────────────────────────────────────
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
