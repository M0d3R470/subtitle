import os
import json
import time
import math
import subprocess
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
CHUNK_SEC = 30  # 청크 길이 (초)


# ── 유틸: 오디오를 청크로 분할 ───────────────────────────
def split_audio(src: str, chunk_sec: int = CHUNK_SEC) -> List[str]:
    """ffmpeg로 오디오를 chunk_sec 단위로 분할, 파일 경로 리스트 반환."""
    # 총 길이 확인
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', src],
        capture_output=True, text=True
    )
    total_sec = float(probe.stdout.strip())
    n_chunks  = math.ceil(total_sec / chunk_sec)

    paths = []
    base  = src.rsplit('.', 1)[0]
    for i in range(n_chunks):
        out = f"{base}_chunk{i:03d}.m4a"
        subprocess.run([
            'ffmpeg', '-y', '-i', src,
            '-ss', str(i * chunk_sec),
            '-t',  str(chunk_sec),
            '-c',  'copy', out
        ], capture_output=True)
        if os.path.exists(out):
            paths.append((out, i * chunk_sec))  # (파일경로, 오프셋)
    return paths


# ── 유틸: 청크 하나를 Gemini STT+번역 ───────────────────
def transcribe_chunk(audio_path: str, offset: float) -> List[dict]:
    """청크 파일을 Gemini로 분석, 타임스탬프에 offset 적용해서 반환."""
    uf = genai.upload_file(path=audio_path, mime_type='audio/mp4')
    wait_until_active(uf)

    prompt = """You are a professional subtitle transcriber and translator.
Listen to this audio clip and produce subtitles. For each segment:
- Transcribe exact spoken words into "orig"
- Translate naturally into Korean (구어체) into "trans"
- Each segment = one full sentence or natural phrase (min 4-5 words)
- Do NOT split single words into separate segments
- Timestamps are relative to the START of this audio clip (start from 0)
- Be as precise as possible with timestamps
- Preserve tone, humor, sarcasm; localize idioms"""

    try:
        resp = stt_model.generate_content(
            [prompt, uf],
            generation_config=types.GenerationConfig(
                response_mime_type='application/json',
                response_schema=list[Segment],
            ),
        )
        raw = (resp.text or '').strip()
        if not raw:
            return []
        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
            raw = raw.strip()
        segs = json.loads(raw)
    except Exception:
        return []
    finally:
        try:
            genai.delete_file(uf.name)
        except Exception:
            pass

    result = []
    for seg in segs:
        orig = (seg.get('orig') or '').strip()
        if not orig:
            continue
        result.append({
            'start': round(float(seg.get('start', 0)) + offset, 3),
            'end':   round(float(seg.get('end',   0)) + offset, 3),
            'orig':  orig,
            'trans': (seg.get('trans') or '').strip(),
        })
    return result

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
    """
    유튜브 자동생성 VTT는 슬라이딩 윈도우 방식이라
    같은 텍스트가 여러 블록에 중복으로 나옴.
    → <c> 태그 기준으로 단어별 타임스탬프를 추출해서 재조립.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # 방법: 블록별로 파싱하되, 각 블록의 start 시각에 새로 등장하는 텍스트만 취함
    blocks = re.split(r'\n\n+', text.strip())

    # (start, end, text) 튜플 수집
    raw = []
    for block in blocks:
        lines = block.strip().splitlines()
        ts_line = None
        text_lines = []
        for line in lines:
            if '-->' in line:
                ts_line = line.split('align:')[0].strip()  # position 메타 제거
            elif ts_line and line.strip() and not re.match(r'^\d+$', line.strip()) and line.strip() != 'WEBVTT':
                # HTML/VTT 태그 전부 제거
                clean = re.sub(r'<[^>]+>', '', line).strip()
                clean = re.sub(r'&nbsp;', ' ', clean).strip()
                if clean:
                    text_lines.append(clean)
        if not ts_line or not text_lines:
            continue
        try:
            start = _vtt_time_to_sec(ts_line.split('-->')[0].strip())
            end   = _vtt_time_to_sec(ts_line.split('-->')[1].strip())
        except Exception:
            continue
        orig = ' '.join(text_lines)
        raw.append((start, end, orig))

    if not raw:
        return []

    # 슬라이딩 윈도우 중복 제거:
    # 이전 블록 텍스트와 완전히 같거나 이전 블록 텍스트를 포함하면 스킵
    segments = []
    prev_text = ''
    for start, end, orig in raw:
        # 이전 텍스트가 현재 텍스트의 접두사면 → 새로 추가된 부분만 취함
        if orig == prev_text:
            continue
        if orig.startswith(prev_text) and prev_text:
            new_part = orig[len(prev_text):].strip()
            if new_part:
                # 새 파트를 이전 세그먼트의 end~현재 end 구간으로
                if segments:
                    segments[-1]['end'] = start
                segments.append({'start': start, 'end': end, 'orig': new_part, 'trans': ''})
        else:
            segments.append({'start': start, 'end': end, 'orig': orig, 'trans': ''})
        prev_text = orig

    return segments


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

        # ② 자막 없으면 Gemini STT 폴백 (청크 방식)
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

        # 오디오를 30초 청크로 분할
        chunks = split_audio(audio_path)
        if not chunks:
            raise RuntimeError("오디오 청크 분할에 실패했습니다.")

        # 각 청크를 순서대로 Gemini STT
        all_segments = []
        for chunk_path, offset in chunks:
            segs = transcribe_chunk(chunk_path, offset)
            all_segments.extend(segs)
            safe_delete_local(chunk_path)

        if not all_segments:
            raise RuntimeError("음성 인식 결과가 없습니다.")

        all_segments = merge_short_segments(all_segments)

        result = [
            {
                'start': seg['start'],
                'end':   seg['end'],
                'orig':  seg['orig'],
                'trans': seg['trans'],
            }
            for seg in all_segments
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

    prompt = f"""문장을 해설해주세요.

문장: "{sentence}"

조건:
1. 문장 구조 해설 (주어, 동사, 목적어, 핵심 문법)
2. 중요 표현·단어·숙어 설명
3. 정중한 한국어 존댓말
4. 짧고 간결하게 해설
5. 줄바꿈을 적극 활용하여 가독성을 높임
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
