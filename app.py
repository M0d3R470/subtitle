import os
import re
import json
import time
import math
import yt_dlp
from flask import Flask, request, jsonify, send_file
import google.generativeai as genai
from google.generativeai import types
from pydantic import BaseModel
from typing import List

app = Flask(__name__)

# ── 환경변수 체크 ─────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")

genai.configure(api_key=GOOGLE_API_KEY)
stt_model   = genai.GenerativeModel('gemini-2.5-flash')
tutor_model = genai.GenerativeModel('gemini-2.5-flash')

# ── Pydantic 스키마 ───────────────────────────────────────
class Segment(BaseModel):
    start: float
    end:   float
    orig:  str
    trans: str

# ── 상수 ─────────────────────────────────────────────────
CHUNK_SEC     = 30
MIN_WORDS     = 4
MAX_WORDS     = 18
MAX_GAP_SEC   = 1.5
SENTENCE_ENDS = {'.', '!', '?', '...'}


# ── 유틸: 파일 안전 삭제 ─────────────────────────────────
def safe_delete_local(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass

def safe_delete_remote(uf):
    try:
        if uf:
            genai.delete_file(uf.name)
    except Exception:
        pass


# ── 유틸: Gemini 파일 ACTIVE 대기 ────────────────────────
def wait_until_active(uf, timeout=120):
    deadline = time.time() + timeout
    while True:
        info  = genai.get_file(uf.name)
        state = info.state.name
        if state == 'ACTIVE':
            return
        if state == 'FAILED':
            raise RuntimeError("구글 서버가 오디오 처리에 실패했습니다.")
        if time.time() > deadline:
            raise TimeoutError("구글 서버 대기 시간 초과.")
        time.sleep(2)


# ── 유틸: 세그먼트 병합 / 분할 ───────────────────────────
def _split_long_segment(seg):
    words = seg['orig'].split()
    if len(words) <= MAX_WORDS:
        return [seg]

    chunks_orig = []
    buf = []
    for word in words:
        buf.append(word)
        stripped = word.rstrip('"\'\u2019')
        if any(stripped.endswith(e) for e in SENTENCE_ENDS) and len(buf) >= MIN_WORDS:
            chunks_orig.append(' '.join(buf))
            buf = []
    if buf:
        if chunks_orig:
            chunks_orig[-1] += ' ' + ' '.join(buf)
        else:
            chunks_orig.append(' '.join(buf))

    trans_words = seg.get('trans', '').split()
    n           = len(chunks_orig)
    chunk_size  = max(1, len(trans_words) // n)
    chunks_trans = []
    for i in range(n):
        s = i * chunk_size
        e = s + chunk_size if i < n - 1 else len(trans_words)
        chunks_trans.append(' '.join(trans_words[s:e]))

    total_dur   = seg['end'] - seg['start']
    total_words = max(1, sum(len(c.split()) for c in chunks_orig))
    result      = []
    cursor      = seg['start']
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
    merged = []
    buf = dict(segments[0])
    for seg in segments[1:]:
        gap        = seg['start'] - buf['end']
        word_count = len(buf['orig'].split())
        if word_count < MIN_WORDS and gap <= MAX_GAP_SEC:
            buf['orig']  += ' ' + seg['orig']
            buf['trans'] += ' ' + seg.get('trans', '')
            buf['end']    = seg['end']
        else:
            merged.append(buf)
            buf = dict(seg)
    merged.append(buf)

    result = []
    for seg in merged:
        result.extend(_split_long_segment(seg))
    return result


# ── 유틸: VTT 파싱 ───────────────────────────────────────
def _vtt_time_to_sec(t):
    parts = t.strip().split(':')
    if len(parts) == 3:
        h, m, s = parts
    else:
        h, m, s = 0, parts[0], parts[1]
    return int(h) * 3600 + int(m) * 60 + float(s.replace(',', '.'))

def parse_vtt(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    blocks = re.split(r'\n\n+', text.strip())
    raw = []
    for block in blocks:
        lines   = block.strip().splitlines()
        ts_line = None
        text_lines = []
        for line in lines:
            if '-->' in line:
                ts_line = line.split('align:')[0].strip()
            elif ts_line and line.strip() and not re.match(r'^\d+$', line.strip()) and line.strip() != 'WEBVTT':
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

    # 슬라이딩 윈도우 중복 제거
    segments  = []
    prev_text = ''
    for start, end, orig in raw:
        if orig == prev_text:
            continue
        if orig.startswith(prev_text) and prev_text:
            new_part = orig[len(prev_text):].strip()
            if new_part:
                if segments:
                    segments[-1]['end'] = start
                segments.append({'start': start, 'end': end, 'orig': new_part, 'trans': ''})
        else:
            segments.append({'start': start, 'end': end, 'orig': orig, 'trans': ''})
        prev_text = orig

    return segments


# ── 유틸: Gemini 번역 (VTT 경로용) ──────────────────────
TRANS_SEP = '\n|||\n'

def gemini_batch_translate(texts):
    if not texts:
        return []
    joined = TRANS_SEP.join(texts)
    prompt = (
        "Translate these English subtitle lines into natural Korean (구어체).\n"
        "Lines are separated by '|||'. Keep the SAME number of lines in the SAME order.\n"
        "Preserve tone, humor, nuance. Return ONLY translated lines separated by '|||'.\n\n"
        + joined
    )
    try:
        resp  = stt_model.generate_content(prompt)
        parts = resp.text.strip().split(TRANS_SEP)
        if len(parts) == len(texts):
            return [p.strip() for p in parts]
        result = []
        for i, t in enumerate(texts):
            result.append(parts[i].strip() if i < len(parts) else t)
        return result
    except Exception:
        return texts


# ── 유틸: yt-dlp 청크 다운로드 ───────────────────────────
def get_video_duration(video_id):
    opts = {'quiet': True, 'skip_download': True, 'noplaylist': True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
        return float(info.get('duration', 0))

def download_chunks(video_id):
    duration = get_video_duration(video_id)
    if duration <= 0:
        return []
    n_chunks = math.ceil(duration / CHUNK_SEC)
    url      = f"https://www.youtube.com/watch?v={video_id}"
    chunks   = []
    for i in range(n_chunks):
        start = i * CHUNK_SEC
        end   = min(start + CHUNK_SEC, duration)
        out   = f"{video_id}_chunk{i:03d}.m4a"
        opts  = {
            'format':      '140/bestaudio[ext=m4a]/bestaudio',
            'outtmpl':     out,
            'quiet':       True,
            'noplaylist':  True,
            'download_ranges': yt_dlp.utils.download_range_func(None, [(start, end)]),
            'force_keyframes_at_cuts': False,
            'extractor_args': {'youtube': {'skip': ['dash', 'hls']}},
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        if os.path.exists(out):
            chunks.append((out, start))
    return chunks


# ── 유틸: 청크 STT ───────────────────────────────────────
def transcribe_chunk(audio_path, offset):
    uf = genai.upload_file(path=audio_path, mime_type='audio/mp4')
    wait_until_active(uf)
    prompt = (
        "You are a professional subtitle transcriber and translator.\n"
        "Listen to this audio clip and produce subtitles. For each segment:\n"
        "- Transcribe exact spoken words into 'orig'\n"
        "- Translate naturally into Korean (informal/conversational) into 'trans'\n"
        "- Each segment = one full sentence or natural phrase (min 4-5 words)\n"
        "- Do NOT split single words into separate segments\n"
        "- Timestamps are relative to the START of this clip (start from 0)\n"
        "- Be as precise as possible with timestamps\n"
        "- Preserve tone, humor, sarcasm; localize idioms naturally"
    )
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
        safe_delete_remote(uf)

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


# ── 라우트 ───────────────────────────────────────────────
@app.route('/')
def index():
    return send_file('index.html')


@app.route('/api/subtitles', methods=['GET'])
def get_subtitles():
    video_id = request.args.get('video_id', '').strip()
    if not video_id:
        return jsonify({'error': '비디오 ID가 없습니다.'}), 400

    url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        # ① 유튜브 자막 시도
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

        vtt_file = None
        for fname in os.listdir('.'):
            if fname.startswith(video_id) and fname.endswith('.vtt'):
                vtt_file = fname
                break

        if vtt_file:
            segments = parse_vtt(vtt_file)
            safe_delete_local(vtt_file)
            if segments:
                segments = merge_short_segments(segments)
                originals    = [s['orig'] for s in segments]
                translations = gemini_batch_translate(originals)
                result = [
                    {'start': s['start'], 'end': s['end'], 'orig': s['orig'], 'trans': t}
                    for s, t in zip(segments, translations)
                ]
                return jsonify(result)

        # ② 자막 없으면 청크 STT 폴백
        chunks = download_chunks(video_id)
        if not chunks:
            raise RuntimeError("오디오 청크 다운로드에 실패했습니다.")

        all_segments = []
        for chunk_path, offset in chunks:
            segs = transcribe_chunk(chunk_path, offset)
            all_segments.extend(segs)
            safe_delete_local(chunk_path)

        if not all_segments:
            raise RuntimeError("음성 인식 결과가 없습니다.")

        all_segments = merge_short_segments(all_segments)
        result = [
            {'start': s['start'], 'end': s['end'], 'orig': s['orig'], 'trans': s['trans']}
            for s in all_segments
        ]
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # 남은 임시 파일 정리
        for fname in list(os.listdir('.')):
            if fname.startswith(video_id):
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
        resp = tutor_model.generate_content(prompt)
        return jsonify({'explanation': resp.text})
    except Exception as e:
        return jsonify({'error': f'AI 응답 오류: {str(e)}'}), 500


# ── 전역 에러 핸들러 ─────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': '404 - 경로를 찾을 수 없습니다.'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': f'500 - 서버 내부 오류: {str(e)}'}), 500

@app.errorhandler(Exception)
def unhandled(e):
    return jsonify({'error': f'예상치 못한 오류: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
