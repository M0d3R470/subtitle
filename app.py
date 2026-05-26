import os
import re
import json
import time
import yt_dlp
from flask import Flask, request, jsonify, send_file
import google.generativeai as genai
from google.generativeai import types
from pydantic import BaseModel
from typing import List

app = Flask(__name__)

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")

genai.configure(api_key=GOOGLE_API_KEY)
stt_model   = genai.GenerativeModel('gemini-2.5-flash')
tutor_model = genai.GenerativeModel('gemini-2.5-flash')

class Segment(BaseModel):
    start: float
    end:   float
    orig:  str
    trans: str

MIN_WORDS   = 4
MAX_WORDS   = 18
MAX_GAP_SEC = 1.5
SENTENCE_ENDS = {'.', '!', '?', '...'}
TRANS_SEP   = '\n|||\n'


# ── 파일 삭제 ─────────────────────────────────────────────
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


# ── Gemini 파일 대기 ──────────────────────────────────────
def wait_until_active(uf, timeout=120):
    deadline = time.time() + timeout
    while True:
        state = genai.get_file(uf.name).state.name
        if state == 'ACTIVE':
            return
        if state == 'FAILED':
            raise RuntimeError("구글 서버 오디오 처리 실패.")
        if time.time() > deadline:
            raise TimeoutError("구글 서버 대기 시간 초과.")
        time.sleep(2)


# ── 세그먼트 병합: 일단 전체 텍스트로 합치기 ───────────────
def merge_all(segments):
    """json3 단어 단위 세그먼트를 타임스탬프와 함께 보존하며 합침."""
    if not segments:
        return []
    # 타임스탬프 인덱스: 각 단어가 몇 초에 시작하는지
    words_with_ts = []
    for seg in segments:
        words = seg['orig'].strip().split()
        if not words:
            continue
        dur_per_word = max(0.01, (seg['end'] - seg['start']) / len(words))
        for i, w in enumerate(words):
            words_with_ts.append({
                'word':  w,
                'start': round(seg['start'] + i * dur_per_word, 3),
                'end':   round(seg['start'] + (i+1) * dur_per_word, 3),
            })
    return words_with_ts


def gemini_sentence_split(words_with_ts):
    """
    Gemini에게 전체 텍스트를 주고 문장 경계 인덱스를 반환받아
    단어 타임스탬프 기준으로 자막 세그먼트를 만듦.
    """
    if not words_with_ts:
        return []

    full_text = ' '.join(w['word'] for w in words_with_ts)

    prompt = (
        "You are a subtitle editor. Split the following transcript into natural subtitle segments.\n"
        "Rules:\n"
        "- Each segment = one complete sentence or natural spoken clause\n"
        "- Split at sentence boundaries based on MEANING and GRAMMAR, not just punctuation\n"
        "- Each segment should be 5-15 words ideally\n"
        "- Return ONLY a JSON array of strings, each string being one subtitle segment\n"
        "- Do NOT translate. Keep the original English.\n\n"
        f"Transcript: {full_text}"
    )

    try:
        resp     = stt_model.generate_content(prompt)
        raw      = (resp.text or '').strip()
        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
            raw = raw.strip()
        sentences = json.loads(raw)
        if not isinstance(sentences, list):
            raise ValueError("not a list")
    except Exception as e:
        print(f"[문장분리 오류] {e}", flush=True)
        # 폴백: 문장부호 기준으로 단순 분리
        sentences = re.split(r'(?<=[.!?])\s+', full_text)

    # 각 문장을 단어 타임스탬프에 매핑
    result   = []
    word_idx = 0
    total    = len(words_with_ts)

    for sentence in sentences:
        s_words = sentence.strip().split()
        if not s_words or word_idx >= total:
            continue

        seg_start = words_with_ts[word_idx]['start']
        end_idx   = min(word_idx + len(s_words) - 1, total - 1)
        seg_end   = words_with_ts[end_idx]['end']

        result.append({
            'start': seg_start,
            'end':   seg_end,
            'orig':  sentence.strip(),
            'trans': ''
        })
        word_idx += len(s_words)

    return result


# ── json3 자막 파싱 ───────────────────────────────────────
def parse_json3(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    segments = []
    for ev in data.get('events', []):
        start_ms = ev.get('tStartMs', 0)
        dur_ms   = ev.get('dDurationMs', 0)
        text     = ''.join(s.get('utf8', '') for s in ev.get('segs', [])).strip()
        text     = re.sub(r'\s+', ' ', text).strip()
        if not text:
            continue
        OFFSET = 0.7  # 유튜브 자막은 실제 발화보다 약간 늦게 설정됨
        start  = max(0, round(start_ms/1000 - OFFSET, 3))
        end    = max(start + 0.1, round((start_ms + dur_ms)/1000 - OFFSET, 3))
        segments.append({'start': start, 'end': end, 'orig': text, 'trans': ''})
    deduped, prev = [], ''
    for seg in segments:
        if seg['orig'] != prev:
            deduped.append(seg)
            prev = seg['orig']
    return deduped


# ── Gemini 번역 ───────────────────────────────────────────
def gemini_batch_translate(texts):
    if not texts:
        return []
    results = []
    BATCH = 60
    for i in range(0, len(texts), BATCH):
        chunk  = texts[i:i+BATCH]
        # 번호 붙여서 보내기 → Gemini가 순서 안 헷갈리고 파싱도 쉬움
        numbered = '\n'.join(f"{j+1}. {t}" for j, t in enumerate(chunk))
        prompt = (
            f"Translate the following {len(chunk)} English subtitle lines into natural Korean (informal 구어체).\n"
            "Each line starts with a number. Return ONLY the translated lines with the SAME numbers.\n"
            "Do NOT add explanations. Preserve tone, humor, nuance.\n\n"
            + numbered
        )
        try:
            resp = stt_model.generate_content(prompt)
            raw  = resp.text.strip()
            # "1. 번역" 형식으로 파싱
            import re
            parsed = {}
            for m in re.finditer(r'^(\d+)\. (.+)$', raw, re.MULTILINE):
                parsed[int(m.group(1))] = m.group(2).strip()
            translated = [parsed.get(j+1, chunk[j]) for j in range(len(chunk))]
            results.extend(translated)
        except Exception as e:
            print(f"[번역 오류] {e}", flush=True)
            results.extend(chunk)
    return results


# ── STT 폴백 ─────────────────────────────────────────────
def transcribe_full(video_id):
    audio_path = f"{video_id}.m4a"
    uf = None
    try:
        opts = {
            'format':      '140/bestaudio[ext=m4a]/bestaudio',
            'outtmpl':     audio_path,
            'quiet':       True,
            'noplaylist':  True,
            'extractor_args': {'youtube': {'skip': ['dash', 'hls']}},
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

        if not os.path.exists(audio_path):
            raise FileNotFoundError("오디오 다운로드 실패.")

        uf = genai.upload_file(path=audio_path, mime_type='audio/mp4')
        safe_delete_local(audio_path)  # 업로드 즉시 로컬 삭제 → 메모리 확보
        wait_until_active(uf)

        prompt = (
            "You are a professional subtitle transcriber and translator. "
            "Listen to this audio and produce subtitles. For each segment: "
            "'orig' = exact spoken words, "
            "'trans' = natural Korean translation (informal/conversational). "
            "Each segment = one full sentence or phrase (min 4-5 words). "
            "Do NOT split single words into separate segments. "
            "Timestamps in seconds from the start. "
            "Preserve tone, humor, sarcasm; localize idioms."
        )
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
        segs = json.loads(raw.strip())
        return [
            {
                'start': round(float(s.get('start', 0)), 3),
                'end':   round(float(s.get('end',   0)), 3),
                'orig':  (s.get('orig') or '').strip(),
                'trans': (s.get('trans') or '').strip(),
            }
            for s in segs if (s.get('orig') or '').strip()
        ]
    finally:
        safe_delete_remote(uf)
        safe_delete_local(audio_path)


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
        # ① json3 자막 시도 (실패해도 STT 폴백으로 넘어감)
        try:
            ydl_sub_opts = {
                'skip_download':     True,
                'writesubtitles':    True,
                'writeautomaticsub': True,
                'subtitleslangs':    ['en', 'en-US', 'en-GB'],
                'subtitlesformat':   'json3',
                'outtmpl':           video_id,
                'quiet':             True,
                'noplaylist':        True,
            }
            with yt_dlp.YoutubeDL(ydl_sub_opts) as ydl:
                ydl.download([url])
        except Exception:
            pass  # 429 등 실패해도 STT 폴백으로 계속 진행

        json3_file = None
        for fname in os.listdir('.'):
            if fname.startswith(video_id) and fname.endswith('.json3'):
                json3_file = fname
                break

        if json3_file:
            segments = parse_json3(json3_file)
            safe_delete_local(json3_file)
            if segments:
                # 단어 단위 타임스탬프 보존하며 합치기
                words_with_ts = merge_all(segments)
                # Gemini가 문맥 보고 문장 단위로 분리
                segments      = gemini_sentence_split(words_with_ts)
                # 번역
                translations  = gemini_batch_translate([s['orig'] for s in segments])
                return jsonify([
                    {'start': s['start'], 'end': s['end'], 'orig': s['orig'], 'trans': t}
                    for s, t in zip(segments, translations)
                ])

        # ② 자막 없으면 Gemini STT 폴백
        all_segments = transcribe_full(video_id)
        if not all_segments:
            raise RuntimeError("음성 인식 결과가 없습니다.")
        all_segments = gemini_sentence_split(merge_all(all_segments))
        return jsonify([
            {'start': s['start'], 'end': s['end'], 'orig': s['orig'], 'trans': s['trans']}
            for s in all_segments
        ])

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        for fname in list(os.listdir('.')):
            if fname.startswith(video_id):
                safe_delete_local(fname)


@app.route('/api/chat', methods=['POST'])
def chat_with_gemini():
    data     = request.get_json(silent=True) or {}
    sentence = data.get('sentence', '').strip()
    if not sentence:
        return jsonify({'error': '문장이 없습니다.'}), 400

    prompt = (
        f'문장을 해설해주세요.\n\n'
        f'문장: "{sentence}"\n\n'
        f'조건:\n'
        f'1. 문장 구조 해설 (주어, 동사, 목적어, 핵심 문법)\n'
        f'2. 중요 표현·단어·숙어 설명\n'
        f'3. 정중한 한국어 존댓말\n'
        f'4. 짧고 간결하게 해설\n'
        f'5. 줄바꿈을 적극 활용하여 가독성을 높임\n'
    )
    try:
        resp = tutor_model.generate_content(prompt)
        return jsonify({'explanation': resp.text})
    except Exception as e:
        return jsonify({'error': f'AI 응답 오류: {str(e)}'}), 500


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
