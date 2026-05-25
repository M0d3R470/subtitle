import os
import json
import yt_dlp
from flask import Flask, request, jsonify, send_file
import google.generativeai as genai
from google.generativeai import types  # 최신 엄격한 데이터 규격용 모듈
from deep_translator import GoogleTranslator

app = Flask(__name__)

# 1. 제미나이 3.0 최신 버전 설정
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    # 음성 인식은 미친 속도의 'gemini-3.0-flash', 해설은 튜터용 최고 성능인 'gemini-3.0-pro'로 세팅!
    stt_model = genai.GenerativeModel('gemini-3.0-flash') 
    tutor_model = genai.GenerativeModel('gemini-3.0-pro')

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/api/subtitles', methods=['GET'])
def get_subtitles():
    video_id = request.args.get('video_id')
    if not video_id:
        return jsonify({'error': '비디오 ID가 없습니다.'}), 400
    
    if not GOOGLE_API_KEY:
        return jsonify({'error': 'Gemini API 키가 설정되지 않았습니다.'}), 500

    audio_filename = f"{video_id}.webm"
    
    try:
        # [과정 1] 유튜브에서 소리만 다운로드
        ydl_opts = {
            'format': 'bestaudio[ext=webm]/bestaudio',
            'outtmpl': audio_filename,
            'quiet': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

        # [과정 2] 구글 서버에 오디오 업로드
        uploaded_file = genai.upload_file(path=audio_filename)
        
        prompt = "Listen to this audio and transcribe it accurately with exact start and end timestamps."
        
        # 💡 Gemini 3.0의 핵심 기능: 자막 데이터 규격을 억까당하지 않게 딱 고정합니다.
        json_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "orig": {"type": "string"}
                },
                "required": ["start", "end", "orig"]
            }
        }
        
        # 3.0 플래시가 오디오를 듣고 지정된 JSON 형식으로 완벽하게 출력합니다.
        response = stt_model.generate_content(
            [prompt, uploaded_file],
            generation_config=types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=json_schema
            )
        )
        
        # 파일 청소
        genai.delete_file(uploaded_file.name)
        if os.path.exists(audio_filename):
            os.remove(audio_filename)

        # 안전하게 데이터 변환
        segments = json.loads(response.text)

        # [과정 3] 다국어 원문을 한국어로 실시간 번역
        merged_subtitles = []
        translator = GoogleTranslator(source='auto', target='ko')
        
        for seg in segments:
            original_text = seg.get('orig', '').strip()
            if original_text:
                try:
                    translated_text = translator.translate(original_text)
                except:
                    translated_text = "(번역 지연)"
                    
                merged_subtitles.append({
                    'start': float(seg.get('start', 0)),
                    'end': float(seg.get('end', 0)),
                    'orig': original_text,
                    'trans': translated_text
                })
                
        return jsonify(merged_subtitles)
        
    except Exception as e:
        if os.path.exists(audio_filename):
            os.remove(audio_filename)
        return jsonify({'error': f'제미나이 3.0 음성 인식 오류: {str(e)}'}), 500

@app.route('/api/chat', methods=['POST'])
def chat_with_gemini():
    data = request.json
    sentence = data.get('sentence', '')
    
    prompt = f"""
    당신은 친절한 언어 튜터입니다. 다음 문장에 대해 한국어로 친절하게 해설해줘. 
    문장: "{sentence}"
    
    조건:
    1. 문장 구조 해설 (주어, 동사, 핵심 문법 등)
    2. 중요 표현이나 단어 설명
    3. 동생에게 알려주듯 다정하고 친근한 말투 사용
    4. 가독성을 위해 HTML 태그(<strong>, <br> 등)를 적절히 사용해서 예쁘게 꾸며줘.
    """
    try:
        if not GOOGLE_API_KEY:
            return jsonify({'error': 'Gemini API 키가 설정되지 않았습니다.'}), 500
        response = tutor_model.generate_content(prompt)
        return jsonify({'explanation': response.text})
    except Exception as e:
        return jsonify({'error': 'AI 응답 오류가 발생했습니다.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
