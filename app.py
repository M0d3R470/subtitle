import os
import json
import time  # 💡 기다림을 위한 시간 마법사 추가!
import yt_dlp
from flask import Flask, request, jsonify, send_file
import google.generativeai as genai
from google.generativeai import types
from deep_translator import GoogleTranslator

app = Flask(__name__)

# 1. 제미나이 설정
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    stt_model = genai.GenerativeModel('gemini-3-flash-preview') 
    tutor_model = genai.GenerativeModel('gemini-3.1-pro-preview')

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
    uploaded_file = None # 에러 처리를 위한 초기화
    
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
        
        # 💡 핵심 추가: 구글 서버가 오디오를 완벽하게 인식(ACTIVE)할 때까지 2초마다 확인하며 기다립니다.
        while uploaded_file.state.name == 'PROCESSING':
            time.sleep(2)
            uploaded_file = genai.get_file(uploaded_file.name)
            
        if uploaded_file.state.name == 'FAILED':
            raise Exception("구글 서버가 오디오 파일을 처리하는 데 실패했습니다.")
        
        prompt = "Listen to this audio and transcribe it accurately with exact start and end timestamps."
        
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
        
        # 플래시가 오디오를 듣고 지정된 JSON 형식으로 완벽하게 출력합니다.
        response = stt_model.generate_content(
            [prompt, uploaded_file],
            generation_config=types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=json_schema
            )
        )
        
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
                
        # 모든 작업이 끝나면 구글 서버에 올렸던 파일과 렌더 서버의 파일을 청소합니다.
        if uploaded_file:
            genai.delete_file(uploaded_file.name)
        if os.path.exists(audio_filename):
            os.remove(audio_filename)
            
        return jsonify(merged_subtitles)
        
    except Exception as e:
        # 에러가 났을 때도 찌꺼기가 남지 않게 청소합니다.
        if uploaded_file:
            try:
                genai.delete_file(uploaded_file.name)
            except:
                pass
        if os.path.exists(audio_filename):
            os.remove(audio_filename)
            
        return jsonify({'error': f'제미나이 음성 인식 오류: {str(e)}'}), 500

@app.route('/api/
