import os
import json
import time  
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

    # 💡 제미나이가 좋아하는 m4a 포맷으로 변경!
    audio_filename = f"{video_id}.m4a"
    uploaded_file = None 
    
    try:
        # [과정 1] 유튜브에서 m4a 형식의 소리만 쏙 빼오기
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/140/bestaudio', # m4a 우선 다운로드 강제
            'outtmpl': audio_filename,
            'quiet': True,
            'noplaylist': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

        # [과정 2] 구글 서버에 오디오 업로드 (mime_type 명시)
        uploaded_file = genai.upload_file(path=audio_filename, mime_type="audio/mp4")
        
        # ACTIVE가 될 때까지 대기
        timeout = 60 
        start_time = time.time()
        
        while True:
            file_info = genai.get_file(uploaded_file.name)
            
            if file_info.state.name == 'ACTIVE':
                break
            elif file_info.state.name == 'FAILED':
                raise Exception("구글 서버가 이 오디오 형식을 처리하는 데 실패했습니다.")
            
            if time.time() - start_time > timeout:
                raise Exception("구글 서버 대기 시간이 초과되었습니다.")
                
            time.sleep(2)
            
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
        
        # 제미나이 플래시 모델로 분석
        response = stt_model.generate_content(
            [prompt, uploaded_file],
            generation_config=types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=json_schema
            )
        )
        
        # 데이터 변환
        segments = json.loads(response.text)

        # [과정 3] 번역
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
                
        # 청소 로직
        if uploaded_file:
            genai.delete_file(uploaded_file.name)
        if os.path.exists(audio_filename):
            os.remove(audio_filename)
            
        return jsonify(merged_subtitles)
        
    except Exception as e:
        if uploaded_file:
            try:
                genai.delete_file(uploaded_file.name)
            except:
                pass
        if os.path.exists(audio_filename):
            try:
                os.remove(audio_filename)
            except:
                pass
            
        return jsonify({'error': f'제미나이 음성 인식 오류: {str(e)}'}), 500

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
