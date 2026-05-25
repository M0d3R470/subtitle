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

    audio_filename = f"{video_id}.webm"
    uploaded_file = None 
    
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
        
        # 🔥 얄짤없는 '무조건 대기' 로직 (ACTIVE가 될 때까지 못 지나감)
        timeout = 120 # 최대 60초 대기
        start_time = time.time()
        
        while True:
            file_info = genai.get_file(uploaded_file.name)
            
            # 1. 소화 완료! (가장 원하던 상태)
            if file_info.state.name == 'ACTIVE':
                break
            # 2. 에러 발생 (구글 서버 뻗음)
            elif file_info.state.name == 'FAILED':
                raise Exception("구글 서버가 오디오 파일을 처리하는 데 실패했습니다.")
            
            # 3. 60초가 넘어가면 무한루프 방지
            if time.time() - start_time > timeout:
                raise Exception("구글 서버 대기 시간이 초과되었습니다.")
                
            time.sleep(2) # 2초 쉬고 다시 확인
            
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
            os.remove(audio_filename)
            
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
