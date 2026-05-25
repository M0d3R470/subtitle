import os
import json
import time  
import yt_dlp
from flask import Flask, request, jsonify, send_file
import google.generativeai as genai
from google.generativeai import types
from deep_translator import GoogleTranslator

app = Flask(__name__)

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
    if not video_id: return jsonify({'error': '비디오 ID 없음'}), 400

    audio_filename = f"{video_id}.m4a"
    uploaded_file = None 
    
    try:
        ydl_opts = {'format': 'bestaudio[ext=m4a]/140/bestaudio', 'outtmpl': audio_filename, 'quiet': True, 'noplaylist': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

        uploaded_file = genai.upload_file(path=audio_filename, mime_type="audio/mp4")
        while uploaded_file.state.name == 'PROCESSING': time.sleep(2); uploaded_file = genai.get_file(uploaded_file.name)
            
        prompt = "Listen to this audio and provide a JSON list of {start, end, orig} objects. Ensure the transcript is natural."
        json_schema = {"type": "array", "items": {"type": "object", "properties": {"start": {"type": "number"}, "end": {"type": "number"}, "orig": {"type": "string"}}, "required": ["start", "end", "orig"]}}
        
        response = stt_model.generate_content([prompt, uploaded_file], generation_config=types.GenerationConfig(response_mime_type="application/json", response_schema=json_schema))
        segments = json.loads(response.text)

        # [전체 맥락 번역 로직] 
        # 문장들을 5개씩 묶어서 제미나이에게 '자연스럽게' 번역하라고 지시
        full_text = "\n".join([seg['orig'] for seg in segments])
        translator_prompt = f"다음 영어 자막들을 전체 문맥을 고려하여 아주 자연스러운 한국어 문어체로 번역해줘. 결과는 줄바꿈으로 구분된 리스트 형식으로만 줘.\n{full_text}"
        trans_response = tutor_model.generate_content(translator_prompt)
        translated_lines = trans_response.text.strip().split('\n')

        merged_subtitles = []
        for i, seg in enumerate(segments):
            merged_subtitles.append({
                'start': float(seg['start']), 'end': float(seg['end']),
                'orig': seg['orig'],
                'trans': translated_lines[i] if i < len(translated_lines) else "번역 오류"
            })
                
        if uploaded_file: genai.delete_file(uploaded_file.name)
        if os.path.exists(audio_filename): os.remove(audio_filename)
        return jsonify(merged_subtitles)
        
    except Exception as e:
        return jsonify({'error': f'서버 오류: {str(e)}'}), 500

@app.route('/api/chat', methods=['POST'])
def chat_with_gemini():
    data = request.json
    sentence = data.get('sentence', '')
    # 튜터 응답도 타임아웃을 피하기 위해 더 가벼운 요청 방식으로 변경
    try:
        response = tutor_model.generate_content(f"문장 '{sentence}'에 대해 동생에게 설명하듯 문법과 뜻을 해설해줘.")
        return jsonify({'explanation': response.text})
    except Exception as e:
        return jsonify({'error': 'AI 튜터 응답 실패: 재시도해주세요.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
