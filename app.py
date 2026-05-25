import os
import json
import time  
import yt_dlp
from flask import Flask, request, jsonify, send_file
import google.generativeai as genai
from google.generativeai import types

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
    if not video_id: return jsonify({'error': 'ID 없음'}), 400

    audio_filename = f"{video_id}.m4a"
    try:
        ydl_opts = {'format': 'bestaudio[ext=m4a]/140', 'outtmpl': audio_filename, 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

        uploaded_file = genai.upload_file(path=audio_filename, mime_type="audio/mp4")
        while uploaded_file.state.name == 'PROCESSING': time.sleep(2); uploaded_file = genai.get_file(uploaded_file.name)
            
        # [수정] 덩어리 번역 대신 각 문장마다 자연스럽게 번역 (전체 맥락유지 위해 튜터 프롬프트 활용)
        prompt = "Listen and transcribe into a JSON list of {start, end, orig} objects."
        response = stt_model.generate_content([prompt, uploaded_file], 
            generation_config=types.GenerationConfig(response_mime_type="application/json"))
        
        segments = json.loads(response.text)
        genai.delete_file(uploaded_file.name)
        os.remove(audio_filename)

        # 개별 번역 수행 (더 빠르고 서버에 부담 적음)
        for seg in segments:
            seg['trans'] = tutor_model.generate_content(f"Translate this subtitle to natural Korean: {seg['orig']}").text
                
        return jsonify(segments)
    except Exception as e:
        return jsonify({'error': f'서버 처리 오류: {str(e)}'}), 500

@app.route('/api/chat', methods=['POST'])
def chat_with_gemini():
    data = request.json
    try:
        # 응답 시간을 줄이기 위해 더 짧은 프롬프트 사용
        resp = tutor_model.generate_content(f"Explain this phrase for a learner: {data.get('sentence')}")
        return jsonify({'explanation': resp.text})
    except:
        return jsonify({'error': '튜터 연결이 불안정합니다.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
