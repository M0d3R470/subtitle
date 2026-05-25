import os
import yt_dlp
from flask import Flask, request, jsonify, send_file
import google.generativeai as genai
from openai import OpenAI
from deep_translator import GoogleTranslator

app = Flask(__name__)

# 1. 제미나이 설정
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-pro') 

# 2. OpenAI Whisper 설정
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/api/subtitles', methods=['GET'])
def get_subtitles():
    video_id = request.args.get('video_id')
    if not video_id:
        return jsonify({'error': '비디오 ID가 없습니다.'}), 400
    
    if not OPENAI_API_KEY:
        return jsonify({'error': 'OpenAI API 키가 설정되지 않았습니다. (Secrets를 확인하세요!)'}), 500

    # m4a 대신 변환기가 필요 없는 webm 포맷으로 변경
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

        # [과정 2] OpenAI Whisper API로 텍스트 변환
        with open(audio_filename, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file, 
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
        
        # 파일 삭제
        if os.path.exists(audio_filename):
            os.remove(audio_filename)

        # [과정 3] 다국어 원문을 한국어로 실시간 번역
        merged_subtitles = []
        translator = GoogleTranslator(source='auto', target='ko')
        
        safe_segments = transcript.segments if transcript.segments else []
        
        for segment in safe_segments:
            original_text = segment.text.strip()
            if original_text:
                try:
                    translated_text = translator.translate(original_text)
                except:
                    translated_text = "(번역 지연)"
                    
                merged_subtitles.append({
                    'start': segment.start,
                    'end': segment.end,
                    'orig': original_text,
                    'trans': translated_text
                })
                
        return jsonify(merged_subtitles)
        
    except Exception as e:
        if os.path.exists(audio_filename):
            os.remove(audio_filename)
        return jsonify({'error': f'AI 음성 인식 중 오류가 발생했습니다: {str(e)}'}), 500

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
        response = model.generate_content(prompt)
        return jsonify({'explanation': response.text})
    except Exception as e:
        return jsonify({'error': 'AI 응답 오류가 발생했습니다.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
