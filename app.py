import os, json, time, yt_dlp
from flask import Flask, request, jsonify, send_file
import google.generativeai as genai

app = Flask(__name__)
genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-3-flash-preview')

@app.route('/')
def index(): return send_file('index.html')

@app.route('/api/subtitles', methods=['GET'])
def get_subtitles():
    video_id = request.args.get('video_id')
    audio_filename = f"{video_id}.m4a"
    try:
        # 1. 유튜브 소리 추출
        ydl_opts = {'format': 'bestaudio[ext=m4a]/140', 'outtmpl': audio_filename, 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        
        audio_file = genai.upload_file(path=audio_filename)
        while audio_file.state.name == 'PROCESSING': time.sleep(2); audio_file = genai.get_file(audio_file.name)
            
        # 2. 자막 원문 추출
        trans_res = model.generate_content(["Extract subtitles as JSON list of {start, end, orig}.", audio_file])
        segments = json.loads(trans_res.text)
        genai.delete_file(audio_file.name); os.remove(audio_filename)

        # 3. 50줄씩 쪼개서 번역 (배열 슬라이싱!)
        translated_segments = []
        chunk_size = 50
        for i in range(0, len(segments), chunk_size):
            chunk = segments[i:i + chunk_size]
            texts = [s['orig'] for s in chunk]
            
            prompt = f"다음 문장들을 문맥을 유지하며 자연스러운 한국어로 번역해. JSON 리스트로 답해: {json.dumps(texts)}"
            trans_res = model.generate_content(prompt)
            translated_texts = json.loads(trans_res.text)
            
            for j, seg in enumerate(chunk):
                seg['trans'] = translated_texts[j] if j < len(translated_texts) else "번역오류"
                translated_segments.append(seg)
            
        return jsonify(translated_segments)
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat_with_gemini():
    data = request.json
    resp = model.generate_content(f"Explain: {data.get('sentence')}")
    return jsonify({'explanation': resp.text})

if __name__ == '__main__': app.run(host='0.0.0.0', port=5000)
