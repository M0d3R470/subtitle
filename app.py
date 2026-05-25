# ... existing code ...
        # 3. 자막 원문 추출 (Structured Outputs 스키마 강제 정의)
        # JSON 포맷 불일치로 인한 파싱 에러를 완벽하게 차단합니다.
        extraction_schema = {
            "type": "ARRAY",
# ... existing code ...
                "required": ["start", "end", "orig"]
            }
        }

        # 오디오 원본 음성 전사 요청
        trans_res = model.generate_content(
            [
                "You are a strict professional transcriber. Transcribe the audio WORD-FOR-WORD. "
                "DO NOT summarize. DO NOT merge sentences. DO NOT skip any dialogue. "
                "Output every single spoken sentence sequentially until the end of the audio.",
                audio_file
            ],
            generation_config=types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=extraction_schema,
                temperature=0.1 # 창작(요약)을 억제하기 위해 온도를 0.1로 극한까지 낮춤
            )
        )

        # 안전하게 파싱을 실행합니다 (스키마 강제로 에러 가능성 극단적으로 감소)
        segments = json.loads(trans_res.text)

        # 4. 번역 연산 최적화 (인덱스 보존형 스키마 적용)
        # 번역 전후의 리스트 개수 불일치 에러를 원천적으로 방지합니다.
        translated_segments = []
        chunk_size = 20 # 50개는 모델이 임의로 문장을 합칠 위험이 크므로 20개로 축소

        # 번역 결과에 매핑할 완벽한 스키마 정의
        translation_schema = {
# ... existing code ...
