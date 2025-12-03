from pydub import AudioSegment

def cut_mp3(input_file, output_file, minutes=2):
    try:
        # MP3 파일 불러오기
        print(f"파일을 로딩 중입니다: {input_file}")
        audio = AudioSegment.from_mp3(input_file)

        # 자를 시간 계산 (pydub은 밀리초 단위를 사용)
        # 2분 = 2 * 60 * 1000 = 120,000ms
        cut_time = minutes * 60 * 1000

        # 처음부터 지정된 시간까지 자르기
        # (파일이 2분보다 짧으면 전체가 선택됩니다)
        cut_audio = audio[:cut_time]

        # 저장하기 (16kHz로 샘플링 레이트 설정)
        cut_audio.export(output_file, format="mp3", parameters=["-ar", "16000"])
        print(f"성공적으로 저장되었습니다: {output_file}")

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

# 사용 예시
if __name__ == "__main__":
    # 입력 파일명과 출력 파일명을 지정하세요
    input_path = "/mnt/ddn/kyudan/Audio-data-centric/podcast-pipeline/𝗣𝗲𝗿𝘀𝗼𝗻𝗮𝗹𝗶𝘁𝗶𝗲𝘀  𝗟𝗲𝗮𝗿𝗻 𝗘𝗻𝗴𝗹𝗶𝘀𝗵 𝗤𝘂𝗶𝗰𝗸𝗹𝘆 𝘄𝗶𝘁𝗵 𝗣𝗼𝗱𝗰𝗮𝘀𝘁  𝗘𝗽𝗶𝘀𝗼𝗱𝗲 109 - English Podcast Zone.mp3" 
    output_path = "test_english_with_music.mp3"
    
    cut_mp3(input_path, output_path)