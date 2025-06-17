import json
from pathlib import Path

# 원본 청크 파일들
files = [
    Path("C:\\Users\\bitcamp\\Desktop\\public\\gl_chunk.json"),
    Path("C:\\Users\\bitcamp\\Desktop\\public\\manual_chunk.json"),
    Path("C:\\Users\\bitcamp\\Desktop\\public\\omnibus_chunk.json"),
    Path("C:\\Users\\bitcamp\\Desktop\\public\\해설서_chunk.json"),
]

# 합쳐서 저장할 파일 경로
merged_path = Path("C:\\Users\\bitcamp\\Desktop\\public\\merged_chunks.json")

# 모든 청크를 담을 리스트
all_chunks = []

# 각 파일 읽어서 all_chunks에 추가
for file_path in files:
    with file_path.open(encoding="utf-8") as f:
        data = json.load(f)
    all_chunks.extend(data)

# merged_chunks.json으로 저장
with merged_path.open("w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

# 최종 청크 개수 확인 및 출력
merged_count = len(all_chunks)
print(f"합쳐진 파일({merged_path.name})의 청크 개수: {merged_count}개")
