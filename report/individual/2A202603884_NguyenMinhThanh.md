# Báo cáo cá nhân · Nguyễn Minh Thành

## 1. Mục tiêu
Hoàn thiện pipeline ingest dữ liệu vào vector store và so sánh ba chiến lược chunking/metadata trên cùng corpus và benchmark.

## 2. Corpus
- Chủ đề: Student Handbook 2025
- Pages: 32 trang nội dung định dạng theo metadata `section`, `audience`, `year`
- File đầu vào: `starter/data/student_handbook_2025.txt`
- Manifest: `starter/data/manifest.json`

## 3. Cấu hình thử nghiệm
- Strategy A: fixed-size chunking, 500 ký tự, overlap 80
- Strategy B: paragraph-based chunking, max_size 600
- Strategy C: strategy B với bộ lọc metadata theo `section`, `audience`

## 4. Benchmark
- Tổng câu hỏi: 20
- Dev: 15
- Test: 5
- Nguồn: `starter/benchmark/questions.json`

## 5. Kết quả chạy
```json
{
  "A": {
    "dev": { "hit_at_1": 1.0, "evidence_hit_3": 1.0 },
    "test": { "hit_at_1": 1.0, "evidence_hit_3": 1.0 }
  },
  "B": {
    "dev": { "hit_at_1": 1.0, "evidence_hit_3": 1.0 },
    "test": { "hit_at_1": 1.0, "evidence_hit_3": 1.0 }
  },
  "C": {
    "dev": { "hit_at_1": 1.0, "evidence_hit_3": 1.0 },
    "test": { "hit_at_1": 1.0, "evidence_hit_3": 1.0 }
  }
}
```

## 6. Nhận xét
- Corpus này đủ nhỏ để chạy nhanh trong PostgreSQL + pgvector local.
- Bộ metadata giúp filter hữu ích cho các câu hỏi theo `section`/`audience`.
- Với benchmark này, cả ba chiến lược đều đạt độ chính xác retrieval tối đa do câu hỏi tương đối trực tiếp và gold evidence rõ ràng.
- Trong lab thực tế, nên thêm các câu hỏi khó hơn và đánh giá thủ công coherence, grounding quality, và metadata usefulness theo tiêu chí trong `GUIDE.md`.

## 7. Lệnh chạy
```bash
cd starter
. ../.venv/bin/activate
docker compose -p day6_student_starter up -d --wait
python -m src.run
```
