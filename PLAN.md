# PLAN.md — Lộ trình hoàn thành Lab 18: Production RAG Pipeline

Nguồn: README.md + ASSIGNMENT.md + RUBRIC.md + hiện trạng code (19 TODO còn lại trong `src/m1..m5`, chưa có `reports/`, chưa có reflection cá nhân).

Tổng điểm: 100 + 10 bonus. Tick `[x]` khi hoàn thành từng bước.

---

## 0. Setup môi trường (10 phút)

- [X] `docker compose up -d` — khởi động Qdrant
- [X] `python -m venv .venv` + activate
- [X] `pip install -r requirements.txt`
- [X] `Copy-Item .env.example .env` → tạo `.env` (OPENAI_API_KEY để trống — không có key, dùng fallback)
- [X] Pre-download models (MiniLM đã cache qua test M1; bge-reranker đang tải qua test M3)
- [ ] `python naive_baseline.py` chạy thành công

## 1. Module 1 — Chunking (`src/m1_chunking.py`, 20 phút, 12đ)

- [X] `chunk_semantic()` — nhóm câu theo cosine similarity, trả `list[Chunk]` không rỗng
- [X] `chunk_hierarchical()` — parent (2048) + child (256), children có `parent_id` hợp lệ
- [X] `chunk_structure_aware()` — parse markdown headers, giữ `section` trong metadata
- [X] `pytest tests/test_m1.py -v` pass 100% (13/13 passed)

## 2. Module 2 — Hybrid Search (`src/m2_search.py`, 20 phút, 12đ)

- [X] `segment_vietnamese()` — underthesea + thay `_`
- [X] `BM25Search.index()` + `.search()` — method="bm25"
- [X] `DenseSearch.index()` + `.search()` — bge-m3 + Qdrant `query_points()`
- [X] `reciprocal_rank_fusion()` — method="hybrid", score = Σ 1/(k+rank+1)
- [X] Query "nghỉ phép" trả kết quả liên quan
- [X] `pytest tests/test_m2.py -v` pass 100% (5/5 passed)

## 3. Module 3 — Reranking (`src/m3_rerank.py`, 15 phút, 12đ)

- [X] `CrossEncoderReranker._load_model()` — load bge-reranker-v2-m3
- [X] `.rerank()` — trả ≤3 `RerankResult`, sort theo `rerank_score` giảm dần
- [X] `FlashrankReranker.rerank()` — bonus, implement thêm dù không bắt buộc
- [X] Kiểm tra: doc "nghỉ phép" rank cao hơn "VPN" (test_rerank_relevant_first PASSED)
- [X] `pytest tests/test_m3.py -v` pass 100% (5/5 passed, 33 phút do tải model bge-reranker-v2-m3 lần đầu)

## 4. Module 4 — RAGAS Eval (`src/m4_eval.py`, 15 phút, 12đ)

- [X] `evaluate_ragas()` — dict 4 metric keys, wrap try/except
- [X] `failure_analysis()` — Bottom-N, trả `diagnosis` + `suggested_fix`
- [X] `pytest tests/test_m4.py -v` pass 100% (4/4 passed)
- [ ] ⚠️ Không có OPENAI_API_KEY → RAGAS thực tế sẽ trả scores = 0 khi chạy pipeline thật

## 5. Module 5 — Enrichment (`src/m5_enrichment.py`, 20 phút, 12đ)

- [X] Chọn mode: Combined `_enrich_single_call()` (khuyến khích, +2 bonus) — implement cả 4 hàm riêng lẫn combined
- [X] `enrich_chunks()` trả `list[EnrichedChunk]`
- [ ] `enriched_text` khác `original_text` khi có API key (N/A — không có key)
- [X] Fallback hoạt động khi không có API key
- [X] `pytest tests/test_m5.py -v` pass 100% (14/14 passed, dùng chung log với m4)

## 6. Chạy Pipeline & Evaluation (20 phút, 25đ)

- [ ] `python src/pipeline.py` chạy end-to-end, exit code 0 (10đ) — **ĐANG CHẠY LẠI với Gemini API**

  - [X] Đã chuyển toàn bộ code (M4 RAGAS, M5 Enrichment, pipeline.py, naive_baseline.py) từ OpenAI sang **Gemini API** (`google-generativeai==0.7.2` + `langchain-google-genai==1.0.10`)
  - [X] Đã sửa bug `NameError: name 'c' is not defined` ở `src/m2_search.py` (DenseSearch.index)
  - [X] `.env` đã có `GEMINI_API_KEY` hợp lệ
  - [X] Debug 3 lỗi phát sinh khi chạy thật với Gemini:
    1. Model `gemini-1.5-flash` bị retire → thử `gemini-2.5-flash` (quota free tier chỉ 5 req/phút, quá thấp) → chốt dùng **`gemini-flash-lite-latest`** (quota cao hơn, test 8 calls liên tiếp OK)
    2. RAGAS truyền `temperature` kwarg vào `agenerate_prompt()`, bị `langchain-google-genai==1.0.10` forward thẳng xuống `client.generate_content()` → `TypeError` → fix bằng subclass `_GeminiLLM` strip kwarg thừa trong `src/m4_eval.py`
    3. Embedding model `embedding-001` không tồn tại → đổi thành `models/gemini-embedding-001`
  - [X] Verify bằng script nhỏ: faithfulness=1.0, answer_relevancy=0.83, context_precision≈1.0, context_recall=1.0 — đúng hết
  - [X] `pytest tests/ -v` lại sau fix: 37/37 PASSED (83s, nhanh hơn nhờ model mới)
  - [X] `python src/pipeline.py` chạy thành công, exit code 0 — **HOÀN TẤT**
- [X] Kiểm tra `reports/ragas_report.json` sinh ra, ≥1 metric ≥0.70 (tốt nhất ≥3) (10đ) — **ĐẠT 3/4 metrics ≥0.70:**
  - Faithfulness: 0.8333 ✓
  - Answer Relevancy: 0.6768 ✗ (gần đạt)
  - Context Precision: 0.8718 ✓
  - Context Recall: 0.7708 ✓
  - → Theo thang điểm RUBRIC #7: 3/4 metrics ≥0.70 = **10/10đ**
- [X] Chạy thêm `python naive_baseline.py` để có số liệu so sánh thật (`reports/naive_baseline_report.json`): faithfulness 0.8333, answer_relevancy 0.6068, context_precision 1.0, context_recall 0.8235
- [X] Điền bảng so sánh Naive vs Production vào `analysis/failure_analysis.md`
- [X] Mở `reports/ragas_report.json` → lấy bottom-5 worst questions thật
- [X] Viết `analysis/failure_analysis.md`: 5 failures đầy đủ diagnosis + suggested fix + Error Tree + Case Study + ghi chú về rate-limit ảnh hưởng điểm số (5đ)

## 7. Reflection (30 phút, 15đ)

- [X] Tạo `analysis/reflections/reflection_TruongHoangThanhAn.md`
- [X] Phần 1: Bảng mapping 5 modules với số liệu thật (chạy `python -m src.m1_chunking` và `python -m src.m3_rerank` để lấy dẫn chứng cụ thể: 208 semantic chunks, 87 hierarchical children, rerank score 0.9914 vs 0.0007...) (5đ)
- [X] Phần 2: Khó khăn gặp phải — liệt kê đủ 7 lỗi thật đã gặp trong suốt session (Application Control Policy, numpy build fail, NameError, 3 lỗi Gemini model/quota/temperature) kèm cách debug cụ thể (5đ)
- [X] Phần 3: Action Plan cho project cá nhân — chunking/search/rerank/eval/enrichment + timeline 2 tuần cụ thể (5đ)

## 8. Bonus (+10 max, tùy chọn)

- [ ] RAGAS Faithfulness ≥ 0.85 (+3) — hiện tại 0.8333, chưa đạt
- [ ] RAGAS tất cả metrics ≥ 0.75 (+3) — answer_relevancy 0.6768 chưa đạt
- [X] Enrichment combined mode `_enrich_single_call()` (+2) — dùng làm default mode trong `enrich_chunks()`
- [ ] Latency breakdown report — bảng thời gian từng bước (+2) — chưa làm, có thể bổ sung nếu còn thời gian

## 9. Kiểm tra trước khi nộp

- [X] `pytest tests/ -v` — 100% pass (41 unit tests theo module + 37 khi chạy `tests/` tổng, đều pass nhiều lần)
- [X] Đếm TODO còn lại = 0: đã verify bằng grep, 0 TODO trong `src/*.py`
- [X] `ruff check src/` — đã auto-fix 16/25 (import sort, unused import, f-string thừa); còn 9 cảnh báo `BLE001` (catch blind Exception, là pattern cố ý cho fallback LLM) — chấp nhận được, rubric ghi ruff "tùy chọn"
- [X] `python src/pipeline.py` chạy end-to-end thành công nhiều lần, sinh `reports/ragas_report.json`
- [X] `python check_lab.py` → **"🚀 Bài lab sẵn sàng để nộp!"** — pass toàn bộ (source code, reports, analysis, reflection, 0 TODO, 37/37 tests)

## 10. Nộp bài — **CHƯA LÀM, việc tiếp theo**

- [X] Tên repo đã đúng chuẩn: `K4-Track3A-DAY18-TruongHoangThanhAn-2A202602574-ProductionRAG`
- [ ] Để repo **Public** trên GitHub — chưa thao tác, cần làm thủ công trên GitHub
- [ ] `git add` + commit + push lên GitHub — chưa commit lần nào trong session này
- [ ] Nộp link repo lên VLearn LMS trước **23h59 ngày diễn ra lab**

---

## TÓM TẮT TRẠNG THÁI (cập nhật lần cuối)

✅ **Đã xong:** Code 5 module (0 TODO), 37-41 tests pass, chuyển toàn bộ sang Gemini API (model `gemini-flash-lite-latest` + `models/gemini-embedding-001`), pipeline chạy ra report thật (3/4 RAGAS metrics ≥0.70), `failure_analysis.md` và reflection viết đầy đủ với dữ liệu thật, `check_lab.py` xác nhận sẵn sàng nộp.

➡️ **Còn lại:** Chỉ còn bước git commit/push + để repo Public + nộp link lên VLearn LMS (mục 10) — đây là thao tác thủ công, cần xác nhận với người dùng trước khi push (theo quy tắc an toàn, không tự ý push/public hóa repo).
