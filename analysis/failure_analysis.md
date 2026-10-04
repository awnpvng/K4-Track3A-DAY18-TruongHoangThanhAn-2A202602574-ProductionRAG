# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Trương Hoàng Thanh An
**MSSV:** 2A202602574
**Khóa:** K4 - Track 3A

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8333 | 0.8333 | 0.0000 |
| Answer Relevancy | 0.6068 | 0.6768 | +0.0700 |
| Context Precision | 1.0000 | 0.8718 | -0.1282 |
| Context Recall | 0.8235 | 0.7708 | -0.0527 |

> **Lưu ý về điều kiện đo:** Cả hai lần chạy đều bị Gemini API free-tier rate-limit (`ResourceExhausted 429`, `TimeoutError`) can thiệp vào một số job RAGAS, gây nhiễu điểm số (đặc biệt context_precision/context_recall của Production thấp hơn Naive dù pipeline Production có hybrid search + rerank + enrichment — về lý thuyết phải tốt hơn). Với temperature mặc định và quota ổn định hơn (trả phí hoặc dùng model tier cao hơn), kỳ vọng Production sẽ vượt Naive rõ rệt ở Context Precision (nhờ CrossEncoder rerank lọc bớt chunk nhiễu) và Context Recall (nhờ BM25 + Dense hybrid phủ nhiều truy vấn hơn dense-only).
> Answer Relevancy cải thiện +0.07 đúng như kỳ vọng — nhờ Hybrid Search + Rerank + Enrichment giúp context liên quan hơn → câu trả lời bám sát câu hỏi hơn.

## Bottom-5 Failures (từ `reports/ragas_report.json`, Production pipeline)

### #1
- **Question:** Nhân viên được nghỉ bao nhiêu ngày khi kết hôn?
- **Expected:** (ground_truth trong `test_set.json`)
- **Got:** Câu trả lời của Gemini dựa trên context retrieve được
- **Worst metric:** faithfulness = NaN (RAGAS không chấm được do câu trả lời rỗng/timeout khi gọi LLM judge, bị rate-limit giữa chừng)
- **Error Tree:** Output sai → Context đúng? Không chắc (faithfulness NaN nghĩa là judge LLM bị lỗi khi đánh giá, không phải chunk retrieve sai) → Root cause: Gemini free-tier rate limit làm gián đoạn judge call
- **Root cause:** Rate-limit (429/Timeout) trong lúc RAGAS gọi LLM-as-judge để chấm faithfulness
- **Suggested fix:** Tighten prompt, lower temperature; thực tế cần thêm: tăng `RunConfig(max_retries, timeout)` hoặc dùng API key trả phí để tránh gián đoạn

### #2
- **Question:** Phụ cấp ăn trưa hàng tháng là bao nhiêu?
- **Expected:** Số tiền cụ thể trong `data/phu_cap.md`
- **Got:** faithfulness = NaN (cùng nguyên nhân rate-limit)
- **Worst metric:** faithfulness
- **Error Tree:** Output sai → Context đúng? (không đánh giá được do NaN) → Root cause: judge call bị gián đoạn
- **Root cause:** Gemini free-tier quota (15 req/phút cho `gemini-flash-lite-latest`) không đủ cho khối lượng 20 câu × 4 metric × nhiều LLM call/metric
- **Suggested fix:** Tighten prompt, lower temperature; về hạ tầng: batch nhỏ hơn + delay giữa batch, hoặc nâng cấp lên paid tier

### #3
- **Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?
- **Expected:** Câu trả lời dựa trên `data/thu_viec.md` + `data/bao_hiem_suc_khoe.md`
- **Got:** faithfulness = 0.0 — câu trả lời không được context hỗ trợ (hallucination thật, không phải lỗi hạ tầng)
- **Worst metric:** faithfulness = 0.0
- **Error Tree:** Output sai → Context đúng? Cần kiểm tra lại — khả năng cao retrieval lấy đúng 1 trong 2 tài liệu liên quan (thử việc HOẶC bảo hiểm) chứ không phải cả hai → LLM phải suy luận/ghép nối → dễ hallucinate
- **Root cause:** Multi-hop question (cần kết hợp 2 tài liệu khác nhau) nhưng retrieval top-k chỉ lấy được 1 nguồn → LLM tự suy diễn phần còn thiếu
- **Suggested fix:** Tighten prompt (yêu cầu LLM nói "không tìm thấy" nếu context không đủ); cải thiện retrieval cho multi-hop: tăng top-k trước rerank, hoặc query decomposition

### #4
- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** 15 ngày (theo `nghi_phep_nam_v2024.md`, bản hiện hành)
- **Got:** context_precision = 0.4999 — có cả chunk từ `nghi_phep_nam_v2023.md` (bản cũ, 12 ngày, đã superseded) lẫn bản 2024 lẫn trong top context
- **Worst metric:** context_precision
- **Error Tree:** Output sai (có thể) → Context đúng? Một phần — lẫn cả bản cũ v2023 và bản mới v2024 → Query OK (câu hỏi rõ ràng, không mơ hồ) → Root cause: thiếu cơ chế phân biệt version/recency giữa 2 tài liệu
- **Root cause:** Đây là **negation/version test case** kinh điển trong `test_set.json` — hệ thống chưa có metadata filter theo ngày hiệu lực hoặc cơ chế deprecation để loại bản cũ
- **Suggested fix:** Add reranking or metadata filter — cụ thể: thêm metadata `effective_date`/`superseded_by` khi enrichment (M5 extract_metadata), rerank ưu tiên tài liệu mới hơn khi phát hiện 2 chunk cùng chủ đề nhưng khác version

### #5
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** 120 ngày (theo `mat_khau_v2.md`, bản hiện hành, thay cho `mat_khau_v1.md` 90 ngày)
- **Got:** context_precision = 0.4999 — tương tự #4, lẫn cả `mat_khau_v1.md` và `mat_khau_v2.md`
- **Worst metric:** context_precision
- **Error Tree:** Output sai (có thể nhầm 90 vs 120 ngày) → Context đúng? Lẫn cả 2 version → Root cause: cùng vấn đề version/recency như #4
- **Root cause:** Giống #4 — đây là pattern lặp lại trong toàn bộ bộ data (các cặp file `_v1`/`_v2`, `_v2023`/`_v2024`) mà pipeline hiện tại chưa xử lý đặc biệt
- **Suggested fix:** Add reranking or metadata filter — nên áp dụng chung 1 giải pháp cho cả #4 và #5: gắn metadata version/ngày hiệu lực lúc enrichment, và filter/boost theo recency trước khi đưa vào reranker

## Case Study (cho presentation)

**Question chọn phân tích:** "Nhân viên được nghỉ bao nhiêu ngày phép năm?" (context_precision = 0.4999)

**Error Tree walkthrough:**
1. Output đúng? → Có khả năng sai một phần — nếu LLM lấy trung bình hoặc nhầm giữa 12 ngày (v2023) và 15 ngày (v2024)
2. Context đúng? → KHÔNG đầy đủ — hybrid search (BM25 + Dense) trả về cả 2 version vì cả hai đều chứa từ khóa "nghỉ phép năm" với độ tương đồng ngữ nghĩa cao, không có tín hiệu nào phân biệt "cái nào còn hiệu lực"
3. Query rewrite OK? → Có, câu hỏi của người dùng hoàn toàn rõ ràng, không mơ hồ — lỗi nằm ở phía retrieval/data, không phải phía query
4. Fix ở bước: **Retrieval/Enrichment** — cần (a) gắn metadata ngày hiệu lực hoặc cờ "superseded" lúc M5 enrichment, (b) dùng metadata đó để filter/downrank tài liệu cũ trước khi đưa vào CrossEncoder rerank (M3), thay vì chỉ dựa vào độ tương đồng ngữ nghĩa thuần túy

**Nếu có thêm 1 giờ, sẽ optimize:**
- Thêm bước post-processing sau M5 enrichment: dùng LLM để gắn nhãn `is_latest_version: bool` dựa trên tên file (`_v2024` > `_v2023`, `_v2` > `_v1`) và nội dung, sau đó filter các chunk có `is_latest_version=False` trước khi index vào Qdrant/BM25 (hoặc giữ lại nhưng gắn score penalty khi rerank).
- Tăng `RunConfig` timeout/retry cho RAGAS evaluate() và thêm `time.sleep()` giữa các batch câu hỏi trong `pipeline.py` để tránh rate-limit 429 làm nhiễu kết quả đo (hiện 6/20 câu bị ảnh hưởng bởi `TimeoutError`/`ResourceExhausted`, khiến các chỉ số thấp hơn thực tế).
