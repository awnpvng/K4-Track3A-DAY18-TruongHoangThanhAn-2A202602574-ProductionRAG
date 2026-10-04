# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Trương Hoàng Thanh An
**MSSV:** 2A202602574
**Khóa:** K4 - Track 3A
**Ngày hoàn thành:** 2026-10-04

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.85 tạo **208 chunks** (avg 99 ký tự, min 6, max 354) so với basic (paragraph-split) chỉ 51 chunks (avg 410 ký tự). Semantic chunking cắt nhỏ hơn nhiều vì mỗi khi độ tương đồng cosine giữa 2 câu liên tiếp < 0.85 là tách chunk mới — với văn bản chính sách tiếng Việt câu ngắn, xen kẽ chủ đề, nên số lượng chunk tăng mạnh. Ưu điểm: không cắt giữa ý; nhược điểm: chunk quá nhỏ (có chunk chỉ 6 ký tự) có thể thiếu context khi retrieve. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Tạo 11 parent chunks (≤2048 ký tự) và 87 child chunks (avg 239, max 255 ký tự, đúng giới hạn child_size=256). Mỗi child có `parent_id` trỏ về đúng parent. Đây là chiến lược dùng trong `pipeline.py` production vì cân bằng tốt giữa độ chính xác retrieval (child nhỏ, cụ thể) và đủ ngữ cảnh khi trả lời (trả về parent lớn hơn). |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | 106 chunks, giữ nguyên header markdown (`##`) trong metadata `section`. Avg 197 ký tự nhưng max lên tới 789 — vì có section dài (bảng, danh sách) không bị cắt giữa chừng, bảo toàn tính toàn vẹn cấu trúc tài liệu tốt hơn semantic/hierarchical thuần text. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | RRF với k=60 kết hợp điểm xếp hạng (không phải điểm thô) của BM25 (lexical, chính xác từ khóa tiếng Việt sau khi segment bằng underthesea) và Dense bge-m3 (semantic). Giải quyết vấn đề: BM25 tốt khi query có từ khóa chính xác ("nghỉ phép", "mật khẩu") nhưng kém với câu hỏi diễn đạt khác đi; Dense tốt với semantic similarity nhưng có thể bỏ sót match từ khóa chính xác (số liệu, tên riêng). RRF cho kết quả ổn định hơn từng phương pháp riêng lẻ. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Test thực tế: câu hỏi "Nhân viên được nghỉ phép bao nhiêu ngày?" — model `bge-reranker-v2-m3` chấm "Nhân viên được nghỉ 12 ngày/năm" = **0.9914**, "Thời gian thử việc là 60 ngày" = 0.0206, "Mật khẩu thay đổi mỗi 90 ngày" = 0.0007. Khoảng cách điểm rất rõ ràng (gần 1.0 vs gần 0.0) cho thấy cross-encoder phân biệt tốt hơn nhiều so với bi-encoder/BM25 vì nó encode query+document cùng lúc (attention chéo) thay vì encode riêng rồi so cosine. Đánh đổi: phải chạy model cho từng cặp (query, doc) nên tốn thời gian hơn — chỉ áp dụng sau khi đã lọc còn ~20 candidates từ Hybrid Search (M2), không chạy trên toàn bộ corpus. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Kết quả thực tế trên Production pipeline (20 câu hỏi, dùng Gemini `gemini-flash-lite-latest` làm LLM judge + `gemini-embedding-001`): Faithfulness=0.8333, Answer Relevancy=0.6768, Context Precision=0.8718, Context Recall=0.7708. Answer Relevancy thấp nhất — cho thấy dù context đủ liên quan (precision/recall khá cao) nhưng câu trả lời của LLM đôi khi lạc đề hoặc quá dài dòng so với câu hỏi gốc. |
| Contextual embeddings / Enrichment | M5 | `contextual_prepend()` / `_enrich_single_call()` | Pipeline dùng combined mode (`_enrich_single_call`, 1 API call/chunk thay vì 4) để sinh summary + hypothesis questions + context prepend + metadata cùng lúc, tiết kiệm chi phí/quota đáng kể — quan trọng vì Gemini free-tier rate-limit rất chặt (15 req/phút), enrichment 100 chunks mà gọi 4 lần/chunk sẽ vượt quota ngay lập tức. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  1. `Program 'pip.exe' failed to run: An Application Control policy has blocked this file` — Windows Application Control Policy chặn `pip.exe` chạy trực tiếp.
  2. `error: metadata-generation-failed ... numpy` khi `pip install -r requirements.txt` — do máy chạy Python 3.13/3.14, chưa có wheel prebuilt cho numpy/ragas/langchain trên Windows, pip phải build từ source qua `meson` nhưng thiếu compiler.
  3. `NameError: name 'c' is not defined` tại `src/m2_search.py` (trong `DenseSearch.index`) — biến `c` dùng ngoài scope trong list comprehension khi tạo `PointStruct`.
  4. `404 models/gemini-1.5-flash is not found` — model Gemini cũ đã bị retire.
  5. `TypeError: GenerativeServiceClient.generate_content() got an unexpected keyword argument 'temperature'` — RAGAS gọi `agenerate_prompt(..., temperature=...)`, nhưng `langchain-google-genai==1.0.10` forward thẳng kwarg `temperature` xuống `client.generate_content()` của Google SDK (API không nhận tham số này ở layer đó) → crash toàn bộ 80 job RAGAS.
  6. `429 ResourceExhausted — Quota exceeded... quota_value: 5` — model `gemini-2.5-flash` free tier chỉ cho 5 request/phút, không đủ cho 20 câu × 4 metric RAGAS.
  7. `404 models/embedding-001 is not found for embedContent` — tên embedding model sai, Gemini API hiện dùng `models/gemini-embedding-001`.

- **Nguyên nhân gốc rễ & Cách debug:**
  - Lỗi (1)(2): dùng `python -m pip` thay vì gọi `pip.exe` trực tiếp để né Application Control Policy; sau đó phát hiện nguyên nhân sâu hơn là Python 3.13/3.14 quá mới, cài thêm Python 3.12 qua `winget install Python.Python.3.12` và tạo lại `.venv` bằng `py -3.12 -m venv .venv`.
  - Lỗi (3): đọc traceback chỉ đúng dòng lỗi, nhận ra biến `c` chỉ tồn tại trong comprehension `texts = [c["text"] for c in chunks]` ở dòng trên, không tồn tại trong comprehension tạo `points` — sửa bằng cách index trực tiếp `chunks[i]`.
  - Lỗi (4)(6)(7): dùng `genai.list_models()` để liệt kê model thực tế khả dụng với API key, lọc theo `supported_generation_methods` chứa `generateContent`/`embedContent`, test từng model bằng gọi thử (`generate_content("Say OK")` lặp 8 lần liên tiếp để kiểm tra quota) trước khi chốt `gemini-flash-lite-latest`.
  - Lỗi (5): đọc source code `langchain_google_genai/chat_models.py` để trace đường đi của kwarg `temperature` — phát hiện `_generate`/`_agenerate` nhận `**kwargs` rồi forward thẳng vào `_chat_with_retry(request=..., **kwargs, generation_method=self.client.generate_content)`. Fix bằng cách tạo subclass `_GeminiLLM(ChatGoogleGenerativeAI)` override `_generate`/`_agenerate` để `kwargs.pop("temperature", None)` trước khi gọi `super()`.

- **Kiến thức còn thiếu & Cách khắc phục:**
  - Chưa quen với việc các thư viện LLM wrapper (`langchain-google-genai`, `ragas`) có version compatibility rất chặt với nhau (`langchain-core<0.3` bắt buộc cho `ragas==0.1.x`) — khắc phục bằng cách luôn kiểm tra `pip check` sau khi cài thêm package mới, và tìm version cụ thể tương thích thay vì cài bản mới nhất.
  - Chưa biết cách tra cứu quota/rate-limit thực tế của từng model Gemini — khắc phục bằng cách viết script nhỏ test trực tiếp bằng vòng lặp gọi API thay vì đoán từ tài liệu (tài liệu có thể không cập nhật kịp các model mới).

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Hệ thống RAG hỏi-đáp chính sách nội bộ doanh nghiệp (tiếp nối từ data mẫu của lab)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Đã có đủ 5 module (Hierarchical Chunking → Hybrid Search BM25+Dense+RRF → CrossEncoder Rerank → Gemini LLM answer → RAGAS eval) chạy end-to-end, dùng Qdrant in-memory (chưa có Docker Desktop lúc build) và Gemini `gemini-flash-lite-latest` (free tier).
- **Vấn đề / Bottlenecks đang gặp:** (1) Context Precision/Recall bị lẫn giữa tài liệu cũ và mới (ví dụ `nghi_phep_nam_v2023.md` vs `v2024.md`, `mat_khau_v1.md` vs `v2.md`) — hệ thống chưa phân biệt được version còn hiệu lực. (2) Rate-limit free tier của Gemini gây nhiễu kết quả đo RAGAS (TimeoutError/429 ảnh hưởng ~6/20 câu).

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Giữ **Hierarchical** (parent 2048 / child 256) làm chính vì cân bằng tốt nhất giữa độ chính xác retrieval và đủ ngữ cảnh trả lời; bổ sung **Structure-aware** cho riêng các tài liệu có bảng/danh sách (policy có bảng mức lương, phụ cấp) để không cắt vỡ bảng.
2. **Search retrieval:** Giữ **Hybrid (BM25 + Dense + RRF)** vì data tiếng Việt có nhiều số liệu/tên riêng cần match chính xác (BM25) lẫn câu hỏi diễn đạt tự do (Dense).
3. **Reranking:** Có dùng — `CrossEncoderReranker` (bge-reranker-v2-m3), vì độ phân tách điểm số rất rõ ràng (0.99 vs 0.0007 trong test thực tế) giúp lọc nhiễu tốt trước khi đưa vào LLM.
4. **Evaluation:** RAGAS 4 metrics làm chuẩn, nhưng bổ sung thêm metric tùy chỉnh "version correctness" — kiểm tra câu trả lời có trích dẫn đúng bản tài liệu mới nhất hay không (dựa trên failure analysis phát hiện lỗi version/recency lặp lại nhiều lần).
5. **Enrichment:** Dùng **combined mode** (`_enrich_single_call`, 1 API call/chunk) để tiết kiệm quota; mở rộng schema metadata để thêm field `is_latest_version` / `superseded_by` nhằm giải quyết vấn đề #4 ở trên.

#### 3. Timeline triển khai
- **Tuần 1:** Thêm metadata version/recency vào M5 enrichment + rerank ưu tiên tài liệu mới nhất; setup Docker Qdrant persistent thay vì in-memory.
- **Tuần 2:** Nâng cấp Gemini API lên paid tier (hoặc thêm `RunConfig` retry/backoff hợp lý) để loại bỏ nhiễu rate-limit khi benchmark; viết thêm test case multi-hop và negation vào bộ eval; tối ưu Answer Relevancy (metric thấp nhất hiện tại) bằng cách tinh chỉnh prompt template yêu cầu LLM bám sát câu hỏi hơn.
