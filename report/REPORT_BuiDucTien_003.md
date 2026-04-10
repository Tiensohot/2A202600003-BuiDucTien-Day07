# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Bùi Đức Tiến
**Nhóm:** Vinno
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity (gần 1.0) có nghĩa là hai vector hướng về cùng một phía trong không gian nhiều chiều — tức là hai văn bản có nội dung/ngữ nghĩa tương đồng nhau, bất kể độ dài của chúng.

**Ví dụ HIGH similarity:**
- Sentence A: "The cat sat on the mat."
- Sentence B: "A cat is resting on the mat."
- Tại sao tương đồng: Cả hai câu cùng mô tả một con mèo đang ngồi/nghỉ trên thảm — các từ khóa và chủ đề trùng nhau, vector embedding hướng gần giống nhau.

**Ví dụ LOW similarity:**
- Sentence A: "The stock market crashed yesterday."
- Sentence B: "She baked a chocolate cake for her birthday."
- Tại sao khác: Hai câu thuộc hoàn toàn hai lĩnh vực khác nhau (tài chính vs. nấu ăn), không có từ khóa hay ngữ nghĩa chung → vector embedding hướng rất khác nhau.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity đo góc giữa hai vector nên không bị ảnh hưởng bởi độ dài văn bản (một đoạn 5 từ và 100 từ nói về cùng chủ đề vẫn có similarity cao). Euclidean distance đo khoảng cách tuyệt đối, nên văn bản dài hơn tự nhiên có vector lớn hơn, dẫn đến bias sai về độ tương đồng.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> step = chunk_size - overlap = 500 - 50 = 450
> số chunks = ceil((L - chunk_size) / step) + 1
          = ceil((10,000 - 500) / 450) + 1
          = ceil(9,500 / 450) + 1
          = ceil(21.11) + 1
          = 22 + 1
          = 23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> step = 500 - 100 = 400
    số chunks = ceil((10,000 - 500) / 400) + 1
          = ceil(9,500 / 400) + 1
          = ceil(23.75) + 1
          = 24 + 1
          = 25 chunks
    Chunk count tăng từ 23 lên 25. Overlap lớn hơn → step nhỏ hơn → các chunk gối nhau nhiều hơn → cần nhiều chunk hơn để phủ toàn bộ document. Muốn overlap nhiều hơn để đảm bảo ngữ cảnh không bị cắt đứt ở ranh giới giữa hai chunk — câu/ý tưởng vắt qua ranh giới vẫn xuất hiện đầy đủ ở ít nhất một chunk, tránh mất thông tin khi retrieval.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Vietnamese Law — Văn bản pháp luật Việt Nam (Thông tư, Nghị định, Nghị quyết)

**Tại sao nhóm chọn domain này?**
> Văn bản pháp luật Việt Nam có cấu trúc cực kỳ đồng nhất (Chương → Điều → Khoản → Điểm), tạo điều kiện lý tưởng để thử nghiệm chunking strategy theo cấu trúc so với fixed-size. Domain này còn có tính thực tiễn cao — tra cứu luật là bài toán RAG thực tế mà các công cụ tìm kiếm thông thường xử lý kém vì câu hỏi thường yêu cầu nội dung chính xác của một Điều cụ thể. Ngoài ra, sự đa dạng về kích thước file (từ 11 KB đến 1 MB) cho phép đánh giá tốt hơn về hiệu năng chunking và embedding ở nhiều mức độ khác nhau.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | TT 15/2026/TT-BTC — Nguyên tắc kế toán tài sản mã hóa | Bộ Tài chính (ban hành 04/03/2026) | 11,498 | source, extension, doc_id, chunk_index |
| 2 | TT 167/2012/TT-BTC | Bộ Tài chính (ban hành 2012) | 18,926 | source, extension, doc_id, chunk_index |
| 3 | TT 19/2014/TT-BTP | Bộ Tư pháp (ban hành 2014) | 32,113 | source, extension, doc_id, chunk_index |
| 4 | TT 25/2014/TT-BTP — Kiểm tra kiểm soát TTHC | Bộ Tư pháp (ban hành 31/12/2014) | 30,868 | source, extension, doc_id, chunk_index |
| 5 | NĐ 63/2010/NĐ-CP — Kiểm soát thủ tục hành chính | Chính phủ (ban hành 08/06/2010) | 80,419 | source, extension, doc_id, chunk_index |
| 6 | NQ 66.7/2025/NQ-CP — Cắt giảm TTHC dựa trên dữ liệu | Chính phủ (ban hành 15/11/2025) | 1,106,010 | source, extension, doc_id, chunk_index |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `source` | `str` | `"raw_data/15_2026_TT-BTC_696722.md"` | Truy nguyên chunk về file gốc; dùng để filter theo văn bản cụ thể khi query có prefix "Theo TT 15/2026..." |
| `extension` | `str` | `".md"` | Phân biệt loại file nếu sau này mix `.md` và `.txt`; filter out file không hỗ trợ |
| `doc_id` | `str` | `"15_2026_TT-BTC_696722"` | Nhóm tất cả chunk của cùng một văn bản; dùng cho `delete_document` và `search_with_filter({"doc_id": ...})` |
| `chunk_index` | `int` | `3` | Biết vị trí chunk trong document gốc; hữu ích để reconstruct ngữ cảnh xung quanh (lấy chunk liền trước/sau khi cần) |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
|15_2026_TT-BTC (8,673 chars) | FixedSizeChunker (`fixed_size`) | 18	| 481.8	|  Không (cắt tại ký tự thứ 500, có thể cắt giữa câu)
|
| | SentenceChunker (`by_sentences`) |16 |540.9	 | Một phần (giữ nguyên câu, nhưng nhóm 3 câu/chunk có thể cắt giữa điều khoản)
 |
| | RecursiveChunker (`recursive`) |26	 |332.5	 | Một phần (tách theo thứ tự \n\n → \n → . nên tôn trọng cấu trúc đoạn văn)|
|167_2012_TT-BTC (14,604 chars)	 | FixedSizeChunker (fixed_size)	 |30	 |486.8		 |  Không|
|	 | SentenceChunker (by_sentences)	 |24	 |607.4		 |  Một phần|
|	 | RecursiveChunker (recursive)	 |42	 |346.6	 |  Một phần|


### Strategy Của Tôi

**Loại:** custom strategy → `VietnamLawArticleChunker`

**Mô tả cách hoạt động:**
> `VietnamLawArticleChunker` tách văn bản dựa trên dấu hiệu cấu trúc pháp lý: regex `\*\*\s*[Đđ]i[eề]u\s+\d+` để nhận diện ranh giới mỗi **Điều** (Article). Mỗi Điều được giữ nguyên là một chunk nếu độ dài ≤ `max_chars`; nếu Điều quá dài thì tiếp tục tách theo **Khoản** (numbered clauses `^\d+\.\s`) với greedy buffer merging — cộng dồn các khoản vào buffer cho đến khi vượt `max_chars`, lúc đó flush và bắt đầu buffer mới. Fallback cuối cùng là fixed-size split cho các Điều có bảng phụ lục không có cấu trúc khoản. Header Điều được prepend vào mọi sub-chunk để kết quả retrieval luôn traceable về điều khoản gốc.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Văn bản luật Việt Nam (Thông tư, Nghị định, Nghị quyết) có cấu trúc **rất đồng nhất và có thứ bậc rõ ràng**: Chương → Điều → Khoản → Điểm. Mỗi Điều là một đơn vị pháp lý khép kín (quy định một nghĩa vụ, quyền hạn, hoặc định nghĩa cụ thể), nên truy vấn của người dùng gần như luôn map về một Điều cụ thể. Tách theo Điều đảm bảo chunk không bao giờ chứa nội dung của hai Điều khác nhau — điều mà FixedSize và Recursive không đảm bảo được.

**Code snippet (nếu custom):**
```python
class VietnamLawArticleChunker:
    def __init__(self, max_chars: int = 1200) -> None:
        self.max_chars = max_chars
        self._article_re = re.compile(r"(?m)(?=\*\*\s*[Đđ]i[eề]u\s+\d+[\.\:])")
        self._khoan_re = re.compile(r"(?m)(?=^\d+\.\s)", re.MULTILINE)

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []
        parts = self._article_re.split(text)
        chunks: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= self.max_chars:
                chunks.append(part)
            else:
                header_match = re.match(r"(\*\*[^\n]+\*\*)", part)
                header = header_match.group(1) + "\n" if header_match else ""
                sub_parts = self._khoan_re.split(part)
                buffer = ""
                for sub in sub_parts:
                    sub = sub.strip()
                    if not sub:
                        continue
                    candidate = (buffer + "\n" + sub).strip() if buffer else sub
                    if len(candidate) <= self.max_chars:
                        buffer = candidate
                    else:
                        if buffer:
                            chunks.append(header + buffer if header not in buffer else buffer)
                        if len(sub) > self.max_chars:
                            for i in range(0, len(sub), self.max_chars):
                                piece = (header + sub[i : i + self.max_chars]).strip()
                                if piece:
                                    chunks.append(piece)
                            buffer = ""
                        else:
                            buffer = sub
                if buffer:
                    chunks.append(header + buffer if header not in buffer else buffer)
        return [c for c in chunks if c.strip()]
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| 15_2026_TT-BTC | best baseline (`recursive`) | 26 | 332.5 | Trung bình — tôn trọng đoạn văn nhưng có thể cắt giữa Điều |
| | **VietnamLawArticleChunker** | **9** | **962.7** | **Cao** — mỗi chunk = 1 Điều hoàn chỉnh, boundary rõ ràng |
| 167_2012_TT-BTC | best baseline (`recursive`) | 42 | 346.6 | Trung bình |
| | **VietnamLawArticleChunker** | **10** | **1469.2** | **Cao** — chunk ít hơn nhưng mỗi chunk đầy đủ ngữ nghĩa pháp lý |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi (Tiến) | SentenceChunker | 8 | chunk dễ đọc, giữ ngữ cảnh tốt | Với văn bản pháp luật dài, chunk đôi khi còn quá lớn |
| Thông | FixedSizeChunker | 7 | Dễ implement, output ổn định | Có thể cắt giữa điều/khoản, làm mất ý |
| Hùng | RecursiveChunker | 9/10 | Giữ ngữ cảnh tốt, overlap liên kết chunks | Chunk count hơi cao, có thể dư thừa |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Nếu xét riêng chất lượng truy hồi cho câu hỏi pháp lý chi tiết, strategy tốt nhất là kết hợp RecursiveChunker + metadata filter. RecursiveChunker giúp bám cấu trúc điều/khoản, còn metadata filter giúp thu hẹp đúng loại văn bản như nghị quyết hay thông tư, từ đó giảm nhiễu rõ rệt.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> *Viết 2-3 câu: dùng regex gì để detect sentence? Xử lý edge case nào?*
Dùng regex (?<=[.!?])\s+|(?<=\.)\n với lookbehind assertion để split tại khoảng trắng sau dấu ., !, ? hoặc xuống dòng sau dấu . — cách này giữ dấu câu lại ở cuối câu thay vì bị cắt rời. Edge case xử lý: lọc bỏ chuỗi rỗng sau split (if s.strip()), và max(1, max_sentences_per_chunk) tránh chunk size = 0. Sau đó group theo range(0, len, max_sentences) rồi join bằng dấu cách.

**`RecursiveChunker.chunk` / `_split`** — approach:
> *Viết 2-3 câu: algorithm hoạt động thế nào? Base case là gì?*
Algorithm ưu tiên separator theo thứ tự ["\n\n", "\n", ". ", " ", ""] — thử tách đoạn văn trước, rồi dòng, rồi câu, rồi từ, cuối cùng ký tự. Base case có hai trường hợp: (1) text đã đủ nhỏ (len <= chunk_size) → trả về nguyên; (2) hết separator → force-split theo ký tự. Với mỗi separator tìm thấy, dùng greedy buffer merging: gộp các phần nhỏ vào buffer cho đến khi vượt chunk_size, lúc đó flush buffer và recurse phần quá lớn với rest_seps.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Viết 2-3 câu: lưu trữ thế nào? Tính similarity ra sao?*
Mỗi Document được embed thành vector qua embedding_fn rồi lưu dưới dạng dict {id, content, embedding, metadata} vào self._store (list in-memory). Khi search, embed query thành vector, tính cosine similarity với toàn bộ stored embeddings qua compute_similarity, sort descending, trả về top_k kết quả kèm score. ChromaDB được ưu tiên nếu available, fallback về in-memory.
**`search_with_filter` + `delete_document`** — approach:
> *Viết 2-3 câu: filter trước hay sau? Delete bằng cách nào?*
search_with_filter filter trước: dùng list comprehension để lọc self._store theo metadata_filter (key-value exact match), sau đó mới chạy similarity search trên tập đã lọc — giảm số lượng vector cần tính. delete_document dùng filter-out pattern: gán lại self._store = [r for r if doc_id != target], trả True nếu length giảm.
### KnowledgeBaseAgent

**`answer`** — approach:
> *Viết 2-3 câu: prompt structure? Cách inject context?*
RAG pattern 3 bước: (1) retrieve top_k chunks từ store bằng search(question); (2) format context bằng cách join các chunks với số thứ tự [1] content\n\n[2] content...; (3) inject vào prompt với instruction rõ ràng: "Answer using ONLY the context provided" để tránh hallucination, và "If context does not contain enough information, say so" để handle trường hợp không có đủ context. Prompt được pass thẳng vào llm_fn — agent không giữ state.

### Test Results

```
# Paste output of: pytest tests/ -v
```

**Số tests pass:** _42 / 42__

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 |The dog is playing in the park.	 |A puppy is running around the garden.	 | high |0.0570 | ✗|
| 2 | Machine learning models require large datasets.	| Deep learning needs a lot of training data.	| high |0.0624 | ✗
|
| 3 | The restaurant serves delicious Italian pasta.	| Quantum physics explains subatomic particles.	| low |0.0959 |✗ (cao hơn pair 1, 2!)
 |
| 4 | She locked the door and went to sleep.	|He closed the window and went to bed.	 | high  |-0.1410	 | ✗ (âm!)
|
| 5 |The economy grew by 3% last quarter.| Stock prices fell sharply this morning.	| low | 0.0225| ✓
|

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:*

---Bất ngờ nhất là Pair 3 (hai câu hoàn toàn không liên quan: nhà hàng vs. vật lý lượng tử) lại có score cao hơn Pair 1 và 2 (các câu ngữ nghĩa rất gần nhau), và Pair 4 (gần như cùng nghĩa) cho ra score âm (-0.14). Điều này phơi bày bản chất của MockEmbedder: nó dùng MD5 hash + LCG để tạo vector ngẫu nhiên có tính deterministic, hoàn toàn mù về ngữ nghĩa — hai câu dù đồng nghĩa hay trái nghĩa đều cho vector gần như vuông góc nhau (score ≈ 0). Đây là lý do tại sao production system phải dùng LocalEmbedder (sentence-transformers) hoặc OpenAIEmbedder — chỉ những model được train trên ngữ nghĩa mới tạo ra vector mà cosine similarity thực sự phản ánh mức độ tương đồng về ý nghĩa.

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | TT 15/2026/TT-BTC hướng dẫn nguyên tắc kế toán cho loại tổ chức nào tham gia thị trường tài sản mã hóa tại Việt Nam? | Tổ chức cung ứng dịch vụ tài sản mã hóa (sàn giao dịch, tổ chức phát hành, ví lưu ký) theo quy định tại TT 15/2026 |
| 2 | Theo NĐ 63/2010/NĐ-CP, 'kiểm soát thủ tục hành chính' được định nghĩa như thế nào? | Hoạt động xem xét, đánh giá, xử lý để đảm bảo TTHC được quy định và thực hiện đúng quy định pháp luật (Điều 3) |
| 3 | NQ 66.7/2025/NQ-CP cho phép thay thế những loại giấy tờ nào bằng thông tin khai thác từ CSDL quốc gia về dân cư? | Giấy khai sinh, giấy kết hôn, CMND/CCCD, sổ hộ khẩu và các giấy tờ nhân thân liệt kê trong Phụ lục đính kèm |
| 4 | Theo TT 25/2014/TT-BTP, tổ chức được kiểm tra phải gửi báo cáo cho Đoàn kiểm tra trước bao nhiêu ngày làm việc? | **05 (năm) ngày làm việc** kể từ ngày Đoàn kiểm tra đến làm việc (trường hợp kiểm tra đột xuất không cần gửi trước) |
| 5 | Khi nào cơ quan giải quyết TTHC được phép yêu cầu cá nhân, tổ chức bổ sung hồ sơ theo NQ 66.7/2025? | Chỉ được yêu cầu bổ sung khi hồ sơ chưa đầy đủ thành phần theo quy định; không được yêu cầu bổ sung giấy tờ đã có trong CSDL quốc gia |

### Kết Quả Của Tôi

> **Lưu ý:** Chạy với `MockEmbedder` (hash-based, không semantic) — retrieval quality thấp là expected behavior. Cần `LocalEmbedder` hoặc `OpenAIEmbedder` để có kết quả thực sự.

| # | Query (tóm tắt) | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | TT 15/2026 — nguyên tắc kế toán tổ chức tài sản mã hóa | `66_7_2025_NQ-CP` — **Điều 10. Hiệu lực thi hành** (bảng phụ lục CSDL) | -0.1892 | Không (sai văn bản) | Context về danh sách CSDL đăng ký doanh nghiệp — không liên quan |
| 2 | NĐ 63/2010 — định nghĩa kiểm soát TTHC | `25_2014_TT-BTP` — **Điều 10. Tổ chức thực hiện kiểm tra** | -0.4070 | Không (sai văn bản) | Context về phân công trách nhiệm kiểm tra — không liên quan |
| 3 | NQ 66.7/2025 — thay thế giấy tờ bằng CSDL dân cư | `66_7_2025_NQ-CP` — **Điều 10. Hiệu lực thi hành** (phụ lục bảng số) | -0.3279 | Không (đúng văn bản, sai Điều) | Context về bảng số liệu CSDL — chưa đúng Điều quy định thay thế |
| 4 | TT 25/2014 — báo cáo trước bao nhiêu ngày | `25_2014_TT-BTP` — **Điều 8. Kinh phí thực hiện kiểm tra** | -0.3539 | Không (đúng văn bản, sai Điều) | Context về kinh phí — Điều đúng là Điều 12 khoản a (05 ngày) |
| 5 | NQ 66.7/2025 — khi nào yêu cầu bổ sung hồ sơ | `66_7_2025_NQ-CP` — **Điều 10. Hiệu lực thi hành** | -0.3474 | Không (đúng văn bản, sai Điều) | Context về bảng CSDL — không trả lời được câu hỏi |

**Bao nhiêu queries trả về chunk relevant trong top-3?** **0 / 5** *(đúng văn bản trong top-3: 3/5 — nhưng không đúng Điều cụ thể nào)*

> **Phân tích:** `MockEmbedder` trả về scores âm (−0.19 đến −0.41) và liên tục trả về `Điều 10. Hiệu lực thi hành` (chunk có vector hash trùng lệch nhất) thay vì nội dung semantic phù hợp. Đây là giới hạn cơ bản của hash-mock — không phải lỗi của `VietnamLawArticleChunker`. Với `LocalEmbedder (sentence-transformers)` retrieval quality sẽ cải thiện đáng kể do model hiểu ngữ nghĩa tiếng Việt.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Một thành viên trong nhóm dùng `SentenceChunker` với `max_sentences_per_chunk=5` thay vì 3, kết hợp lọc câu quá ngắn (< 20 ký tự) trước khi gom — cách này giảm đáng kể số chunk "rác" từ các tiêu đề và số điều khoản ngắn trong văn bản luật. Tôi nhận ra rằng tham số `max_sentences` không phải giá trị cố định tốt nhất mà phụ thuộc vào độ dài câu trung bình của domain; văn bản luật Việt Nam có câu rất dài nên 3 câu/chunk có thể đã vượt ngưỡng embedding model. Điều này thúc đẩy tôi thêm bước pre-filtering câu ngắn vào `VietnamLawArticleChunker` trong lần cải tiến tiếp theo.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Một nhóm làm về domain y tế (hướng dẫn điều trị) đã dùng **metadata filtering theo loại tài liệu** (thông tư vs. nghị định vs. hướng dẫn lâm sàng) trước khi chạy similarity search — retrieval precision tăng rõ rệt vì câu hỏi thường chỉ liên quan đến một loại văn bản nhất định. Tôi nhận ra `search_with_filter` của mình đã implement sẵn tính năng này nhưng chưa khai thác trong benchmark, trong khi đó chính là lợi thế lớn nhất của việc gán metadata có cấu trúc. Nếu query có prefix "Theo Thông tư..." thì filter `{"doc_type": "thong_tu"}` trước sẽ loại bỏ phần lớn noise từ các nghị định không liên quan.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ convert các file `.docx` sang Markdown có cấu trúc đầy đủ ngay từ đầu thay vì dùng raw text — nhiều file luật trong `law_data_docx/` có bảng biểu và phụ lục dạng bảng mà khi convert sang plain text bị vỡ thành chuỗi số vô nghĩa, làm `VietnamLawArticleChunker` tạo ra các chunk phụ lục nhiễu rất lớn (thấy rõ qua avg_length = 1469 chars của 167_2012_TT-BTC). Ngoài ra, tôi sẽ thêm trường metadata `dieu_number` (số Điều) và `chuong` (Chương) vào mỗi chunk ngay trong bước chunking — điều này cho phép filter chính xác theo cấu trúc pháp lý thay vì chỉ theo tên file.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | / 10 |
| Similarity predictions | Cá nhân | / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |
