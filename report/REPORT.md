# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Bùi Trần Gia Bảo
**Nhóm:** F1
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**

> Cosine similarity đo độ tương đồng theo góc giữa hai véc-tơ. High cosine similarity nghĩa là hai véc-tơ có hướng gần nhau (gần 1). Low (gần 0) nghĩa là hướng lệch nhiều; -1 nghĩa là đối nhau.

**Ví dụ HIGH similarity:**

> - Sentence A: Hợp đồng này có hiệu lực kể từ ngày ký.
> - Sentence B: Thỏa thuận này bắt đầu có hiệu lực từ thời điểm được ký kết.
> - Tại sao tương đồng: Vì cả hai câu đều diễn đạt cùng một ý nghĩa về thời điểm hiệu lực của văn bản.

**Ví dụ LOW similarity:**

> - Sentence A: Quả chuối đang ở trên bàn.
> - Sentence B: Con người đã lên mặt trăng.
> - Tại sao khác: Vì hai câu nói về hai chủ đề hoàn toàn khác nhau, không liên quan về ngữ nghĩa.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**

> Vì cosine similarity loại bỏ sự ảnh hưởng của độ dài véc-tơ vì đã được normalized nên không bị ảnh hưởng. Nếu dùng Euclidean distance, text vector có thể bị coi là xa hơn chỉ vì nhiều từ hơn.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

> Nếu tổng số ký tự bé hơn chunk_size thì chỉ có 1 chunk. Nếu lớn hơn thì số chunk tính theo công thức:

>     Số chunk = ceil((tổng số ký tự - chunk_size) / Số step) + 1
>     Trong đó, số step = chunk_size - overlap

> => ceil((10,000 - 500) / (500 - 50)) + 1
> Đáp án: 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**

> Nếu overlap tăng lên 100 tăng lên 100 => Số step = 400

> => ceil((10_000 - 500)/400) + 1
> Đáp án: 25 chunks.

> Muốn overlap nhiều hơn vì giữ được ngữ cảnh giữa các chunk, tránh mất thông tin khi chunk bị cắt giữa câu/ý, và ải thiện retrieval accuracy trong RAG

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Luật lao động Việt Nam

**Tại sao nhóm chọn domain này?**

> Luật lao động Việt Nam là một lĩnh vực phức tạp với nhiều quy định khác nhau. Việc áp dụng RAG cho domain này sẽ giúp người dùng dễ dàng tra cứu thông tin và giải đáp các thắc mắc liên quan đến luật lao động.

### Data Inventory

| #   | Tên tài liệu          | Nguồn                                                              | Số ký tự | Metadata đã gán |
| --- | --------------------- | ------------------------------------------------------------------ | -------- | --------------- |
| 1   | Bộ luật lao động 2019 | https://datafiles.chinhphu.vn/cpp/files/vbpq/2019/12/45.signed.pdf | 193202   | Không có        |

### Metadata Schema

| Trường metadata             | Kiểu   | Ví dụ giá trị                      | Tại sao hữu ích cho retrieval?                                                                                                                                                                                                                    |
| --------------------------- | ------ | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| document_name (Tên văn bản) | String | "Bộ luật Lao động 2019"            | Hữu ích khi vector database lưu trữ nhiều bộ luật/tài liệu khác nhau. Nó cho phép ứng dụng dùng bộ lọc (filter) chỉ tìm kiếm trong văn bản tương ứng trước khi search bằng vector, tránh bị nhiễu do bị lẫn lộn giữa các luật.                    |
| chuong (Tên Chương)         | String | "Chương III HỢP ĐỒNG LAO ĐỘNG"     | Giúp thu hẹp ngữ cảnh truy vấn vào một nhóm chủ đề lớn. Nếu user hỏi về hợp đồng thì hệ thống có thể ưu tiên lấy các chunk thuộc metadata chương này (Pre-filtering).                                                                             |
| muc (Tên Mục)               | String | "Mục 3 CHẤM DỨT HỢP ĐỒNG"          | Giúp nhóm các điều luật có sự liên kết chặt chẽ với nhau ở mức độ chi tiết hơn hẳn so với Chương.                                                                                                                                                 |
| dieu (Số Điều)              | String | "Điều 15", "Điều 34"               | Cực kỳ hữu ích cho các câu hỏi tra cứu đích danh (VD: "Theo Điều 34 quy định những gì?"). Hệ thống có thể chuyển câu này thành Exact Keyword Match đối chiếu với metadata số Điều để lấy kết quả chính xác 100% thay vì chỉ dùng Semantic search. |
| tieu_de_dieu (Tiêu đề Điều) | String | "Các trường hợp chấm dứt hợp đồng" | Tiêu đề là câu tóm lược chính xác nhất đoạn text bên dưới. Việc gán metadata tiêu đề giúp vector embedding nắm bắt ý nghĩa của chunk này tốt hơn, tăng trọng số và điểm số Similarity score khi tra cứu.                                          |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu               | Strategy                         | Chunk Count | Avg Length | Preserves Context? |
| ---------------------- | -------------------------------- | ----------- | ---------- | ------------------ |
| Luật lao động Việt Nam | FixedSizeChunker (`fixed_size`)  | 1074        | 199.87     | Không              |
| Luật lao động Việt Nam | SentenceChunker (`by_sentences`) | 554         | 346.92     | Có                 |
| Luật lao động Việt Nam | RecursiveChunker (`recursive`)   | 1652        | 115.27     | Có                 |

### Strategy Của Tôi

**Loại:** Document structure-based chunking

**Mô tả cách hoạt động:**

> Strategy này chia tài liệu dựa trên cấu trúc markdown (các heading như #, ##, ###). Mỗi heading được xem như điểm bắt đầu của một section, sau đó các dòng phía dưới sẽ được gom lại thành một chunk. Các section liền kề sẽ được nhóm lại thành chunk theo max_sections_per_chunk. Nếu tài liệu không có heading, hệ thống fallback về fixed-size chunking để đảm bảo vẫn chia được văn bản.

**Tại sao tôi chọn strategy này cho domain nhóm?**

> Văn bản luật có cấu trúc rõ ràng theo chương, mục, điều nên việc chunk theo structure giúp giữ nguyên ngữ cảnh logic của từng điều luật. Điều này giúp retrieval chính xác hơn khi truy vấn theo nội dung hoặc theo từng điều cụ thể.

**Code snippet (nếu custom):**

```python
class DocumentStructureChunker:
    def __init__(self, max_sections_per_chunk: int = 3, fallback_chunk_size: int = 500) -> None:
        self.max_sections_per_chunk = max(1, max_sections_per_chunk)
        self.fallback_chunk_size = fallback_chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        lines = text.splitlines(keepends=True)
        md_header_re = re.compile(r"^\s{0,3}(#{1,6})\s+.*$")

        sections: list[str] = []
        current: list[str] = []
        found_md_heading = False

        for line in lines:
            if md_header_re.match(line):
                if current:
                    sections.append("".join(current).strip())
                current = [line]
                found_md_heading = True
            else:
                current.append(line)

        if current:
            sections.append("".join(current).strip())

        if not found_md_heading or len(sections) == 0:
            return FixedSizeChunker(chunk_size=self.fallback_chunk_size, overlap=0).chunk(text)

        chunks: list[str] = []
        step = self.max_sections_per_chunk
        for i in range(0, len(sections), step):
            chunk = "\n\n".join(s for s in sections[i : i + step] if s).strip()
            if chunk:
                chunks.append(chunk)
        return chunks
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu         | Strategy      | Chunk Count | Avg Length | Retrieval Quality? |
| ---------------- | ------------- | ----------- | ---------- | ------------------ |
| luat_lao_dong.md | best baseline | 555         | 346.29     | Good               |
| luat_lao_dong.md | **của tôi**   | 94          | 2053.37    | Good               |

### So Sánh Với Thành Viên Khác

| Thành viên              | Strategy                                | Retrieval Score (/10) | Điểm mạnh                                                                 | Điểm yếu                                                                                                                                                |
|------------------------|------------------------------------------|----------------------|---------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| Lê Duy Anh             | Custom Strategy (Regex Based Chunking)   | 8.5                  | Bảo toàn ngữ cảnh tốt                                                     | Khi điều luật quá dài, chunk vượt quá context window; hao phí embedding; dư thừa khi truy xuất                                                          |
| Lại Gia Khánh          | Semantic Chunking                        | 8                    | Giữ nguyên đơn vị nghĩa; cải thiện độ chính xác truy vấn; giảm nhiễu       | Phụ thuộc embedding & threshold; cần tuning; tốn tài nguyên; chunk không đồng đều                                                                        |
| Mạc Phương Nga         | FixedSizeChunker                         | 10                   | Đơn giản, nhanh; kiểm soát tốt token                                      | Phụ thuộc chunk_size & overlap; cần thử nghiệm nhiều để tối ưu                                                                                           |
| Nguyễn Phạm Trà My     | AgenticChunker                           | 8                    | Linh hoạt quản lý ngữ cảnh                                                | Chi phí cao; chậm do gọi API LLM cho từng đoạn                                                                                                          |
| Trương Minh Sơn        | Parent–Child Chunking                    | 7.8                  | Trả lời khá chính xác; Top-1 thường chứa đáp án                           | Một số query lan man; có case mất mạch thông tin; Top-K chứa nhiều chunk không liên quan → nhiễu context                                                 |
| Bùi Trần Gia Bảo       | DocumentStructureChunker                 | 6                    | Giữ cấu trúc tài liệu; phù hợp markdown pháp lý                           | Phụ thuộc chất lượng markdown; nếu cấu trúc kém hoặc quá dài → chunk mất cân bằng, ảnh hưởng retrieval                                                  |

**Strategy nào tốt nhất cho domain này? Tại sao?**

> Với domain luật lao động, FixedSizeChunker là lựa chọn tốt nhất vì giúp kiểm soát chặt chẽ độ dài chunk, đảm bảo phù hợp với context window của LLM và tối ưu hiệu suất embedding. Dù không bảo toàn hoàn toàn ngữ nghĩa như semantic chunking, nhưng trong thực tế nó cho kết quả ổn định, nhanh và ít bị lỗi do chunk quá dài hoặc không đồng đều.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:

> Sử dụng regex (?<=[.!?])\s+ để detect sentence boundary dựa trên dấu câu kết thúc (., !, ?) theo sau bởi whitespace. Trước khi split, normalize trường hợp ".\n" thành ". " để tránh bị tách sai khi xuống dòng. Các câu sau đó được group thành chunk theo max_sentences_per_chunk và đồng thời loại bỏ empty hoặc whitespace-only sentences để tránh noise.

**`RecursiveChunker.chunk` / `_split`** — approach:

> Chunker hoạt động theo kiểu recursive splitting với các separators (\n\n, \n, . , " ", "") để giữ được context tốt nhất. Base case là khi độ dài text nhỏ hơn chunk_size thì return trực tiếp, hoặc khi không còn separator thì fallback về fixed-size chunking. Ở mỗi bước, text được split theo separator hiện tại, sau đó recursively xử lý từng phần nhỏ hơn để đảm bảo chunk cuối cùng không vượt quá giới hạn.

### EmbeddingStore

**`add_documents` + `search`** — approach:

> Mỗi chunk được embed thành vector và normalize về unit length trước khi lưu vào store (in-memory hoặc ChromaDB). Khi search, query cũng được embed và normalize, sau đó tính similarity bằng dot product (tương đương cosine similarity do đã normalize). Các kết quả được sort theo score giảm dần và trả về top_k chunks có độ tương đồng cao nhất.

**`search_with_filter` + `delete_document`** — approach:

> Với search_with_filter, hệ thống thực hiện pre-filter trên metadata trước (ví dụ theo doc_id, chuong, …), sau đó mới chạy similarity search trên tập đã lọc để giảm nhiễu. Hàm delete_document xóa toàn bộ chunks thuộc một document bằng cách loại bỏ các record có metadata["doc_id"] tương ứng khỏi store.

### KnowledgeBaseAgent

**`answer`** — approach:

> Áp dụng RAG pipeline: đầu tiên retrieve top_k chunks liên quan từ vector store, sau đó inject vào prompt dưới dạng CONTEXT kèm source và score. Prompt được thiết kế với instruction rõ ràng yêu cầu model chỉ sử dụng context để trả lời, tránh hallucination. Cuối cùng, toàn bộ context + question được truyền vào LLM để generate câu trả lời grounded.

### Test Results

```
============================== test session starts ==============================
platform win32 -- Python 3.11.9, pytest-9.0.3, pluggy-1.6.0 -- E:\Desktop\lockin\Day-07-Lab-Data-Foundations\venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: E:\Desktop\lockin\Day-07-Lab-Data-Foundations
plugins: anyio-4.13.0
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED     [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED      [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED     [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================== 42 passed in 0.11s ===============================
```

**Số tests pass:** ** 42/42 **

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A                             | Sentence B                                     | Dự đoán | Actual Score | Đúng? |
| ---- | -------------------------------------- | ---------------------------------------------- | ------- | ------------ | ----- |
| 1    | "Anh ấy đang đọc sách trong thư viện." | "Người đàn ông đang ngồi đọc sách ở thư viện." | High    | 0.79         | Có    |
| 2    | "Công ty đã tăng lương cho nhân viên." | "Nhân viên được giảm lương trong năm nay."     | Low     | 0.78         | Sai   |
| 3    | "Hôm nay trời rất lạnh."               | "Nhiệt độ giảm mạnh trong ngày hôm nay."       | High    | 0.53         | Sai   |
| 4    | "Tôi thích ăn pizza vào cuối tuần."    | "Cuối tuần tôi thường đi bơi với bạn."         | Low     | 0.57         | Có    |
| 5    | "Chiếc xe này chạy rất nhanh."         | "Tốc độ của chiếc xe này rất cao."             | High    | 0.66         | Có    |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**

> Kết quả bất ngờ nhất là Cặp 2, khi hai câu mang ý nghĩa trái ngược (tăng lương vs giảm lương) nhưng similarity vẫn khá cao. Điều này cho thấy embedding model phụ thuộc nhiều vào lexical overlap và chủ đề chung hơn là hiểu logic phủ định. Nó có thể nhận diện các từ liên quan như “lương”, “nhân viên” nhưng không nắm rõ quan hệ ngữ nghĩa đối lập.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| #   | Query                                                                                                                                                                                                                                   | Gold Answer                                                                                                                                                                                                                                                    |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Bộ luật Lao động năm 2019 (Luật số 45/2019/QH14) chính thức có hiệu lực thi hành kể từ ngày tháng năm nào?                                                                                                                              | Ngày 01 tháng 01 năm 2021.                                                                                                                                                                                                                                     |
| 2   | Theo Bộ luật Lao động 2019, hợp đồng lao động được phân loại thành mấy loại chính? Đó là những loại nào?                                                                                                                                | Gồm 02 loại chính: Hợp đồng lao động không xác định thời hạn và Hợp đồng lao động xác định thời hạn (thời hạn không quá 36 tháng). (Lưu ý: Đã bỏ Hợp đồng lao động theo mùa vụ hoặc theo một công việc nhất định có thời hạn dưới 12 tháng so với bộ luật cũ). |
|     |
| 3   | Quy định pháp luật không cho phép áp dụng thời gian thử việc đối với trường hợp người lao động giao kết loại hợp đồng lao động nào?                                                                                                     | Không áp dụng thử việc đối với người lao động giao kết hợp đồng lao động có thời hạn dưới 01 tháng.                                                                                                                                                            |
| 4   | Theo quy định, thời gian thử việc tối đa đối với công việc của người quản lý doanh nghiệp (theo quy định của Luật Doanh nghiệp, Luật Quản lý, sử dụng vốn nhà nước đầu tư vào sản xuất, kinh doanh tại doanh nghiệp) là bao nhiêu ngày? | Không quá 180 ngày.                                                                                                                                                                                                                                            |
| 5   | Trong dịp lễ Quốc khánh 02/9, người lao động được nghỉ làm việc và hưởng nguyên lương tổng cộng bao nhiêu ngày?                                                                                                                         | 02 ngày (Bao gồm ngày 02 tháng 9 dương lịch và 01 ngày liền kề trước hoặc sau ngày 02 tháng 9).                                                                                                                                                                |
|     |

### Kết Quả Của Tôi

| #   | Query                                                                                                      | Top-1 Retrieved Chunk (tóm tắt)                     | Score | Relevant? | Agent Answer (tóm tắt)             |
| --- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ----- | --------- | ---------------------------------- |
| 1   | Bộ luật Lao động năm 2019 (Luật số 45/2019/QH14) chính thức có hiệu lực thi hành kể từ ngày tháng năm nào? | Header văn bản luật, không chứa trực tiếp đáp án    | 0.77  | Có        | Ngày 01/01/2021                    |
| 2   | Theo Bộ luật Lao động 2019, hợp đồng lao động được phân loại thành mấy loại chính? Đó là những loại nào?   | Nội dung liên quan đến phân loại hợp đồng lao động  | 0.73  | Có        | Gồm 2 loại hợp đồng lao động chính |
| 3   | Quy định pháp luật không cho phép áp dụng thời gian thử việc đối với trường hợp nào?                       | Đoạn liên quan đến quy định thử việc (không đầy đủ) | 0.80  | Có        | Tham khảo Điều 27                  |
| 4   | Thời gian thử việc tối đa đối với người quản lý doanh nghiệp là bao nhiêu ngày?                            | Nội dung không liên quan đến thời gian thử việc     | 0.70  | Không     | Không tìm thấy thông tin           |
| 5   | Trong dịp lễ Quốc khánh 02/9, người lao động được nghỉ bao nhiêu ngày?                                     | Nội dung điều luật không liên quan trực tiếp        | 0.72  | Không     | Không xác định rõ số ngày nghỉ     |

**Bao nhiêu queries trả về chunk relevant trong top-3?** \_\_ 3 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**

> Việc lựa chọn chunking strategy ảnh hưởng rất lớn đến quality của retrieval, không chỉ về độ chính xác mà còn về độ ổn định. Đặc biệt, FixedSizeChunker tuy đơn giản nhưng lại cho kết quả rất consistent và dễ kiểm soát trong thực tế. Điều này giúp tôi hiểu rằng giải pháp tốt nhất không phải lúc nào cũng là giải pháp phức tạp nhất.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**

> Qua demo của nhóm khác, tôi nhận ra tầm quan trọng của việc kết hợp nhiều kỹ thuật như semantic chunking và metadata filtering để cải thiện retrieval. Một số nhóm còn tối ưu prompt và cách inject context để giảm hallucination, giúp câu trả lời chính xác và grounded hơn.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**

> Nếu làm lại, tôi sẽ giảm kích thước chunk hoặc kết hợp thêm overlap để tránh việc chunk quá lớn làm giảm hiệu quả retrieval. Ngoài ra, tôi cũng sẽ cải thiện metadata (như gán rõ điều, chương) để hỗ trợ filtering tốt hơn. Tôi sẽ thử hybrid approach (kết hợp structure + fixed-size) để cân bằng giữa context và performance.

---

## Tự Đánh Giá

| Tiêu chí                    | Loại    | Điểm tự đánh giá |
| --------------------------- | ------- | ---------------- |
| Warm-up                     | Cá nhân | 5 / 5            |
| Document selection          | Nhóm    | 9 / 10           |
| Chunking strategy           | Nhóm    | 13 / 15          |
| My approach                 | Cá nhân | 8 / 10           |
| Similarity predictions      | Cá nhân | 5 / 5            |
| Results                     | Cá nhân | 6 / 10           |
| Core implementation (tests) | Cá nhân | 30 / 30          |
| Demo                        | Nhóm    | 5 / 5            |
| **Tổng**                    |         | **81 / 100**     |
