# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)

Temperature thay đổi output của model, nhưng không đổi chất lượng câu chữ: 0.0 và 0.7 cho gần như cùng nội dung (36 phố phường), 1.2 nhảy sang "Hà Nội có hơn 70 hồ nước", 1.8 sang truyền thuyết Hồ Gươm. Không mức nào kém mạch lạc, kể cả 1.8 — em nghĩ vì `top_p=0.9` đã cắt đuôi phân phối trước khi temperature kịp phát huy.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**

Hợp đồng pháp lý: 0.0. Những tài liệu liên quan đến hợp đồng, tài liệu pháp lý cần sự chính xác 100%.

Slogan: 1.2–1.5, sinh nhiều rồi chọn. Slogan hay content liên quan đến sáng tạo thì cần để temp cao để model có nhiều lựa chọn sáng tạo và đa dạng hơn.

Khác biệt: hợp đồng tối ưu để tránh trường hợp xấu nhất, slogan tối ưu để chạm trường hợp tốt nhất.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**

20M token output/ngày. Theo bảng giá trong template: gpt-4o $200/ngày, gpt-4o-mini $12/ngày — chênh 16,7 lần (~5.640 USD/tháng). Cặp Gemini em chạy: flash $50, flash-lite $8.

Model lớn: rà soát hợp đồng trước khi ký — bỏ sót một điều khoản có thể mất hàng trăm triệu, 200 USD/ngày là số lẻ.

Model nhỏ: phân loại ticket hỗ trợ vào 8 nhóm — sai thì ticket chỉ đi nhầm hàng đợi, chi phí lỗi gần bằng không.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)

Độ dài chênh 2,3 lần — nhà thơ 288 từ, kỹ sư 655 từ — dù không prompt nào nhắc đến độ dài. Giọng văn: nhà thơ mở bằng "Ôi bạn hiền" và ẩn dụ xuyên suốt; kỹ sư trung tính, có heading, bullet, ví dụ bộ lọc thư rác. Mức kỹ thuật: bản nhà thơ không một thuật ngữ, bản kỹ sư có ngay "explicitly programmed", "patterns". Vậy system prompt điều khiển giọng văn, độ dài, thuật ngữ và cả cấu trúc trình bày — nhưng không đổi phần lõi kiến thức, cả hai đều nói đúng ý "học quy luật từ dữ liệu".

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**

Đoạn văn em dùng dài 184 từ (mô tả Hà Nội, đủ dấu tiếng Việt).

**Chênh nhau 5,3%:** tiktoken đếm 259 token, ước lượng `184 / 0.75` cho 245 token.

**Dự toán THIẾU**, vì ước lượng cho ra số nhỏ hơn số thật — hóa đơn cuối tháng sẽ cao hơn con số em dự trù khoảng 5%.

**Vì sao:** ước lượng `/0.75` giả định 1,33 token cho mỗi từ, nhưng đoạn tiếng Việt này thực tế tốn **1,41 token/từ**. Em thử đối chiếu một đoạn tiếng Anh cùng kiểu nội dung thì chỉ 1,14 token/từ. Chênh lệch nằm ở chỗ bộ mã hóa được huấn luyện chủ yếu trên tiếng Anh: từ tiếng Anh thông dụng thường có sẵn một token nguyên, còn chữ tiếng Việt có dấu bị tách thành nhiều mảnh. Con số 0.75 trong công thức vốn là quy tắc rút ra từ tiếng Anh, mang sang tiếng Việt thì luôn lệch về phía thiếu.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)

Hưởng lợi nhất là (b) trợ lý giọng nói, hơn cả chatbot. Với chatbot, streaming chỉ cải thiện *cảm nhận* — người dùng thấy chữ chạy nên biết hệ thống còn sống, nhưng vẫn phải đợi hết câu mới hiểu. Với giọng nói, streaming rút ngắn thời gian *thực*: có câu đầu là đẩy sang TTS phát ngay trong lúc model còn đang sinh, hai công đoạn chạy song song thay vì nối tiếp; mà ngưỡng chịu đựng của người nghe cũng khắt khe hơn — im lặng 3 giây trong hội thoại là bất thường, còn con trỏ nhấp nháy 3 giây thì bình thường. (c) pipeline chạy đêm không cần chút nào, thậm chí streaming còn có hại: khó retry hơn và chặn mất đường dùng batch API vốn rẻ hơn ~50%.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**

**Backoff giúp gì so với delay cố định:** server đang quá tải cần được giảm tải để hồi phục. Với delay cố định 1 giây, 1.000 client sẽ gửi đúng 1.000 request mỗi giây, mãi mãi — server không bao giờ được nghỉ. Với backoff nhân đôi, cùng 1.000 client đó gửi 1.000 request ở giây thứ nhất, rồi thứ hai, thứ tư, thứ tám... Tải do retry gây ra tự giảm một nửa sau mỗi vòng, cho server khoảng trống để đứng dậy.

**Jitter giải quyết vấn đề gì còn sót lại:** backoff làm tải thưa dần *theo thời gian*, nhưng không trải nó ra *bên trong* mỗi khoảng chờ. Nếu 1.000 client cùng nhận lỗi tại một thời điểm (server vừa sập), tất cả cùng tính ra delay y hệt nhau — 0,1s rồi 0,2s rồi 0,4s. Kết quả là chúng vẫn ập vào cùng lúc, chỉ là thưa hơn: 1.000 request dồn vào một khoảnh khắc, rồi im lặng, rồi lại 1.000 request dồn vào một khoảnh khắc. Server nhận các cú đấm tập trung đúng lúc nó yếu nhất.

Jitter phá vỡ sự trùng khớp đó bằng cách cho mỗi client chờ một khoảng ngẫu nhiên:

```python
time.sleep(random.uniform(0, base_delay * (2 ** attempt)))
```

Client A chờ 0,03s, client B chờ 0,27s, client C chờ 0,11s. Cùng 1.000 request đó giờ rải đều trên toàn bộ cửa sổ chờ thay vì dồn cục — server thấy dòng chảy đều thay vì từng đợt sóng.

`retry_with_backoff` em viết ở Task 3.2 chưa có jitter, đúng theo yêu cầu đề bài. Nếu đưa lên production với nhiều client thì đây là thứ phải thêm đầu tiên.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**

> Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt.

Em xóa từng cụm rồi hỏi lại cùng một câu để đo, thay vì đoán:

"ngắn gọn" — đòn bẩy mạnh nhất. Hỏi "Embedding trong NLP hoạt động thế nào?": bản đầy đủ trả lời 120 từ, bỏ cụm này ra 698 từ — dài gấp 5,8 lần, kèm heading và mục đánh số. Hậu quả không chỉ là khó đọc: mỗi phản hồi đều được nhồi ngược vào `history`, nên token input của mọi lượt sau tăng theo, chi phí leo thang dần suốt phiên chat.

"thân thiện" — đổi giọng chứ không đổi độ dài. Hỏi "Em vẫn chưa hiểu overfitting là gì": bản đầy đủ mở bằng "Chào bạn!" rồi giải thích bằng ví von "học vẹt", "học thuộc lòng bài kiểm tra" (82 từ). Bỏ "thân thiện" đi thì mất lời chào, mất ví von, còn lại định nghĩa khô khốc "Overfitting là hiện tượng mô hình học quá sát, thậm chí cả nhiễu..." (50 từ). Với người mới học thì bản sau khó tiếp thu hơn hẳn.

Điều bất ngờ: cụm "bằng tiếng Việt" em tưởng quan trọng nhất thì lại gần như không có tác dụng đo được. Bỏ nó ra, kể cả khi hỏi bằng tiếng Anh "How does embedding work in NLP?", trợ lý vẫn trả lời tiếng Việt. Lý do là phần còn lại của system prompt ("Bạn là trợ giảng thân thiện của khóa AI") vốn đã viết bằng tiếng Việt nên đã neo ngôn ngữ rồi — ràng buộc này bị trùng lặp, không phải nó vô dụng về nguyên tắc.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**

Em chạy thật một phiên tư vấn ăn uống 6 lượt với persona "trợ lý dinh dưỡng":

| Lượt | Người dùng nói |
|---|---|
| 1 | "Tôi bị dị ứng hải sản nặng, tuyệt đối không ăn được tôm cua." |
| 2 | "Tôi đang tập gym, cần nhiều protein." |
| 3 | "Ngân sách khoảng 100 nghìn một bữa." |
| 4 | "Tôi nấu ăn khá vụng, chỉ làm được món đơn giản." |
| 5 | "Nhà tôi có nồi chiên không dầu." |
| 6 | "Bạn còn nhớ tôi bị dị ứng với thứ gì không?" |

Đến lượt 6, `history[-8:]` chỉ còn giữ từ lượt 3 trở đi — thông tin dị ứng ở lượt 1 đã bị cắt. Trợ lý trả lời nguyên văn:

> "Tôi chưa có thông tin cụ thể về việc bạn bị dị ứng với loại thực phẩm nào cả."

Đây là kiểu hỏng nguy hiểm vì hai lẽ. Thứ nhất, thông tin quan trọng nhất thường được nói sớm nhất — người dùng khai ràng buộc ở đầu phiên rồi mới đi vào chi tiết, mà cửa sổ 4 lượt lại cắt đúng phần đầu. Thứ hai, trợ lý không biết là mình đã quên: nếu lượt 6 em hỏi "gợi ý món tối nay" thay vì hỏi về dị ứng, nó sẽ gợi ý món có tôm với giọng hoàn toàn tự tin, và người dùng không có cách nào biết là lời khuyên đó đã mất một ràng buộc.

Khắc phục: tóm tắt lũy tiến. Thay vì vứt các lượt bị đẩy ra khỏi cửa sổ, nén chúng thành vài câu bằng model rẻ rồi ghép vào ngay sau system prompt:

```python
if len(history) > 8:
    tom_tat = call_openai(
        "Tóm tắt 3 câu, giữ lại mọi ràng buộc cá nhân của người dùng "
        "(dị ứng, kiêng khem, ngân sách, thiết bị sẵn có):\n" + str(history[:-8]),
        model=OPENAI_MINI_MODEL,
    )[0]
    history = history[-8:]
```

Cách này hợp lý vì thứ hay bị mất khi cắt history đúng là loại nén rất tốt — "dị ứng hải sản, ngân sách 100k, có nồi chiên không dầu" gói gọn trong một dòng. Còn thứ cần giữ nguyên văn thì lại là các lượt gần nhất, vốn vẫn nằm đủ trong cửa sổ. Chi phí nén giao cho model nhỏ nên không đáng kể.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
