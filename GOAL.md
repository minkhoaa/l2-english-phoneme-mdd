# DATASET R&D GOAL — Dữ liệu quy mô lớn cho chẩn đoán lỗi phát âm

**Ngày chốt định hướng:** 2026-11-15  
**Trạng thái:** `SCALE_AGREED — DETAILED_R&D_PENDING`  
**Mục đích:** lưu quyết định và bàn giao để phiên sau bắt đầu R&D chi tiết, đánh giá tính khả thi và lập kế hoạch đầu tư effort.

> **Định hướng đã được người dùng đồng ý: xây dựng/tuyển chọn khoảng 220.000 bản thu có nhãn, từ khoảng 2.200 người nói độc lập. Không tối ưu kế hoạch quanh quy mô tối thiểu vài nghìn câu.**
>
> Đây là **mục tiêu planning**, không phải xác nhận dữ liệu đã có sẵn, ngân sách đã được duyệt, hay bằng chứng quy mô này bảo đảm một phương pháp tốt hơn. File này không khởi động một campaign training hoặc thay thế các GOAL vận hành hiện hữu.

## 1. Mục tiêu nghiên cứu

Xây dựng một nền tảng dữ liệu đủ đa dạng, có nhãn đáng tin và đánh giá độc lập để:

1. Phát triển phương pháp phát hiện/chẩn đoán lỗi âm vị tiếng Anh L2: **Substitution (S), Omission (O), Insertion (I)**, cùng các trường hợp phát âm đúng.
2. Phân biệt cải thiện do **dữ liệu**, do **phương pháp**, và do thay đổi ngưỡng/đánh đổi precision–recall.
3. Chọn phương pháp dựa trên khả năng khái quát sang người nói/nội dung/điều kiện thu mới và chi phí cảnh báo sai, không chỉ điểm đẹp trên một fold phát triển.
4. Giữ SM13 làm baseline có thể tái lập; không mặc định bỏ baseline hoặc kết luận phương pháp mới tốt hơn trước khi có đối chứng.

**Đơn vị quy mô:** “mẫu/dataset” trong trao đổi này được hiểu là **bản thu/câu nói (utterance)**, không phải hàng nghìn corpus. Số corpus thực tế là quyết định của R&D; ít nguồn có chất lượng và tương thích nhãn có thể tốt hơn nhiều nguồn ghép không kiểm chứng.

**Giả định phương pháp hiện tại:** dùng/fine-tune backbone pretrained. Quy mô đề xuất không phải thiết kế cho huấn luyện speech foundation model từ đầu.

## 2. Quy mô đã thống nhất làm mục tiêu

| Phần dữ liệu | Người nói độc lập | Câu/người dự kiến | Tổng bản thu | Vai trò |
|---|---:|---:|---:|---|
| Train | 1.400 | 100 | **140.000** | Huấn luyện phương pháp |
| Dev | 200 | 100 | **20.000** | Chọn phương pháp, hyperparameter, checkpoint, calibration và ngưỡng |
| Test nội bộ khóa | 400 | 100 | **40.000** | Đánh giá sau khi khóa phương pháp/policy |
| Test ngoài miền | 200 | 100 | **20.000** | Kiểm tra trên nguồn thu/điều kiện hoặc population độc lập phù hợp phạm vi claim |
| **Tổng** | **2.200** | **100** | **220.000** | |

- Corpus chính: **200.000 bản thu / 2.000 người nói**; tập ngoài miền: **20.000 bản thu / 200 người nói**.
- Nếu trung bình mỗi bản thu dài **4–6 giây**, tổng audio khoảng **244–367 giờ**. Đây là quy đổi giả định; phải đo duration thực tế và phân biệt audio thô với audio có gold annotations.
- Ưu tiên mở rộng **người nói độc lập và diversity**, không thay bằng hàng trăm nghìn câu từ vài chục người.
- 100 câu/người là định mức planning, không phải quota cứng buộc loại bỏ bản thu tốt hay lặp câu để đủ số.
- Có thể triển khai theo giai đoạn; pilot nhỏ là bước giảm rủi ro, **không tự hạ mục tiêu cuối về kế hoạch tối thiểu 100–150 người nói**.
- Thay đổi đáng kể quy mô hoặc tỷ lệ split phải được giải thích và chốt lại với người dùng trước thực hiện.

## 3. Vì sao mở hướng dữ liệu này?

Audit SM13 LoRA hiện tại cho thấy chất lượng số liệu có thể tái lập nhưng bằng chứng generalization còn hạn chế:

- Fit **9 người / 1.350 câu**; held **3 người / 450 câu**, ba người held đều nam.
- **245/245 held prompt IDs** đã xuất hiện trong Fit; đây là speaker-disjoint nhưng không phải text-disjoint evaluation.
- Latest được kiểm chứng ở thời điểm audit là **LoRA epoch 12**: pooled precision **51,25%**, recall **44,38%**, F1 **47,57%**, **2,2044 FP/câu**.
- Kết quả pooled dùng **LoRA target + gap predictions Dropout 2× cũ**; chưa chứng minh LoRA cải thiện insertion.
- Các event cùng câu/người nói có tương quan; không coi 30.082 held events là 30.082 quan sát độc lập.

**Diễn giải:** đây là lý do đầu tư vào dữ liệu và protocol tốt hơn, không phải chứng minh rằng thiếu dữ liệu là nguyên nhân duy nhất hoặc SM13 đã hết tiềm năng. Các số trên là snapshot audit, không phải monitor checkpoint mới trong tương lai.

Bằng chứng: [audit SM13](reports/SM13_LORA_RECHECK_20261115/REPORT_VI.md), [metrics](reports/SM13_LORA_RECHECK_20261115/RESULTS.json), [supplemental](reports/SM13_LORA_RECHECK_20261115/SUPPLEMENTAL.json).

## 4. Tiêu chuẩn dữ liệu phải thiết kế và kiểm chứng

### 4.1. Đúng bài toán và nhãn

- Có audio, intended transcript, canonical phone sequence và mapping rõ ràng tới event.
- Target labels **C/S/O**; gap labels **insertion/no-insertion**, theo chuẩn **N targets / N+1 gaps** nếu tiếp tục giao thức exact-gap SM13.
- Có đầy đủ correct/non-insertion events; không chỉ tập hợp những lỗi được phát hiện.
- Thống nhất cách xử lý multiple insertions, skipped/repeated words, deletion, silence, noise và trường hợp không chắc chắn.
- Phân biệt **accent nhưng phát âm được chấp nhận** với lỗi cần sửa; xử lý accepted pronunciation variants/allophones theo rubric đã chốt.
- Không tự đổi điểm phát âm thành S/O/I khi nguồn không tách được các khái niệm này. Missing/uncertain annotation không mặc định là Correct.
- Giữ raw annotations và versioned adapters; mọi nhãn chuyển đổi phải truy vết được về nguồn.

### 4.2. Ground truth và annotation reliability

- Evaluation gold phải được chuyên gia kiểm tra, không lấy ASR/forced-aligner/teacher output làm chân lý để xác nhận chính model.
- Định hướng chất lượng: hai annotator độc lập và adjudication khi bất đồng cho gold evaluation; kế hoạch review chéo train theo sampling đại diện và lỗi hiếm cần thiết kế ở phiên R&D.
- Annotator không thấy dự đoán model khi tạo gold; lưu rater provenance, disagreement, adjudication và uncertain cases.
- Báo agreement theo từng C/S/O/I/phone/subgroup, không dùng overall agreement bị đa số Correct chi phối làm chứng chỉ duy nhất.
- Có thể dùng weak/pseudo labels ở train nếu nghiên cứu thấy phù hợp, nhưng phải tách tầng dữ liệu và báo riêng. **Không tính weak labels thành gold tương đương hoặc làm nguồn evaluation độc lập.**

### 4.3. Diversity đúng đối tượng sử dụng

- Chốt population đích trước khi phân quota: người Việt học tiếng Anh, đa accent, người lớn/trẻ em, trình độ và ngữ cảnh sử dụng nào?
- Coverage về người nói, proficiency, độ tuổi, giới, L1/accent, microphone/thiết bị, điều kiện âm học, độ dài và tốc độ nói.
- Có correct-accented speech, lỗi khó/hiếm và điều kiện thực tế để đo false alarms, không chỉ clean speech và lỗi rõ.
- Nếu phục vụ người Việt, ưu tiên diversity người Việt tương ứng; không tự thay bằng nhiều accent khác chỉ để tăng tổng speaker.
- Speech tự phát/hội thoại nếu nằm trong phạm vi sản phẩm phải có protocol/nhãn riêng phù hợp; không mặc định mọi nguồn tương thích với read-speech S/O/I.

### 4.4. Split và chống leakage

- Không giao người nói giữa train/dev/test nội bộ/ngoài miền; phát hiện cùng người mang mã khác giữa nhiều nguồn.
- Kiểm exact/near-duplicate audio, session overlap, transcript/prompt overlap và corpus lineage.
- Thiết kế bài đánh giá **speaker mới + prompt mới**; có thể giữ một track same-prompt để đối chiếu benchmark nhưng phải báo tách biệt.
- Fit preprocessing/normalization/calibration chỉ trên phần được phép; lựa chọn ngưỡng trên dev, không trên test.
- Tập ngoài miền phải thực sự phù hợp claim chuyển miền. Nếu dùng nó để adaptation/tuning thì không còn là confirmation untouched; cần phần đánh giá khác.
- Gold alignment được phép phục vụ annotation/audit, không trở thành oracle input cho model nếu deployment chỉ có predicted alignment.

### 4.5. Audio, nguồn gốc, pháp lý

- Kiểm decode, sample rate/channel, clipping, truncation, duration, silence/noise và alignment geometry. Chuẩn hóa audio có lineage; không upsample rồi coi là tăng chất lượng.
- Manifest có corpus/release, speaker/session/prompt/utterance/event IDs, source path, hashes, label/rater version, split và QC flags.
- License/consent/quyền thu thập, xử lý, huấn luyện, chia sẻ và mục đích thương mại phải được kiểm tra trước sử dụng; đặc biệt với trẻ em và dữ liệu nhận diện người nói.
- Không tự upload dữ liệu cá nhân lên dịch vụ ngoài; không đưa raw audio/nhãn nhạy cảm vào Git.

## 5. Lựa chọn nguồn — cần nghiên cứu, chưa chốt mua/thu thập

Ưu tiên ít nhất **hai vai trò**: corpus chính để phát triển và tập xác nhận độc lập. Có thể kết hợp corpus có sẵn, annotation bổ sung hoặc thu mới sau khi đánh giá tính khả thi/quyền sử dụng.

Các điểm đã xác nhận cần giữ khi khảo sát:

- **L2-ARCTIC:** 24 người nói, 27,1 giờ audio tổng; chỉ **3.599 câu có manual error annotations**. Không gọi toàn bộ 27 giờ là gold S/O/I. Official license hiện ghi CC BY-NC 4.0.
- **SpeechOcean762:** 250 người nói / 5.000 câu; phone score 0–2, score0 gộp “incorrect or missed”; detailed annotations có insertion `[]`. Không mặc định score0 là Omission hoặc corpus đã có exhaustive exact-gap gold như SM13.
- **EpaDB:** nguồn ứng viên cần kiểm release/license, độ đầy đủ annotation và mapping insertion/deletion trước khi quyết định dùng.
- Không cộng speaker/câu từ nhiều corpus rồi mặc định có một gold dataset đồng nhất. Kiểm ontology, tiêu chuẩn accent, mức annotation, domain và overlaps trước.

Nguồn public và phân tích sâu: [báo cáo yêu cầu dữ liệu](reports/SM13_DATASET_REQUIREMENTS_20261115/REPORT_VI.md). **Mốc 100–150 speaker trong báo cáo đó là đề xuất ban đầu; mục tiêu planning hiện tại đã nâng lên 2.200 speaker theo trao đổi mới này.** Các nhận xét nguồn/nhãn trong báo cáo vẫn hữu ích, nhưng cần recheck release thực tế ở phiên sau.

## 6. Đánh giá và cách chọn phương pháp

- Không dùng số mẫu lớn như bằng chứng một phương pháp sẽ tốt; train sufficiency phải được kiểm tra bằng learning curve.
- So SM13 và phương pháp mới ở các mức train dự kiến **10.000 → 50.000 → 100.000 → 140.000 câu**, lựa chọn subset có kiểm soát speaker/diversity, dev cố định.
- Đối chứng phải mô tả cùng data access/label mapping/splits và tuning/compute budget; ghi rõ khác biệt pretrained sources và recipe, không quy gain của nhiều thay đổi đồng thời cho một thành phần.
- Nhiều seed; định mức khởi điểm **ít nhất 3 seed nếu ngân sách cho phép**. Seed không tạo thêm người nói độc lập.
- Primary decision: **recall tại cùng ngân sách false warnings**, đi kèm precision, pooled detection F1 và typed S/O/I AP/recall.
- Báo FP/câu và normalization theo số phone/thời lượng phù hợp khi câu dài khác nhau; thêm calibration, subgroup performance, latency và tài nguyên.
- CI của paired differences phải phản ánh speaker clusters; không bootstrap từng âm vị như i.i.d. Challenge set làm giàu lỗi hiếm báo riêng, không thay test prevalence thực tế.
- Chốt minimum useful improvement và yêu cầu precision/recall/false warnings với người dùng trước campaign. **Các ngưỡng chấp nhận sản phẩm hiện vẫn TBD, không tự đặt để gọi PASS.**

Ví dụ ngoại suy để hiểu độ lớn: 40.000 câu test có thể chứa khoảng **12.000 insertion** nếu tỷ lệ giống held hiện tại. Đây không phải quota chắc chắn, power calculation hoặc CI đã kiểm chứng. Cần kiểm support và uncertainty thực tế theo từng nhóm; không cố tạo lỗi để làm đẹp độ đủ mẫu của natural test.

## 7. Công việc mở đầu cho phiên R&D tiếp theo

1. **Chốt problem/population/rubric:** phát hiện loại lỗi nào, cho ai, chuẩn phát âm nào; đâu là lỗi cần can thiệp và đâu là biến thể được chấp nhận.
2. **Lập evidence-backed source inventory:** corpus/release, audio hours vs gold hours, speaker độc lập, nhãn S/O/I, domain, license, khả năng truy cập, annotation quality và overlaps. Tách verified / claimed / unknown.
3. **Gap analysis với mục tiêu 220.000/2.200:** phần nào có thể tái sử dụng, cần re-annotate, cần thu mới; không giả định đã có toàn bộ từ nguồn public.
4. **Thiết kế data contract và annotation handbook:** schema, versioned mapping, uncertainty, QC, double annotation/adjudication; kiểm sample trước khi annotation quy mô lớn.
5. **Thiết kế sampling/splits:** quotas theo population, speaker/prompt/session/source disjointness, natural test và challenge tracks; giữ nguyên các split hiện đang được bảo vệ.
6. **Lập feasibility/budget:** tuyển người nói, thu âm, annotation/review, license, bảo mật, lưu trữ, backup và compute. Phân biệt audio hours với annotator hours; không bịa chi phí khi chưa có pilot time study/báo giá.
7. **Đề xuất pilot có giới hạn:** kiểm consent/recruitment, annotation agreement, event support, adapter fidelity và chi phí thực trước khi đầu tư lớn. Quy mô/budget pilot cần được chốt riêng.
8. **Trình kế hoạch rollout và gates:** pilot → mở rộng → freeze/evaluate; xác định điều kiện dừng/sửa dữ liệu, trách nhiệm và đầu ra từng phase.

### Đầu ra mong đợi của phiên R&D

- Source inventory và claim/evidence matrix.
- Quyết định population/label policy và các trade-off còn mở.
- Data schema/annotation handbook draft, QC và acceptance tests dự kiến.
- Phương án sampling/split và quyền truy cập test.
- So sánh các phương án reuse/re-annotate/collect với effort, rủi ro, license và timeline.
- Kế hoạch pilot, ngân sách đề xuất và các quyết định cần người dùng phê duyệt.

## 8. Ranh giới ủy quyền và bảo tồn công việc hiện tại

**Yêu cầu hiện tại chỉ là ghi tài liệu để bàn giao.** Việc đồng ý mục tiêu lớn không tự cấp quyền chi tiền, tuyển người tham gia, nhận/đẩy dữ liệu cá nhân, mua license, gán nhãn qua dịch vụ ngoài hoặc khởi chạy training/download corpus lớn.

- Không tự mở official test/DEV_CAL đang sealed; không biến test cũ thành train để đạt quota.
- Không thay đổi corpus gốc, nhãn gốc, checkpoints, reports lịch sử hay split hiện hành.
- Không dừng/restart/chiếm tài nguyên của các nhánh CROTTC/SM13 hiện hữu. Trạng thái process/GPU phải kiểm thực tế trước tác vụ compute tương lai, không dựa snapshot trong tài liệu.
- [GOAL.md](GOAL.md), [GOAL_SM13.md](GOAL_SM13.md), [GOAL_CROTTC_IF.md](GOAL_CROTTC_IF.md) và [HANDOFF.md](HANDOFF.md) tiếp tục giữ vai trò hiện có. File này là **định hướng dataset R&D cho phiên sau**, không tự thay lịch điều phối hoặc các gate của chúng.
- Chưa chốt kiến trúc mới, nhà cung cấp dữ liệu, cách mua/thu thập, ngân sách, timeline, demographic quotas hay ngưỡng triển khai. Những nội dung đó thuộc phiên R&D chi tiết.

## 9. Câu lệnh bàn giao gợi ý cho phiên sau

> Đọc `DATASET_R&D_GOAL.md`, báo cáo dataset requirements và audit SM13 được liên kết. Bắt đầu R&D chi tiết cho mục tiêu **220.000 bản thu có nhãn / 2.200 người nói độc lập**, theo train140k/dev20k/test40k/external20k. Trước hết chốt population/rubric, khảo sát nguồn có bằng chứng, audit khả năng chuẩn hóa S/O/I và license, rồi đề xuất phương án thu thập/annotation, protocol, effort/budget và pilot. Không tự hạ mục tiêu xuống kế hoạch tối thiểu, cũng không coi quy mô là bảo đảm chất lượng. Không mở sealed data, đổi split, chi tiền, chia sẻ dữ liệu hoặc chạy training khi chưa có quyền tương ứng.
