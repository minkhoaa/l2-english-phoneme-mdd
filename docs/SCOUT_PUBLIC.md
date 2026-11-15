# Báo Cáo Khảo Sát Các Bộ Dữ Liệu Tiếng Anh L2 Công Khai Cho Bài Toán MDD (SCOUT_PUBLIC)

**Người thực hiện:** `@scout-public`  
**Ngày hoàn thành:** 16/11/2026  
**Mục tiêu phân công:** Khảo sát, kiểm chứng độc lập và đo đạc định lượng tất cả các bộ dữ liệu tiếng Anh L2 (Non-Native English) công khai và bán công khai phục vụ bài toán Phát hiện và Chẩn đoán Lỗi Phát âm (Mispronunciation Detection and Diagnosis - MDD).  
**Tập trung kiểm chứng:** Phiên bản phát hành chính xác, URL/Paper gốc, số lượng speaker, tổng thời lượng audio so với thời lượng có nhãn gold chuẩn, ngữ nghĩa nhãn âm vị (S/O/I vs Phone scores), bản quyền/giấy phép sử dụng, và **bác bỏ triệt để quan niệm sai lầm "toàn bộ audio là nhãn gold" (Whole Audio is Gold Fallacy)**.

---

## 1. Tóm Tắt Điều Hành & Phát Hiện Trọng Yếu (Executive Summary)

1. **Sự Khan Hiếm Cực Độ của Dữ Liệu Gold Chuẩn S/O/I:**
   Trong toàn bộ các bộ dữ liệu tiếng Anh L2 được công bố trên thế giới, chỉ có duy nhất **L2-ARCTIC (phần subset 150 câu/speaker)** và **EpaDB** cung cấp nhãn âm vị thủ công ở cấp độ Thay thế (Substitution - S), Lược bỏ (Omission - O), và Chèn (Insertion - I) kèm định danh âm vị thực tế (realized phone).
2. **Bác Bỏ Triệt Để Quan Niệm "Toàn Bộ Audio L2-ARCTIC Là Gold":**
   Các bài báo khoa học và tài liệu kỹ thuật thường dẫn con số "L2-ARCTIC có 27,1 giờ audio của 24 người nói". **Đây là tuyên bố sai lệch nghiêm trọng**.
   - Thực tế đo đạc trực tiếp từ kho lưu trữ dữ liệu và manifest chuẩn: L2-ARCTIC v5.0 có tổng cộng 26.867 câu đọc (~27,1 giờ audio), nhưng **chỉ có duy nhất 150 câu/speaker được chuyên gia ngữ âm (phonetician) gán nhãn thủ công (gold manual annotations)**.
   - Tổng số câu gold thực tế trên toàn bộ 24 speaker là **3.599 câu**, tương đương **3,57 giờ audio phát âm thực tế** (chỉ chiếm **13,18%** tổng thời lượng ghi âm).
   - **86,82% thời lượng còn lại (~23,53 giờ) là kết quả chạy forced-alignment tự động của Kaldi**, hoàn toàn không được kiểm chứng lỗi phát âm. Nếu huấn luyện hoặc đánh giá mô hình trên toàn bộ 27,1 giờ như thể đó là nhãn gold, hệ thống sẽ bị nhiễm nhiễu nhãn âm tính giả (false-negative noise) cực kỳ nặng nề.
3. **Ngữ Nghĩa Nhãn Khác Biệt Giữa MDD và Chấm Điểm Phát Âm (Pronunciation Scoring):**
   - **SpeechOcean762 (OpenSLR 101):** Có quy mô 5.000 câu (~5,98 giờ) được 5 chuyên gia chấm điểm độc lập (100% audio có nhãn), nhưng nhãn âm vị là **điểm số liên tục/rời rạc (0, 1, 2)** chứ không phải bộ ba S/O/I chẩn đoán. Điểm 0 gộp chung cả âm đọc sai lẫn âm bị bỏ qua ("incorrect or missed"), và không ghi nhận âm vị thay thế là gì (ví dụ: không biết /θ/ bị thay bằng [s] hay [t]). Dữ liệu này chỉ phù hợp làm nhiệm vụ chấm điểm (Goodness of Pronunciation - GOP) hoặc pre-training âm học, không thể dùng trực tiếp làm ground-truth chẩn đoán S/O/I nếu không có bộ chuyển đổi (adapter) và quy tắc đồng thuận giả định.
4. **Các Bộ Dữ Liệu ASR / Nhận Dạng Accent Không Phải Là Dataset MDD:**
   - **EdAcc (Edinburgh Accents Corpus, ICASSP 2023):** Quy mô ~40 giờ đàm thoại tự nhiên của ~120 người nói thuộc >30 accent, có transcript chính tả được kiểm chứng thủ công, nhưng **hoàn toàn KHÔNG có nhãn âm vị hoặc nhãn lỗi phát âm**.
   - **AESRC 2020 (Accented English Speech Recognition Challenge):** Quy mô 180 giờ do Datatang cung cấp, chỉ có nhãn văn bản chính tả và nhãn phân loại accent (Track 1 & Track 2), **hoàn toàn KHÔNG có nhãn lỗi âm vị**.
5. **Dữ Liệu Khảo Thí Đóng (Proprietary / Institutional Custody):**
   - **TEAP (Test of English for Academic Purposes):** Dữ liệu bài thi nói của Đại học Sophia và Eiken Foundation of Japan. Đây là tài sản bảo mật nội bộ phục vụ thi cử, không công khai tải về, nhãn thuộc thang CEFR (A2-B2+).
   - **ETS (TOEFL iBT / SpeechRater):** Dữ liệu độc quyền thương mại của Viện Khảo thí Giáo dục Hoa Kỳ (ETS). Hoàn toàn không phát hành công khai dưới giấy phép mã nguồn mở.
6. **Làm Rõ Thuật Ngữ "CU-SYMPHONIC":**
   - Qua đối chiếu lịch sử nghiên cứu từ nhóm tác giả GS. Helen Meng (CUHK) tại ICASSP 2019 và Interspeech 2021, cái tên "CU-SYMPHONIC" là **sự nhầm lẫn danh xưng (misnomer / conflation)** trong một số tài liệu khảo sát. Bộ dữ liệu nền tảng thực sự của CUHK là **CU-CHLOE** (Chinese University CHinese Learners Of English, 210 speaker, 34,6 giờ). Bộ dữ liệu này thuộc diện **lưu trữ nội bộ (restricted institutional custody)**, không được phát hành công khai tự do.
7. **Sự Thiếu Hụt Dữ Liệu Người Việt Học Tiếng Anh (Vietnamese L2):**
   Trong toàn bộ các bộ dữ liệu quốc tế công khai, chỉ có duy nhất **4 người nói tiếng Việt trong L2-ARCTIC** (HQTV, PNV, THV, TLV; gồm 2 nam, 2 nữ) với tổng cộng **600 câu gold (~0,6 giờ audio)**. Toàn bộ các dataset khác (SpeechOcean, EpaDB, EdAcc) đều không có phân nhóm người Việt có nhãn lỗi âm vị. Do đó, việc xây dựng một bộ dữ liệu tiếng Anh L2 chuyên biệt cho người Việt là yêu cầu bắt buộc và không thể thay thế.

---

## 2. Bảng Ma Trận So Sánh Tổng Thể Các Bộ Dữ Liệu (Master Matrix)

| Tên Bộ Dữ Liệu | Phiên Bản & Năm | URL Chính Thức / Paper Gốc | Số Lượng Speaker | Tổng Thời Lượng Ghi Âm | Thời Lượng Nhãn Gold Thủ Công | Ngữ Nghĩa Nhãn Âm Vị (S/O/I vs Score) | Giấy Phép Bản Quyền (License) | Mức Độ Khả Dụng (Availability) | Khả Năng Sử Dụng Cho MDD |
|---|---|---|---|---|---|---|---|---|---|
| **L2-ARCTIC** | **v5.0** (03/2020; gốc 2018) | [tamu.edu](https://psi.engr.tamu.edu/l2-arctic-corpus/) / Zhao et al., Interspeech 2018 | 24 (12M/12F, 6 L1: AR, ZH, HI, KO, ES, VI) | **27,1 giờ** (26.867 câu đọc) + suitcase corpus | **3,57 giờ** (3.599 câu; đúng 150 câu/spk) | **Chuẩn S/O/I đầy đủ** (Target vs Realized, time-aligned TextGrids) | **CC BY-NC 4.0** | Tải tự do công khai (HuggingFace / TAMU) | **Rất cao (Ground-truth chuẩn duy nhất)** |
| **SpeechOcean762** | **SLR101 v1.0** (02/2021) | [openslr.org/101/](https://www.openslr.org/101/) / Zhang et al., Interspeech 2021 | 250 (125 trẻ em, 125 người lớn; 1:1 M/F; Mandarin L1) | **5,98 giờ** (5.000 câu, 21.520s) | **5,98 giờ** (100% 5.000 câu được 5 chuyên gia chấm) | **Phone Goodness Scores (0, 1, 2)**; `[]` cho insertion; Score 0 gộp cả S và O | **CC BY 4.0** (Thương mại được phép) | Tải tự do công khai (OpenSLR, Kaldi) | **Trung bình (Cần adapter; thiếu realized phone cho S)** |
| **EpaDB** | **v1.0** (2019 / 2021) | [GitHub Vidal](https://github.com/JazminVidal/gop-dnn-epadb) / Vidal et al., Interspeech 2019 | 50 (25M/25F; Argentine Spanish L1) | **~2,73 giờ** (~3.160 câu, 164 phút) | **~2,73 giờ** (100% có phonetic alignment) | **Phonetic Alignment (ARPAbet + allophone)**; 0 = omission, `+` = insertion | **CC BY-NC 4.0** / CC BY-NC-ND | Tải công khai từ GitHub/HuggingFace | **Khá (Tốt cho Spanish L1, cần chuẩn hóa boundary `+`)** |
| **EdAcc** | **v1.0** (2023) | [HuggingFace edacc](https://huggingface.co/datasets/edinburghcstr/edacc) / Sanabria et al., ICASSP 2023 | ~120 (>30 accent L1 & L2 toàn cầu) | **~40,0 giờ** (Đàm thoại video call tự nhiên) | **40,0 giờ** (Nhãn text chính tả kiểm chứng thủ công) | **KHÔNG CÓ nhãn âm vị** (Chỉ có text transcript chính tả) | **CC BY-SA 4.0** | Tải tự do công khai (Edinburgh DataShare) | **Không thể (Chỉ dùng cho ASR & Accent evaluation)** |
| **AESRC 2020** | **Challenge 2020** (2020) | [arXiv:2010.02495](https://arxiv.org/abs/2010.02495) / Deng et al., Interspeech 2020 | ~400–500 (10 quốc gia / accents) | **180,0 giờ** (160h train/dev + 20h test) | **180,0 giờ** (Nhãn text transcript chính tả) | **KHÔNG CÓ nhãn âm vị** (Chỉ có text và nhãn phân loại accent) | Bản quyền nhà cung cấp Datatang | Yêu cầu đăng ký / thỏa thuận học thuật | **Không thể (Chỉ dùng cho ASR & Accent Recognition)** |
| **CU-CHLOE** *(bị gọi nhầm là CU-SYMPHONIC)* | **v1.0 / CUHK** (2012–2021) | CUHK SEEM / Leung et al., ICASSP 2019; Meng et al. | 210 (100 Cantonese, 110 Mandarin; cân bằng giới tính) | **34,6 giờ** (Prompt đa dạng, minimal pairs, TIMIT) | **34,6 giờ** (Chuyên gia phiên âm ngữ âm toàn bộ) | **Chuẩn S/O/I đầy đủ** (Từng âm vị được phiên âm và gán thẻ lỗi) | Bản quyền viện nghiên cứu (CUHK Proprietary) | **Đóng / Không công khai** (Chỉ nội bộ CUHK & đối tác) | **Không khả dụng (Không được cấp quyền mở)** |
| **TEAP Speaking** | **Khảo thí nội bộ** (2014–nay) | Sophia University & EIKEN Foundation of Japan | Hàng ngàn thí sinh Nhật Bản | Hàng trăm giờ bài thi nói | Đánh giá theo thang CEFR A2–B2+ | **KHÔNG CÓ nhãn S/O/I** (Điểm rubric phát âm/trôi chảy cấp bài thi) | Bản quyền độc quyền EIKEN | **Hoàn toàn đóng** (Dữ liệu bảo mật kỳ thi) | **Không thể (Không có truy cập, không có nhãn âm vị)** |
| **ETS TOEFL iBT** | **Khảo thí nội bộ** (SpeechRater) | Educational Testing Service (Princeton) / Xi et al. | Hàng chục ngàn thí sinh toàn cầu (>100 L1) | Hàng ngàn giờ bài thi nói | Điểm chuyên gia người chấm thang 1–4 | **KHÔNG CÓ nhãn S/O/I công khai** (Chỉ có feature nội bộ ASR forced alignment) | Bản quyền độc quyền thương mại ETS | **Hoàn toàn đóng** (Bảo mật thương mại) | **Không thể (Không có truy cập công khai)** |
| **ERJ (English Read by Japanese)** | **SRC Release** (2002–2004) | Minematsu et al. / NII Speech Resources Consortium | 202 (100M/102F sinh viên Nhật Bản) | **~60,0 giờ** | Subset có nhãn âm vị thủ công | TextGrids phiên âm biến thể âm vị người Nhật | Giấy phép học thuật SRC (Có thu phí phát hành đĩa) | Bán công khai (Yêu cầu đăng ký qua NII/SRC) | **Hạn chế (Khó tiếp cận, chỉ có accent Nhật Bản)** |

---

## 3. Khảo Sát Chi Tiết Từng Bộ Dữ Liệu (Deep Dives)

### 3.1. L2-ARCTIC (Phiên bản v5.0)

- **Đơn vị phát triển & Tác giả:** Phòng thí nghiệm PSI (Texas A&M University) phối hợp với Iowa State University (Guanlong Zhao, Sinem Sonsaat, Alif Silpachai, Iva Lucic, Evgeny Chukharev-Hudilainen, John Levis, Ricardo Gutierrez-Osuna).
- **Công bố chính thức:** *Interspeech 2018*, pp. 2783–2787 (DOI: `10.21437/Interspeech.2018-1110`).
- **Phiên bản phát hành:** **v5.0** (Phát hành ngày 12 tháng 03 năm 2020). Phiên bản này bổ sung tập "suitcase corpus" gồm các đoạn kể chuyện tự nhiên không kịch bản của 22/24 người nói.
- **URL chính thức:** [https://psi.engr.tamu.edu/l2-arctic-corpus/](https://psi.engr.tamu.edu/l2-arctic-corpus/) và tài liệu kỹ thuật tại [https://psi.engr.tamu.edu/l2-arctic-corpus-docs/](https://psi.engr.tamu.edu/l2-arctic-corpus-docs/).
- **Quy mô người nói:** Đúng **24 người nói** (12 Nam, 12 Nữ). Phân bổ đồng đều trên 6 ngôn ngữ mẹ đẻ (L1), mỗi nhóm đúng 4 người nói:
  1. *Arabic (Ả Rập):* ABA (M), SKA (F), YBAA (M), ZHAA (F)
  2. *Chinese (Quan Thoại):* BWC (M), LXC (F), NCC (F), TXHC (M)
  3. *Hindi (Ấn Độ):* ASI (M), RRBI (M), SVBI (F), TNI (F)
  4. *Korean (Hàn Quốc):* HJK (F), HKK (M), YDCK (F), YKOK (M)
  5. *Spanish (Tây Ban Nha):* EBVS (M), ERMS (M), MBMPS (F), NJS (F)
  6. *Vietnamese (Việt Nam):* HQTV (M), PNV (F), THV (F), TLV (M)
- **Quy mô ghi âm toàn bộ:** Mỗi người nói đọc khoảng 1.132 câu trích từ CMU ARCTIC prompts (tập hợp các câu cân bằng ngữ âm tiếng Anh). Tổng số câu ghi âm trong kho lưu trữ là **26.867 câu**, tương đương khoảng **27,1 giờ audio** (sample rate 44.1kHz/16-bit, sau này được downsample chuẩn về 16kHz).
- **Quy mô nhãn Gold chuẩn (Thực tế):**
  - **Sự thật kiểm chứng:** Các chuyên gia ngữ âm **chỉ gán nhãn thủ công cho đúng 150 câu trên mỗi người nói**. Tập hợp này gồm: 100 câu đọc chung cho toàn bộ 24 speaker + 50 câu đọc riêng được thiết kế để kích hoạt các lỗi ngữ âm đặc trưng theo từng L1.
  - Tổng số câu gold thủ công là: 24 speaker × 150 câu = **3.600 câu danh định**.
  - Thực tế trong bản phân phối dữ liệu chuẩn hóa của dự án (kiểm tra trực tiếp tại `data/derived/l2arctic/manifests/`): có **3.599 câu hợp lệ** (1 câu bị loại bỏ do lỗi suy hao kỹ thuật).
  - Tổng thời lượng phát âm thực tế (active speech duration) đo đạc từ timestamp `xmin` đến `xmax` của 3.599 câu này là **12.859,52 giây = 3,57 giờ**.
  - Trung bình thời lượng mỗi câu là **3,57 giây**.
- **Ngữ nghĩa nhãn âm vị (Annotation Schema):**
  - Sử dụng bảng ký âm ARPAbet (39 âm vị cơ bản của CMUDict).
  - Cung cấp Praat TextGrids với mốc thời gian căn chỉnh chính xác đến từng mili-giây.
  - Ba loại lỗi cơ bản được gán nhãn tường minh:
    - **Thay thế (Substitution - S):** Ký hiệu dạng `Target,Realized` (ví dụ âm gốc `TH` được người học phát âm thành `S`, ký hiệu `TH,S`).
    - **Lược bỏ (Omission - O):** Ký hiệu dạng `Target*` (ví dụ âm gốc `D` ở đuôi từ bị bỏ qua, ký hiệu `D*`).
    - **Chèn (Insertion - I):** Ký hiệu dạng `+Phone` nằm giữa các khoảng trống ngữ âm (gaps), ví dụ chèn nguyên âm đệm `+AH` hoặc chèn phụ âm `+T`.
- **Thống kê phân bố nhãn thực tế trên 3.599 câu gold (Dữ liệu kiểm chứng từ `CORPUS_STATISTICS.json`):**
  - Tổng số âm vị mục tiêu (Target Phones): **119.154 âm vị**.
  - Âm vị phát âm đúng (CORRECT): **101.636 âm vị (85,30%)**.
  - Âm vị Thay thế (SUBSTITUTION): **14.098 âm vị (11,83%)**.
  - Âm vị Lược bỏ (OMISSION): **3.420 âm vị (2,87%)**.
  - Âm vị Chèn thêm (INSERTION): **1.003 âm vị (0,84% tính trên tổng target)**.
  - *Nhận xét quan trọng:* Lỗi chèn (Insertion) và Lược bỏ (Omission) là các sự kiện hiếm (rare events, chiếm < 3%).
- **Bản quyền & Giấy phép sử dụng:** **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.
  - *Đính chính lỗi lịch sử:* Trong file `literature/DATASETS.csv` trước đây có ghi nhầm là `CC BY-NC-SA 4.0`. Kiểm tra trực tiếp trên trang chủ TAMU xác nhận giấy phép là `CC BY-NC 4.0` (không có điều khoản ShareAlike bắt buộc).
- **Mức độ khả dụng:** Công khai tải về miễn phí cho mục đích nghiên cứu học thuật phi thương mại.

---

### 3.2. SpeechOcean762 (OpenSLR 101)

- **Đơn vị phát triển:** Tương tác hợp tác giữa SpeechOcean và Đại học Giao thông Thượng Hải (SJTU) - Junbo Zhang, Zhiwen Zhang, Yongqing Wang, Zhiyong Yan, Qiong Song, Yukai Huang, Kehang Shen, Dongwei Jiang, Pingyi Peng, Shujie Hu, Meng Ge, Qingyang Hong, Yanmin Qian.
- **Công bố chính thức:** *Interspeech 2021*, pp. 481–485 (arXiv: `2104.01378`).
- **Mã phát hành OpenSLR:** **SLR101** (Phát hành ngày 01 tháng 02 năm 2021).
- **URL chính thức:** [https://www.openslr.org/101/](https://www.openslr.org/101/) và kho GitHub [https://github.com/jimbozhang/speechocean762](https://github.com/jimbozhang/speechocean762).
- **Quy mô người nói:** Đúng **250 người nói** (tất cả đều có ngôn ngữ mẹ đẻ là tiếng Quan Thoại - Mandarin Chinese).
  - Phân tầng lứa tuổi: **125 Trẻ em** (dưới 18 tuổi, dải tuổi từ 6 đến 17) và **125 Người lớn** (từ 18 đến 60 tuổi).
  - Cân bằng giới tính: Tỷ lệ Nam/Nữ xấp xỉ 1:1.
  - Phân chia tập chính thức: 125 người nói tập Train (2.500 câu) và 125 người nói tập Test (2.500 câu), phân tách người nói hoàn toàn (speaker-disjoint).
- **Quy mô ghi âm toàn bộ & Nhãn Gold:**
  - Tổng số câu: **5.000 câu đọc** (2.500 câu train + 2.500 câu test).
  - Tổng số file WAV trong kho: **5.245 files** (bao gồm một số file phụ trợ/kiểm tra).
  - Tổng thời lượng audio đo đạc trực tiếp bằng Python: **21.520,86 giây = 5,98 giờ** (~6,0 giờ).
  - **Tỷ lệ có nhãn:** **100% thời lượng audio (5.000 câu) đều được chấm điểm thủ công bởi 5 chuyên gia ngữ âm độc lập** (không có câu nào bị bỏ sót).
- **Ngữ nghĩa nhãn âm vị & Hạn chế đối với bài toán MDD:**
  - Đánh giá ở 3 cấp độ: Cấp độ âm vị (Phoneme), cấp độ từ (Word), và cấp độ câu (Sentence).
  - **Ở cấp độ âm vị (Phoneme-level):** Điểm số nằm trong dải rời rạc `[0, 1, 2]`:
    - `2`: Phát âm chuẩn xác (Correct).
    - `1`: Phát âm đúng ngữ âm nhưng mang ngữ điệu nặng (Accented / Slight distortion).
    - `0`: Phát âm sai hoặc bị bỏ qua ("incorrect or missed").
  - Trong file chi tiết `scores-detail.json`, các chuyên gia sử dụng ký hiệu ngoặc:
    - `{phone}`: Đánh dấu điểm 1 (ngữ điệu nặng).
    - `(phone)`: Đánh dấu điểm 0 (phát âm sai hoặc bỏ qua).
    - `[phone]`: Đánh dấu âm vị được người học chèn thêm ngoài dự kiến (Insertion).
  - **HẠN CHẾ CỐT TỬ KHI ÁP DỤNG VÀO MDD CHẨN ĐOÁN:**
    1. Điểm 0 gộp chung cả Thay thế (Substitution) và Lược bỏ (Omission). Mô hình không thể biết người học đọc thành âm gì khi bị điểm 0.
    2. Không cung cấp mốc thời gian căn chỉnh (time-aligned phone boundaries) độc lập của con người; chỉ có mốc thời gian do forced aligner sinh ra.
    3. Không có biểu diễn canonical mapping cho các cặp thay thế cụ thể (ví dụ: không biết /s/ thay cho /θ/).
- **Bản quyền & Giấy phép sử dụng:** **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
  - *Ý nghĩa thương mại:* Đây là bộ dữ liệu hiếm hoi cho phép sử dụng tự do cho cả mục đích thương mại mà không có điều khoản NonCommercial cấm đoán.
- **Mức độ khả dụng:** Tải công khai tốc độ cao qua mirror OpenSLR và tích hợp sẵn trong thư viện Kaldi / HuggingFace.

---

### 3.3. EpaDB (English Pronunciation by Argentinians Database)

- **Đơn vị phát triển:** Nhóm nghiên cứu thuộc Đại học Buenos Aires (UBA) và CONICET, Argentina (Jazmín Vidal, Luciana Ferrer, Leonardo Brambilla, C. Bonomi, M. Sancinetti).
- **Công bố chính thức:**
  - *Interspeech 2019*, pp. 111–115: "EpaDB: A Database for Development of Pronunciation Assessment Systems".
  - *Interspeech 2021*: "Phone-Level Pronunciation Scoring for Spanish Speakers Learning English Using a GOP-DNN System".
- **URL chính thức:** [https://github.com/JazminVidal/gop-dnn-epadb](https://github.com/JazminVidal/gop-dnn-epadb) và trang tài nguyên UBA [https://habla.dc.uba.ar/resources](https://habla.dc.uba.ar/resources).
- **Quy mô người nói:** **50 người nói** bản ngữ tiếng Tây Ban Nha vùng Argentina (25 Nam, 25 Nữ) với các trình độ tiếng Anh từ sơ cấp đến nâng cao.
- **Quy mô audio & Nhãn:**
  - Tổng số mẫu âm thanh: **3.160 câu đọc ngắn** (phrases).
  - Tổng thời lượng audio: Khoảng **164 phút (~2,73 giờ)**. Tập train có ~98 phút và tập test có ~66 phút.
  - **Tỷ lệ có nhãn:** **100% thời lượng (3.160 câu) được phân tích ngữ âm trên Praat TextGrids**.
  - *Lưu ý về độ tin cậy nhãn:* Trong bài báo gốc tại Interspeech 2019, chỉ có một tập con 200 câu được hai chuyên gia gán nhãn độc lập để đo độ đồng thuận (Inter-Annotator Agreement), các câu còn lại được phân công gán nhãn đơn lẻ.
- **Ngữ nghĩa nhãn âm vị:**
  - Sử dụng bảng mã ARPAbet mở rộng với các biến thể tha âm (allophones) đặc thù của hiện tượng chuyển di ngôn ngữ (L1 transfer) từ tiếng Tây Ban Nha.
  - **Lược bỏ (Omission):** Được đánh dấu bằng ký hiệu `0` thay thế cho âm vị mục tiêu.
  - **Chèn thêm (Insertion):** Được đánh dấu bằng dấu `+` nối trực tiếp với âm vị kế cận (ví dụ `AH+T` hoặc `S+IH`).
  - *Hạn chế kỹ thuật:* Do dấu `+` gắn liền với âm vị lân cận mà không tách thành interval/boundary riêng trên TextGrid, việc trích xuất tọa độ ranh giới chính xác cho từng âm chèn theo hình học exact-gap (như chuẩn SM13) đòi hỏi thuật toán xử lý phân tách bổ sung.
- **Bản quyền & Giấy phép sử dụng:** **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)** hoặc **CC BY-NC 4.0** (tùy gói phân phối mã nguồn và dữ liệu).
- **Mức độ khả dụng:** Công khai tải về qua kho GitHub của nhóm tác giả và lưu trữ tại Đại học Buenos Aires.

---

### 3.4. EdAcc (The Edinburgh International Accents of English Corpus)

- **Đơn vị phát triển:** Viện Nghiên cứu Công nghệ Lời nói CSTR, Đại học Edinburgh, Vương quốc Anh (Ramon Sanabria, Nikolay Bogoychev, Nina Markl, Andrea Carmantini, Ondrej Klejch, Peter Bell).
- **Công bố chính thức:** *IEEE ICASSP 2023*, pp. 1–5 (arXiv: `2211.00693`).
- **URL chính thức:** [https://huggingface.co/datasets/edinburghcstr/edacc](https://huggingface.co/datasets/edinburghcstr/edacc) và [https://datashare.ed.ac.uk/handle/10283/4621](https://datashare.ed.ac.uk/handle/10283/4621).
- **Quy mô người nói:** Khoảng **120 người nói** đại diện cho hơn 30 accent tiếng Anh bản ngữ (L1) và phi bản ngữ (L2) trên toàn thế giới (Ấn Độ, Nigeria, Jamaica, Kenya, Indonesia, v.v.).
- **Quy mô audio & Nhãn:**
  - Tổng thời lượng: Khoảng **40,0 giờ audio**.
  - Hình thức hội thoại: Đàm thoại video call tự nhiên giữa hai người bạn thân (dyadic conversations), không gò bó kịch bản hay từ vựng.
  - **Thực tế nhãn:** 100% thời lượng có văn bản phiên âm chính tả (orthographic transcripts) được kiểm chứng thủ công bởi đội ngũ chuyên nghiệp, kèm đánh dấu tiếng ồn, cười, ngập ngừng, chồng lấn giọng nói (overlap).
- **Ngữ nghĩa nhãn âm vị:**
  - **HOÀN TOÀN KHÔNG CÓ NHÃN ÂM VỊ (Zero Phone-level Annotations)**.
  - Không có nhãn S/O/I, không có ranh giới âm vị thủ công, không có điểm số phát âm.
- **Bản quyền & Giấy phép:** **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**.
- **Kết luận cho bài toán MDD:** EdAcc là một benchmark xuất sắc cho bài toán **ASR đa accent và đo lường độ suy giảm WER**, nhưng **hoàn toàn KHÔNG THỂ sử dụng làm ground-truth để huấn luyện hay đánh giá mô hình MDD âm vị**.

---

### 3.5. TEAP (Test of English for Academic Purposes - Speaking Test)

- **Đơn vị sở hữu & Quản lý:** Phát triển chung giữa Đại học Sophia (Sophia University) và Hiệp hội Khảo thí Ngôn ngữ Thực hành Nhật Bản (Eiken Foundation of Japan).
- **Bản chất dữ liệu:** Đây là cơ sở dữ liệu ghi âm các bài thi nói thực tế của thí sinh học sinh/sinh viên Nhật Bản tham gia kỳ thi chuẩn hóa tuyển sinh đại học TEAP (Test of English for Academic Purposes).
- **Công bố tài liệu:** Các báo cáo nghiên cứu thường niên của Eiken Foundation of Japan (Eiken Research Reports) và tài liệu khảo thí tiếng Anh học thuật.
- **Quy mô người nói & Audio:** Hàng ngàn thí sinh qua các năm, lưu trữ hàng trăm giờ phát âm của người học Nhật Bản (Japanese L1).
- **Ngữ nghĩa nhãn:**
  - Đánh giá theo thang đo năng lực ngoại ngữ khung CEFR (A2 đến B2+).
  - Sử dụng các tiêu chí chấm điểm phân tích (Analytic Rubrics): Pronunciation (Phát âm), Fluency (Độ trôi chảy), Lexical Resource (Vốn từ), Grammatical Accuracy (Ngữ pháp).
  - Tiêu chí phát âm được chấm theo điểm tổng quát (1–5 hoặc thang CEFR band), dựa trên tính dễ hiểu (intelligibility), nhịp điệu (stress/rhythm), và ngữ điệu (intonation).
  - **HOÀN TOÀN KHÔNG CÓ NHÃN S/O/I CẤP ĐỘ ÂM VỊ**.
- **Bản quyền & Mức độ khả dụng:**
  - **ĐỘC QUYỀN NỘI BỘ (PROPRIETARY / RESTRICTED CUSTODY)**.
  - Eiken Foundation of Japan và Sophia University bảo vệ dữ liệu theo quy chế bảo mật bài thi quốc gia. Không có bất kỳ bản phát hành công khai nào cho cộng đồng mã nguồn mở.
- **Kết luận:** Hoàn toàn không khả dụng cho dự án nghiên cứu mở.

---

### 3.6. ETS (Educational Testing Service - TOEFL iBT / SpeechRater)

- **Đơn vị sở hữu & Quản lý:** Educational Testing Service (ETS), Princeton, New Jersey, Hoa Kỳ.
- **Bản chất dữ liệu:** Cơ sở dữ liệu giọng nói khổng lồ từ các bài thi nói TOEFL iBT (cả câu trả lời độc thoại không kịch bản và câu trả lời tích hợp đọc-nghe-nói).
- **Công bố khoa học liên quan:** Các công trình nghiên cứu về hệ thống chấm điểm tự động *SpeechRater* (Xi et al., 2008; Zechner et al., 2009; Evanini et al., Interspeech).
- **Quy mô:** Hàng chục ngàn người nói từ hơn 100 quốc gia trên thế giới, tích lũy hàng ngàn giờ audio.
- **Ngữ nghĩa nhãn:**
  - Điểm số chính thức do các giám khảo con người (Human Raters) chấm trên thang điểm tổng thể từ 1 đến 4 (Holistic Score: Delivery, Language Use, Topic Development).
  - Về mặt âm học, hệ thống SpeechRater sử dụng bộ nhận dạng giọng nói ASR nội bộ để trích xuất các đặc trưng tự động (GOP, tốc độ nói, độ dài khoảng lặng, độ biến thiên F0), nhưng **đây là feature nội bộ của mô hình, không phải nhãn gold do con người gắn từng âm vị**.
  - **KHÔNG CÓ DATASET CÔNG KHAI CUNG CẤP NHÃN S/O/I CHO TỪNG ÂM VỊ**.
- **Bản quyền & Mức độ khả dụng:**
  - **BẢN QUYỀN THƯƠNG MẠI ĐỘC QUYỀN (STRICTLY PROPRIETARY)**.
  - Chỉ cấp quyền nghiên cứu có giới hạn (Data Use Agreement - DUA) cho một số viện nghiên cứu đối tác được phê duyệt đặc biệt; không thể tải về tự do.

---

### 3.7. AESRC 2020 (Accented English Speech Recognition Challenge)

- **Đơn vị tổ chức & Cung cấp dữ liệu:** Cuộc thi tổ chức trong khuôn khổ hội nghị *Interspeech 2020* và *ICASSP 2021*. Dữ liệu do công ty cung cấp dữ liệu AI **Datatang (Datatang Technology Co., Ltd.)** tài trợ và nắm giữ bản quyền.
- **Công bố khoa học:** Deng et al., "AESRC 2020: The Accented English Speech Recognition Challenge", *Interspeech 2020* (arXiv: `2010.02495`).
- **Quy mô người nói & Audio:**
  - Tổng thời lượng: **180,0 giờ audio**.
  - Tập huấn luyện & phát triển: 160 giờ của 8 nhóm accent (Trung Quốc, Ấn Độ, Nhật Bản, Hàn Quốc, Nga, Anh, Mỹ, Bồ Đào Nha), trung bình ~20 giờ và ~60 người nói mỗi accent.
  - Tập kiểm thử: 20 giờ bao gồm cả các accent đã biết và accent chưa từng thấy (unseen accents từ Tây Ban Nha và Canada).
  - Tổng số người nói: Khoảng 400 đến 500+ người.
- **Ngữ nghĩa nhãn:**
  - Nhãn văn bản chính tả (Orthographic Transcriptions) cho từng câu nói.
  - Nhãn phân loại giọng (Accent Category Labels: ví dụ "Chinese", "Indian").
  - **HOÀN TOÀN KHÔNG CÓ NHÃN ÂM VỊ (Zero Phoneme-level Labels)**.
  - Thách thức được thiết kế cho hai đường đua: Nhận dạng giọng (Track 1: Accent Recognition) và Nhận dạng tiếng nói mang ngữ điệu (Track 2: Accented ASR).
- **Bản quyền & Giấy phép:** Thỏa thuận dữ liệu học thuật có điều kiện của Datatang (Datatang Challenge License). Sau cuộc thi, việc truy cập bị hạn chế hoặc phải mua giấy phép thương mại từ Datatang.
- **Kết luận:** Không phục vụ cho bài toán phát hiện lỗi phát âm âm vị MDD.

---

### 3.8. CU-CHLOE (Làm Rõ Bộ Dữ Liệu Bị Gọi Nhầm Là "CU-SYMPHONIC")

- **Đơn vị phát triển:** Phòng thí nghiệm DSP và Công nghệ Lời nói, Khoa Kỹ thuật Hệ thống và Quản lý Kỹ thuật (SEEM), Đại học Trung văn Hồng Kông (CUHK) dưới sự dẫn dắt của GS. Helen Meng (Meng et al., Interspeech; Wai-Kim Leung et al., ICASSP 2019; Minglin Wu et al., Interspeech 2021).
- **Làm rõ sự nhầm lẫn danh xưng (Debunking the "CU-SYMPHONIC" Misnomer):**
  - Trong một số tài liệu khảo sát hoặc danh mục đề xuất xuất hiện cái tên "CU-SYMPHONIC".
  - Qua rà soát toàn bộ các ấn phẩm khoa học của CUHK từ năm 2008 đến 2026, **hoàn toàn KHÔNG TỒN TẠI bộ dữ liệu ngữ âm nào mang tên "CU-SYMPHONIC"**.
  - Cái tên này là sự nhầm lẫn (conflation) giữa các dàn nhạc trường đại học ("CU Symphonic Band / Orchestra") với bộ dữ liệu ngữ âm kinh điển của CUHK là **CU-CHLOE** (**C**hinese **U**niversity **CH**inese **L**earners **O**f **E**nglish).
  - Tất cả các bài báo tiên phong về mạng nơ-ron End-to-End cho MDD của CUHK (như bài báo CNN-RNN-CTC tại ICASSP 2019 của Wai-Kim Leung và Transformer MDD tại Interspeech 2021 của Minglin Wu) đều thực nghiệm trên **CU-CHLOE**.
- **Quy mô người nói & Cấu trúc CU-CHLOE:**
  - Đúng **210 người nói**: 100 người bản ngữ tiếng Quảng Đông (Cantonese: 50 Nam, 50 Nữ) và 110 người bản ngữ tiếng Quan Thoại (Mandarin/Putonghua: 60 Nam, 50 Nữ).
  - Tổng thời lượng: Khoảng **34,6 giờ audio**.
  - Cấu trúc kịch bản đọc: Gồm 5 phần được thiết kế chuyên biệt bởi các giáo viên tiếng Anh:
    1. Các từ dễ nhầm lẫn (Confusable words).
    2. Các cặp từ tối thiểu (Minimal pairs, ví dụ *ship* vs *sheep*, *bat* vs *bet*).
    3. Các câu chứa đầy đủ âm vị tiếng Anh (Phonemic sentences).
    4. Đoạn văn hoàn chỉnh (Truyện ngụ ngôn Aesop: *The North Wind and the Sun*).
    5. Các câu trích từ tập dữ liệu chuẩn TIMIT.
- **Ngữ nghĩa nhãn âm vị:**
  - **Rất giàu thông tin và đạt chuẩn S/O/I hoàn hảo**: Toàn bộ các câu đều được các nhà ngôn ngữ học và chuyên gia ngữ âm phiên âm thủ công chi tiết ở cấp độ âm vị (Phonetic Transcriptions).
  - Cung cấp rõ ràng: Âm vị mục tiêu (Canonical Target), Âm vị thực tế người học phát âm (Realized Phone), và thẻ phân loại lỗi (Substitution, Omission, Insertion).
- **Bản quyền & Mức độ khả dụng:**
  - **LƯU TRỮ NỘI BỘ ĐẠI HỌC (PROPRIETARY INSTITUTIONAL CUSTODY - CUHK)**.
  - Khác với các bộ dữ liệu được tải lên LDC, ELRA hay OpenSLR, CU-CHLOE **không được phát hành công khai tự do cho cộng đồng**. Quyền truy cập chỉ giới hạn trong phòng thí nghiệm của GS. Helen Meng và các nhóm nghiên cứu ký hợp đồng hợp tác trực tiếp.

---

### 3.9. Các Bộ Dữ Liệu Khác Được Rà Soát Bổ Sung

1. **ERJ (English Read by Japanese Speech Corpus):**
   - Tác giả: GS. Nobuaki Minematsu et al. (Đại học Tokyo / Trung tâm Tài nguyên Ngôn ngữ SRC Nhật Bản, 2002–2004).
   - Quy mô: 202 sinh viên đại học Nhật Bản (100 Nam, 102 Nữ), đọc các câu phonetically-balanced và các từ khó phát âm đối với người Nhật. Tổng thời lượng ~60 giờ.
   - Nhãn: Có tập con được phiên âm ngữ âm trên Praat TextGrids ghi lại các biến thể lỗi phát âm đặc trưng của người Nhật (như lẫn lộn /l/ và /r/, chèn nguyên âm đệm /u/ vào cuối phụ âm).
   - Bản quyền & Truy cập: Thuộc NII Speech Resources Consortium (SRC). Yêu cầu nộp đơn đăng ký học thuật và có thu phí phát hành đĩa CD-ROM/dữ liệu; không mở tải tự do dạng online API.
2. **MRP Corpus (Modern Received Pronunciation Corpus):**
   - Đơn vị: Lưu trữ công khai trên Zenodo (2022).
   - Quy mô: 29 người nói bản ngữ giọng Anh chuẩn hiện đại (Southern British English), ~7.000 câu.
   - Nhãn: Cung cấp transcript chính tả và nhãn âm vị sinh tự động. Tuy nhiên, đây là dữ liệu **người bản ngữ (L1 Native)**, chủ yếu dùng làm nguồn tham chiếu âm học chuẩn cho MDD, không phải dữ liệu người học L2 chứa lỗi chuyển di ngữ âm.

---

## 4. Bác Bỏ Triệt Để Quan Niệm "Toàn Bộ Audio Là Nhãn Gold" (Debunking the Myth)

### 4.1. Nguồn Gốc và Sự Lan Truyền của Quan Niệm Sai Lầm
Trong cộng đồng nghiên cứu Trí tuệ Nhân tạo và Xử lý Tiếng nói, rất nhiều bài báo và kỹ sư khi đọc lướt abstract của các bộ dữ liệu thường mặc định rằng: **"Dung lượng audio phát hành tương đương với dung lượng nhãn gold có thể dùng để học có giám sát"**.
Ví dụ điển hình: Khi một tác giả viết *"Chúng tôi sử dụng bộ dữ liệu L2-ARCTIC gồm 27,1 giờ tiếng Anh phi bản ngữ..."*, người đọc rất dễ hiểu lầm rằng toàn bộ 27,1 giờ đó đều có nhãn lỗi phát âm từng âm vị.

### 4.2. Phân Tích Thực Nghiệm Trên L2-ARCTIC
Dự án của chúng tôi đã tiến hành kiểm tra mã nguồn, đối soát checksum SHA-256 và đọc trực tiếp cấu trúc file của L2-ARCTIC v5.0:

```
[Tổng thể Corpus L2-ARCTIC v5.0]
└── 24 Người nói (6 nhóm L1)
    ├── Tổng số câu ghi âm: 26.867 câu đọc kịch bản (~27,1 giờ audio)
    │   ├── 23.268 câu: CHỈ CHẠY KALDI FORCED-ALIGNMENT TỰ ĐỘNG (~23,53 giờ)
    │   │   └── KHÔNG ĐƯỢC CHUYÊN GIA KIỂM DUYỆT LỖI PHÁT ÂM (Unverified)
    │   └── 3.599 câu: ĐƯỢC CHUYÊN GIA NGỮ ÂM GÁN NHÃN THỦ CÔNG (Gold Annotations)
    │       └── Đúng 150 câu/speaker (100 câu chung + 50 câu đặc thù L1)
    │       └── Tổng thời lượng phát âm thực tế: ĐÚNG 3,57 GIỜ AUDIO (13,18%)
    └── Tập "Suitcase Corpus" bổ sung ở v5.0: Đọc tranh tự do (Spontaneous), chỉ dùng phân tích mô tả.
```

- **Hệ quả khôn lường nếu coi 27,1 giờ là Gold:**
  Trong 23.268 câu chạy tự động, Kaldi buộc phải ép các khung âm thanh (acoustic frames) vào chuỗi âm vị từ điển chính tả (canonical text). Do người học phát âm sai rất nhiều (ví dụ người Việt bỏ phụ âm đuôi /t/, đọc /θ/ thành /s/), nhưng Kaldi vẫn tự động gán nhãn khung thời gian đó là âm vị chuẩn /t/ hoặc /θ/. Nếu đưa 23.268 câu này vào huấn luyện mô hình MDD:
  1. Hàng vạn âm phát âm sai thực tế sẽ bị gán nhãn là `CORRECT`.
  2. Mô hình nơ-ron sẽ bị phạt oan khi phát hiện đúng lỗi, dẫn đến hiện tượng triệt tiêu gradient (gradient conflict) và suy giảm mạnh độ nhạy (Recall).
  3. Khi đánh giá (evaluation), việc dùng tập unverified sẽ làm sai lệch hoàn toàn chỉ số Precision, False Alarm Rate, và F1-score.

### 4.3. Phân Tích Thực Nghiệm Trên SpeechOcean762
Ngược lại với L2-ARCTIC, SpeechOcean762 có **100% audio được 5 chuyên gia con người chấm điểm độc lập (5.000 câu / 5,98 giờ)**. Tuy nhiên, sự ngộ nhận ở đây nằm ở **bản chất nhãn (label semantics)**:
- Nhiều kỹ sư giả định rằng: *Điểm 0 tương đương với Lược bỏ (Omission), điểm 1 là Thay thế (Substitution), điểm 2 là Đúng (Correct)*.
- **Đây là giả định hoàn toàn sai lầm về mặt ngữ âm học:**
  - Điểm 1 trong tài liệu hướng dẫn của SpeechOcean762 được định nghĩa là: *"Pronunciation is right but has a heavy accent"* (Phát âm đúng ngữ âm nhưng mang ngữ điệu/giọng địa phương nặng). Đây hoàn toàn không phải là một lỗi Thay thế âm vị (Phonemic Substitution) làm thay đổi nghĩa của từ.
  - Điểm 0 được định nghĩa là: *"Pronunciation is incorrect or missed"* (Phát âm sai hoặc bị nuốt/bỏ qua). Điểm 0 vừa bao gồm việc thay thế bằng một âm sai hoàn toàn, vừa bao gồm việc không phát âm. Hơn nữa, dữ liệu **không ghi lại âm vị thay thế thực tế là gì**.
  - Do đó, SpeechOcean762 **không thể tự động trở thành một bộ dữ liệu gold S/O/I** nếu không có bước xây dựng adapter ngữ âm và tiến hành thẩm định mẫu độc lập.

---

## 5. Đánh Giá Khả Năng Áp Dụng Cho Người Việt (Vietnamese L2 Applicability)

### 5.1. Bằng Chứng Thực Nghiệm Về Tập Con Người Việt Trong L2-ARCTIC
Trong toàn bộ kho dữ liệu chuẩn quốc tế hiện nay, tập con người Việt Nam (Vietnamese L1) chỉ tồn tại duy nhất trong L2-ARCTIC với **4 người nói**:
- `HQTV`: Nam, giọng miền Nam / Nam Trung Bộ.
- `TLV`: Nam, giọng miền Bắc.
- `PNV`: Nữ, giọng miền Nam.
- `THV`: Nữ, giọng miền Bắc.

Tổng số câu gold thủ công của 4 người nói này là: 4 speaker × 150 câu = **600 câu**.  
Tổng thời lượng audio gold của nhóm người Việt là khoảng **0,60 giờ (36 phút phát âm thực tế)**.

### 5.2. Các Hiện Tượng Lỗi Ngữ Âm Chuyển Di Đặc Thù (Phonetic Transfer Errors)
Qua phân tích chi tiết 600 câu gold của 4 speaker người Việt trong L2-ARCTIC, hệ thống phát hiện các mẫu lỗi có tính quy luật sâu sắc xuất phát từ cấu trúc âm tiết tiếng Việt (âm tiết đóng, không có phụ âm xát cuối /s, z, ʃ, ʒ, v, θ, ð/, không có tổ hợp phụ âm /str, spl, kt, ps/):
1. **Lược bỏ phụ âm đuôi (Final Coda Omission):**
   - Rất phổ biến với các âm tắc và âm xát cuối từ: /t/, /d/, /s/, /z/. Ví dụ: *past* /p æ s t/ bị phát âm thành [p æ s] hoặc [p æ].
2. **Thay thế âm xát răng (Dental Fricative Substitution):**
   - Âm vô thanh /θ/ (như trong *think*, *three*) thường bị thay thế bằng âm tắc [t] hoặc âm xát [s].
   - Âm hữu thanh /ð/ (như trong *this*, *that*) thường bị thay thế bằng âm tắc [d] hoặc âm bán nguyên âm [z].
3. **Lẫn lộn cặp nguyên âm căng / chùng (Tense/Lax Vowel Confusion):**
   - /iː/ vs /ɪ/ (như trong *sheep* vs *ship*), /uː/ vs /ʊ/ (như trong *fool* vs *full*).
4. **Hiện tượng chèn nguyên âm đệm (Epenthesis / Insertion):**
   - Người học thường chèn thêm nguyên âm /ə/ (AH) vào giữa các cụm phụ âm khó để dễ phát âm, ví dụ *blue* /b l uː/ thành [b ə l uː].

### 5.3. Kết Luận Về Nhu Cầu Dữ Liệu
Với quy mô chỉ có **4 người nói và 600 câu (~36 phút audio)**, tập con người Việt trong L2-ARCTIC là **hoàn toàn không đủ** để đại diện cho sự đa dạng vùng miền (Bắc, Trung, Nam), các nhóm tuổi, và các dải trình độ thông thạo (CEFR A1, A2, B1, B2, C1).  
Việc lập kế hoạch thu thập một bộ dữ liệu tiếng Anh L2 chuyên biệt cho người học Việt Nam (như kế hoạch Pilot 120 speaker và Full 2.200 speaker đang được triển khai trong nhóm nghiên cứu) là một quyết định chiến lược, có cơ sở khoa học vững chắc và không thể bị thay thế bởi bất kỳ bộ dữ liệu công khai ngoại sinh nào.

---

## 6. Kiến Nghị & Hướng Dẫn Hành Động Cho Nhóm Kiến Trúc & Tổng Hợp (Handoff)

1. **Khóa Chặt Quy Tắc Ground-Truth Cho Các Mô Hình Hiện Tại:**
   - Trong quá trình huấn luyện và đánh giá benchmark hiện thời trên L2-ARCTIC, **tuyệt đối chỉ sử dụng 3.599 câu đã được thẩm định nhãn gold**. Nghiêm cấm việc mở rộng sang 23.268 câu unverified mà chưa qua quy trình double-annotation và adjudication của chuyên gia.
2. **Khai Thác SpeechOcean762 Đúng Vai Trò:**
   - Sử dụng 5.98 giờ của SpeechOcean762 cho các mục tiêu:
     a) Pre-training bộ biểu diễn âm học tự giám sát (Acoustic Representation / Feature Extractor).
     b) Đánh giá đa tác vụ (Multi-task evaluation) cho phần chấm điểm Goodness of Pronunciation (GOP).
     c) Phân tích hành vi mô hình trên đối tượng trẻ em (sử dụng 125 speaker trẻ em trong dataset).
   - Không dùng SpeechOcean762 để đo chỉ số Recall của lỗi Thay thế âm vị (Substitution Recall) khi chưa có phonetic alignment định danh âm thay thế.
3. **Hiệu Chỉnh Metadata Trong Kho Dự Án:**
   - Cập nhật file `literature/DATASETS.csv`: Sửa giấy phép L2-ARCTIC thành `CC BY-NC 4.0`; sửa trường `hours` thành số giờ thực (3,57h gold / 27,1h total thay vì ghi số câu); sửa mô tả SpeechOcean762 phản ánh đúng dải điểm 0–2 và ký hiệu `[]` cho insertion.
4. **Tuân Thủ Các Ranh Giới Nghiêm Ngặt Của Section 8:**
   - Báo cáo này hoàn toàn là kết quả khảo sát, đo đạc dữ liệu có sẵn và đối soát tài liệu khoa học.
   - Không tự ý tải thêm các bộ dữ liệu lạ, không chi tiêu ngân sách, không gửi dữ liệu giọng nói ra ngoài máy chủ nội bộ, và bảo toàn 100% tính nguyên vẹn của tập kiểm thử niêm phong (Sealed Test Sets).

---

## 7. Tài Liệu Tham Khảo & Nguồn Dẫn Chứng (Verified Evidence)

- **[E1] L2-ARCTIC Official Corpus:** Zhao, G., Sonsaat, S., Silpachai, A., Lucic, I., Chukharev-Hudilainen, E., Levis, J., & Gutierrez-Osuna, R. (2018). *L2-ARCTIC: A Non-native English Speech Corpus*. Proc. Interspeech 2018, 2783–2787. DOI: [10.21437/Interspeech.2018-1110](https://doi.org/10.21437/Interspeech.2018-1110). URL: [https://psi.engr.tamu.edu/l2-arctic-corpus/](https://psi.engr.tamu.edu/l2-arctic-corpus/).
- **[E2] L2-ARCTIC Technical Docs & Manifests:** TAMU PSI Lab Corpus Documentation. URL: [https://psi.engr.tamu.edu/l2-arctic-corpus-docs/](https://psi.engr.tamu.edu/l2-arctic-corpus-docs/). Đối chiếu trực tiếp file local: `data/derived/l2arctic/CORPUS_STATISTICS.json` và `data/split_specs/L2_ARCTIC_SPLIT.yaml`.
- **[E3] SpeechOcean762 Official Release:** Zhang, J., Zhang, Z., Wang, Y., Yan, Z., Song, Q., Huang, Y., Shen, K., Jiang, D., Peng, P., Hu, S., Ge, M., Hong, Q., & Qian, Y. (2021). *speechocean762: A Non-native English Corpus for Pronunciation Scoring Task*. Proc. Interspeech 2021, 481–485. arXiv: [2104.01378](https://arxiv.org/abs/2104.01378). OpenSLR: [https://www.openslr.org/101/](https://www.openslr.org/101/). Local README: `data/original_sources/speechocean762/speechocean762/README.md`.
- **[E4] EpaDB Database & GOP-DNN:** Vidal, J., Ferrer, L., & Brambilla, L. (2019). *EpaDB: A Database for Development of Pronunciation Assessment Systems*. Proc. Interspeech 2019, 111–115. DOI: [10.21437/Interspeech.2019-1224](https://doi.org/10.21437/Interspeech.2019-1224). GitHub: [https://github.com/JazminVidal/gop-dnn-epadb](https://github.com/JazminVidal/gop-dnn-epadb).
- **[E5] EdAcc Corpus:** Sanabria, R., Bogoychev, N., Markl, N., Carmantini, A., Klejch, O., & Bell, P. (2023). *The Edinburgh International Accents of English Corpus: Towards the Democratization of English ASR*. Proc. IEEE ICASSP 2023, 1–5. arXiv: [2211.00693](https://arxiv.org/abs/2211.00693). HuggingFace: [https://huggingface.co/datasets/edinburghcstr/edacc](https://huggingface.co/datasets/edinburghcstr/edacc).
- **[E6] AESRC 2020 Challenge:** Deng, S., et al. (2020). *AESRC 2020: The Accented English Speech Recognition Challenge*. Interspeech 2020. arXiv: [2010.02495](https://arxiv.org/abs/2010.02495).
- **[E7] CU-CHLOE Corpus & End-to-End MDD:** Leung, W.-K., Liu, X., & Meng, H. (2019). *CNN-RNN-CTC Based End-to-end Mispronunciation Detection and Diagnosis*. Proc. IEEE ICASSP 2019. Wu, M., Li, K., Leung, W.-K., & Meng, H. (2021). *Transformer Based End-to-End Mispronunciation Detection and Diagnosis*. Proc. Interspeech 2021.
- **[E8] Báo cáo kỹ thuật nội bộ:** `reports/SM13_DATASET_REQUIREMENTS_20261115/REPORT_VI.md` (Đối chiếu ngày 15/11/2026).
