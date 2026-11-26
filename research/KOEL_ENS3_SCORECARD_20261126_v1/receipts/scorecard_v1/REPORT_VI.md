# Scorecard chốt Koel ENS3 — lỗi phát âm theo từng benchmark

**Mô hình cố định:** `S0_clean_ens3_o5_o4` · `clean_11` · ngưỡng target `0,35`, phân loại O `0,50`, gap `0,38`. Chỉ tái đếm hoặc chạy suy luận mới, không train/chọn ngưỡng, `test_sealed` giữ đóng.

Các dòng con của Spontaneous và TIMIT control không cộng với tập cha. Nội dung này là benchmark chẩn đoán, không phải xác nhận độc lập EXP‑6.

## 1. Kết quả chính theo từng tập

| Benchmark | Câu / người | Slot được chấm / tổng | Lỗi thật | Strict P / R / F1 | Macro F1 | Binary P / R / F1 | KTC 95% F1 theo người |
|---|---:|---:|---:|---|---:|---|---|
| L2-ARCTIC Held‑6 (đọc) | 802 / 6 | 53.496 / 53.496 | 5.148 | 62,67% / 70,24% / **66,24%** | 63,24% | 66,93% / 75,02% / 70,75% | [58,05%; 71,78%] |
| EpaDB TEST | 1.266 / 20 | 42.147 / 43.096 | 4.807 | 64,76% / 73,89% / **69,02%** | 56,06% | 67,99% / 77,57% / 72,46% | [67,50%; 70,37%] |
| L2-ARCTIC tự phát — ngoài Fit/Cal | 24 / 5 | 3.612 / 3.612 | 517 | 63,74% / 65,96% / **64,83%** | 52,16% | 75,33% / 77,95% / 76,62% | — |
| L2-ARCTIC tự phát — người gặp ở Fit/Cal | 85 / 13 | 11.485 / 11.485 | 1.288 | 60,37% / 61,26% / **60,81%** | 49,79% | 69,63% / 70,65% / 70,13% | [56,45%; 63,93%] |
| L2-ARCTIC tự phát — tổng trộn | 109 / 18 | 15.097 / 15.097 | 1.805 | 61,35% / 62,60% / **61,97%** | 50,47% | 71,28% / 72,74% / 72,00% | [59,02%; 64,52%] |
| SAA‑27/Stella | 27 / 27 | 12.015 / 12.015 | 1.139 | 39,91% / 48,81% / **43,92%** | 39,39% | 43,43% / 53,12% / 47,79% | [39,51%; 48,06%] |
| SAA‑18/Stella | 18 / 18 | 8.010 / 8.010 | 804 | 47,26% / 52,61% / **49,79%** | 41,31% | 52,96% / 58,96% / 55,80% | [44,66%; 54,13%] |
| L2-ARCTIC intake DEV532 | 532 / 8 | 35.783 / 35.786 | 2.603 | 60,51% / 66,88% / **63,54%** | 59,48% | 64,82% / 71,65% / 68,07% | [56,91%; 68,72%] |
| L2-ARCTIC intake TRAIN bổ sung | 262 / 4 | 17.503 / 17.506 | 1.181 | 58,17% / 59,10% / **58,63%** | 55,48% | 59,75% / 60,71% / 60,23% | — |
| Cal L2-ARCTIC | 260 / 3 | 17.446 / 17.446 | 1.245 | 56,59% / 56,87% / **56,73%** | 50,69% | 58,19% / 58,47% / 58,33% | — |
| Cal EpaDB | 255 / 4 | 8.537 / 8.679 | 1.139 | 79,25% / 83,14% / **81,15%** | 72,68% | 82,43% / 86,48% / 84,40% | — |
| TIMIT Core (bản ngữ) | 192 / 24 | 12.940 / 12.940 | 0 | — / — / **—** | — | — / — / — | — |
| Cal TIMIT (sạch) | 120 / 60 | 7.680 / 7.680 | 0 | — / — / **—** | — | — / — / — | — |
| TIMIT control (prompt thay âm) | 50 / 7 | 3.352 / 3.352 | 279 | 90,94% / 93,55% / **92,23%** | — | 94,43% / 97,13% / 95,76% | [88,37%; 95,67%] |

Strict: khớp loại và vị trí; nhầm loại tính cả FP và FN strict. Binary: chỉ khớp vị trí. KTC bootstrap 2.000 lượt theo người (≥6 người), không suy rộng sang population khác. Macro chỉ khi đủ cả S/O/I; với toàn âm đúng, xem báo oan thay F1.

## 2. Theo loại lỗi và mức báo sai

| Benchmark | F1 S | F1 O | F1 I | AP đích / khe | Báo oan C/N | Báo oan/câu | Bỏ sót (không cờ) | S→O / O→S |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| L2-ARCTIC Held‑6 (đọc) | 65,88% | 69,59% | 54,25% | 0,775 / 0,562 | 1908/48348 | 2,379 | 1286 | 124 / 122 |
| EpaDB TEST | 71,81% | 56,01% | 40,37% | 0,815 / 0,374 | 1756/37340 | 1,387 | 1078 | 88 / 89 |
| L2-ARCTIC tự phát — ngoài Fit/Cal | 68,34% | 56,14% | 32,00% | 0,826 / 0,540 | 132/3095 | 5,500 | 114 | 32 / 30 |
| L2-ARCTIC tự phát — người gặp ở Fit/Cal | 63,89% | 53,74% | 31,75% | 0,780 / 0,298 | 397/10197 | 4,671 | 378 | 47 / 74 |
| L2-ARCTIC tự phát — tổng trộn | 65,18% | 54,41% | 31,82% | 0,794 / 0,339 | 529/13292 | 4,853 | 492 | 79 / 104 |
| SAA‑27/Stella | 44,38% | 50,30% | 23,47% | 0,471 / 0,217 | 788/10876 | 29,185 | 534 | 42 / 7 |
| SAA‑18/Stella | 52,15% | 53,45% | 18,32% | 0,584 / 0,219 | 421/7206 | 23,389 | 330 | 39 / 12 |
| L2-ARCTIC intake DEV532 | 65,31% | 57,94% | 55,20% | 0,755 / 0,548 | 1012/33180 | 1,902 | 738 | 85 / 39 |
| L2-ARCTIC intake TRAIN bổ sung | 60,00% | 51,57% | 54,88% | 0,661 / 0,596 | 483/16322 | 1,844 | 464 | 15 / 4 |
| Cal L2-ARCTIC | 59,06% | 41,38% | 51,61% | 0,642 / 0,489 | 523/16201 | 2,012 | 517 | 15 / 5 |
| Cal EpaDB | 83,21% | 69,71% | 65,12% | 0,928 / 0,652 | 210/7398 | 0,824 | 154 | 17 / 21 |
| TIMIT Core (bản ngữ) | — | — | — | — / — | 108/12940 | 0,562 | 0 | 0 / 0 |
| Cal TIMIT (sạch) | — | — | — | — / — | 3/7680 | 0,025 | 0 | 0 / 0 |
| TIMIT control (prompt thay âm) | 95,08% | — | — | 0,986 / — | 16/3073 | 0,320 | 8 | 10 / 0 |

AP được đọc cùng tỷ lệ lỗi gốc của từng miền; không so AP đích với AP khe như cùng mẫu số. P/R/F1, AP, TP/FP/FN từng lớp: [THEO_LOAI.csv](THEO_LOAI.csv). Ma trận C/S/O và N/I, Brier theo miền, báo oan/1.000 vị trí đúng: `RESULTS.json` (cục bộ).

## 3. Lát /t,d/ cuối từ và gợi ý bỏ âm

Cùng quy cách Flow 1 trên câu đọc L2‑ARCTIC. Không áp quy tắc vị trí từ cho corpus thiếu word mapping được đối soát.

| Tập | Âm | Câu có vị trí / câu map được | Vị trí chấm | S đúng / S thật | O đúng / O thật | Báo oan | Strict F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| L2-ARCTIC Held‑6 (đọc) | T+D | 409/573 | 641 | 41/108 | 74/112 | 51 | 57,36% |
| L2-ARCTIC Held‑6 (đọc) | D | 241/573 | 281 | 20/48 | 34/52 | 31 | 55,67% |
| L2-ARCTIC Held‑6 (đọc) | T | 266/573 | 360 | 21/60 | 40/60 | 20 | 58,94% |
| L2-ARCTIC intake DEV532 | T+D | 276/397 | 428 | 13/29 | 12/29 | 41 | 38,46% |
| L2-ARCTIC intake DEV532 | D | 166/397 | 193 | 9/15 | 8/11 | 25 | 48,57% |
| L2-ARCTIC intake DEV532 | T | 176/397 | 235 | 4/14 | 4/18 | 16 | 26,67% |
| L2-ARCTIC intake TRAIN bổ sung | T+D | 145/199 | 222 | 4/18 | 9/25 | 9 | 39,39% |
| L2-ARCTIC intake TRAIN bổ sung | D | 77/199 | 93 | 4/11 | 4/7 | 3 | 53,33% |
| L2-ARCTIC intake TRAIN bổ sung | T | 101/199 | 129 | 0/7 | 5/18 | 6 | 27,78% |
| Cal L2-ARCTIC | T+D | 136/185 | 211 | 4/28 | 4/12 | 7 | 28,57% |
| Cal L2-ARCTIC | D | 81/185 | 98 | 4/14 | 3/8 | 3 | 42,42% |
| Cal L2-ARCTIC | T | 88/185 | 113 | 0/14 | 1/4 | 4 | 8,70% |

### Gợi ý bỏ âm ở mọi vị trí đích (không phải chấm giáo viên)

| Tập | O thật | O đúng / gợi ý sai với ngưỡng model | O đúng / gợi ý sai với chốt demo 0,70 |
|---|---:|---:|---:|
| L2-ARCTIC Held‑6 (đọc) | 1137 | 842 / 441 | 393 / 74 |
| EpaDB TEST | 459 | 268 / 230 | 187 / 78 |
| L2-ARCTIC tự phát — ngoài Fit/Cal | 106 | 64 / 58 | 18 / 13 |
| L2-ARCTIC tự phát — người gặp ở Fit/Cal | 285 | 158 / 145 | 62 / 25 |
| L2-ARCTIC tự phát — tổng trộn | 391 | 222 / 203 | 80 / 38 |
| SAA‑27/Stella | 173 | 124 / 196 | 39 / 39 |
| SAA‑18/Stella | 130 | 89 / 114 | 26 / 17 |
| L2-ARCTIC intake DEV532 | 407 | 281 / 282 | 127 / 40 |
| L2-ARCTIC intake TRAIN bổ sung | 141 | 74 / 72 | 26 / 8 |
| Cal L2-ARCTIC | 121 | 60 / 109 | 13 / 11 |
| Cal EpaDB | 121 | 84 / 36 | 52 / 14 |

Lát cắt từng âm/khe, người, L1 và vai trò Fit/Cal: `LAT_CAT.csv` (cục bộ). Nhóm ít người chỉ mô tả ca; không dùng để xếp hạng giọng.

## 4. Nhận dạng âm vị, đối chứng bản ngữ và độ trễ

PER dưới đây là **phép đo lịch sử của recognizer Koel Base đóng băng**, không phải điểm MDD head, không gộp inventory 39/79 và không phải WER:

| Tập / chuẩn âm | PER | Số edit / âm tham chiếu |
|---|---:|---:|
| dev18_common39 | 11,30% | 6011/53179 |
| saa27_common39 | 21,95% | 1321/6017 |
| epadb_test_native79 | 18,42% | 3827/20779 |
| timit_core_surface39 | 12,11% | 698/5764 |
| timit_core_canonical39 | 16,90% | 1077/6374 |

- Core: 106 cờ âm đích + 2 cờ khe trên 192 câu (0,562/câu). 78 câu có ≥1 cờ. Nhãn all‑C/N có thể gồm biến thể bản ngữ.
- Control: 271/279 lỗi **cài ở prompt** có cờ (97,13%), không phải người nói thật mắc lỗi.
- Độ trễ lịch sử của bundle trên GTX 1650: câu ngắn Held‑6 50 mẫu P50/P95 `155,8/220,6 ms`, Stella 5 mẫu P50 `1.477,6 ms`; CPU 10 câu ngắn P50 `1.791,1 ms`. Không dùng lượt recount để đo dịch vụ.

## 5. Phạm vi và cách đọc

- **1.447 WAV mới + 2.446 lượt archive**; đã ghi đủ 235.103 vị trí nguồn trước mask. Trong 265 ứng viên TRAIN, 3 trượt đối chiếu dạng sóng trước suy luận; lý do từng ca trong `LOCK.json`.
- Eval lịch sử: head không học các người trong Held‑6, EpaDB TEST, SAA‑27, TIMIT Core; Spontaneous tách ngoài/đã gặp Fit/Cal. Tách người ở head **không** chứng minh không phơi nhiễm khi pretrain Koel; recipe công bố có L2‑ARCTIC đọc và EpaDB. EpaDB/SAA dùng nhãn quy đổi, không đồng nhất chuẩn nhãn với L2‑ARCTIC.
- Cal gồm ba bộ đã dùng để hiệu chỉnh. DEV532 và TRAIN bổ sung không trùng chính bản thu Fit/Cal, nhưng dùng người nói đã xuất hiện ở Fit/Cal; có thể đọc lại cùng văn bản. SAA‑18/27 là các người khác nhau đọc cùng một bài Stella; SAA‑18 từng dùng hiệu chỉnh nghiên cứu cũ, không dùng hiệu chỉnh head này.
- Chi tiết âm thực phát đúng ký hiệu, chất lượng lời sửa, tiếng Anh người Việt mới và WER không được xác nhận qua chỉ số S/O/I.
- Không đưa nguồn không có nhãn strict, dữ liệu giả, hoặc `test_sealed` vào bảng. SpeechOcean762 TEST không có bản thu trong cache đã kiểm; nhãn điểm 0 không tách S/O.

Tái lập: `PROTOCOL_VI.md` (workspace nghiên cứu) · `LOCK.json` (cục bộ) · `SEAL.json` (cục bộ) · [Biên nhận kiểm](VERIFICATION.json).
