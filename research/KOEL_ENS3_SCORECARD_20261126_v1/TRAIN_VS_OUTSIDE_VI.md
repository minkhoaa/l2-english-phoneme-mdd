# Koel ENS3 — báo cáo trong và ngoài tập huấn luyện

**Mốc 26/11/2026:** `S0_clean_ens3_o5_o4`, cùng bundle và ngưỡng trên mọi tập. Strict P/R/F1 yêu cầu đúng vị trí **và đúng loại**; S = thay âm, O = bỏ âm, I = chèn âm. FA/câu chỉ đếm cảnh báo trên vị trí nhãn đúng, không tính nhầm S↔O; phụ thuộc độ dài câu. Dấu `—` nghĩa là không có lớp lỗi thật phù hợp để báo F1.

## 1. Trong tập train (Fit)

| Tập | Câu / người | Strict P / R / F1 | F1 S / O / I | FA/câu |
|---|---:|---|---|---:|
| L2-ARCTIC Fit | 1.042 / 11 | 68,13 / 78,13 / **72,79%** | 73,01 / 74,33 / 61,35% | 1,532 |
| EpaDB Fit | 1.639 / 26 | 81,54 / 90,24 / **85,67%** | 86,41 / 79,28 / 84,47% | 0,717 |
| TIMIT Fit, prompt can thiệp | 1.000 / 334 | 88,29 / 87,18 / **87,73%** | 91,29 / 82,61 / 79,67% | 0,312 |

TIMIT Fit là đối chứng huấn luyện bằng cách can thiệp prompt, không phải tập lỗi L2 tự nhiên. EpaDB Fit loại 1.026 vị trí `unknown` trước khi tính điểm.

## 2. Người nói ngoài Fit và Cal của head

| Tập | Câu / người | Strict P / R / F1 | F1 S / O / I | FA/câu |
|---|---:|---|---|---:|
| L2-ARCTIC Held-6, đọc | 802 / 6 | 62,67 / 70,24 / **66,24%** | 65,88 / 69,59 / 54,25% | 2,379 |
| EpaDB TEST, đọc | 1.266 / 20 | 64,76 / 73,89 / **69,02%** | 71,81 / 56,01 / 40,37% | 1,387 |
| L2-ARCTIC tự phát, lát không trùng người | 24 / 5 | 63,74 / 65,96 / **64,83%** | 68,34 / 56,14 / 32,00% | 5,500 |
| SAA-27, Stella | 27 / 27 | 39,91 / 48,81 / **43,92%** | 44,38 / 50,30 / 23,47% | 29,185 |
| SAA-18, Stella, từng dùng trong nghiên cứu cũ | 18 / 18 | 47,26 / 52,61 / **49,79%** | 52,15 / 53,45 / 18,32% | 23,389 |
| TIMIT Core sạch | 192 / 24 | — | — | 0,562 |
| TIMIT Core, 50 prompt thay âm trên cùng WAV | 50 / 7 | 90,94 / 93,55 / **92,23%** | 95,08 / — / — | 0,320 |

Chỉ tách người ở **head**; recipe backbone có L2-ARCTIC đọc và EpaDB, không có bằng chứng tách tuyệt đối từng bản ghi lúc tiền huấn luyện. Hai tập SAA dùng cùng một bài đọc dài, không phải hai kiểu bài khác nhau. TIMIT Core sạch có **108 cảnh báo trên 192 câu**; control phát hiện theo vị trí 271/279 lỗi cài vào prompt, không phải lỗi người nói thực sự mắc. EpaDB TEST có 949 vị trí `unknown` không tính.

## 3. Ngoài mẫu train nhưng không độc lập theo quy trình/người nói

| Tập | Câu / người | Strict P / R / F1 | F1 S / O / I | FA/câu |
|---|---:|---|---|---:|
| Cal L2-ARCTIC | 260 / 3 | 56,59 / 56,87 / **56,73%** | 59,06 / 41,38 / 51,61% | 2,012 |
| Cal EpaDB | 255 / 4 | 79,25 / 83,14 / **81,15%** | 83,21 / 69,71 / 65,12% | 0,824 |
| Cal TIMIT sạch | 120 / 60 | — | — | 0,025 |
| DEV532 intake, người đã gặp Fit/Cal | 532 / 8 | 60,51 / 66,88 / **63,54%** | 65,31 / 57,94 / 55,20% | 1,902 |
| Intake bổ sung nguồn TRAIN, người đã gặp Fit/Cal | 262 / 4 | 58,17 / 59,10 / **58,63%** | 60,00 / 51,57 / 54,88% | 1,844 |
| L2-ARCTIC tự phát, lát trùng người Fit/Cal | 85 / 13 | 60,37 / 61,26 / **60,81%** | 63,89 / 53,74 / 31,75% | 4,671 |

Cal đã dùng để hiệu chỉnh ngưỡng, nên không coi là test độc lập. Tổng tự phát 109 câu đạt 61,97% F1; đây là tổng của hai lát tự phát trên, không phải tập bổ sung. Intake “nguồn TRAIN” là tên split của repo nguồn, không phải mẫu đã dùng huấn luyện head ENS3.

## 4. Kết luận để báo cáo

- Từ Fit sang tập người nói giữ riêng cùng corpus, Strict F1 giảm **6,55 điểm phần trăm** ở L2-ARCTIC và **16,65 điểm** ở EpaDB. Đây là chênh lệch quan sát, không quy toàn bộ cho overfit vì người nói và phân bố lỗi khác nhau.
- **Chèn âm** có F1 thấp nhất trong S/O/I tại mọi tập L2 ngoài Fit/Cal trong bảng. EpaDB giảm mạnh nhất ở chèn âm: **84,47% → 40,37%**. SAA chỉ **43,92–49,79%** Strict F1; điểm tổng trên bài đọc khác không bảo đảm đúng từng nhóm âm.
- Chạy lại từ WAV trên 2.446 lượt cũ tái lập **139.996/139.996 quyết định**. Đây là kiểm tra tái lập, không phải thêm một test mới. Không có F1 chung gộp Fit/Cal/test; `test_sealed` chưa mở và chưa có điểm độc lập trên câu ngắn người Việt mới hoặc mock IELTS.

**Tệp số đếm:** [Fit và WAV replay](receipts/wav_replay_fit_v1/REPORT_VI.md) · [TP/FP/FN và P/R/F1 cho Fit/replay](receipts/wav_replay_fit_v1/METRICS.csv) · [Scorecard theo từng benchmark](receipts/scorecard_v1/REPORT_VI.md) · [P/R/F1 và AP từng loại ngoài Fit](receipts/scorecard_v1/THEO_LOAI.csv). Các kết quả từng vị trí và bảng lát cắt chi tiết lưu cục bộ trong `runs/`.
