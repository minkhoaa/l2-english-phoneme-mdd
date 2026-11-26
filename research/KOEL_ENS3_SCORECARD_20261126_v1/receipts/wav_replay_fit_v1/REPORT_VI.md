# ENS3 từ WAV — đối chiếu benchmark và chấm toàn Fit

Model/ngưỡng như scorecard gốc, không huấn luyện. 2.446 lượt archive (gồm 50 prompt control trên WAV Core đã có) + 3.681 lượt Fit. Fit chỉ đo trên mẫu đã học. Không mở `test_sealed`.

## Benchmark cũ: chạy lại từ WAV

| Tập | Câu | Call khớp / tổng (đích; khe) | Δ điểm số lớn nhất | Strict F1 cũ → mới |
|---|---:|---|---:|---|
| EpaDB TEST | 1266 | 20,915/20,915; 22,181/22,181 | 1.96e-06 | 69,02% → 69,02% |
| L2-ARCTIC Held‑6 (đọc) | 802 | 26,347/26,347; 27,149/27,149 | 1.42e-06 | 66,24% → 66,24% |
| SAA‑27/Stella | 27 | 5,994/5,994; 6,021/6,021 | 0.000102 | 43,92% → 43,92% |
| L2-ARCTIC tự phát — tổng trộn | 109 | 7,494/7,494; 7,603/7,603 | 7.14e-07 | 61,97% → 61,97% |
| TIMIT Core (bản ngữ) | 192 | 6,374/6,374; 6,566/6,566 | 5.79e-07 | — → — |
| TIMIT control (prompt thay âm) | 50 | 1,651/1,651; 1,701/1,701 | 5.59e-07 | 92,23% → 92,23% |
| L2-ARCTIC tự phát — ngoài Fit/Cal | 24 | lát con | — | 64,83% → 64,83% |
| L2-ARCTIC tự phát — người gặp ở Fit/Cal | 85 | lát con | — | 60,81% → 60,81% |

## Fit: kết quả trong tập train

| Tập | Câu / người | Vị trí chấm / tổng | Strict P / R / F1 | F1 S / O / I | Báo oan |
|---|---:|---:|---|---|---:|
| Fit L2-ARCTIC | 1042 / 11 | 70,046/70,046 | 68,13% / 78,13% / 72,79% | 73,01% / 74,33% / 61,35% | 1596 |
| Fit EpaDB | 1639 / 26 | 54,609/55,635 | 81,54% / 90,24% / 85,67% | 86,41% / 79,28% / 84,47% | 1175 |
| Fit TIMIT (prompt can thiệp) | 1000 / 334 | 68,560/68,560 | 88,29% / 87,18% / 87,73% | 91,29% / 82,61% / 79,67% | 312 |

Fit TIMIT dùng nhãn/prompt can thiệp anti-shortcut, không phải lỗi L2 tự nhiên. Không gộp Fit với Eval.

[P/R/F1 và TP/FP/FN](METRICS.csv) · `DIFF.csv` (cục bộ) · `SEAL.json` (cục bộ) · `RESULTS.json` (cục bộ)
