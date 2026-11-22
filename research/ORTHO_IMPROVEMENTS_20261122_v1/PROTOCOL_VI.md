# Giao thức Gói Cải tiến Trực giao Sau Chiến dịch Backbone (`ORTHO_IMPROVEMENTS_20261122_v1`)

**Trạng thái:** `PLAN_AND_CODE_READY_CPU_VERIFIED` · **Tách biệt hoàn toàn** khỏi `BACKBONE_REP_CAMPAIGN_20261122_v1` (không thay đổi mã hay hash phục vụ `--resume` của lượt `full_clean11_v1`).

## 1. Bằng chứng nền (từ phân tích CPU chỉ đọc trên `iso_no_hwin` 3 seed & `S0_clean` Cal)
1. **Bug lấy mẫu đều ở `TIMIT FIT` (`train_and_eval_seq_mdd_v2.py:270-276`):** `wrong` (`S`) và `bogus` (`O`) được chọn đều `1/38` trên `ARPA_39` trong khi tần suất âm tiếng Anh theo luật Zipf → các âm hiếm bị ngập lỗi giả trên TIMIT FIT (`ZH 95,6%`, `UH 74,0%`, `TH 70,6%`, `JH 70,1%`, `CH 67,5%`, `AW 64,2%`, `NG 55,1%`, `UW 21,6%` vs `T/R 4,1–4,2%`), trực tiếp biến `UW` thành #1 Substitution FA trên `TIMIT Core` (`FPR 12,8%` vs `1,1%` Held-6).
2. **Shortcut Corpus-Conditioned Phone Prior (`r = +0,726` Held-6, `+0,641` SAA-27, `+0,505` EpaDB):** Chênh lệch tỷ lệ lỗi Fit giữa `EpaDB` và `L2-ARCTIC` tới `6–17×` (`DH 3,3%` vs `57,2%`, `AH 54,9%` vs `8,8%`, `AO 48,5%` vs `6,4%`, `Z 80,4%` vs `48,6%`). Ngay cả khi khống chế `gold == C` và `can_lp_vit_max > -0,2`, `DH` trên Held-6 vẫn bị phạt `13,6%` (`FPR 12,1%`) vs `2,1%` (`0,6%`) trên EpaDB; trên `SAA-27` tỷ lệ cờ/lỗi lệch `0,42×–2,85×` giữa các người nói.
3. **Lỗ hổng `S ↔ O` mất `3,36–12,20 pp` giữa Binary F1 và Strict F1:** `Spontaneous unseen H6` (`75,70% → 63,50%`), `Held-6` (`71,30% → 66,59%`, riêng `R+L` chiếm `33,3%` swap dù `LIQ_GLIDE AP = 0,835`), `SAA-27` (`44 S→O` vs `7 O→S`), `EpaDB` (`72,68% → 69,32%`).
4. **Overconfidence của `GapHead` do `gap_pos_weight = 8,0`:** Dịch chuyển Bayes `ln(8) = +2,079` nén ngưỡng `tau_g` vào vùng bão hòa `0,875–0,90` (`sigmoid(ln 8) = 0,889`), làm độ phân giải lưới ngưỡng thô hơn `~2,5×` so với vùng `0,50` và gây lệch khi chuyển sang `SAA-27` (`oracle 0,60`).
5. **Variance giữa các seed (`1,39M` head params trên cùng 1 lượt forward backbone):** Trung bình 3 seed tăng đồng loạt trên `S0_clean` Cal (`tAP +0,0052`, `gAP +0,0301`, `L1 FA@R70 -25%`) và trên bộ lịch sử (`Held-6 +0,60 pp`, `EpaDB +0,50 pp`, `SAA-27 +0,79 pp`, `Spont H6 +0,41 pp`, `TIMIT FA -4%`, `Ctrl +0,95 pp`) với `< 2 ms` độ trễ thêm.

## 2. Thiết kế đối chứng cô lập 1 biến (áp lên bất kỳ backbone nào thắng `full_clean11_v1`)
Dùng đúng split sạch `clean_11` (`3.681 Fit / 635 Cal / 2.446 prompt Eval`, không chạm `test_sealed`, giữ nguyên checkpoint lịch sử và `GOAL.md`):

| Nhánh | Biến thay đổi duy nhất (so với `O0`) | Module triển khai | Giả thuyết kiểm định |
|---|---|---|---|
| **`O0_control`** | Đối chứng chuẩn (backbone thắng từ `full_clean11_v1`, `tau_la=0`, `lambda_so_cond=0`, `rho_o=0,5`, `gap_pos_weight=8,0`) | `losses.py` (parity `0,0` với baseline) | Mốc ghép cặp cùng seed/init/batch order |
| **`O1_timit_freq`** | Thay mẫu đều TIMIT Fit bằng tần suất sạch `P_timit(phone)` (giữ nguyên 1.000 WAV/speaker/tỷ lệ tổng) | `timit_freq_corrupt.py` | Giảm FA bản ngữ trên âm hiếm (`UW/NG/TH/AW`) mà không tụt Control Recall |
| **`O2_logit_adj`** | Logit Adjustment theo `(source, phone)` (`tau_la ∈ {0,5, 1,0}`, Empirical-Bayes `α=20` ước lượng thuần trên Fit) | `prior_debias.py`, `losses.py` | Giảm FPR trên âm `C` (`DH/Z/AH/IH/AO/AE`), tăng tổng quát hóa miền mới (`SAA-27`) |
| **`O2b_phone_drop`** | Phone-ID dropout (`p=0,4` lúc train, tắt lúc eval) — tách khỏi `O2` | `prior_debias.py` | Kiểm so sánh trực tiếp regularize embedding vs Logit Adjustment |
| **`O3_cond_so`** | Thêm loss điều kiện `S vs O` trên slot lỗi (`lambda_so_cond=0,5`) + hiệu chỉnh `rho_o` trên Cal | `losses.py`, `calibration.py` | Thu hẹp khoảng cách Binary→Strict F1 (`R/L`, Spontaneous, SAA-27) |
| **`O4_gap_platt`** | Hiệu chỉnh Platt/Temperature `(T_gap, b_gap)` thuần trên Cal (0 train lại) | `calibration.py` | Kéo `tau_g` về vùng tuyến tính `~0,4–0,6`, giảm ECE/NLL Gap và ổn định `I F1` |
| **`O5_head_ens3`** | Gộp 3 seed head trên cùng 1 lượt forward backbone, khóa ngưỡng thuần trên Cal | `calibration.py` | Giảm variance FP/Gap (`+0,4–0,8 pp` F1, `gAP +0,01–0,03`, `< 2 ms` phụ trội) |
| **`O_combo`** | **Chỉ ghép** các nhánh đơn lẻ vượt cổng Cal + Guard vào xác nhận 3 seed ghép cặp | tất cả module | Xác nhận cộng hưởng mà không gộp mù như `fix123_full` |

## 3. Quy chế đánh giá & chốt an toàn
- **Chọn mọi siêu tham số, `(tau_t, tau_g, rho_o, T_gap, b_gap)` và nhánh vào `O_combo` thuần trên `Cal`.**
- **Niêm phong Parquet dự đoán Eval trước khi ghép nhãn lịch sử**; đếm độc lập `S/O/I` và `Control Recall` ở cả hai điểm vận hành (Native Cal-locked và Cal L2 Recall-Matched).
- **Guard bắt buộc mọi seed:** `Control Recall ≥ 95%`, số FA `TIMIT Core` không tăng so với `O0`, và không làm tăng độ trễ backbone.
- **Không tự promote** hay mở `test_sealed.jsonl`.
