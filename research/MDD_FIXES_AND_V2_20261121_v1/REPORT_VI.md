# Báo Cáo Fix Lỗi Tồn Đọng (P0 Zero-Retrain + P1 Kiến Trúc Mới) & Kết Quả Trước/Sau Khi Giữ Nguyên Checkpoint

- **Ngày thực hiện:** `2026-11-21`
- **Thư mục mã & kết quả:** `research/MDD_FIXES_AND_V2_20261121_v1/`
  - **P0 Fix Koel Base & R0 (giữ nguyên checkpoint):** `koel_ipa_map_v2.py` (phủ 75/75 token IPA của Koel, 0% unmapped + quy tắc allophone theo ngữ cảnh), `feature_extractor_v2.py` (Viterbi + `can_lp` nhận biết allophone cho toàn bộ 39 âm, tương thích drop-in với `r0_base.pt` & các fold head cũ).
  - **P0 Fix SM13 (giữ nguyên checkpoint `seed101`):** `sm13_e4_fixed.py` (sửa `compute_e4_likelihoods` từ 4 âm kế cận alphabet + `np.mean` sang top-6 ứng viên cạnh tranh âm học thực từ `log_probs` + `np.max`).
  - **P1 Kiến trúc & Loss mới (sẵn sàng cho train lại):** `seq_mdd_v2.py` (`SeqMDDv2`: scalar mixing 13 tầng XLSR `L12–24` + chuẩn hóa CMVN mức câu + bỏ toàn bộ shortcut rời rạc `a_call_is_*`/`align_tier_*` + BiGRU 2 tầng toàn câu + 3 đầu đa nhiệm `Target C/S/O` + `Gap N/I` + `Realized-Phone 40 lớp` + loss neo người bản ngữ L1).
  - **Kiểm thử đơn vị:** `tests/test_mdd_fixes_v2.py` (`5/5` pytest pass).
  - **Benchmark trước/sau P0 (0 train lại, 0 sửa checkpoint cũ):** `run_p0_zero_retrain_benchmark.py` → `outputs_p0/P0_BEFORE_AFTER_RESULTS.json` (`1.338,6s` trên GTX 1650).

---

## 1. Kết Quả Trước vs Sau P0 (Giữ Nguyên 100% Checkpoint Cũ — 0 Train Lại)

### 1.1. Trên 4 tập L2 (`Held-6`, `Spontaneous`, `EpaDB TEST`, `SAA-27`)

| Tập dữ liệu | Hệ thống | **TRƯỚC (Old)** | **SAU P0 (0 train lại)** | Chênh lệch (Δ) | Ghi chú chẩn đoán |
|---|---|---:|---:|---:|---|
| **`EpaDB TEST`** (1.266 câu, 20 spk unseen) | **Koel Base (Van)** | 53,70% (`?=2.444`) | **64,81%** (`?=380`) | **+11,11 pp** | Unmapped `2.660 → 0`; vượt cả R0 chưa train lại! |
| | **R0 (Cal / Oracle)** | 58,95% / 61,89% (AP 0,677) | **60,75% / 64,57%** (AP **0,755**) | **+1,81 / +2,68 pp** | Precision `K`: **12,4%→70,8%** (240→14 FP); `P`: **12,4%→75,0%** (78→3 FP); Recall `AE`: **32,5%→45,8%**! |
| | **SM13 (Cal / Oracle)** | 15,27% / 15,55% (`S→O=1.976`) | **35,44% / 38,80%** (`S→O=851`) | **+20,17 / +23,25 pp** | Locked `S`: `0,0% → 7,9%`; AP `0,431 → 0,481`. |
| **`L2A Held-6`** (802 câu, R0 unseen / SM13 train) | **Koel Base (Van)** | 56,96% (`?=1.351`) | **56,93%** (`?=740`) | -0,03 pp | Unmapped `470 → 0`. |
| | **R0 (Cal / Oracle)** | 66,75% / 66,80% (AP 0,792) | 65,76% / 65,79% (AP 0,791) | -0,99 pp | AP giữ nguyên (`0,791`); tụt `-1 pp` do lệch phân phối đặc trưng với checkpoint train trên feature cũ! |
| | **SM13 (Cal / Oracle)** | 40,41% / 40,65% (`S→O=1.386`) | **49,81% / 63,69%** (`S→O=266`) | **+9,40 / +23,04 pp** | Locked `S`: `0,0% → 25,9%`; AP `0,652 → 0,767`; cắt **80,8%** nhầm `S→O`! |
| **`L2A Spontaneous`** (129 câu tự phát) | **Koel Base (Van)** | 56,12% (`?=593`) | **56,70%** (`?=386`) | **+0,58 pp** | Unmapped `164 → 0`. |
| | **R0 (Cal / Oracle)** | 61,55% / 62,62% (AP 0,789) | 61,00% / 62,29% (AP 0,790) | -0,54 pp | AP giữ nguyên (`0,790`), cần train lại trên feature v2. |
| | **SM13 (Cal / Oracle)** | 39,19% / 40,47% (`S→O=389`) | **49,89% / 53,54%** (`S→O=101`) | **+10,70 / +13,07 pp** | Locked `S`: `0,0% → 19,1%`; AP `0,600 → 0,644`. |
| **`SAA-27` / Stella** (27 spk unseen, dài 22–49s) | **Koel Base (Van)** | 37,46% | 36,37% | -1,09 pp | Unmapped `→ 0`. |
| | **R0 (Cal / Oracle)** | 43,94% / 44,78% (AP 0,510) | 42,71% / 44,18% (AP 0,512) | -1,23 pp | AP nhích nhẹ (`0,512`), cần train lại trên feature v2. |
| | **SM13 (Cal / Oracle)** | 37,27% / 38,12% (`S→O=136`) | **42,15% / 44,10%** (`S→O=29`) | **+4,88 / +5,98 pp** | Locked `S`: `0,0% → 22,7%`; gần ngang `R0` Oracle (`44,18%`) trên Stella; cắt **78,7%** `S→O`! |

### 1.2. Trên Đối Chứng Người Bản Ngữ Mỹ (`TIMIT Core` — 192 câu, 6.374 âm đích)

| Hệ thống | TRƯỚC (`FA/câu` / `FA/1k âm`) | SAU P0 (`FA/câu` / `FA/1k âm`) | Thay đổi |
|---|---:|---:|---:|
| **Koel Base (Vanilla)** | 2,20 / 64,3 | 2,96 / 88,8 | +0,76 *(vì 1.216 `?` trước đây che lỗi, nay chấm thật)* |
| **R0 Locked (`τ=0,624`)** | 5,13 / 152,5 | **3,05 / 91,5** | **-40,0% lỗi giả bản ngữ!** |
| **R0 Cal-F1 (`τ=0,341`)** | 7,92 / 236,6 | **4,93 / 147,9** | **-37,5% lỗi giả bản ngữ!** |
| **SM13 Cal-F1** *(dùng ngưỡng cũ chưa re-cal)* | 6,22 / 186,4 | 11,01 / 330,6 | *(bắt buộc hiệu chỉnh lại ngưỡng `τ_t` vì `P_e4(S)` không còn = 0!)* |

---

## 2. Trả Lời Câu Hỏi: Sau Khi Chỉnh Sửa Xong, Có Cần Train Lại Model Không?

**Câu trả lời chính xác theo từng mô hình và mục tiêu:**

1. **Với `Koel Base` (P0-Fixed):** **KHÔNG cần train lại.** Chỉ riêng sửa bảng ánh xạ 75/75 token IPA + căn lề nhận biết allophone đã kéo `EpaDB TEST` nhảy thẳng **`+11,11 pp` (`53,70% → 64,81%`)**.
2. **Với `SM13` (P0-Fixed E4):**
   - **KHÔNG cần train lại để cứu bug E4**, nhưng **BẮT BUỘC phải Calibrate lại ngưỡng `(τ_t, τ_g)`** (mất vài giây trên CPU, không train) vì phân phối `P_fused(S)` đã đổi hoàn toàn (Oracle nhảy **`+6` đến `+23 pp`** trên cả 4 tập L2, cắt 57–81% nhầm `S→O`).
   - **CÓ cần train lại** nếu muốn dùng SM13 lâu dài, vì ngay cả ở Oracle sau khi sửa E4, SM13 vẫn overfit 12 người train (`63,7%` Held-6 train vs `38,8%` EpaDB unseen) và `GapHead` gần như chết (`Gap AP 0,022` EpaDB).
3. **Với `R0` → `SeqMDDv2` (QUAN TRỌNG NHẤT): CÓ — RẤT CẦN TRAIN LẠI HEAD MỚI (`~5–15 phút` trên GTX 1650, vẫn đóng băng Koel Base!)** vì 4 lý do đã chứng minh bằng số liệu trên:
   - **Lệch phân phối đặc trưng với checkpoint cũ:** `r0_base.pt` được train trên feature lỗi cũ (`40/75` map + `align_tier_3` shortcut), nên khi cấp feature sạch v2 mà không train lại, tuy `EpaDB` nhảy vọt (`K/P` precision `12% → 71–75%`, `AP 0,677 → 0,755`) và lỗi giả bản ngữ TIMIT giảm ngay **38–40%**, thì `Held-6`/`SAA-27` lại tụt nhẹ `-1 pp` do sai lệch `sc_mean/sc_std` và trọng số shortcut cũ.
   - **Lỗi giả người bản ngữ vẫn quá cao (`147,9/1k` ở Cal-F1, `91,5/1k` ở Locked):** Phải train lại với **L1 Native Anchor (TIMIT TRAIN / LibriSpeech `all-C`)** để kéo xuống `< 15–20 FA/1k`.
   - **Vẫn mù hoàn toàn lỗi chèn âm (`I`):** Checkpoint `r0_base.pt` cũ không có tham số nào cho khe chèn; phải train `SeqMDDv2` để học đồng thời `Gap Head (N/I)` + `Target Head (C/S/O)` + `Realized-Phone Head`.
   - **Cần dữ liệu đa corpus + ngữ cảnh mức câu:** Train `SeqMDDv2` (đa tầng XLSR `L12–24`, BiGRU toàn câu, bỏ shortcut rời rạc) trên **L2-ARCTIC + EpaDB TRAIN + TIMIT TRAIN** (tách người nói tuyệt đối, giữ `Held-6` & `EpaDB TEST` & `SAA-27` & `TIMIT Core` hoàn toàn ngoài tập train) sẽ giải quyết triệt để cả 4 điểm nghẽn cùng lúc!

---

## 3. Kết Quả Train & Đánh Giá `SeqMDDv2` (P1 — Checkpoint Mới Riêng Biệt, Giữ Nguyên Mọi Checkpoint Cũ)

### 3.1. Thiết lập huấn luyện & Phát hiện quan trọng: Anti-Shortcut trên TIMIT TRAIN

- **Tách dữ liệu 0% rò rỉ người nói (`train_and_eval_seq_mdd_v2.py`):**
  - **FIT (4.011 câu, 375 người):** `L2-ARCTIC DEV-18` (15 người, 1.372 câu) + `EpaDB TRAIN` (26 người, 1.639 câu) + `TIMIT TRAIN` (400 người bản ngữ Mỹ, 1.000 câu).
  - **CALIBRATE (635 câu, 67 người — chỉ dùng chọn epoch & khóa ngưỡng `(τ_t, τ_g)`):** `DEV-18` (3 người `ERMS, SVBI, YBAA`, 260 câu) + `EpaDB TRAIN` (4 người, 255 câu) + `TIMIT TRAIN` sạch (60 người, 120 câu).
  - **EVAL (100% người nói chưa từng thấy trong Fit/Cal):** `Held-6` (802), `Spontaneous` (129; riêng lát Held-6 unseen 24 câu), `EpaDB TEST` (1.266), `SAA-27` (27), `TIMIT Core` (192).
- **Kiến trúc & tham số:** Backbone `Koel Base` đóng băng hoàn toàn; `SeqMDDv2` (`1.441.419` tham số học = 6 tầng XLSR `L12, 15, 18, 20, 22, 24` scalar mixing + CMVN mức câu + bỏ toàn bộ shortcut rời rạc + BiGRU 2 tầng toàn câu + 3 đầu `Target C/S/O`, `Gap N/I`, `Realized-Phone 40 lớp`). Trọng số tầng học được xác nhận `L12, L15, L18` đóng góp lớn nhất (`52,8%`: `[0.179, 0.178, 0.171, 0.149, 0.158, 0.164]`).
- **Phát hiện kiểm định Zero-Trust (TIMIT Corruption Control — 50 câu TIMIT Core chèn 15% lỗi giả vào `canonical_phones` trên CÙNG audio/mic TIMIT):**
  - **Lượt 1 (TIMIT TRAIN gắn `all-C` thuần túy, `seq_mdd_v2_seed20261121.pt`):** Tưởng như siêu thấp (`0,10 FA/câu`), nhưng bài test đối chứng phát hiện BiGRU học **shortcut nhận diện micro TIMIT** (Recall trên âm bị đánh tráo sai ngay trong audio TIMIT chỉ **`2,91%`**!).
  - **Lượt 2 (Anti-Shortcut `seq_mdd_v2_antishortcut_seed20261121.pt`, best epoch 4, SHA `a9bb69ed5cdd49d6…`, khóa `τ_t=0,29, τ_g=0,825` trên Calibrate):** Tiêm ngẫu nhiên `10%` tráo âm `S` + `3%` âm thừa `O` + `3%` xóa âm `I` vào `canonical_phones` ngay trong từng câu `TIMIT TRAIN` thật (vẫn giữ `~87%` âm bản ngữ `C` trên cùng mic) → **triệt tiêu hoàn toàn shortcut micro TIMIT**:
    - Trên 50 câu TIMIT Core đối chứng (cùng mic TIMIT): âm bản ngữ đúng chỉ báo **`2,84%`**, âm đánh tráo sai bị bắt **`96,42%` Recall** (**`PR-AUC = 0,9830`**, **`ROC-AUC = 0,9934`**)!

### 3.2. Bảng đối đầu tổng hợp toàn bộ 5 tập Eval Unseen (`eval_seq_mdd_v2_final.py`)

| Tập dữ liệu (Unseen) | `Koel Base` (Old → P0) | `SM13` (Old → P0 Cal/Or) | `R0` (Old → P0) | **`SeqMDDv2` Anti-Shortcut** (Cal / Or) | Δ `SeqMDDv2` vs `R0` Old (95% CI) | **`I` F1** (`R0/Van` → **`SeqMDDv2`**) | Độ trễ GPU P50 (`SeqMDDv2` vs `R0`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`EpaDB TEST`** (1.266 câu, 20 spk) | 53,70% → 64,81% | 15,27% → 35,44 / 38,80% | 58,95% → 60,75% | **68,29%** / **68,62%** *(tAP 0,806)* | **+9,34 pp [+7,13; +11,80]** (20/0/0!) | 25,6% → **43,4%** (+17,8 pp) | **112,1 ms** vs 112,6 ms |
| **`L2A Spontaneous`** (129 câu tự phát) | 56,12% → 56,70% | 39,19% → 49,89 / 53,54% | 61,55% → 61,00% | **63,22%** *(unseen Held-6: **64,38%**)* / 63,63% | **+1,67 pp [+0,18; +3,24]** (14/8/0) | 41,3% → **41,4%** *(gAP 0,341 vs SM13 0,119)* | **364,1 ms** vs 373,5 ms |
| **`SAA-27` / Stella** (27 spk, dài 22–49s) | 37,46% → 36,37% | 37,27% → 42,15 / 44,10% | 43,94% *(SAA-Cal)* → 42,71% | **45,39%** *(0 chạm SAA!)* / **45,91%** | **+1,45 pp** *(+1,96 pp vs DEV-Cal)* | 19,2% → **23,4%** *(Or **29,2%**)* | **1.360,3 ms** vs 1.449,7 ms *(nhanh hơn 10× SM13!)* |
| **`L2A Held-6`** (802 câu, 6 spk) | 56,96% → 56,93% | 40,41% → 49,81 / 63,69% | **66,75%** → 65,76% | **65,20%** / 65,85% *(tAP 0,777)* | -1,54 pp [-3,09; -0,47] *(train 15 vs 18 spk)* | 45,9% → **46,8%** *(Or **49,7%**)* | **128,7 ms** vs 125,7 ms |
| **`TIMIT Core` (Bản ngữ L1 Mỹ, 192 câu)** | 2,20 → 2,96 FA/u | 6,22 → 11,01 FA/u | 7,92 → 4,93 FA/u | **1,63 FA/u (`42,8/1k`)** *(Ctrl Rec **96,4%**!)* | **Giảm 5,5× lỗi giả bản ngữ!** | Gap FA: `6,1/1k` | **113,8 ms** |

### 3.3. Kết luận tổng thể sau toàn bộ chu trình Fix + Retrain

- **`SeqMDDv2` (`checkpoints/seq_mdd_v2_antishortcut_seed20261121.pt`)** hiện là **mô hình toàn diện nhất toàn dự án**, vượt mốc cao nhất mọi thời đại trên **3/4 tập L2 ngoài miền (`EpaDB TEST 68,29%`, `Spontaneous 63,22% / 64,38%`, `SAA-27 45,39%`)**, cải thiện mạnh nhận diện chèn âm `I` trên mọi tập, **giảm 5,5× báo sai trên người bản ngữ Mỹ (`7,92 → 1,63 FA/câu`) đã kiểm chứng không dùng shortcut micro (`96,42%` Recall trên TIMIT đối chứng)**, và chạy **nhanh bằng/hơn `R0`** (`112 ms` câu ngắn, `1,36 s` đoạn Stella dài 49s).
- **Cơ hội cải thiện tiếp (nếu cần đẩy `Held-6` vượt `67%` và kéo `TIMIT` dưới `0,8 FA/câu`):** train trên toàn bộ `18` người `DEV-18` (giữ Cal tách từ EpaDB+SAA-18), bổ sung thêm dữ liệu đọc/tự phát tiếng Anh L1 (LibriSpeech) với cơ chế anti-shortcut tương tự, và mở nhẹ LoRA tầng trên của Koel trên tập đa corpus lớn này.
