# Full Benchmark 3 Mô Hình (`Koel Base` vs `R0` vs `SM13`) Trên Toàn Bộ Dataset & Mổ Xẻ Chuyên Sâu Nguyên Nhân Mô Hình Còn Kém

- **Ngày thực hiện:** `2026-11-21`
- **Thư mục thực nghiệm:** `research/FULL_3MODEL_BENCHMARK_AND_DEEP_DIAGNOSIS_20261121_v1/`
- **Script suy luận & phân tích tái lập:** `run_full_inference.py`, `analyze_and_dissect.py`
- **Biên nhận toàn vẹn (SHA-256) & JSON đầy đủ:** `outputs/INFERENCE_RECEIPT.json`, `outputs/FULL_BENCHMARK_AND_DISSECTION.json`
- **Phạm vi dữ liệu (toàn bộ 6 tập thật / 4 corpus tiếng Anh trên đĩa):** **`4.066` câu, `113` người nói, `~249.600` slot âm vị & khe chèn**, gồm:
  1. **L2-ARCTIC `DEV-18`** (`1.632` câu, `18` người, `6` L1, `109.900` slot, `7.558` lỗi gold) — dùng dự đoán niêm phong đã xác minh (`dev18_koel_vs_sm13_20261120`); `R0` chạy **6-fold OOF tách người nói**; `SM13` có **6/18 người nằm ngay trong tập TRAIN của SM13**.
  2. **L2-ARCTIC `Held-6`** (`802` câu, `6` người `ABA, ASI, BWC, EBVS, HQTV, LXC`, `5` L1, `53.496` slot, `5.148` lỗi gold) — **chạy mới theo chỉ thị đánh giá toàn dataset** (chỉ suy luận đóng băng, **0** train lại, **0** chỉnh ngưỡng trên Held-6). **100% người nói mới với `R0`** (`r0_base.pt`), trong khi **cả 6/6 người nằm trong tập TRAIN của `SM13`**.
  3. **L2-ARCTIC `Spontaneous`** (`129` câu nói tự phát, `22` người, `18.135` slot, `2.174` lỗi gold) — chạy mới; `R0` ghép đúng checkpoint **tách người nói 100%** (fold OOF giữ ngoài tương ứng cho 17 người DEV-18, `r0_base.pt` cho 5 người Held-6); `SM13` có 12/22 người trong TRAIN.
  4. **EpaDB `TEST`** (`1.266` câu đọc ngắn, `20` người L1 Tây Ban Nha Argentina, `43.096` slot, `4.807` lỗi gold sạch sau khi đánh dấu `unknown` cho 2 ký hiệu bị trộn ở nguồn `θʰ`/`sʰ` — độ phủ `95,5%`; kèm kiểm nhạy không che `5.362` lỗi) — **100% người nói & câu đọc mới** với cả `R0` và `SM13`.
  5. **SAA-27 / Stella** (`27` đoạn đọc dài `21,8–49,0s`, `27` người, `6` L1, `12.015` slot, `1.139` lỗi gold GMU) — dùng dự đoán niêm phong (`saa27_koel_vs_sm13_20261120`) + chạy bổ sung tách `SM13 neural vs E4`. **100% người nói & bài đọc mới** với cả `R0` và `SM13`.
  6. **TIMIT `Core` (Đối chứng người bản ngữ L1 Mỹ)** (`192` câu, `24` người bản ngữ Mỹ, `6.374` âm đích + `6.566` khe = `12.940` slot, **`0` lỗi phát âm L2**) — bài kiểm tra **báo sai trên người bản ngữ (Native L1 False-Alarm Stress Test)**.
  - *Lưu ý:* `New-Method-Research/.../test_sealed.jsonl` không bị chỉnh sửa hay mở lại; 4 người nói trong đó (`MBMPS, TLV, TNI, ZHAA`) vốn đã nằm trong `DEV-18` (330/600 câu).

---

## 1. Kết Luận Nhanh (Executive Verdict)

1. **Mô hình nào nhận diện lỗi phát âm tốt nhất hiện tại?**
   - **`R0` (Frozen Base Koel + `SmallMDDHead`, ngưỡng hiệu chỉnh `Cal-F1`) thắng tuyệt đối cả `Koel Base` lẫn `SM13` trên MỌI tập L2 (5/5 tập, thắng `100%` người nói trước SM13 trên DEV-18 `18/18`, Held-6 `6/6`, Spontaneous `22/22`, EpaDB `20/20`, và `22/27` trên SAA-27)**, kể cả trên **`Held-6` nơi cả 6/6 người nói nằm trong tập TRAIN của SM13** (`66,75%` vs `40,41%` strict μF1; `PR-AUC 0,7922` vs `0,6524`), đồng thời nhanh hơn SM13 từ `1,1×` (câu rất ngắn 2s) đến `9,4×` (đoạn dài Stella).
   - **`Koel Base (Vanilla)` đứng thứ 2** (`37,46%–56,96%` strict μF1), ổn định hơn `SM13` ở mọi tập nhưng Recall thấp (`30,8%–48,3%`).
   - **`SM13 formal seed 101` đứng cuối toàn diện** (`1,50%–16,98%` ở policy gốc `fixed/0.2` với **`0,00%` Recall lớp `S` trên mọi tập**; và chỉ `15,27%–40,41%` sau khi hiệu chỉnh lại ngưỡng `Cal-F1`).

2. **Nhưng CẢ 3 MÔ HÌNH ĐỀU CHƯA ĐẠT CHUẨN SẢN PHẨM (PRODUCTION-READY)!**
   - Ngay cả mô hình tốt nhất là **`R0`** vẫn bộc lộ **4 điểm yếu nghiêm trọng**:
     1. **Trần F1 vẫn thấp khi ra ngoài L2-ARCTIC:** Strict μF1 đạt `65,1%–66,8%` trong miền L2-ARCTIC đọc, rơi xuống `61,6%` trên nói tự phát, **`58,95%` trên EpaDB TEST** (người nói + câu đọc mới), và chỉ **`43,94%` trên SAA-27** (đoạn dài). Ngay cả dùng **ngưỡng Oracle tối ưu ăn gian trên chính Eval**, `R0` cũng chỉ đạt trần `44,78%` (SAA-27) và `61,89%` (EpaDB).
     2. **Thảm họa báo sai trên người bản ngữ Mỹ (`TIMIT Core`):** `R0 (Cal-F1)` báo tới **`7,92` lỗi giả mỗi câu (`236,6` lỗi giả / 1.000 âm — gần 1/4 số âm của người bản ngữ bị tố phát âm sai!!)**; ngay cả `R0 Locked (τ=0,624)` vẫn báo **`5,13` lỗi giả / câu (`152,5 / 1.000` âm)** — gấp `2,4×–3,7×` Koel Base (`2,20 / câu`).
     3. **Hoàn toàn mù lỗi chèn âm (`I`):** `R0` không hề có gap head học được; F1 lớp `I` kẹt cứng ở **`19,15%`** (SAA-27), **`25,58%`** (EpaDB), **`41,32%`** (Spontaneous), **`45,4%–45,9%`** (L2-ARCTIC).
     4. **Phụ thuộc lối tắt (`shortcut`) vào nhãn rời rạc của Koel Base & sập hoàn toàn khi gặp 35/75 token IPA chưa ánh xạ (`unmapped`)**: trên `EpaDB TEST`, độ chính xác (Precision) của `R0` cho âm `K` và `P` tụt xuống còn **`12,4%`** (`240 FP` trên `K` và `78 FP` trên `P`) chỉ vì Koel Base phát ra `kʰ`, `pʰ` mà bảng `pilot.MAP` không ánh xạ!

---

## 2. Bảng Điểm Tổng Hợp Toàn Bộ 6 Tập Dữ Liệu

### 2.1. Bảng chính: Strict micro-F1 (S/O/I) & Threshold-Free Target PR-AUC (AP)

*Ghi chú chế độ ngưỡng:*
- **Locked (Lịch sử):** `R0` dùng `τ_t = 0,6236` (trung bình khóa FP-budget DEV-18; riêng SAA-27 dùng `0,8020` khóa SAA-18); `SM13` dùng policy gốc `fixed/0.2` (`τ_t = 0,856953`, `τ_g = 0,866081`).
- **Cal-F1 (Chuyển giao công bằng, 0 chạm Eval):** dùng đúng ngưỡng `Cal-F1` học từ DEV-18 (`R0 τ_t = 0,3407`; `SM13 τ_t = 0,3598`, `τ_g = 0,7178`) áp thẳng không chỉnh sang `Held-6`, `Spontaneous`, `EpaDB TEST`, `TIMIT Core` (và đối chiếu thêm trên `SAA-27` cạnh bản hiệu chỉnh `SAA-18`).
- **Oracle-τ (Chẩn đoán trần trên):** ngưỡng tốt nhất chọn trực tiếp trên Eval — dùng để tách *"lệch ngưỡng"* khỏi *"năng lực mô hình kém"*.
- **`SM13 Neural-Only` (Chẩn đoán mổ xẻ):** tắt hẳn nhánh E4 bị bug (`0.5*neural + 0.5*e4` → `1.0*neural`), chấm bằng Oracle-τ.

| Tập dữ liệu (Utts / Spks / Lỗi Gold) | `Koel Base` (Vanilla) | `R0` Locked | **`R0` Cal-F1** | `R0` Oracle-τ | `SM13` Locked | `SM13` Cal-F1 | `SM13` Oracle-τ | *`SM13 Neural` (No E4, Or)* | **`R0` AP** | `SM13` AP | *`SM13 Neu` AP* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **L2A `DEV-18`** (1.632 / 18 / 7.558) *(R0 OOF; SM13 6/18 train)* | 56,12% | 59,16% | **65,10%** | 65,18% | 8,53% | 31,02% | 31,30% | *50,96%* (180 sub) | **0,7544** | 0,4827 | *0,5560* |
| **L2A `Held-6`** (802 / 6 / 5.148) *(R0 unseen; SM13 **6/6 train**)* | 56,96% | 62,62% | **66,75%** | 66,80% | 16,98% | 40,41% | 40,65% | *68,50%* *(memorized!)* | **0,7922** | 0,6524 | *0,7867* |
| **L2A `Spontaneous`** (129 / 22 / 2.174) *(R0 unseen spk)* | 56,12% | 52,10% | **61,55%** | 62,62% | 3,77% | 39,19% | 40,47% | *43,60%* | **0,7894** | 0,5999 | *0,5426* |
| **EpaDB `TEST`** (1.266 / 20 / 4.807) *(Unseen R0 & SM13)* | 53,70% | 44,43% | **58,95%** | 61,89% | 1,50% | 15,27% | 15,55% | *38,20%* | **0,6773** | 0,4311 | *0,4183* |
| *EpaDB `TEST` (không che `θʰ/sʰ`, 5.362 lỗi)* | 50,19% | 42,58% | **57,61%** | 61,16% | 1,42% | 14,92% | 15,13% | *38,45%* | **0,6142** | 0,4368 | *0,4226* |
| **`SAA-27` / Stella** (27 / 27 / 1.139) *(SAA-18 Cal)* | 37,46% | 27,09% | **43,94%** | 44,78% | 7,02% | 37,27% | 38,12% | *40,00%* | **0,5102** | 0,4535 | *0,4207* |
| **`SAA-27` / Stella** *(chuyển ngưỡng DEV-18 Cal, 0 chạm SAA)* | 37,46% | 27,09% | **43,43%** | 44,78% | 7,02% | 37,88% | 38,12% | *40,00%* | **0,5102** | 0,4535 | *0,4207* |

### 2.2. Chi tiết theo từng loại lỗi `S / O / I`, Nhầm `S↔O`, Bootstrap theo người nói & Độ trễ GPU (chế độ `Cal-F1`)

| Tập | Mô hình | Strict P / R / **μF1** | Bin F1 | **S F1** (P/R) | **O F1** (P/R) | **I F1** (P/R) | Strict FP/câu | `S→O` / `O→S` | Độ trễ GPU P50 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **DEV-18** | Vanilla | 69,8 / 46,9 / **56,12%** | 58,24% | 57,1% (71/48) | 54,7% (66/47) | 45,4% (63/36) | 0,94 | 76 / 58 | ~117 |
| | **R0 Cal-F1** | 62,2 / 68,2 / **65,10%** | 68,14% | **66,1%** (62/70) | **64,8%** (62/68) | 45,4% (63/36) | 1,92 | 124 / 117 | ~118 |
| | SM13 Cal-F1 | 24,3 / 42,8 / **31,02%** | 51,03% | 41,1% (57/32) | 22,5% (13/91) | 33,0% (34/32) | 6,17 | **2.077** / 11 | — |
| **Held-6** | Vanilla | 71,9 / 47,2 / **56,96%** | 59,73% | 57,6% (71/48) | 57,3% (75/46) | 45,9% (69/34) | 1,18 | 67 / 51 | 123,6 |
| | **R0 Cal-F1** | 63,7 / 70,2 / **66,75%** | 71,00% | **66,8%** (62/73) | **70,2%** (70/71) | 45,9% (69/34) | 2,57 | 107 / 123 | 125,7 |
| | SM13 Cal-F1 | 33,4 / 51,1 / **40,41%** | 61,73% | 50,2% (73/38) | 32,3% (19/96) | 38,4% (36/41) | 6,54 | **1.386** / 3 | 172,4 (1,37×) |
| **Spontaneous** | Vanilla | 74,9 / 44,9 / **56,12%** | 59,80% | 58,5% (78/47) | 50,5% (66/41) | 41,3% (60/32) | 2,54 | 37 / 27 | 370,8 |
| | **R0 Cal-F1** | 62,5 / 60,6 / **61,55%** | 71,35% | **64,4%** (65/64) | **54,7%** (54/56) | 41,3% (60/32) | 6,13 | 89 / 121 | 373,5 |
| | SM13 Cal-F1 | 38,6 / 39,8 / **39,19%** | 58,07% | 40,5% (59/31) | 40,1% (28/72) | 18,8% (14/28) | 10,69 | **389** / 28 | 861,5 (2,31×) |
| **EpaDB TEST** | Vanilla | 74,0 / 42,1 / **53,70%** | 55,53% | 56,8% (75/46) | 37,8% (63/27) | 25,6% (85/15) | 0,56 | 25 / 44 | 110,7 |
| | **R0 Cal-F1** | 62,3 / 56,0 / **58,95%** | 63,11% | **61,4%** (63/60) | **49,3%** (56/44) | 25,6% (85/15) | 1,29 | 94 / 96 | 112,6 |
| | SM13 Cal-F1 | 12,7 / 19,2 / **15,27%** | 48,22% | 20,1% (40/13) | 11,9% (6/81) | 2,0% (4/1) | 5,01 | **1.976** / 13 | 123,3 (1,09×) |
| **SAA-27** | Vanilla | 47,8 / 30,8 / **37,46%** | 40,34% | 35,9% (45/30) | **53,8%** (54/53) | 19,2% (55/12) | 14,22 | 25 / 2 | 1.313,2 |
| | **R0 Cal-F1** | 40,9 / 47,4 / **43,94%** | 48,25% | **44,7%** (41/50) | 51,0% (41/69) | 19,2% (55/12) | 28,85 | 45 / 8 | 1.449,7 |
| | SM13 Cal-F1 | 33,5 / 42,0 / **37,27%** | 48,65% | 40,3% (40/40) | 36,0% (23/79) | 15,3% (50/9) | 35,11 | **136** / 10 | 13.666,1 (9,43×) |

**Kiểm định ghép cặp theo người nói (Bootstrap 95% CI, chế độ `Cal-F1`):**
- `R0 − SM13`: DEV-18 **`+34,08 pp [+29,69; +37,72]` (18/0/0)** · Held-6 **`+26,34 pp [+20,22; +29,81]` (6/0/0)** · Spontaneous **`+22,35 pp [+19,95; +25,16]` (22/0/0)** · EpaDB **`+43,67 pp [+41,37; +45,63]` (20/0/0)** · SAA-27 **`+6,67 pp [+4,24; +9,29]` (22/5/0)**.
- `R0 − Vanilla`: DEV-18 **`+8,98 pp [+7,30; +10,66]` (18/0/0)** · Held-6 **`+9,78 pp [+6,96; +12,58]` (6/0/0)** · Spontaneous **`+5,42 pp [+2,96; +8,16]` (19/3/0)** · EpaDB **`+5,25 pp [+2,44; +7,82]` (17/3/0)** · SAA-27 **`+6,48 pp [+3,95; +9,08]` (21/5/1)**.

### 2.3. Đối chứng Người Bản Ngữ L1 Mỹ (`TIMIT Core`: 192 câu, 24 người bản ngữ, 6.374 âm đích)

| Hệ thống | Tổng báo sai (`S / O / I`) | **Báo sai / câu** | **Báo sai đích / 1.000 âm** | % báo sai rơi vào biến thể tự nhiên của người bản ngữ (`surface39 ≠ canonical39`) | Top âm bản ngữ bị tố sai |
|---|---:|---:|---:|---:|---|
| **Koel Base (Vanilla)** | 423 (315 / 95 / 13) | **2,20** | **64,32** | 63,2% | `AA(104), AH(56), D(29), IH(28), AE(27)` |
| **R0 Locked (`τ=0,624`)** | 985 (627 / 345 / 13) | **5,13** | **152,49** | 55,9% | `AH(249), AA(100), T(92), D(74), L(62), N(57)` |
| **R0 Cal-F1 (`τ=0,341`)** | **1.521** (1.025 / 483 / 13) | **7,92** | **236,59** | 56,4% | `AH(410), AA(151), T(123), IH(103), D(92), N(74), L(71)` |
| **SM13 Locked (`fixed/0.2`)** | 12 (0 / 11 / 1) | 0,06 *(mù S)* | 1,73 | 81,8% | *(tê liệt toàn bộ lớp S)* |
| **SM13 Cal-F1** | **1.195** (189 / **999** / 7) | **6,22** | **186,38** | 45,8% | `AA(128), T(126), Z(96), D(86), R(78), DH(72)` |

---

## 3. Mổ Xẻ Chuyên Sâu: Tại Sao Cả 3 Mô Hình Vẫn Kém?

Đây không phải vấn đề "chỉnh thêm ngưỡng" — **Oracle-τ đã chứng minh trần F1 chỉ tăng thêm `+0,05` đến `+2,94 pp` so với `Cal-F1`**. Nguyên nhân nằm ở **lỗi thiết kế thuật toán, khuyết tật kiến trúc, quy trình train và giới hạn dữ liệu/nhãn**. Dưới đây là bằng chứng định lượng cho từng mô hình:

---

### 3.1. Mổ xẻ `SM13` (`LANGFENS-SM13-FUSION-V1`) — 4 Nguyên Nhân Khiến SM13 Sụp Đổ

#### (1) LỖI THUẬT TOÁN CHÍ MẠNG TRONG MODULE `E4` (`agent_a_sm13_engine.py`, dòng 236 & 281)
Trong `compute_e4_likelihoods()`:
```python
# 4 candidate substitutions (offset modulo 42)
sub_cands = [(phone_id + k) % 42 for k in range(1, 5)]
...
sub_scores = [scores_out[idx] for idx in sub_map[i]]
max_sub = np.mean(sub_scores)   # <-- lấy trung bình cộng của 4 âm kế cận theo thứ tự alphabet!
sllr = max_sub - can_score
```
- **Hậu quả toán học khủng khiếp:** Thay vì tìm âm thay thế có xác suất cao nhất (`max` trên 41 âm khác), `E4` lấy **4 âm đứng liền sau theo thứ tự chữ cái** (ví dụ `T=33` chỉ so với `TH, UH, UW, V`; `D=9` chỉ so với `DH, EH, ER, ERR`!) rồi lấy **TRUNG BÌNH CỘNG (`np.mean`)** log-likelihood!
- Khi người học thay thế thật (ví dụ phát `DH → D` hoặc `V → W`), `can_score` giảm mạnh, nhưng 4 âm alphabet ngẫu nhiên cũng sai bét nên `sllr` vẫn cực âm (`mean P_e4(S)` trên lỗi `S` thật chỉ là **`0,0307`** trên DEV-18 và **`0,0264`** trên Held-6!). Trong khi đó giả thuyết xóa (`hyp_del`) được tính thật nên `ollr` thắng tuyệt đối:
  - Trên **DEV-18 (180 câu mẫu)**: `E4` cho `P_e4(O) > P_e4(S)` trên **`99,54%` các lỗi thay âm (`S`) thật** (`mean P_e4(O) = 0,5657` vs `P_e4(S) = 0,0307`)! Trong khi chính `ScalarMixingTargetHead` (`neural`) đoán đúng `P_neural(S) > P_neural(O)` tới **`96,61%`**!
  - Khi trộn cứng `fused = 0,5 * neural + 0,5 * e4`:
    1. `P_fused(S)` bị ép xuống tối đa `~0,50` (trên toàn DEV-18-180 `max P_fused(S) = 0,6146 < 0,856953`) → **tê liệt toán học 100% lớp `S` ở policy gốc `fixed/0.2` (`0/5.823` DEV-18, `0/3.738` Held-6, `0/4.056` EpaDB, `0/811` SAA-27)**.
    2. Ở ngưỡng hạ `Cal-F1`, `E4` lật **`47,46%` lỗi `S` thật thành `O`**, tạo ra **`2.077 S→O` trên DEV-18**, **`1.386 S→O` trên Held-6**, **`1.976 S→O` trên EpaDB** (kéo Precision lớp `O` tụt còn **`6,4%–19,4%`**!).
  - **Chứng minh bằng cách tháo bỏ E4 (`SM13 Neural-Only`):** Chỉ cần bỏ phép trộn `0.5*E4`, strict μF1 nhảy vọt ngay: DEV-18 **`31,30% → 50,96%` (+19,7 pp)**, Held-6 **`40,65% → 68,50%` (+27,9 pp)**, EpaDB **`15,55% → 38,20%` (+22,7 pp)**!

#### (2) NGAY CẢ KHI ĐÃ THÁO BỎ BUG E4, `SM13 Neural-Only` VẪN OVERFIT NẶNG NGƯỜI NÓI & YẾU CROSS-DOMAIN
Phân rã `SM13 Neural-Only` theo mức độ phơi nhiễm người nói trong tập TRAIN của SM13 (12 người `L2_ARCTIC_SPLIT.yaml`):

| Nhóm người nói / Tập | Phơi nhiễm với SM13 | `SM13 Neural` AP | `SM13 Neural` Oracle μF1 | So với `R0` AP / μF1 |
|---|---|---:|---:|---:|
| **L2A `Held-6` (6 người)** | **6/6 TRONG TRAIN SM13** | 0,7867 | **68,50%** | 0,7922 / 66,75% *(R0 unseen!)* |
| **DEV-18 `TRAIN` (6 người)** | **6/6 TRONG TRAIN SM13** | 0,6701 | **66,55%** | 0,7244 / 62,64% *(R0 OOF!)* |
| **DEV-18 `DEV_SEARCH` (4 người)** | **Unseen với SM13** | **0,4032** | **42,08%** *(-24,5 pp!)* | 0,6734 / 59,59% |
| **DEV-18 `TEST_SEALED` (4 người)** | **Unseen với SM13** | **0,5305** | **43,74%** *(-22,8 pp!)* | 0,7935 / 67,61% |
| **DEV-18 `UNSEEN` (2 người)** | **Unseen với SM13** | **0,4957** | **50,91%** *(-15,6 pp!)* | 0,7200 / 64,21% |
| **L2A `Spontaneous` (22 người)** | Khác phong cách nói | **0,5426** | **43,60%** | 0,7894 / 61,55% |
| **EpaDB `TEST` (20 người)** | **Corpus + câu mới** | **0,4183** | **38,20%** | 0,6773 / 58,95% |
| **SAA-27 (27 người)** | **Corpus + bài đọc mới** | **0,4207** | **40,00%** | 0,5102 / 43,94% |

- **Lý do gốc rễ kiến trúc & train:**
  1. **Backbone sai bài toán (`facebook/wav2vec2-base-960h`):** Chỉ 95M tham số, tiền huấn luyện tự giám sát trên LibriSpeech L1 rồi fine-tune **ASR chữ cái tiếng Anh bản ngữ** (không phải nhận dạng âm vị đa ngôn ngữ như XLSR-53 315M của Koel). Khi gặp giọng L2 mới hoặc L1 Tây Ban Nha (`EpaDB`), đặc trưng tầng ẩn bị lệch miền nặng (`PR-AUC` tụt từ `0,787` xuống `0,418`).
  2. **Kiến trúc `ScalarMixingTargetHead` quá thô sơ & không có ngữ cảnh chuỗi:** Lấy căn lề bằng vòng lặp DP Python $O(T \cdot 2N)$ chậm gấp `9,4×` trên đoạn dài (`CharSiu en_w2v2_fc_10ms`), cắt cửa sổ cứng 11 frame rồi gộp tĩnh `[mean, std, last-first]` (2304D) + one-hot 39D của riêng âm đích → MLP 256-128-3. **Hoàn toàn không biết âm trước/sau là gì, không có attention, không có sequence encoder**, và train 12 epoch với trọng số mất cân bằng cực đoan `[1.0, 2.5, 4.0]` trên đúng 12 người nói (59k slot) → học thuộc giọng 12 người TRAIN và tụt `-23..25 pp` ngay trên người nói mới cùng corpus!
  3. **`ScalarMixingGapHead` gần như không hội tụ:** Trọng số 13 tầng sau 5 epoch gần như phẳng lì `[0.0749 .. 0.0790]` (gần hệt khởi tạo đều `1/13 = 0.0769`!), khiến `Gap PR-AUC` sụp xuống **`0,0216`** trên EpaDB, **`0,1193`** trên Spontaneous, **`0,1932`** trên SAA-27!

---

### 3.2. Mổ xẻ `Koel Base` (`Vanilla KoelLabs/xlsr-english-01` + Greedy DP) — 3 Nguyên Nhân Cốt Lõi

#### (1) THẤT THOÁT NGHIÊM TRỌNG DO **35 / 75 TOKEN IPA CHƯA ĐƯỢC ÁNH XẠ** TRONG `pilot.MAP`
Từ vựng CTC của Koel Base có 79 token (4 token đặc biệt + **75 ký hiệu IPA thật**), nhưng `pilot.MAP` chỉ ánh xạ **40 token** — bỏ rơi **35 token IPA quan trọng** (`ɝ`, `ɨ`, `ʔ`, `ɾ`, `l̩`, `n̩`, `m̩`, `kʰ`, `pʰ`, `sʰ`, `θʰ`, `a`, `o`, `e`, `β`, `x`, `ɣ`, `r`, `ə̃`, `ĩ`, ...):
- **Trên `Held-6`:** `470` token unmapped (`467` là âm `ɝ` — stressed ER!), gây ra `1.351` vị trí bị ép thành **`?` (ABSTAIN = không báo lỗi)**, trong đó có tới **`688` lỗi thật (`413 S + 275 O`) bị nuốt mất trắng**!
- **Trên `Spontaneous`:** `164` token unmapped (`159` là `ɝ`, `5` là `ɾ`), gây `593` vị trí `?` nuốt mất **`334` lỗi thật (`196 S + 138 O`)**!
- **Trên `EpaDB TEST`:** **`2.660 / 20.722` (`12,84%` tổng số token phát ra, ảnh hưởng `1.105 / 1.266` câu!!)** bị unmapped (`ɝ: 532`, `θʰ: 456`, `a: 414`, `sʰ: 375`, `o: 287`, `kʰ: 232`), tạo ra **`2.444` vị trí `?` nuốt mất `1.204` lỗi thật (`1.045 S + 159 O`)**!
- **Trên `TIMIT Core` (bản ngữ Mỹ):** **`915 / 6.169` (`14,83%` token, `187/192` câu!)** bị unmapped (`ɨ: 387`, `ʔ: 137`, `ɾ: 96`, `ɝ: 94`, `l̩: 54`, `ʉ: 47`) tạo `1.216` vị trí `?`. (Trớ trêu là chính lỗ hổng này vô tình "che" bớt lỗi giả cho Vanilla trên TIMIT nhưng lại **kích nổ 884 lỗi giả cho R0**!)

#### (2) BỘ NHẬN DẠNG ÂM VỊ (ASR/PER) ≠ BỘ PHÁT HIỆN LỖI PHÁT ÂM (MDD)
- Koel Base là bộ nhận dạng CTC **không điều kiện theo âm đích (unconditioned)**: lúc suy luận nó không biết người nói *đáng lẽ phải đọc âm gì*, và giải mã `greedy argmax` vứt bỏ toàn bộ thông tin xác suất/độ tự tin (`margin`, `entropy`).
- Khi lỗi phát âm ở vùng ranh giới (ví dụ nguyên âm `IH/AH/AE` hơi lệch hoặc phụ âm cuối `T/D` bật yếu), CTC greedy vẫn chọn âm chuẩn (`Recall` chỉ kẹt `30,8%–48,3%`). Ngược lại, khi người bản ngữ nói nhanh tạo biến âm tự nhiên (`AA↔AO`, `AH↔IH`, `T/D` flap), CTC trung thực ghi nhận âm bề mặt và DP lập tức tố sai (`63,2%` số lỗi giả trên TIMIT rơi đúng vào biến âm tự nhiên!).

#### (3) CĂN LỀ DP RỜI RẠC DỄ LỆCH DÂY CHUYỀN & MÙ CHÈN ÂM (`I`)
- Levenshtein DP trên một chuỗi greedy duy nhất không dùng biên thời gian âm học: một âm bị nuốt hoặc một ký hiệu unmapped đứng cạnh làm sinh ra nhiều đường căn lề đồng chi phí → ép thành `?` (bỏ sót lỗi) hoặc đổi `S↔O`, đồng thời Recall chèn âm `I` chỉ đạt `11,6%` (SAA-27) – `15,1%` (EpaDB) – `35,6%` (DEV-18).

---

### 3.3. Mổ xẻ `R0` (`Frozen Base Koel + SmallMDDHead`) — 5 Điểm Nghẽn Khiến R0 Chạm Trần & Sập Trên TIMIT / EpaDB / SAA-27

#### (1) HỌC LỐI TẮT (`SHORTCUT LEARNING`) VÀO NHÃN RỜI RẠC CỦA KOEL BASE & THẢM HỌA `align_tier_3`
Kiểm tra trực tiếp trọng số mạng (`r0_base.pt`) và hành vi phân tầng chứng minh `R0` phụ thuộc cực mạnh vào nhãn rời rạc và xác suất CTC của Koel Base:
- Trong tầng `classifier.2.weight`, khối đặc trưng vô hướng `h_scalar (64D)` có **RMS trên mỗi trọng số cao hơn `24%`** mọi khối đặc trưng âm học (`0,0264` vs `h_attn 0,0212`, `h_vit 0,0214`). Top đặc trưng có chuẩn L2 cao nhất trong `scalar_proj.0`: `#1 frame_margin_vit (0,964)`, `#2 entropy_vit_mean (0,913)`, `#3 can_lp_win_max (0,905)`, **`#4 align_tier_3 (0,869)`**, `#5 align_tier_0 (0,851)`, `#6 a_call_is_C (0,848)`.
- **Hệ quả 1 — Tàn phá người bản ngữ Mỹ trên `TIMIT Core` (`7,92` lỗi giả / câu!):**
  - Khi `Vanilla == "?"` trên TIMIT (`1.216` âm bản ngữ bị unmapped: `ɨ, ʔ, ɾ, ɝ, l̩, ʉ`), **`R0 Cal-F1` báo sai tới `884 / 1.216` (`72,7%`!!)** (và `R0 Locked` vẫn báo sai `593 / 1.216 = 48,8%`!), chiếm **`58,6%` tổng số lỗi giả** trên người bản ngữ! Lý do: trong tập train L2-ARCTIC, `Vanilla == "?"` hầu như chỉ xuất hiện ở ca lỗi khó (`51%` lỗi thật trên Held-6), nên `R0` học máy móc quy tắc *"cứ thấy `a_call_is_Q` / `align_tier_3` + `can_lp` thấp là báo lỗi!"*.
- **Hệ quả 2 — Precision `12,4%` cho `K` và `P` trên `EpaDB TEST`:**
  - Koel Base nhận diện đúng âm bật hơi chuẩn tiếng Anh `kʰ` (`232` lần) và `pʰ` (`~75` lần), nhưng vì chúng bị unmapped → rơi vào `align_tier_3` (`1.760` slot, `PR-AUC` sập xuống **`0,4300`**, kém hơn tung đồng xu!) → `R0` tố sai **`240` âm `K` đúng** (`34 TP / 240 FP`, Precision `12,4%`) và **`78` âm `P` đúng** (`11 TP / 78 FP`, Precision `12,4%`)!
- **Hệ quả 3 — Khả năng "cứu" lỗi bị Base bỏ sót (`Vanilla == C`) rất hạn chế:**
  - Trên `Held-6`: trong `1.735` lỗi thật mà Base gọi `C`, `R0 Cal-F1` chỉ cứu được `714` TP nhưng phải trả giá **`933` FP** (Precision vùng cứu chỉ `43,4%`), vẫn bỏ sót `1.021` lỗi!
  - Trên `EpaDB TEST`: trong `1.261` lỗi thật mà Base gọi `C`, `R0 Cal-F1` chỉ cứu được **`197` TP (`15,6%`)** với **`368` FP** (Precision `34,9%`)! Trong khi trên các lỗi giả của Base (`Vanilla == S, gold == C`), điểm trung bình của `R0` vẫn rất cao (`0,605` trên Held-6, `0,536` trên EpaDB).

#### (2) MÙ HOÀN TOÀN VỚI LỖI CHÈN ÂM (`I`)
- `R0` **chỉ train head `C/S/O` trên vị trí âm đích**, giữ nguyên `100%` quyết định khe chèn rời rạc của Koel Base. Trên `SAA-27`, lỗi `I` chiếm **`13,6%` tổng số lỗi** (`155/1.139`), nhưng `R0` chỉ bắt được `18/155` (`Recall 11,6%`, `F1 19,15%`). Trên `EpaDB TEST` (`292` lỗi `I`), Recall chỉ `15,1%` (`F1 25,58%`).

#### (3) TRẦN ĐẶC TRƯNG ĐÓNG BĂNG + CỬA SỔ CỤC BỘ `±80ms` + THIẾU NGỮ CẢNH CÂU & CHUẨN HÓA NGƯỜI NÓI
- `R0` đóng băng hoàn toàn `Koel Base` (chỉ trích `H18`, `H24`, `lp`), cắt cửa sổ cứng **9 frame (`±4 frame = ±80ms`)** quanh đỉnh căn lề rồi dự đoán độc lập từng slot:
  1. **Không có bộ mã hóa chuỗi mức câu (Utterance-level Sequence Encoder / Transformer / BiLSTM over slots):** Không biết tốc độ nói, không chuẩn hóa được giọng nền của từng người nói (speaker accent baseline), không mô hình hóa tương quan giữa các slot kế nhau (ví dụ một âm bị dịch ranh giới dẫn đến cặp `O+I` hoặc `S+S`).
  2. **Chỉ 2 tầng (`H18, H24`) và không có đặc trưng âm học/thanh điệu tường minh:** Các tầng trên cùng của mô hình CTC đã nén mạnh thông tin âm học mịn về hướng phân lớp chữ/âm vị chuẩn, làm mờ khác biệt giữa **giọng nặng nhưng chấp nhận được (`C`)** và **lỗi thay âm (`S`)** trên các nhóm âm chí tử lặp lại ở MỌI tập:
     - **Nguyên âm lơi/yếu (`AH`, `IH`, `AE`, `EH`, `OW`, `UW`):** `F1` chỉ `47%–65%` trên mọi tập; riêng `AE` trên EpaDB Recall tụt còn `32,5%` (`286 FN`), `UW` Recall còn **`12,0%` (`95 FN`)**!
     - **Âm xát hữu thanh/răng (`DH`, `Z`, `V`):** Nguồn FP số 1 trên L2-ARCTIC (`DH`: `485 FP` DEV-18, `177 FP` Held-6; `Z`: `345 FP` DEV-18, `136 FP` Held-6).
     - **Âm tắc đầu lưỡi (`T`, `D`):** Recall chỉ `49,5%–66,4%` do âm cuối không bật hơi, flap (`ɾ`), glottal stop (`ʔ`).

#### (4) QUY TRÌNH TRAIN `R0` QUÁ HẸP & THIẾU TÍN HIỆU GIÁM SÁT PHỤ TRỢ
Kiểm tra `run_pilot.py` và `r0_base.pt`:
1. **Dữ liệu train siêu nhỏ và đơn miền:** Chỉ train trên **`12–18` người nói L2-ARCTIC** (`32.917–54.134` slot âm đích, **cùng bộ câu đọc Arctic lặp đi lặp lại**, `0` câu nói tự phát, `0` câu bản ngữ L1 sạch làm negative anchor!). Chính việc **thiếu hoàn toàn dữ liệu bản ngữ L1 sạch (như TIMIT/LibriSpeech gắn nhãn `all-C`)** trong lúc train `SmallMDDHead` là lý do khiến `R0` không biết phân biệt biến âm tự nhiên của người bản ngữ với lỗi L2 → báo `7,92` lỗi giả/câu trên TIMIT!
2. **Mất mát 3 lớp phẳng `CE(C, S, O)` bỏ phí nhãn âm thực phát (`realized_phone`):** L2-ARCTIC và EpaDB có nhãn **âm thực tế người học đã phát ra**, nhưng `R0` vứt bỏ hoàn toàn thông tin đó, chỉ ép head học 3 lớp gộp `C/S/O` nên head không phân biệt được *thay âm hợp lệ do biến âm (allophone)* với *thay âm sai nghiêm trọng* và dễ nhầm `S↔O` (`241` ca trên DEV-18, `230` ca trên Held-6, `210` ca trên Spontaneous).
3. **Train cố định 6 epoch không điều chỉnh độ nhạy miền / không temperature scaling theo độ dài câu:** Điểm phân phối `r0_score` trên lỗi `S` thật tụt mạnh khi chuyển miền: từ `0,607` (Held-6) → `0,527` (Spontaneous) → **`0,474` (EpaDB)**, làm ngưỡng `R0 Locked (0,624)` tụt F1 thảm hại trên EpaDB (`44,43%`) và SAA-27 (`27,09%`).

#### (5) GIỚI HẠN CỐ HỮU CỦA NHÃN GROUND-TRUTH GIỮA CÁC CORPUS (ANNOTATION CEILING)
Ngay cả khi sửa sạch lỗi kỹ thuật, vẫn tồn tại **độ vênh chuẩn gán nhãn giữa các corpus** kéo giảm F1 đo được:
- **L2-ARCTIC:** Chứa các nhãn `S` cùng mã âm gốc sau chuẩn hóa (nhãn có dấu `*` chỉ lệch giọng nhẹ — `12/41` ca `S` trong gói audit 96 ca DEV-18) và tỷ lệ chênh lệch mạnh giữa người chấm (`AA↔AO`, `AH↔IH`).
- **SAA-27 / GMU:** Phiên âm IPA hẹp theo phong cách ngữ âm học cho đoạn Stella dài 222 âm (chứa nhiều biến thể nối âm tự nhiên), khiến cả 3 hệ đều chạm trần `PR-AUC ≤ 0,51` và `Oracle μF1 ≤ 44,8%`.
- **EpaDB:** Gán nhãn hẹp giọng Argentina nhưng bản nhập nguồn `KoelLabs/EpaDB` gộp chồng ký hiệu `θʰ` (vừa là `tʰ` bật hơi đúng vừa là `θ`) và `sʰ` (vừa là `ʃ` vừa là hậu tố của `ɔɪ`).

---

## 4. Bảng Tổng Hợp Nguyên Nhân Gốc Rễ & Lộ Trình Khắc Phục Ưu Tiên (Theo Tác Động / Chi Phí)

| Ưu tiên | Nguyên nhân gốc rễ đã chứng minh | Mô hình bị ảnh hưởng | Giải pháp kỹ thuật cụ thể (có thể kiểm chứng ngay) |
|---|---|---|---|
| **P0 (Sửa lỗi ngay — 0 cần dữ liệu mới)** | **35/75 token IPA của Koel Base bị unmapped** (`ɝ, ɨ, ʔ, ɾ, l̩, n̩, kʰ, pʰ, ...`) → mất `1.204` lỗi EpaDB, `688` lỗi Held-6, gây **`884` lỗi giả TIMIT** và **Precision `12,4%` trên `K/P` EpaDB** | `Koel Base` & `R0` | Chuẩn hóa đầy đủ bảng ánh xạ allophone/IPA cho toàn bộ 75 token Koel (`ɝ→ER`, `kʰ→K`, `pʰ→P`, `l̩→AH L/L`, `n̩→AH N/N`, `ɾ→flap(T/D)`, `ɨ→IH/AH`, `ʔ→allophone(T)`...) và loại bỏ shortcut `align_tier_3` trong `R0`. |
| **P0 (Nếu giữ nhánh SM13 làm baseline)** | **Bug `E4` (`(phone_id + 1..4) % 42` + `np.mean`)** giết chết 100% lớp `S` ở policy gốc & gây `2.077 S→O` | `SM13` | Sửa `compute_e4_likelihoods` dùng `max` (hoặc `logsumexp`) trên toàn bộ 41 âm cạnh tranh (hoặc top-K theo posterior CTC khung tại vị trí `i`), không dùng 4 âm kế cận alphabet; kiểm chứng lại so với `SM13 Neural-Only`. |
| **P1 (Cải thiện kiến trúc & huấn luyện `R0` → `R1-v2`)** | **Thiếu negative bản ngữ L1 trong train** → báo `7,92` lỗi giả/câu trên TIMIT; **chỉ train trên 12–18 người L2-ARCTIC** → sập trên EpaDB/SAA | `R0` | 1. Bổ sung **L1 Native Anchor (TIMIT TRAIN / LibriSpeech gắn nhãn `C` theo quy ước allophone)** vào tập train MDD Head để ép tỷ lệ báo sai bản ngữ `< 15–20 FA/1k`.<br>2. **Huấn luyện đa corpus (L2-ARCTIC + EpaDB TRAIN + L1 Native)** tách người nói chặt để tránh overfit 1 corpus.<br>3. **Bỏ hoặc dropout mạnh các feature shortcut rời rạc (`a_call_is_*`, `align_tier_*`)** để ép head học từ đặc trưng âm học liên tục & CTC log-prob! |
| **P1 (Lấp lỗ hổng kiến trúc)** | **Mù lỗi chèn âm (`I`)** (`I` F1 chỉ `19–45%`); **mô hình hóa từng slot độc lập** (không có ngữ cảnh mức câu & chuẩn hóa người nói) | `R0` | 1. Thay MLP độc lập từng slot bằng **Sequence MDD Encoder (BiGRU / 2-layer Transformer trên chuỗi slot $0..L-1$ + các khe $0..L$)** dùng **đa tầng trung gian (weighted sum layers 12–24 của XLSR, không chỉ H18/H24)** + chuẩn hóa Z-score theo câu (utterance-level CMVN).<br>2. **Huấn luyện đồng thời Target Head (`C/S/O`) + Gap Head (`N/I`) + Auxiliary Realized-Phone Head (39 lớp)** (multi-task) trên cùng một lượt forward Koel! |
| **P2 (Dữ liệu & đánh giá chuẩn hóa)** | **Độ vênh chuẩn gán nhãn giữa corpus & thiếu tập xác nhận câu ngắn sạch** | Cả 3 | Định nghĩa rõ **chính sách chấp nhận allophone (acceptability policy)** thống nhất giữa L1/L2 (tách *lỗi sai âm vị ảnh hưởng hiểu* khỏi *biến âm bản ngữ/giọng nhẹ*), hoàn tất kiểm chứng nghe mù và chấm trên tập câu ngắn mới có cả người nói L2 lẫn đối chứng L1 bản ngữ. |
