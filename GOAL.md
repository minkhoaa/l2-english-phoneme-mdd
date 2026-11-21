# MASTER R&D GOAL & EXPERIMENTAL EXECUTION PLAN — Pronunciation Mispronunciation Detection & Diagnosis (MDD)

**Cập nhật:** `2026-11-21` (sau commit `cc3572b` và campaign forensic)

**Trạng thái:** `SEQ_MDD_V3_ROADMAP_READY — PLAN_ONLY`. `EXP-1` → `EXP-6` chưa chạy; các runner/CLI V3 bên dưới là đặc tả triển khai, không phải lệnh đã kiểm thử. Các mốc định lượng là mục tiêu đề xuất, không phải kết quả.

**Đối chứng phiên:** [Bàn giao và lệnh tra cứu](research/CHECKPOINT.md#2026-11-21--bàn-giao-và-đối-chứng-phiên-session_ref_20261121_full_bench_seqmddv2_autopsy_goal), Pi `01a1fa39-20f8-770b-8701-1e62fc1e9335`. Các tập đã chấm là dữ liệu chẩn đoán, không phải tập xác nhận mới; full Spontaneous có trùng người nói Fit/Cal, cần báo riêng lát Held-6. Không dùng oracle Eval để triển khai.

**Mục đích:** Bàn giao ngữ cảnh, kế hoạch, cách chạy, đầu ra và cổng quyết định cho từng thí nghiệm. Mục tiêu dài hạn **220.000 bản thu / 2.200 người nói** được nối ở `EXP-6`; xem kế hoạch dataset gốc đầy đủ với `git show cc3572b:GOAL.md`.

---

## 1. Ngữ Cảnh Hiện Tại & Mốc Chuẩn Đã Xác Minh (Baseline State of Record)

### 1.1. Bảng điểm trên 6 cohort (`4.048` lượt câu cộng theo từng bảng)

Không cộng số người nói qua các cohort để suy ra số người độc lập. Đối chiếu số đếm, policy và kết quả theo từng cohort tại:
- `research/FULL_3MODEL_BENCHMARK_AND_DEEP_DIAGNOSIS_20261121_v1/outputs/FULL_BENCHMARK_AND_DISSECTION.json`
- `research/MDD_FIXES_AND_V2_20261121_v1/outputs_p0/P0_BEFORE_AFTER_RESULTS.json`
- `research/MDD_FIXES_AND_V2_20261121_v1/outputs_seq_mdd_v2/SEQ_MDD_V2_ANTISHORTCUT_RESULTS.json`

| Tập Eval (100% Unseen trừ ghi chú) | `Koel Base` (Old → P0) | `SM13` (Old → P0 Cal/Or) | `R0` (Old → P0) | **`SeqMDDv2` Anti-Shortcut (Lead hiện tại)** | Ghi chú chính |
|---|---:|---:|---:|---:|---|
| **L2A `DEV-18`** (1.632 câu, 18 spk, 7.558 lỗi) | 56,12% | 31,02% *(6/18 train)* | 65,10% *(6-fold OOF)* | *(15 spk Fit + 3 spk Cal)* | Mốc phát triển cũ |
| **L2A `Held-6`** (802 câu, 6 spk, 5.148 lỗi) | 56,96% → 56,93% | 40,41% → 49,81 / 63,69% *(6/6 train)* | **66,75%** → 65,76% | **65,20%** / Or 65,85% *(I: **46,8%**, tAP 0,777)* | `SeqMDDv2` +101 TP nhưng +415 `C` FP (dồn `EBVS/ABA`) |
| **L2A `Spontaneous`** (129 câu tự phát, 22 spk) | 56,12% → 56,70% | 39,19% → 49,89 / 53,54% | 61,55% → 61,00% | **63,22%** *(Held-6 unseen: **64,38%**)* / Or 63,63% | Cao nhất mọi thời đại (`+1,67 pp` vs R0) |
| **EpaDB `TEST`** (1.266 câu, 20 spk Spanish L1) | 53,70% → 64,81% | 15,27% → 35,44 / 38,80% | 58,95% → 60,75% | **68,29%** / Or 68,62% *(I: **43,4%**, tAP 0,806)* | Cao nhất (`+9,34 pp`, thắng `20/20` spk vs R0) |
| **SAA-27 / Stella** (27 spk, 6 L1, dài 22–49s) | 37,46% → 36,37% | 37,27% → 42,15 / 44,10% | 43,94% *(SAA-Cal)* → 42,71% | **45,39%** *(0 chạm SAA!)* / Or 45,91% *(I: **23,4%**)* | Cao nhất; P50 `1,36 s` (nhanh hơn `10×` SM13) |
| **TIMIT `Core`** (192 câu bản ngữ Mỹ, đối chứng) | 2,20 → 2,96 FA/u | 6,22 → 11,01 FA/u | 7,92 → 4,93 FA/u | **1,63 FA/câu**; **42,83 FA/1k âm đích** | FA/câu gồm đích và khe; đối chứng nhiễu prompt: recall **96,42%**, AP `0,983` |

### 1.2. Các Checkpoint Đang Giữ Nguyên (TUYỆT ĐỐI KHÔNG GHI ĐÈ / XÓA)

| Tên mốc | Đường dẫn | Vai trò |
|---|---|---|
| **Lead hiện tại (`SeqMDDv2` Anti-Shortcut)** | `research/MDD_FIXES_AND_V2_20261121_v1/checkpoints/seq_mdd_v2_antishortcut_seed20261121.pt` (SHA `a9bb69ed5cdd49d6…`, `τ_t=0,29, τ_g=0,825`) | Ứng viên dẫn đầu để so sánh với `SeqMDDv3` |
| **R0 Full-DEV-18 (Mốc cũ)** | `research/KOEL_R0_FULL_CONFIG_BENCHMARK_20261120_v1/runs/full_config_20261120/checkpoints/saa27/r0_base.pt` | Mốc so sánh 1-luồng cũ |
| **R0 6-Fold OOF** | `research/KOEL_TOP_LAYER_ABLATION_20261120_v1/runs/top_layer_20261120/checkpoints/R0/seed_20261119_fold_{0..5}.pt` | Mốc OOF DEV-18 |
| **SM13 Formal Seed 101** | `research/KOEL_SM13_BENCHMARK_20261120_v1/seed101/` (và bản vá E4 `research/MDD_FIXES_AND_V2_20261121_v1/sm13_e4_fixed.py`) | Baseline lịch sử Wav2Vec2+CharSiu |
| **Koel Base Backbone** | `KoelLabs/xlsr-english-01` (SHA `b657c0cf…`) + bản vá 75/75 IPA `koel_ipa_map_v2.py` | Backbone đóng băng & mốc Vanilla |

### 1.3. Phần cứng, Môi trường & Luật Bảo Vệ Dữ Liệu

- **Phần cứng:** 1× `NVIDIA GeForce GTX 1650 (4 GiB VRAM)`, `31 GiB RAM` (`~25 GiB` trống), `/home` còn `~103 GiB` trống.
- **Ràng buộc VRAM:** Chỉ nạp **tối đa 1 backbone lớn (`Koel` hoặc `SM13`) lên GPU tại một thời điểm** (`peak reserved <= 3.500 MiB`). Giải phóng sạch (`del model; gc.collect(); torch.cuda.empty_cache()`) trước khi chuyển pha.
- **Biến môi trường chuẩn cho mọi lệnh chạy:**
  ```bash
  export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  ```
- **Luật chống rò rỉ tuyệt đối:**
  1. `L2-ARCTIC Held-6` (`ABA, ASI, BWC, EBVS, HQTV, LXC`), `EpaDB TEST` (20 spk), `SAA-27` (27 spk), `TIMIT Core` (24 spk) **KHÔNG BAO GIỜ** được xuất hiện trong `FIT` hoặc `CALIBRATE`.
  2. `../New-Method-Research/data/derived/l2arctic/manifests/test_sealed.jsonl` **giữ nguyên đóng, không chạm**.
  3. Mọi mô hình có neo bản ngữ L1 **BẮT BUỘC** vượt qua bài test **TIMIT Corruption Anti-Shortcut Control** (`50` câu TIMIT Core đánh tráo 15% âm vị ngay trên cùng audio/mic TIMIT; yêu cầu `Corrupt Recall >= 90%`, `PR-AUC >= 0,95`).

---

## 2. Bản Đồ Kiến Thức Lịch Sử: Những Gì Đã Thất Bại / Bị Bác Bỏ (CẤM LẶP LẠI!)

Mọi phiên sau **PHẢI đọc bảng này trước khi đề xuất ý tưởng mới** để không đốt thời gian vào các ngõ cụt đã có bằng chứng bác bỏ:

| Hướng đã thử / kiểm định | Kết quả bằng chứng | Kết luận (KHÔNG lặp lại) |
|---|---|---|
| **Fine-tune CTC thuần / L1-Anchor / KD-LoRA đơn corpus** | L1-Anchor sập TIMIT `18,34%`; KD-LoRA tăng FP `Held-6` (`831→910`), tệ `13/20` spk EpaDB, thua `R0` `-7 TP` SAA-27. | Không fine-tune CTC thuần trên tập L2 nhỏ. |
| **Dual-Stream (`C`), 59 cấu hình Cascade (Logistic/MLP), Dual-Fusion, KD Distillation** | Mọi cascade thua `R0` (`-34..-55 TP` DEV-18, `-3..-18 TP` SAA-27), `2×` độ trễ; 2 luồng chung backbone mắc lỗi tương quan. | Loại bỏ vĩnh viễn Dual-Stream / Cascade 2 lượt Koel. |
| **Lọc lại Candidate Pool (`V2-05a/07b/09c`, `Omission-Verifier`, `D/T` waveform check)** | Bỏ sót `2.942` lỗi ngoài pool; `D/T` waveform chỉ `AUC 0,47–0,59`; `05a` thua `R0` `-30 TP` SAA-27. | Không lọc trên pool greedy; luôn chấm mọi slot đích & khe. |
| **Dữ liệu tổng hợp ghép âm (`pilot20k` `S0/S1/Mix`, `l2_guided_18k`)** | `S0` sập `27,16%` F1 trên Real (`-2.475 TP`); `Mix 75/25` giảm `-88 TP` vì học shortcut vết ghép. | Không dùng audio cắt ghép tổng hợp thay dữ liệu thật. |
| **Oversampling ca khó `4×` (`R1`) & tách `SmallSOTypeHead` trên feature cũ** | `R1` sập **`-745 TP`** (phóng đại nhãn mập mờ/gắn sao); `SmallSOTypeHead` cho `-0,18 pp`. | Không oversampling ca khó trên không gian đặc trưng cũ. |
| **Neo bản ngữ `all-C` thuần (`SeqMDDv2 Lượt 1`)** | `0,10 FA/câu` TIMIT nhưng **bị bắt sống shortcut micro** (`2,91%` Recall trên âm đánh tráo trong cùng audio TIMIT!). | Mọi dữ liệu L1 đưa vào train **BẮT BUỘC** tiêm lỗi anti-shortcut trong câu. |
| **Thêm tầng thấp `L6, L9` hoặc đặc trưng sóng thô `RMS/ZCR/FFT` (`Probe 2 & 6`)** | `L6 < L9 < L18 < SeqMDDv2` trên 6/6 âm khó (`DH, Z, AH, IH, T, D`); `O vs S` RMS `AUC = 0,5286` (nhiễu thuần!). | **Loại bỏ** `L6–L9` và đặc trưng sóng thủ công `RMS/ZCR/FFT`. |
| **Cửa sổ trượt BiGRU trên `SAA-27` (`Probe 4`)** | `L=40` đạt `45,23%` (`AP 0,4756`) ≈ cả bài `45,39%` (`AP 0,4729`); đoạn cuối bài có `AP 0,510` cao nhất. | **Loại bỏ** giả thuyết trôi độ dài BiGRU trên SAA-27. |
| **Thay backbone bằng `HuPER` hoặc giữ bug `SM13 E4`** | `HuPER` thua Koel (`28/96` vs `43/96` `S`, `6×` FA trên `C`); `SM13 E4` cũ chết `S` (`0%`) và `Wav2Vec2-Base-960h` overfit 12 người (`-24,5 pp`). | Giữ `KoelLabs/xlsr-english-01` làm backbone chính. |

### 5 tín hiệu chẩn đoán và giả thuyết cần kiểm chứng

Đọc các cơ chế dưới đây như giả thuyết để thiết kế ablation, không phải bằng chứng nhân quả. Xem [số đo và giới hạn](research/EXPERIMENTS.md): ví dụ `span_frames_mean ≈ 1` ở nhóm S/O không chứng minh `h_vit == h_peak` ở 100% slot; `68,9%` chỉ là tỷ lệ trên **cảnh báo âm đích** TIMIT trùng vị trí canonical–surface khác nhau, không xác nhận nguyên nhân `AA↔AO`.

1. **Bug "1-Frame CTC Spike (`span_frames_mean = 1,00` = 20ms)" (`Probe 2 & 7`):** CTC nhọn làm `vs == ve` cho ~100% âm vị → **`h_vit ≡ h_peak`** (mù hoàn toàn thân âm vị `60–200ms`!), `vit_dur` vô dụng (`AUC 0,4994`), và khe chèn `[prev+1..next-1]` kéo dài `12–26` frame nuốt trọn thân 2 âm láng giềng (`39/40` FA khe nội câu TIMIT).
2. **Lệch chuẩn gán nhãn đa corpus (`Probe 3 & 5`):** `1.639` câu `EpaDB TRAIN` (gán IPA hẹp cực gắt) lây sang `EBVS` (`+190 FP` trên Held-6) và `ABA` (`+107 FP`), cộng ngưỡng `τ_t = 0,29` hơi thấp cho L2-ARCTIC và `ECE = 0,080` trên `SAA-27`.
3. **Neo L1 `TIMIT TRAIN` dùng nhãn bề mặt thay vì `CMUdict Canonical` (`Probe 1`):** `68,9%` FA trên `TIMIT Core` là biến âm nối từ bản ngữ chuẩn; riêng âm **`AA` chiếm `68/273` (`24,9%`)** (hòa âm `AA↔AO`).
4. **Nhầm `S↔O` làm rơi `-4,7` đến `-9,3 pp` (`Binary F1 → Strict F1`) & bỏ phí `realized_head` (`Probe 2 & 7`):** `realized_head` đạt `95,5%` đúng chuẩn trên `C` và `68,1%` `<DEL>` trên `O`, nhưng chưa dùng lúc suy luận; thiếu tín hiệu chuỗi `CTC Deletion LLR (OLLR)` trên nhóm `R/L/D/N/T`.
5. **Trần biểu diễn đóng băng trên nguyên âm khó (`Probe 6`):** `IH` (`AP 0,534`) và `AH` (`AP 0,649`) vẫn thấp dù tăng độ rộng head, cần kiểm chứng mở nhẹ `LoRA L16–L23` khi đã có `400+` người nói đa corpus.

---

## 3. Kế Hoạch Thực Nghiệm Chi Tiết (`EXP-1` → `EXP-6`)

> **Quy tắc chạy tuần tự (Ablation Discipline):** Mỗi thí nghiệm `EXP-1` đến `EXP-5` phải **thay đổi có kiểm soát từng nhóm nguyên nhân**, so sánh ghép cặp trên **cùng 5 tập Eval Unseen + bài test TIMIT Corruption Control** so với mốc **`SeqMDDv2` (`cc3572b`)**, và chỉ gộp vào nhánh chính (`SeqMDDv3`) những thành phần **thực sự vượt tiêu chí Đạt (GO)**.

---

### `EXP-1`: `SeqMDDv3-Window` — Vá Bug "1-Frame CTC Spike" Bằng Cửa Sổ Âm Học `180ms`

- **Mục tiêu & Giả thuyết:** Chứng minh rằng việc khôi phục thân âm vị `180ms` (bị mất do `span=1,00` frame trong `SeqMDDv2`) giúp lấy lại và vượt `R0` trên `L2A Held-6` (`65,20% → >= 67,0%`) đồng thời cải thiện đồng loạt `Spontaneous` và `EpaDB TEST`.
- **Thiết kế kỹ thuật:**
  - Thư mục làm việc mới: `research/SEQ_MDD_V3_CAMPAIGN_20261121_v1/`
  - Thay cặp trùng lặp `(h_vit, h_peak)` (6 tầng × 1 frame) bằng **cửa sổ 5 điểm cố định quanh đỉnh CTC `vp + [-4, -2, 0, +2, +4]` (`180ms`)** trên **4 tầng XLSR hiệu quả nhất `(L12, L15, L18, L24)`** (`4 × 5 × 1024 × 2B = 40 KB/slot`, vừa gọn trong 25 GiB RAM!).
  - Dùng **4-head Cross-Attention** (query = `phone_emb + prev_emb + next_emb`, key/value = 5 điểm XLSR `h` + 9-frame `lp_win` + relative pos emb) trước khi đưa vào `2-layer BiGRU`.
  - Thay đặc trưng chết `vit_dur_log` (`AUC 0,4994`) bằng **khoảng cách liên đỉnh thực tế** `log1p(next_vp - prev_vp)`, tỷ lệ tốc độ nói cục bộ `(next_vp - vp) / frames_per_phone`, và **trung bình `h` trên vùng trung điểm Voronoi `[mid(vp_{i-1}, vp_i) .. mid(vp_i, vp_{i+1})]`** (thực sự phủ toàn bộ thân âm vị!).
  - Giữ nguyên **y hệt** dữ liệu Fit/Cal (`4.011 / 635` câu), seed (`20261121`), optimizer và loss của `SeqMDDv2` để cô lập 100% tác động của bản vá 1-frame spike!
- **Hướng dẫn chạy:**
  ```bash
  export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  # 1. Kiểm thử đơn vị shape + gradient + zero-NaN (CPU)
  CUDA_VISIBLE_DEVICES='' .venv/bin/pytest -q tests/test_seq_mdd_v3.py
  # 2. Trích đặc trưng 1 lần & train EXP-1 (GPU GTX 1650, ~12–15 phút, VRAM < 2.8 GiB)
  .venv/bin/python -u research/SEQ_MDD_V3_CAMPAIGN_20261121_v1/run_exp1_window.py \
    --device cuda --seed 20261121 --epochs 14 --batch-size 16 \
    --out-dir research/SEQ_MDD_V3_CAMPAIGN_20261121_v1/runs/exp1_window
  ```
- **Yêu cầu đầu ra bắt buộc (`runs/exp1_window/`):**
  1. `checkpoint_best.pt` + `CHECKPOINT_CARD.json` (SHA-256, `best_epoch`, `locked_tau_t`, `locked_tau_g`, `learned_layer_weights`).
  2. `RESULTS.json`: đầy đủ Strict μF1, Binary F1, S/O/I P/R/F1, `S→O`/`O→S`, Target `PR-AUC`, Gap `PR-AUC`, `ECE`, GPU P50/P95 ms trên cả **5 tập Eval** (`held6`, `spontaneous`, `epadb_test`, `saa27`, `timit_core`) + **Paired Speaker Bootstrap 95% CI vs `SeqMDDv2` & `R0`** + **TIMIT Corruption Anti-Shortcut Control (50 utts)**.
  3. `eval_predictions/*.parquet` + `PREDICTIONS_SEAL.json` (SHA-256).
  4. `REPORT_VI.md` (ghi rõ đối chiếu trên 6 âm `AH, D, R, IH, T, OW` của `Held-6`).
- **Tiêu chí Đạt (GO):**
  - `Held-6` Strict μF1 tăng **`>= +1,5 pp`** so với `SeqMDDv2` (`>= 66,7%`, san bằng/vượt `R0 66,75%`), đồng thời `EpaDB TEST` không giảm `> 0,5 pp` (`>= 67,8%`), `TIMIT Core` `<= 1,70 FA/câu`, và `TIMIT Corruption Control Recall >= 94%`.

---

### `EXP-2`: `SeqMDDv3-L1Canon` — Neo L1 Anti-Shortcut Theo Đúng Chuẩn `CMUdict Canonical`

- **Mục tiêu & Giả thuyết:** `Probe 1` chứng minh `68,9%` lỗi giả bản ngữ trên `TIMIT Core` (`AA = 68/273 = 24,9%`) sinh ra vì `TIMIT TRAIN` dùng nhãn bề mặt (`realized_ipa_tokens`) thay vì `CMUdict canonical`. Bổ sung dữ liệu bản ngữ L1 có **phiên âm `CMUdict canonical` thật + tiêm lỗi anti-shortcut trong câu** sẽ kéo lỗi giả `TIMIT Core` từ `1,63 FA/câu (42,8/1k)` xuống **`< 0,60 FA/câu (< 18/1k)`** mà không làm giảm Recall L2.
- **Thiết kế kỹ thuật:**
  - **Nguồn L1 Canonical sẵn có / tự động chuẩn hóa (0 trùng `TIMIT Core`):**
    1. **`MOCHA-TIMIT`** (`920` câu, `2` người bản ngữ đã có sẵn trên đĩa trong `normalized_recognizer_manifest.jsonl` với `canonical_ipa_tokens` đầy đủ!).
    2. **`LibriSpeech train-clean-100`** (tải 1 phần hoặc dùng `download_librispeech.sh` — chọn `1.500` câu phủ `200+` người bản ngữ Mỹ khác nhau, chạy `CMUdict` G2P từ file `.trans.txt` gốc → `canonical_phones` chuẩn từ điển 100%!). *(Nếu mạng/đĩa giới hạn, có thể dùng phương án dự phòng: giải mã `.WRD` của `TIMIT TRAIN` qua `CMUdict` để chuyển chính 1.000 câu `TIMIT TRAIN` từ surface sang `CMUdict canonical`!)*
  - **Quy tắc Anti-Shortcut BẮT BUỘC:** Tiêm `10% S` + `3% O` + `3% I` vào `canonical_phones` ngay bên trong từng câu L1 thật (khác hoàn toàn `all-C` thuần). **Không** thêm các câu mà G2P OOV/không có trong CMUdict.
  - Thêm kiểm định phân tích phụ: đo riêng `AA` FA trên `TIMIT Core` (kỳ vọng giảm từ `68` xuống `< 15`).
- **Hướng dẫn chạy:**
  ```bash
  # 1. Chuẩn bị manifest L1 CMUdict-Canonical (MOCHA-TIMIT + LibriSpeech hoặc TIMIT-WRD CMUdict)
  .venv/bin/python -u research/SEQ_MDD_V3_CAMPAIGN_20261121_v1/prepare_l1_canonical_anchor.py \
    --out-manifest research/SEQ_MDD_V3_CAMPAIGN_20261121_v1/manifests/l1_canonical_anchor.jsonl
  # 2. Train & Eval EXP-2 kế thừa kiến trúc đã đạt ở EXP-1 (~15 phút GPU)
  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -u research/SEQ_MDD_V3_CAMPAIGN_20261121_v1/run_exp2_l1_canon.py \
    --device cuda --seed 20261121 --epochs 14 --batch-size 16 \
    --l1-manifest research/SEQ_MDD_V3_CAMPAIGN_20261121_v1/manifests/l1_canonical_anchor.jsonl \
    --out-dir research/SEQ_MDD_V3_CAMPAIGN_20261121_v1/runs/exp2_l1_canon
  ```
- **Yêu cầu đầu ra bắt buộc (`runs/exp2_l1_canon/`):**
  1. `l1_canonical_anchor_audit.json` (số câu, số người nói, tỷ lệ OOV=0, xác nhận 0 trùng `TIMIT Core` speaker/WAV SHA-256).
  2. `checkpoint_best.pt`, `CHECKPOINT_CARD.json`, `RESULTS.json` (đủ 5 tập Eval + Bootstrap vs `EXP-1` + phân rã FA theo âm trên `TIMIT Core` đặc biệt là `AA` + **TIMIT Corruption Control**).
  3. `REPORT_VI.md`.
- **Tiêu chí Đạt (GO):**
  - `TIMIT Core` giảm xuống **`<= 0,65 FA/câu (<= 18/1k)`** (giảm `> 2,5×` so với `1,63 FA/câu`), riêng `AA` FA giảm `>= 60%`.
  - **TIMIT Corruption Anti-Shortcut Control:** `Corrupt Recall >= 93%` và `PR-AUC >= 0,96` (chứng minh không dính shortcut!).
  - Điểm L2 trên `Held-6`, `Spontaneous`, `EpaDB TEST`, `SAA-27` giữ nguyên hoặc tăng (không tụt quá `-0,4 pp` ở bất kỳ tập nào).

---

### `EXP-3`: `SeqMDDv3-GapSO` — Cô Lập Lõi Khe Chèn (`I`) + Focal Loss + Koel-CTC `OLLR` + Hợp Nhất `Realized-Phone Head`

- **Mục tiêu & Giả thuyết:** Giải quyết đồng thời 2 lỗ hổng lớn từ `Probe 2 & 7`:
  1. **Khe chèn (`I`):** Khe `[prev_spike+1..next_spike-1]` hiện nuốt thân 2 âm láng giềng (`12–26` frame) → chỉ lấy **Top-2 frame có `log_prob` non-blank (loại bỏ blank + allophone 2 âm kề) cao nhất** trong khe + đổi sang **Focal Loss (`γ=2.0, α=0.75`)** thay vì `pos_weight=8.0` CE → đẩy `I F1` tăng `+5..10 pp` và cắt `> 50%` lỗi giả khe trên TIMIT (`40 → < 15`).
  2. **Nhầm `S↔O` (`-4,7` đến `-9,3 pp` so với `Binary F1`):** Bổ sung đặc trưng chuỗi **Exact Koel-CTC Deletion LLR (`OLLR = log P_ctc(can \ {i}) - log P_ctc(can)`)** (chạy C++ kernel `ctc_batch.so` có sẵn của dự án trên `lp` 79 lớp của Koel — chỉ tốn `< 5 ms/câu`!) + kiểm thử **kết hợp xác suất `realized_head`** (`P_realized(<DEL>)` và `1 - P_realized(can)`) với `target_head` khi phân định `S` vs `O` (chọn trọng số `α_real ∈ [0, 0.5]` duy nhất trên Calibrate!).
- **Hướng dẫn chạy:**
  ```bash
  export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  # 1. Test C++ Koel-CTC OLLR kernel parity & latency (< 5ms/utt)
  CUDA_VISIBLE_DEVICES='' .venv/bin/pytest -q tests/test_koel_ollr_and_gapcore.py
  # 2. Train & Eval EXP-3 trên nền thắng của EXP-1/EXP-2 (~15 phút GPU)
  .venv/bin/python -u research/SEQ_MDD_V3_CAMPAIGN_20261121_v1/run_exp3_gap_so.py \
    --device cuda --seed 20261121 --epochs 14 --batch-size 16 \
    --out-dir research/SEQ_MDD_V3_CAMPAIGN_20261121_v1/runs/exp3_gap_so
  ```
- **Yêu cầu đầu ra bắt buộc (`runs/exp3_gap_so/`):**
  1. `checkpoint_best.pt`, `CHECKPOINT_CARD.json`, `RESULTS.json`, `REPORT_VI.md`.
  2. Bảng tách riêng tác động của từng thành phần: `(a) Top-2 Gap-Core + Focal` đối chiếu `Gap PR-AUC` & `I F1` & `TIMIT gap FA`; `(b) OLLR + Realized-head fusion` đối chiếu số ca `S→O`, `O→S` và độ chênh `Binary F1 − Strict F1` trên cả 4 tập L2.
  3. Đo độ trễ GPU P50/P95 xác nhận `OLLR` không làm vượt ngân sách độ trễ (`<= 145 ms` câu ngắn, `<= 1.600 ms` Stella).
- **Tiêu chí Đạt (GO):**
  - `I F1` tăng trung bình **`>= +4 pp`** trên 4 tập L2 (`Held-6 >= 51%`, `EpaDB >= 47%`) và `TIMIT gap FA <= 15` (từ `40`).
  - Tổng số ca nhầm `S↔O` giảm **`>= 15%`** trên `Held-6` & `Spontaneous`, nâng Strict μF1 tổng thể thêm **`>= +0,8 pp`**.

---

### `EXP-4`: `SeqMDDv3-CalFull` — Cân Bằng Trọng Số Đa Corpus, Hiệu Chỉnh Nhiệt Độ & Xác Minh 3 Seed Đủ 18 Người `DEV-18`

- **Mục tiêu & Giả thuyết:** `Probe 3 & 5` chỉ ra 2 vấn đề vận hành: (a) `EpaDB TRAIN` (`1.639` câu gắt) áp đảo `DEV-18` (`1.372` câu, thiếu 3 người) gây `+190 FP` trên `EBVS`; (b) ngưỡng đơn `τ_t=0,29` gây `ECE = 0,080` trên `SAA-27`. Thí nghiệm này:
  1. **Cân bằng trọng số lấy mẫu/loss giữa 3 nguồn** (`L2-ARCTIC 45% : EpaDB 30% : L1-Canonical Anti-Shortcut 25%`) + kiểm thử **Domain-Conditioned Temperature Scaling** (hiệu chỉnh nhiệt độ `T` trên Calibrate để giảm `ECE < 0,03`).
  2. Chạy **2 biến thể minh bạch:**
     - `EXP-4a (DEV-18 6-Fold OOF)`: Đánh giá sạch toàn bộ `18` người `DEV-18` (để so trực tiếp với mốc `R0 OOF 65,10%`).
     - `EXP-4b (Final Multi-Seed — 3 seed ghép cặp 20261121, 20261122, 20261123)`: Train trên đủ `18/18` người `DEV-18` + `26` người `EpaDB TRAIN` + `L1 Canonical Fit`, giữ Calibrate hoàn toàn tách người nói (`4` người `EpaDB Cal` + `60` người `L1 Cal` + báo thêm độ nhạy với `SAA-18 Cal` riêng biệt cho track `SAA-27`).
- **Hướng dẫn chạy:**
  ```bash
  export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  .venv/bin/python -u research/SEQ_MDD_V3_CAMPAIGN_20261121_v1/run_exp4_multiseed_cal.py \
    --device cuda --seeds 20261121 20261122 20261123 --epochs 14 \
    --out-dir research/SEQ_MDD_V3_CAMPAIGN_20261121_v1/runs/exp4_multiseed
  ```
- **Yêu cầu đầu ra bắt buộc (`runs/exp4_multiseed/`):**
  1. `checkpoints/seq_mdd_v3_seed{20261121,20261122,20261123}.pt` + `CHECKPOINT_CARD.json`.
  2. `RESULTS.json` báo **Trung bình ± Độ lệch chuẩn (3 seed)** + **KTC 95% ghép cặp theo cụm người nói** so với `R0` và `SeqMDDv2` trên cả `DEV-18 OOF` và 5 tập Eval Unseen (`Held-6`, `Spontaneous`, `EpaDB TEST`, `SAA-27`, `TIMIT Core`), kèm bảng riêng `EBVS` & `ABA` FP và `ECE` trước/sau Temperature Scaling.
  3. `REPORT_VI.md`.
- **Tiêu chí Đạt (GO — Khóa mốc `SeqMDDv3` Backbone Đóng Băng):**
  - Thắng đồng thời `R0` lẫn `SeqMDDv2` trên **cả 5/5 tập Eval** (kể cả `Held-6 >= 67,5%`, `Spontaneous >= 64,5%`, `EpaDB TEST >= 69,5%`, `SAA-27 >= 46,5%`, `TIMIT Core <= 0,60 FA/câu`) với độ lệch giữa 3 seed `< 0,6 pp` và `TIMIT Corruption Recall >= 94%`.

---

### `EXP-5`: `SeqMDDv3-LoRA` — Kiểm Định Mở Khóa `Joint Multi-Corpus LoRA (L16–L23)` Có Guard L1 CTC-KD *(Chỉ chạy SAU khi `EXP-4` hoàn tất!)*

- **Mục tiêu & Giả thuyết:** Kiểm chứng xem khi đã có **`400+` người nói đa corpus** (thay vì `12` người như trước đây), việc mở nhẹ `LoRA (r=8, α=16, L16–L23)` train chung với `SeqMDDv3` kèm mất mát chưng cất khung **L1 CTC-KD (`λ_kd = 0,5`, từ Frozen Base Koel)** có phá được trần biểu diễn đóng băng trên các nguyên âm khó (`IH`, `AH`, `AE`, `T`) mà **không làm hỏng L1 PER** hay không.
- **Thiết kế kỹ thuật:**
  - Dùng split-autograd đã chứng minh an toàn VRAM trên GTX 1650 (`no_grad` trên CNN + `L0..15`, autograd chỉ trên `L16..23` + `SeqMDDv3`, `peak VRAM <= 3.300 MiB`, `lr_lora = 3e-5`, `lr_head = 5e-4`, `6` epoch, `gradient_accumulation = 4`).
  - Mất mát tổng: $\mathcal{L} = \mathcal{L}_{\text{SeqMDDv3}} + 0.5 \cdot \mathcal{L}_{\text{CTC-KD}}(\text{Base} \to \text{LoRA})$.
  - **Cổng an toàn bắt buộc:** Đo **TIMIT Core Surface-39 PER** và **EpaDB TEST native79 PER** của luồng CTC sau khi ghép LoRA!
- **Hướng dẫn chạy:**
  ```bash
  export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  # 1. Smoke test VRAM & gradient trên 8 câu dài nhất trước khi chạy full
  .venv/bin/python -u research/SEQ_MDD_V3_CAMPAIGN_20261121_v1/run_exp5_joint_lora.py \
    --device cuda --smoke-only --out-dir research/SEQ_MDD_V3_CAMPAIGN_20261121_v1/runs/exp5_smoke
  # 2. Full run (chỉ chạy nếu smoke peak_reserved <= 3400 MiB & 0 NaN)
  .venv/bin/python -u research/SEQ_MDD_V3_CAMPAIGN_20261121_v1/run_exp5_joint_lora.py \
    --device cuda --seed 20261121 --epochs 6 \
    --out-dir research/SEQ_MDD_V3_CAMPAIGN_20261121_v1/runs/exp5_joint_lora
  ```
- **Yêu cầu đầu ra bắt buộc (`runs/exp5_joint_lora/`):**
  1. `SMOKE_VRAM.json` (`peak_allocated`, `peak_reserved`, xác nhận Base weights & `lm_head` diff `= 0.0`).
  2. `lora_adapter.pt` + `seq_mdd_v3_joint_head.pt` + `CHECKPOINT_CARD.json`.
  3. `PER_GUARD.json` (`TIMIT Core Surface-39 PER` & `EpaDB TEST PER` so với `Base Koel 12,110% / 18,418%`).
  4. `RESULTS.json` & `REPORT_VI.md` (so ghép cặp trực tiếp với `EXP-4` đóng băng trên cả 5 tập Eval + AP riêng của `IH, AH, AE, T`).
- **Tiêu chí Đạt (GO / NO-GO nghiêm ngặt):**
  - **BẮT BUỘC** `TIMIT Core Surface-39 PER <= 12,30%` (`Δ <= +0,19 pp`) và `EpaDB PER <= 18,60%`.
  - Phải tăng **`>= +1,0 pp` Strict μF1** trên ít nhất 3/4 tập L2 so với `EXP-4` đóng băng (với KTC 95% dương), và không làm tăng `TIMIT Core` FA. Nếu không đạt → **NO-GO**, giữ nguyên backbone đóng băng của `EXP-4`!

---

### `EXP-6`: `Real-Eval VN` — Xác Nhận Độc Lập Trên Người Học Việt Nam (Câu Ngắn Mới) & Lộ Trình Dữ Liệu 220K

- **Mục tiêu & Giả thuyết:** `Held-6` chỉ có 1 người Việt (`HQTV`), `EpaDB` có `0`, `SAA-27` chỉ có 3 người đọc cùng 1 bài dài. Đây là **cổng quyết định triển khai sản phẩm (Deployment Gate)** trên đúng quần thể đích (người Việt nói câu đọc tiếng Anh ngắn mới, kèm đối chứng người bản ngữ).
- **Thiết kế & Công cụ đã sẵn sàng trong repo:**
  - Bộ câu đề xuất (`60` câu mới): `research/KOEL_SM13_BENCHMARK_20261120_v1/PROMPT_BANK_CANDIDATE.tsv`
  - Quy cách thu pilot (`6–10` người × `15–20` câu mới, 2 người chấm mù + phân xử): `research/KOEL_TOP_LAYER_ABLATION_20261120_v1/CONFIRMATION_COLLECTION_SPEC_VI.md`
  - Công cụ kiểm duyệt rò rỉ & tạo phiếu chấm mù: `confirm_data_admission.py`, `prepare_sentence_annotation.py`
  - Trình chạy xác nhận khóa ngưỡng & kiểm đếm CPU độc lập: mở rộng `research/KOEL_SM13_BENCHMARK_20261120_v1/run_fresh_confirmation.py` để chấm song song **4 hệ đã khóa**: `Koel Base P0`, `R0`, `SM13 P0 (re-cal)`, và **ứng viên thắng `EXP-4/5` (`SeqMDDv3`)**.
- **Hướng dẫn chạy (khi đã có audio & phiếu chấm đã phân xử):**
  ```bash
  export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  # 1. Kiểm duyệt đầu vào (0 trùng speaker/prompt/SHA-256)
  .venv/bin/python research/KOEL_TOP_LAYER_ABLATION_20261120_v1/confirm_data_admission.py \
    --manifest <fresh_manifest.jsonl> --out <run_dir>/DATA_ADMISSION.json
  # 2. Suy luận đóng băng, niêm phong Parquet trước khi ghép gold, kiểm đếm CPU độc lập
  .venv/bin/python research/SEQ_MDD_V3_CAMPAIGN_20261121_v1/run_real_confirmation_v3.py \
    --device cuda --cal-manifest <cal.jsonl> --eval-manifest <eval.jsonl> --out-dir <run_dir>
  ```
- **Yêu cầu đầu ra bắt buộc:**
  1. `DATA_ADMISSION.json` (`overlap=0`), `INTER_RATER_AGREEMENT.json` (Cohen's κ theo từng lớp `C/S/O/I` & tỷ lệ `unknown`).
  2. `FROZEN.json`, `thresholds.json`, `eval_predictions.parquet` + `PREDICTIONS_SEAL.json` (niêm phong trước ghép gold).
  3. `RESULTS.json` & `REPORT_VI.md` (Strict μF1 S/O/I, P/R/F1 từng lớp, FP/câu, KTC 95% theo cụm người nói + độ nhạy theo cụm câu đọc, độ trễ GPU, và bảng đối chứng FA trên người bản ngữ).
- **Lộ trình mở rộng dữ liệu dài hạn (bảo lưu từ định hướng `2026-11-15`):** Nếu kết quả `EXP-6` xác nhận mô hình cần mở rộng quy mô dữ liệu thật có nhãn chuẩn chuyên gia, triển khai theo cấu trúc đã chốt: **220.000 bản thu / 2.200 người nói độc lập** (`Train 140k / Dev 20k / Sealed Test 40k / External 20k`), đo learning curve ở các mốc `10k → 50k → 100k → 140k` và ưu tiên diversity người học Việt Nam kèm kiểm soát bản quyền & double-annotation.

---

## 4. Quy Trình Nghiệm Thu & Ghi Nhận Bắt Buộc Sau MỖI Thí Nghiệm (Theo `AGENTS.md`)

Sau khi kết thúc **bất kỳ** thí nghiệm nào trong `EXP-1` → `EXP-6`, bắt buộc thực hiện đủ 5 bước trước khi chuyển bước tiếp theo:

1. **Chạy toàn bộ unit test liên quan** (`pytest -q tests/test_mdd_fixes_v2.py tests/test_seq_mdd_v3.py`) và kiểm tra GPU rảnh (`nvidia-smi` = `0%`, không treo process).
2. **Lưu biên nhận SHA-256** cho mọi checkpoint mới và mọi file dự đoán Eval (`CHECKPOINT_CARD.json`, `PREDICTIONS_SEAL.json`) — **không bao giờ ghi đè checkpoint cũ**.
3. **Cập nhật đầy đủ 4 hồ sơ nghiên cứu trong `research/`:**
   - `research/EXPERIMENTS.md` — Ghi rõ 5 mục: Objective, Setup, Results, Conclusion, Next Steps.
   - `research/SUMMARY.md` — Cập nhật bảng điểm & phát hiện mới nhất.
   - `research/DECISIONS.md` — Ghi rõ phán quyết `GO / NO-GO` kèm bằng chứng định lượng.
   - `research/CHECKPOINT.md` — Ghi rõ trạng thái bàn giao, đường dẫn checkpoint mới nhất, tài nguyên đĩa/GPU.
4. **Không tự bịa số liệu hay tối ưu ngưỡng trên tập Eval:** mọi ngưỡng `(τ_t, τ_g)` và trọng số hợp nhất đều phải được khóa trên `CALIBRATE` trước khi mở `EVAL`; mọi điểm `Oracle` phải dán nhãn rõ là *"chẩn đoán trần trên"*.
5. **Commit sạch có phạm vi** (chỉ commit code, test, report, JSON kết quả, và checkpoint thắng đạt cổng GO; giữ file cache `.parquet` lớn ngoài Git).
