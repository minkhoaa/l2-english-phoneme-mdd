# Pronunciation-Dataset-220K — Nghiên Cứu & Engine Chẩn Đoán Lỗi Phát Âm Tiếng Anh L2 (MDD)

Dự án nghiên cứu độc lập xây dựng hệ thống **Phát hiện & Chẩn đoán Lỗi Âm vị Tiếng Anh L2 (Mispronunciation Detection and Diagnosis — MDD)** trên đầy đủ 3 loại lỗi **Substitution (S), Omission (O), Insertion (I)** theo chuẩn topo Exact-Gap ($N$ target slots / $N+1$ gap slots), cùng lộ trình dài hạn xây dựng bộ dữ liệu chuẩn quy mô lớn (220.000 câu / 2.200 người nói).

> **Trạng thái hiện tại (22/11/2026 — Mốc lớn `MAJOR_MILESTONE_20261122_CLEAN11_KOEL_ENS3_O5_O4`):**
> - **Đã hoàn thiện & đóng gói Engine suy luận nghiên cứu / pilot mạnh nhất** trên dữ liệu công khai hiện có: [`deploy/koel_ens3_mdd/`](deploy/koel_ens3_mdd/README.md) (`15,94 MiB`, đối khớp `997/997` slot từ `.wav` thô, độ trễ đầu-cuối `155,8 ms` trên GTX 1650).
> - **Chưa phê duyệt triển khai production đại trà cho người học Việt Nam:** các tập điểm bên dưới là benchmark chẩn đoán lịch sử (chủ yếu L2-ARCTIC & EpaDB Tây Ban Nha); cần qua xác nhận độc lập trên câu ngắn người Việt (`EXP-6`) và lớp G2P đa biến thể phát âm chuẩn trước khi phát hành đại trà.
> - **Các con số 220.000 câu / 2.200 người nói** ở phần lộ trình bên dưới là mục tiêu quy hoạch dài hạn (`PLAN_ONLY`), không phải dữ liệu đã thu xong. Toàn bộ dữ liệu niêm phong bảo vệ (`test_sealed.jsonl`) giữ nguyên chưa mở.

---

## 1. Benchmark Thực Tế Đã Xác Minh (Tất cả số liệu từ hiện vật niêm phong & đếm độc lập)

Mọi mô hình dưới đây được chấm theo **Strict micro-F1 (S/O/I)** (phải đúng cả vị trí lẫn loại lỗi `S / O / I`), hiệu chỉnh ngưỡng **thuần trên tập Calibrate** (không chọn theo Eval), niêm phong Parquet trước khi ghép nhãn, và kiểm tra đồng thời 2 chốt an toàn bản ngữ:
- **`TIMIT Core` (192 câu bản ngữ Mỹ chuẩn):** số cảnh báo sai trung bình mỗi câu (**FA/câu**, càng thấp càng tốt).
- **`TIMIT Corruption Control` (50 câu bản ngữ có cài lỗi canonical):** tỷ lệ bắt đúng lỗi cài sẵn (**Ctrl Recall**, bắt buộc `≥ 95%` để loại trừ shortcut nhận diện micro TIMIT).

### 1.1. Bảng tiến hóa mô hình (từ các hệ cũ đến `SeqMDDv2.1` trên split 15 người L2-ARCTIC)

| Hệ thống / Mốc nghiên cứu | Held-6 F1 | EpaDB TEST F1 | Spont (H6) F1 | SAA-27 F1 | TIMIT Core (FA/câu) | Ctrl Recall | Ghi chú & Hiện vật |
|---|---:|---:|---:|---:|---:|---:|---|
| `SM13` formal seed 101 (locked `fixed/0.2`) | `16,98%` | `1,50%` | `~8–15%` | `7,02%` | — | — | Bug E4 diệt lớp `S` (`0%` S); Stella P50 `13,67 s` |
| `SM13` formal seed 101 (Cal-F1) | `40,41%` | `15,27%` | `~31–40%` | `37,27%` | `6,22` | — | Overfit 12 người train, `GapHead` không hội tụ |
| `Koel Base` (Vanilla gốc) | `56,96%` | `53,70%` | `~55%` | `37,46%` | `2,20` | — | 35/75 token IPA chưa ánh xạ |
| `Koel Base` + **bản vá P0** (75/75 IPA, 0 train) | — | **`64,81%`** | — | — | — | — | `+11,11 pp` EpaDB, `0` token unmapped |
| `R0` (Frozen Koel + `SmallMDDHead`, Cal-F1) | `66,75%` | `58,95%` | `61,55%`* | `43,94%` | **`7,92`** | — | Shortcut `align_tier_3` báo sai gần 1/4 âm bản ngữ; mù lớp `I` |
| `SeqMDDv2 Anti-Shortcut` (`cc3572b`) | `65,20%` | `68,29%` | `64,38%` | `45,39%` | `1,63` | `96,42%` | BiGRU 6 tầng XLSR + GapHead + TIMIT anti-shortcut ([card](research/MDD_FIXES_AND_V2_20261121_v1/checkpoints/CHECKPOINT_CARD.json)) |
| `SeqMDDv2.1 Fix #1 (iso_no_hwin)` (`aef0d8a`, TB 3 seed) | `65,99±0,31%` | `68,82±0,52%` | `63,10%` | `45,01±0,70%` | **`0,710`** | `96,18%` | Bỏ utterance CMVN + chuẩn hóa từng tầng + `scale_lp` ([card](research/SEQ_MDD_V3_CAMPAIGN_20261121_v1/checkpoints/CHECKPOINT_CARD.json)) |

*\*`R0` `61,55%` tính trên full Spontaneous; các dòng sau tách riêng lát unseen Held-6 (`24` câu). Nguồn: [`FULL_BENCHMARK_AND_DISSECTION.json`](research/FULL_3MODEL_BENCHMARK_AND_DEEP_DIAGNOSIS_20261121_v1/outputs/FULL_BENCHMARK_AND_DISSECTION.json), [`SEQ_MDD_V2_ANTISHORTCUT_RESULTS.json`](research/MDD_FIXES_AND_V2_20261121_v1/outputs_seq_mdd_v2/SEQ_MDD_V2_ANTISHORTCUT_RESULTS.json), [`CONFIRM_3SEED_RESULTS.json`](research/SEQ_MDD_V3_CAMPAIGN_20261121_v1/runs/confirm_fix1_3seed_20261121_v1/CONFIRM_3SEED_RESULTS.json).*

### 1.2. Bảng chuẩn tách sạch tuyệt đối `test_sealed` (`clean_11`: 11 người L2A Fit + 26 EpaDB + 334 TIMIT) & Mốc Đóng Gói Chính

Sau khi kiểm toán phát hiện split 15 người cũ trùng 4 người nói với danh sách công khai `test_sealed` của repo trước (`ZHAA, TNI, MBMPS, TLV`), toàn bộ vòng đánh giá chuẩn cuối cùng được khóa trên split sạch **`clean_11`** (`3.681 Fit / 635 Cal / 2.446 prompt Eval`, `0%` trùng người nói với `test_sealed`):

| Hệ thống trên `clean_11` | Held-6 F1 (`tAP / gAP`) | EpaDB TEST F1 (`tAP`) | Spont H6 F1 (`gAP`) | SAA-27 F1 (`tAP`) | TIMIT Core (FA/câu) | Ctrl Recall |
|---|---:|---:|---:|---:|---:|---:|
| `S0_clean` (`iso_no_hwin`, TB 3 seed) | `65,32±0,70%` (`0,7687 / 0,5306`) | `68,57±0,05%` (`0,8110`) | `64,68±1,25%` (`0,4938`) | `43,05±1,31%` (`0,4664`) | `0,578±0,070` | `95,82%` |
| **`S0_clean_ens3_o5_o4`** *(3-seed Ensemble + Gap Platt, Cal L1 ≤ 1/1k)* | **`66,24%`** (`0,7753 / 0,5622`) | **`69,02%`** (`0,8149`) | **`64,83%`** (`0,5401`) | **`43,92%`** (`0,4709`) | **`0,562`** (`106t+2g`) | **`97,13%`** |
| **Chênh lệch `O5+O4` vs 1-seed TB** | **`+0,92 pp`** (`gAP +0,0316`) | **`+0,45 pp`** | **`+0,15 pp`** (`gAP +0,0463`) | **`+0,87 pp`** | **`-0,016` (giảm FA)** | **`+1,31 pp`** |

- **Giảm `14×` báo sai bản ngữ so với `R0` ban đầu** (`7,92 → 0,562 FA/câu` trên `TIMIT Core`) trong khi giữ `Control Recall = 97,13%`.
- **Đối khớp từ file `.wav` thô (`15` câu thực tế, `997` slot) vs Parquet niêm phong:** Target **`491/491`** (`max |Δscore| = 5,5e-7`), Gap **`506/506`** (`max |Δscore| = 1,02e-6`).
- **Độ trễ đầu-cuối (`WAV → Koel Base → Pool → 3-head Ensemble → C/S/O/I`):**
  - **GTX 1650 (50 câu ngắn Held-6):** P50 **`155,8 ms`** / P95 **`220,6 ms`** (`Koel 132,7 ms + 3-head 11,1 ms`, đỉnh VRAM **`1.646 MiB`**).
  - **GTX 1650 (5 đoạn Stella dài 22–49s):** P50 **`1.477,6 ms`** (`3.498 MiB` VRAM).
  - **CPU-only (10 câu ngắn):** P50 **`1.791,1 ms`**.
- **Biên nhận:** [`deploy/koel_ens3_mdd/BUNDLE_CARD.json`](deploy/koel_ens3_mdd/BUNDLE_CARD.json) · [`DEPLOY_VERIFICATION.json`](deploy/koel_ens3_mdd/DEPLOY_VERIFICATION.json) · [`S0_clean_ens3 RESULTS.json`](research/ORTHO_IMPROVEMENTS_20261122_v1/receipts/RESULTS.json).

---

## 2. Tổng Hợp Các Hướng Đã Kiểm Chứng & Đóng (`NO_GO` — Lưu Biên Nhận Kết Quả)

Thay vì giữ lại mã thử nghiệm dư thừa, mọi hướng thử nghiệm không đạt cổng đều đã được kiểm chứng có kiểm soát và lưu biên nhận định lượng để tránh lặp lại:

| Chiến dịch / Giả thuyết | Cấu hình đã thử | Kết quả thực đo chính | Kết luận & Biên nhận |
|---|---|---|---|
| **Dual-Stream / KD-LoRA Cascade** | 59 cấu hình DEV-18 & SAA-27 (Logistic/MLP Cascade, Veto, Joint-LoRA) | Không vượt `R0` trong ngân sách FP; gọi luồng KD `48–100%` câu (tăng độ trễ ~2×) | **`NO_GO`** — [SUMMARY](research/SUMMARY.md) |
| **`SeqMDDv3` (`EXP-1 → EXP-5`)** | Cửa sổ 5 điểm, CMUdict L1, Gap-core/Focal/OLLR, Koel LoRA tầng 20–23 | 5/5 trượt cổng Held-6/TIMIT; `EXP-5` LoRA trên Koel làm vỡ PER (`12,4% / 18,9%`) | **`NO_GO`** — [EXPERIMENTS](research/EXPERIMENTS.md) |
| **Synthetic 3 hướng trên frozen Koel (`S1/S2/S3`)** | S1 prompt perturbation (1.500 câu), S2 Kokoro TTS + Sham-C (600 cặp), S3 Koel+HuPER consensus (1.450 câu) | 0/3 đạt cổng phân bố âm học (`can_lp -7,3..-8,96` quá dễ vs Fit `-1,68`); S3 chỉ giữ 34 câu (`O 64,7%`) | **`NO_GO`** — [receipts](research/SYNTH_3DIR_ISO_V1_20261122/receipts/TECHNICAL_ADMISSION.json) |
| **Backbone Campaign `full_clean11_v1` (8h43m GPU)** | 4 encoder đóng băng (`WavLM-Large`, `WavLM-Base+`, `HuPER`, `Whisper-Small`), DualRep `A1/A3`, Frame Cross-Attn `B1/B2`, `WavLM-Large` LoRA + 18 AF (`C1/C2`) | Chỉ `WavLM-Base+` qua probe khe; **`0/6` nhánh train qua cổng Cal** (`Held-6 64,02–65,17%` vs `S0 65,96%`; `C1/C2` TIMIT vọt `1,16–1,65 FA/câu`) | **`NO_GO`** — [báo cáo](research/BACKBONE_REP_CAMPAIGN_20261122_v1/receipts/CAMPAIGN_REPORT_VI.md) |
| **Chẩn đoán & Sửa 197 `realized_id` DP (`FIT_ONLY_PATCH`)** | Đối chứng 3 seed ghép cặp sửa đúng 197 slot `S` sai ID phụ trong L2A Fit | Held-6 `-0,24 pp` (`0/3`), Spont H6 `-0,25 pp`, TIMIT `+0,025/câu` | **`NO_GO`** — [chẩn đoán](research/BACKBONE_CHECKPOINT_AUTOPSY_20261122_v1/DIAGNOSIS_VI.md) · [kết quả](research/ORTHO_IMPROVEMENTS_20261122_v1/receipts/GPU_3SEED_VERIFY_RESULTS.json) |
| **Cải tiến trực giao (`O1/O2/O3-cal/O3_cond_so`)** | `O1` TIMIT freq-matched, `O2` Logit Adjustment `(src, phone)`, `O3-cal` `rho_o`, `O3_cond_so` (`λ=0,25/0,5`) | `O1` TIMIT tăng 2×; `O2` tụt tới `-5,45 pp`; `O3-cal` overfit Cal; `O3_cond_so` tăng EpaDB `+0,34 pp` nhưng tụt Spont & tăng TIMIT (`ens3` thua `F0`) | **`NO_GO`** *(chỉ giữ `O5+O4`)* — [receipts](research/ORTHO_IMPROVEMENTS_20261122_v1/receipts/CPU_ABLATION_RESULTS.json) |
| **Mở rộng dữ liệu L2 thật `SpeechOcean762 TRAIN`** | `+125` người L2 thật (`2.500` câu, `47.076` âm, 5 chuyên gia), exact marginal/soft binary CE (`strict_all`, `adult_only`, `soft_all`, 3 seed) | Nhích nhẹ `tAP` đọc ngắn (`+0,003`), nhưng giảm Held-6 (`-0,24..-0,40 pp`), tụt nặng `SAA-27` (`-2,27..-7,43 pp`) & Spont H6 (`-1,07..-2,66 pp`) do lệch miền `80%` `C` trên câu ngắn | **`NO_GO`** — [receipts](research/REAL_L2_SO762_20261122_v1/receipts/SO762_3SEED_RESULTS.json) |

**Bài học cốt lõi đã chứng minh định lượng:** Trần `~66–69%` hiện tại không còn nằm ở kiến trúc head hay thiếu tham số backbone, mà ở **(1)** độ phủ người nói L2 Fit hẹp (`37` người), và **(2)** tỷ lệ lớn lỗi `S_same` (nhãn ghi `S` dù mã âm realized trùng canonical — chiếm `25,3%` lỗi `S` trên L2-ARCTIC Cal vs `2,24%` Fit; `75,5%` lỗi cả 3 seed bỏ sót nằm trong lát Koel tự tin `can_lp > -0,2`).

---

## 3. Chạy Thử Engine Suy Luận (`deploy/koel_ens3_mdd/`)

Yêu cầu: Python 3.11 + PyTorch + Base Koel cục bộ (offline, 0 upload dữ liệu ra ngoài).

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m deploy.koel_ens3_mdd.engine \
  --wav path/to/sample_16k.wav \
  --phones "HH AH L OW" \
  --device cuda
```

Hoặc dùng trực tiếp từ Python:

```python
from deploy.koel_ens3_mdd import KoelEns3MDDEngine

engine = KoelEns3MDDEngine(device="cuda")  # hoặc device="cpu"
result = engine.score_wav("path/to/sample_16k.wav", ["HH", "AH", "L", "OW"])
print(result["summary"], result["targets"], result["gaps"])
```

Chi tiết schema đầu ra, giới hạn và lệnh tái kiểm parity: [`deploy/koel_ens3_mdd/README.md`](deploy/koel_ens3_mdd/README.md).

---

## 4. Cấu Trúc Thư Mục Chính Được Lưu Giữ Trong Repo

```text
Pronunciation-Dataset-220K/
├── README.md                                        # Tổng quan & bảng benchmark xác minh
├── GOAL.md                                          # Đặc tả lộ trình & ranh giới ủy quyền
├── deploy/koel_ens3_mdd/                            # Engine suy luận đóng gói (15,94 MiB bundle + API + parity/latency)
├── research/
│   ├── SUMMARY.md / EXPERIMENTS.md / DECISIONS.md / CHECKPOINT.md   # Hồ sơ R&D xuyên suốt
│   ├── FULL_3MODEL_BENCHMARK_AND_DEEP_DIAGNOSIS_20261121_v1/        # Benchmark gốc 6 tập / 3 mô hình
│   ├── MDD_FIXES_AND_V2_20261121_v1/                # Bản vá P0 75/75 IPA + checkpoint SeqMDDv2 Anti-Shortcut
│   ├── SEQ_MDD_V3_CAMPAIGN_20261121_v1/             # Cô lập Fix #1 + checkpoint SeqMDDv2.1 (iso_no_hwin)
│   ├── ORTHO_IMPROVEMENTS_20261122_v1/              # O5 3-seed Ensemble + O4 Gap Platt + biên nhận CPU/GPU
│   ├── BACKBONE_REP_CAMPAIGN_20261122_v1/           # Mã & biên nhận chiến dịch 4 backbone / XAttn / LoRA (NO_GO)
│   ├── BACKBONE_CHECKPOINT_AUTOPSY_20261122_v1/     # Phân rã lát mù S_same & audit 197 realized_id Fit
│   ├── REAL_L2_SO762_20261122_v1/                   # Mã & biên nhận thí nghiệm SpeechOcean762 TRAIN (NO_GO)
│   └── SYNTH_3DIR_ISO_V1_20261122/receipts/         # Chỉ lưu biên nhận kết quả NO_GO của Synthetic S1/S2/S3
├── docs/                                            # Hồ sơ thiết kế quy mô 220K, pháp lý, QC & audit chống rò rỉ
└── tests/                                           # Bộ kiểm thử đơn vị & hợp đồng dữ liệu (45/45 pass ở mốc mới nhất)
```

---

## 5. Các Bước Bắt Buộc Tiếp Theo Để Lên Production & Lộ Trình Dài Hạn 220K

1. **Nghe mù đối soát nhóm `S_same`** (trên Fit/Cal) để chốt định nghĩa sản phẩm: có cần cảnh báo lệch sắc thái nhẹ trong cùng âm vị hay chỉ bắt lỗi đổi/mất/thêm âm rõ.
2. **Xác nhận độc lập trên người học Việt Nam câu ngắn (`EXP-6`):** thu âm thực tế trên micro phổ thông, gán nhãn kép mù & phân xử bởi người thật, rồi chấm trực tiếp bằng `KoelEns3MDDEngine` để đo F1 & FA thực tế.
3. **Bổ sung lớp tiền/hậu xử lý sản phẩm:** G2P hỗ trợ đa biến thể phát âm chuẩn (tránh phạt oan các cách đọc từ điển hợp lệ), lọc im lặng/nhiễu nền, và gom lỗi theo từ/âm tiết.
4. **Lộ trình dữ liệu dài hạn (quy hoạch, chưa triển khai):** 220.000 câu / 2.200 người nói (`140k Train / 20k Dev / 40k Test Int / 20k Test OOD`), xem chi tiết tại [`docs/MASTER_DATASET_RD_REPORT.md`](docs/MASTER_DATASET_RD_REPORT.md).

### Quy tắc bảo toàn & quản trị (Section 8)
- **Zero Cloud API Uploads:** 100% xử lý cục bộ on-premises, không đưa audio lên API đám mây.
- **Zero Leakage & Sealed Benchmarks:** `test_sealed.jsonl` giữ nguyên đóng; không chọn siêu tham số/ngưỡng trên Eval.
- **Git Hygiene:** Không commit audio thô, feature cache nặng hay thư mục `runs/`; chỉ lưu mã tái lập, bundle chính thức và biên nhận JSON/Markdown.
