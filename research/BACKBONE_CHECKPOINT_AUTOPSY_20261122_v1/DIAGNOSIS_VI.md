# Vì sao F1 chưa tăng? Kiểm tra trực tiếp checkpoint `full_clean11_v1`

**Kết luận trên Cal:** thay encoder không tự động giải quyết phần target khó. Koel đã phân biệt tốt lỗi mà bộ nhận dạng nghe ra một âm khác. Nhiều lỗi còn lại được gán `S` dù âm realized trong manifest nguồn có cùng mã với âm chuẩn. Loại này ít trong Fit L2-ARCTIC nhưng phổ biến ở ba người Cal. Các nhánh WavLM **có học**; chúng thay đổi dự đoán, nhưng chưa xếp các lỗi thật lên trên âm đúng tốt hơn S0. Không có bằng chứng rằng đây là trần của âm thanh, rằng mọi nhãn khó đều sai, hoặc LoRA đã thất bại.

## 1. So đúng model, đúng người, đúng điểm vận hành

Seed `20261121`, cùng `clean_11`. Cal có 515 câu L2 (3 người L2-ARCTIC + 4 người EpaDB), 120 câu native. Strict F1 đếm đúng *loại* S/O/I; tAP xếp hạng có lỗi S/O, gAP xếp hạng insertion. “Lát mù” = `can_lp_vit_max > −0,2`, tức Koel rất tự tin vào âm canonical. Không dùng nhãn Eval.

| Checkpoint | Epoch | Cal strict F1 | tAP | gAP | AP lát mù | F1 khi giữ số target/gap positive phát hiện bằng S0¹ |
|---|---:|---:|---:|---:|---:|---:|
| Koel S0 | 5/14 | 67,82% | 0,8000 | 0,5286 | 0,2187 | 67,82% |
| A1 + WavLM-Base+ | 5/14 | 67,57% | 0,7948 | 0,4946 | 0,2064 | 67,25% |
| A3 thêm CharSiu/prototype | 5/14 | 67,65% | 0,7963 | 0,4822 | 0,2150 | 67,17% |
| B1 attention frame WavLM | 3/12 | 66,68% | 0,7931 | 0,4654 | 0,1964 | 66,26% |
| B2 attention frame WavLM + Koel | 4/12 | 66,94% | 0,7866 | 0,4735 | 0,1598 | 66,34% |

¹ Khớp đúng `1.660/2.282` target và `58/102` gap positive Cal; chỉ là phép so hậu kiểm, không đổi policy. S0 cả 3 seed: `67,82 / 68,54 / 68,10%`. A/B chỉ một seed ở đây; B có ngân sách tối ưu/kiến trúc khác, không coi là ablation một biến. Ước lượng bootstrap mô tả ghép cặp theo người, trong từng nguồn, với policy cố định: ΔF1 A1–S0 `−0,25 pp [−0,49; +0,07]`, A3–S0 `−0,17 pp [−0,70; +0,58]`. Chỉ 7 người và cũng chính là Cal dùng để chọn epoch/ngưỡng.

**Vì sao probe tăng mà F1 không tăng?** Stage 1 so hai *Ridge*: Base+ giúp gap AP `0,3881 → 0,4132`. Gap AP của **S0 neural** đã là `0,5286`. Chỉ `102/2.384` lỗi Cal là I; ngay cả một oracle sửa *mọi* gap, giữ nguyên target S0, chỉ đưa strict F1 từ `67,82%` lên `69,65%`. Ở các ngưỡng đã lưu, A1 cứu/mất **81/37 S, 17/6 O, 3/14 I** so với S0, nhưng tổng FP tăng `836 → 940`. A3 cứu/mất `61/44 S, 14/7 O, 3/22 I`. Tăng số phát hiện S hoặc một chỉ số probe không đồng nghĩa tăng F1.

## 2. Lỗi còn lại nằm ở đâu?

- Trên lát Koel tự tin, S0 chỉ báo được **39/451** target lỗi (`8,65%`); ngoài lát đó, báo được **1.621/1.831** (`88,53%`). Tỷ lệ lỗi hai lát là `4,70%` và `59,56%`; không so AP giữa lát mà bỏ qua prevalence. Cả 3 S0 cùng bỏ qua **526** S/O, trong đó **397 (75,5%)** thuộc lát mù. Đây không chủ yếu là dao động giữa seed.
- L2-ARCTIC Cal: `270/1.066 S (25,33%)` có mã âm realized **giống** canonical trong chú thích nguồn (`S_same`); Fit chỉ `80/3.566 (2,24%)`. `187/270` ví dụ Cal này ở riêng SVBI. S0 gọi đúng S ở `43/270` trường hợp, so với `587/796` nếu realized khác canonical. S0 strict F1 `56,34%` trên L2-ARCTIC, `80,55%` trên EpaDB. Fit và Cal tách người, nhưng lần lượt `133/260` và `250/255` câu Cal có chuỗi canonical đã gặp trong Fit; không quy chênh lệch hai corpus chỉ cho accent.
- [Phân rã 526 lỗi chung](supervision/CAL_SLICES.json): `208 S_same`, `158 S_alt` và `44 O` ở L2-ARCTIC; `106 S` và `10 O` ở EpaDB. Trong 208 `S_same`, có 192 ở lát mù. A1/A3/B1/B2 chỉ cứu `13/9/4/2` trong 208 tại ngưỡng riêng. Nếu lấy mỗi `S` so với **các C cùng người, cùng âm canonical**, S0 xếp `S_same` trên C trong `61,9%` cặp (tính trung bình đều theo S, ties ½); với `S_alt` là `84,1%`. B2: `62,1% / 84,0%`. Không phải hoàn toàn không có tín hiệu, nhưng thêm frame chưa cải thiện rõ khả năng tách loại khó. Không coi các cặp dùng chung C là các quan sát độc lập.
- Mức đóng góp **số học**, không phải can thiệp: với S0, tỷ lệ `S → C` ở L2-ARCTIC tăng từ `20,53%` Fit lên `39,12%` Cal. Giữ tỷ lệ bỏ sót Cal trong từng nhóm nhưng cho `S_same/S_alt` trọng số như Fit: còn **26,48%** (`39,12 − 26,48 = 12,64 pp`). Hiệu ứng mix tương ứng A1/A3 `12,22/12,33 pp`. Phần còn lại vẫn tồn tại **trong** nhóm; không suy từ phép tính này rằng thay phân bố train sẽ tăng F1 từng đó. Công thức: `Σ_g (n_Fit,g/n_Fit,S) × (S→C_Cal,g/n_Cal,g)`, từ [`head_counts`](supervision/AUX.json).
- Trong `270` ví dụ `S_same`, realized head của S0 đoán đúng mã âm ở `264`, trong khi head MDD gọi `C` ở `216`; **215** slot vừa đoán đúng mã realized vừa bỏ sót lỗi. Trái lại, trong `201` lỗi `S_alt` bị gọi `C`, realized head chỉ đoán đúng âm khác `7` lần. Không thể lấy “đoán đúng mã âm” làm đồng nghĩa với “phát âm đạt”. Manifest nguồn đã chuẩn hóa ký hiệu; chưa có người nghe độc lập xác nhận tiểu loại của 270 slot. **Không đổi `S_same` thành `C`.**

## 3. Một lỗi supervision thật, nhưng khác với dịch chuyển ở trên

Đối chiếu theo UID, SHA WAV, thứ tự canonical, target và gap của **1.042 Fit + 260 Cal L2-ARCTIC** với manifest nguồn có thời gian: nhãn MDD khớp. Nhưng [builder](../MDD_FIXES_AND_V2_20261121_v1/train_and_eval_seq_mdd_v2.py#L169) căn lại *chuỗi* realized bằng DP, lấy `real_ids` của alignment mới rồi chỉ ép lại C/O. Ở slot S, vì thế ID phụ có thể rơi vào âm bên cạnh hoặc `<DEL>`.

- Sai khác thực tế: **197/3.566 S Fit** (129 ID là `<DEL>`) và **25/1.066 S Cal** (10 `<DEL>`). [`supervision/AUX.json`](supervision/AUX.json) có khóa `(role, uid, slot_index)` và hai ID cho đủ 222 trường hợp, **chưa áp dụng**. Không dùng 25 nhãn Cal để train.
- Trên 197 slot Fit, thay *chỉ* ID trong phép tính CE phụ thì tỷ lệ cosine gradient âm giữa target loss và realized loss tại cùng vector sau BiGRU đổi `39,1% → 25,9%` (S0), `59,9% → 21,8%` (A1), `43,7% → 25,4%` (A3). Đây là gradient cục bộ của checkpoint cố định, không đo gradient toàn bộ mạng sau retrain. ID sai là bug có thể sửa; **chưa đo được nó làm giảm bao nhiêu F1**.

## 4. Nhánh mới có chết không? Có phải chỉ pool sai biên?

- A1 có gradient hữu hạn, khác 0 ở projection WavLM target/gap; trọng số đã rời khởi tạo. A1/A3 không nhận phase gap-only vì không vượt mốc Cal; target drift `0`. `af_head` không có gradient ở A là đúng [loss Stage 2](../BACKBONE_REP_CAMPAIGN_20261122_v1/stages_0_3.py#L787), khác Stage 4 có loss AF.
- **Can thiệp trên checkpoint, không retrain:** xáo activation trong cùng người và âm (gap: cặp âm kề), 3 derangement. A1 tAP `0,7948 → 0,7921` khi xáo WavLM target+gap, nhưng `→ 0,4948` khi xáo scalar Koel target. gAP của A1 khi xáo WavLM `0,4946 → 0,4615`; tín hiệu không bằng 0 nhưng chưa đủ hơn S0. A3 tAP `0,7963 → 0,7960` khi xáo WavLM, `→ 0,7936` khi xáo gói CharSiu/prototype. Đây là *độ nhạy với thông tin riêng của token trong các nhóm này*, không phải hiệu ứng độc lập khi bỏ encoder rồi train lại.
- **Biên thời gian:** `12.459/12.663` target Cal L2 chỉ có một frame ở span Viterbi và cũng có `h_vit == h_peak`. Tuy nhiên peak vẫn nằm trong đoạn tham chiếu ở `6.591/7.406 C`, `690/796 S_alt`, `226/270 S_same`; cho dung sai ±20 ms là `7.283/7.406`, `779/796`, `262/270`. Có `53/59` tâm insertion L2-ARCTIC trong khoảng hai peak (57/59 với dung sai). Không thể từ “một frame” suy ra cả hidden state vô dụng.
- **Đối chứng biên trên cả Fit và Cal:** Ridge cố định trên L2-ARCTIC C/S, cùng Koel + cùng tensor WavLM L8, thay duy nhất cửa sổ mean. AP tổng `0,5940` (Voronoi), `0,5950` (CharSiu), **`0,5926` (biên tham chiếu)**, so với `0,6005` chỉ Koel. Riêng `S_same` vs C, AP `0,0687 / 0,0678 / 0,0675` (prevalence `3,52%`). CharSiu cải thiện IoU nhưng biên tốt hơn **không đủ** trong phép thử này; không phải cận trên cho encoder trainable hay alignment khác. [Kết quả và các đối chứng](timing/WINDOW_PROBE.json).
- **Checkpoint B1 thực sự dùng attention:** B1 chỉ cứu **20/526** lỗi cả ba S0 đều bỏ, đồng thời bỏ **65/1.521** lỗi cả ba từng phát hiện (binary, ở ngưỡng B1). Ở cùng recall với S0 seed đầu: cứu `31/526`, trong đó `16/397` trên lát mù. Attention WavLM target trung bình qua head trải rộng (entropy chuẩn hóa bình quân `0,974`, từng head `0,823`). Nhưng ép từng head lấy đều, giữ nguyên frame/mask/V/out, làm B1 tAP **`0,7931 → 0,6751`**, F1 `66,68% → 56,80%`. Chỉ ép đều gap: gAP `0,4654 → 0,2657`, không đổi target. **Attention rộng không có nghĩa attention vô dụng**; phụ thuộc vào trọng số đã học không suy ra biên đúng hoặc hơn Koel. [Replay B1](frames/B1.json).

- **B2 dùng cả hai nhánh:** [replay B2](frames/B2.json) cứu `23/526` target FN chung ở ngưỡng riêng nhưng bỏ `59/1.521` target cả 3 S0 đã phát hiện. Khớp recall: `46/526` được cứu, riêng lát mù `11/397`. Ép đều *riêng* attention target của WavLM hoặc Koel: tAP từ `0,7866` xuống `0,7658` hoặc `0,7613`; ép đều gap WavLM đưa gAP `0,4735 → 0,2238`. B2 không đơn thuần bỏ qua nhánh mới, nhưng thông tin nó dùng chưa khắc phục lỗi trên Cal.

## 5. Epoch, phạm vi và phép thử đáng làm tiếp

Trên mẫu Fit cố định (96 UID/nguồn lấy theo SHA; số liệu F1/tAP chỉ tính L2), A1 tAP `0,9087` so với S0 `0,8948` nhưng trên Cal lại kém. Sau epoch 5→14 ở A1, loss Fit `0,227 → 0,098` trong khi tAP Cal `0,795 → 0,756`; A3 cũng giảm tAP `0,796 → 0,765`. Ở B1, tAP `0,793 → 0,736` từ epoch 3→12 (loss B có cả thưởng entropy, không so trị số loss với A/S0). **Đây là dấu hiệu khớp Fit quá mức, không phải thiếu epoch.** Epoch lưu tối đa `tAP + 0,15·gAP`, không phải strict F1; không có logits mọi epoch để loại trừ một epoch khác có F1 tốt hơn.

**Kiểm tra lưới ngưỡng:** [Cực đại chính xác trên Cal](operating_point/THROUGH_B1.json) với mọi cặp ngưỡng target/gap, giữ nguyên thứ tự điểm, ties và quyết định loại S/O; không áp cả guard native nên đây là đối chứng lạc quan. Seed đầu: S0 tối đa `67,86%`, A1 `67,64%`, A3 `67,75%`, B1 `66,81%`. [B2](operating_point/B2.json) tối đa `67,02%`. Cả bốn model mới vẫn dưới **F1 ở ngưỡng đã lưu** của S0 (`67,82%`); không thể đạt gate `+0,5 pp` bằng cách làm mịn lưới. Không xuất hay áp ngưỡng mới. Phép thử không loại trừ cách thay đổi thứ tự xếp hạng hoặc cách phân biệt S với O.

1. Để nguyên run GPU, ngưỡng, `GOAL.md`, `test_sealed` và các checkpoint cũ. Cal chưa phải kết quả tổng quát. Chờ LoRA và cổng 3 seed khóa sẵn.
2. Sau campaign: đối chứng **chỉ sửa 197 ID phụ Fit** trên 144 câu bằng [`FIT_ONLY_PATCH.json`](supervision/FIT_ONLY_PATCH.json), tạo độc lập, không dùng `AUX.json` có Cal. Loader [`load_fit`](supervision/fit_only.py) kiểm SHA Fit, chỉ số, loại S, ID cũ, hash các trường bất biến và trả bản sao trong RAM. `F0` y hệt vs `F1`, 3 seed, cùng thứ tự/HP; giữ nguyên Cal và toàn bộ S/O/I. Tách khỏi mọi thay đổi sampler/loss/threshold. So tAP, gAP, strict F1, native FA, AP/recall lát mù và `S_same` theo người; không nhận gain nếu chỉ đổi ngưỡng hoặc một người gánh hết.
3. Với `S_same`, ưu tiên đối soát mù audio + ký hiệu TextGrid gốc trên Fit, có mẫu C và S_alt ghép người/âm, trước khi gọi đây là lỗi annot hoặc tạo nhãn phụ mới. Nếu nhãn nhất quán, bổ sung ví dụ L2 thật theo biến thể *trong* cùng âm trên Fit/nguồn mới. Nếu người chấm bất đồng, xử lý độ chắc chắn của nhãn thay vì đẩy head học thuộc. Không chọn lại bằng Eval đã dùng ở dự án lịch sử.

## Tái lập và giới hạn

Trên git `aef0d8a` + 55 hash source của run; SHA `RUN_BINDINGS.json` bắt đầu `ec082d39`. Chỉ đọc Fit/Cal, cache theo UID và metadata seal; toàn bộ phép đo không đọc dự đoán hoặc gold Eval. CPU `torch 2.6.0+cu124` không lộ CUDA, Python `3.11.17`. CPU/GPU max |Δp| cho năm checkpoint S0/A ≤ `6,86e−7`; tính cả B1/B2 ≤ `1,14e−6`. Strict F1 replay trùng card tới 0,01 pp ở cả bảy checkpoint. Tái dựng attention từng head khớp output PyTorch trong `1,91e−6`. Ba seed S0 không thay thế ba seed cho A/B. Các lát phân tích sau khi biết Cal không phải kiểm định độc lập.

```bash
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
P=research.BACKBONE_CHECKPOINT_AUTOPSY_20261122_v1
.venv/bin/python -m "$P.audit" --out /tmp/cal-audit-fresh.json
.venv/bin/python -m "$P.timing.pool" --out /tmp/windows-fresh.json
.venv/bin/python -m "$P.supervision.check_aux" --out /tmp/aux-fresh.json
.venv/bin/python -m "$P.frames.check" --arm B1 --out /tmp/b1-fresh.json
.venv/bin/python -m "$P.operating_point.frontier" --arms S0_20261121 S0_20261122 S0_20261123 A1 A3 B1 --out /tmp/f1-envelope-fresh.json
.venv/bin/python -m "$P.supervision.fit_only" --out /tmp/fit-only-patch-fresh.json
.venv/bin/python -m "$P.frames.check" --arm B2 --out /tmp/b2-fresh.json
.venv/bin/python -m "$P.operating_point.frontier" --arms B2 --out /tmp/b2-envelope-fresh.json
.venv/bin/python -m "$P.supervision.slices" --out /tmp/conditional-ranks-fresh.json
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_backbone{_checkpoint_autopsy,_autopsy_frame_and_thresholds,_autopsy_fit_patch,_rep_campaign}.py tests/test_ortho_improvements_v1.py
```

Kiểm thử: `41 passed` (3 cảnh báo thư viện), gồm so trần F1 với vét cạn có ties, parity MHA và chiếu V có mask, bất biến dữ liệu khi sửa ID. `ruff format --check` và `ruff check --select F,E9` đạt cho autopsy; không tuyên bố đạt toàn bộ quy tắc style Ruff mặc định.

Dùng tên output chưa tồn tại. [`CAL_AUDIT_V2.json`](CAL_AUDIT_V2.json) là receipt dùng cho báo cáo. `CAL_AUDIT.json` cũ giữ nguyên lịch sử: 6 ô đếm `peak_inside` / `inside_20ms` trên ba lát bị rút gọn theo OR do `object` chứa `numpy.bool_`. V2 sửa bộ đếm, có regression test và replay đầy đủ; ngoài timestamp/hash code và 6 ô này mọi kết quả không đổi.
