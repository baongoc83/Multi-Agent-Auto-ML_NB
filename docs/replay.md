# Replay một run sang môi trường khác

Mỗi run ghi ra một bundle đủ để **chạy lại toàn trình** ở máy/môi trường khác:
từ lúc chia train/valid/oot, qua các apply transform, tới train lại model cuối
bằng đúng feature list đã chốt.

Không có bước nào ở replay *quyết định lại* điều gì. LLM, FLAML, Optuna, RFE,
PSI, stability và SHAP prune đều là các bước **tìm kiếm** trên dữ liệu train;
kết quả của chúng đã nằm trong bundle rồi, nên replay đọc kết quả đó thay vì
tìm lại. Đó là lý do kết quả tái lập được chứ không chỉ "gần giống".

## Bundle gồm gì — 6 file

Trong `outputs/<YYYY-MM-DD>/run_NN/`. Đây là **toàn bộ** những gì driver đọc;
copy đúng 6 file này sang thư mục rỗng là chạy được (đã kiểm chứng).

| File | Agent | Vai trò | retrain | score |
|------|-------|---------|:---:|:---:|
| `replay_pipeline.py` | — | Driver. Chạy file này. | ✅ | ✅ |
| `replay_manifest.json` | — | Mọi quyết định đã chốt: key chia, estimator, best_params, **final feature list**, metrics gốc để đối chiếu. | ✅ | ✅ |
| `split_assignment.parquet` | 0 (split) | Mỗi dòng gốc → partition nào, vị trí thứ mấy (`_split_pos_`), occurrence rank (`_key_occ_`). | ✅ | — |
| `cleaning_spec.pkl` | 1 | CleaningSpec: drops, dtype fixes, clip bounds + `row_ops` train-only. | ✅ | ✅ |
| `feature_spec.pkl` | 2 | FeatureSpec: interactions, encoders, WoE maps, selected features. | ✅ | ✅ |
| `final_model.pkl` | 3 | Model + ensemble + calibrator + cat encoders + versions lúc train. | ✅ | ✅ |

Score mode không cần `split_assignment.parquet` → 5 file.

Hai thứ **không** nằm trong bundle mà bạn phải tự mang: **file input gốc** và
**repo** (xem phần dưới). Thiếu file nào thì driver liệt kê ngay từ đầu rồi
dừng, không chạy nửa chừng mới chết.

Các file còn lại trong RUN_DIR (`*_report.json`, `final_report.md`, `charts/`,
`shap_*`, `psi_report.csv`, `pipeline_process_*.py`, `agent_execution.log`) là
báo cáo/chẩn đoán — hữu ích để đọc, không cần cho replay.

## Chạy

```bash
# Train lại và đối chiếu với metrics của run gốc
python replay_pipeline.py <file_input_gốc> --mode retrain --repo /path/to/repo

# Chỉ score dữ liệu mới bằng model đã train
python replay_pipeline.py <file_mới> --mode score --out scores.parquet --repo /path/to/repo
```

`--repo` trỏ tới thư mục repo (hoặc đặt biến môi trường `AUTOML_REPO`). Bỏ qua
được nếu run dir vẫn nằm trong repo.

Cuối `--mode retrain`, driver in bảng so sánh từng metric giữa replay và run gốc
rồi kết luận một trong ba:

```
Reproduced exactly.
Reproduced within 2.3e-05 — typically a library-version difference.
DID NOT reproduce: largest metric gap 0.0147.
```

## Vì sao cần repo

Hai lý do, cả hai đều cố ý:

1. `cleaning_spec.pkl` / `feature_spec.pkl` là pickle của class trong repo —
   unpickle cần class đó import được.
2. `--mode retrain` gọi thẳng `TrainModelAgent.replay_fit` → `_tool_train_final_model`,
   tức **đúng hàm** run gốc đã dùng, không phải bản sao. Nếu viết lại logic
   refit-on-train+valid / multi-seed bagging / OOF calibration trong driver thì
   hai bản sẽ lệch nhau lúc nào không biết — mà đó đúng là thứ bundle này sinh
   ra để ngăn.

Đổi lại: phải mang repo theo, **đúng version**. Pin thư viện theo `versions`
trong `final_model.pkl` (driver tự cảnh báo khi lệch) — sklearn/LightGBM khác
version có thể đổi hành vi của model đã pickle.

## Chuẩn hoá đầu đọc (CSV / parquet / Excel / SQL)

Cùng một bảng đọc bằng reader khác nhau cho dtype khác nhau, và khác biệt đó
**đổi model** chứ không chỉ đổi cách hiển thị:

| Ca | CSV / parquet | SQL / pyarrow / Excel |
|---|---|---|
| Cột int có null | `float64` → token `'1.0'` | `Int64` → token `'1'` |
| Null trong cột chữ | `'nan'` / `'None'` | `'<NA>'` |
| Cột ngày | text (CSV) | `datetime64` → `to_numeric` ra epoch nanosecond |
| SQL NUMERIC | — | `decimal.Decimal` trong cột object |
| Categorical chứa số | `int64` | `category` → object → bị label-encode |

Hệ quả thật: label encoder dựng ra classes khác nhau, `get_dummies` sinh tên cột
khác nhau, và một cột ngày trở thành feature số khổng lồ hay bị bỏ qua tuỳ định
dạng.

`BaseAgent.normalize_loaded_frame` gộp hết về một dạng chuẩn ngay lúc load, nên
không component nào phía sau phải tự phòng vệ. Quy ước chọn theo `pd.read_csv`
vì CSV là định dạng ít metadata nhất — thứ giàu hơn quy về nó được, ngược lại
thì không. Mọi agent và pipeline đều load qua `BaseAgent.load_dataframe` nên
không thể đi vòng.

Chi phí: 0.05s cho 50k × 557 cột, và là no-op với frame vốn đã chuẩn.
Guard: `tests/test_reader_normalization.py` (4 reader × 8 kiểu cột × 5 phép
biến đổi, cộng mọi cặp fit-reader → replay-reader).

## Khi replay báo không tái lập được

Driver phân loại trường trước khi kết luận, vì so delta tuyệt đối trên mọi
trường số là vô nghĩa (`best_iteration` lệch 9 cây không cùng thang với AUC):

- **structural** (`final_train_rows`, `n_seeds`) — lệch là **fail cứng**, vì nó
  có nghĩa dữ liệu replay không phải dữ liệu đã train.
- **score** (auc/brier/ks/gini/...) — có dung sai; `<1e-9` là khớp tuyệt đối,
  `<1e-3` là khớp trong sai số thư viện.
- **info** (`best_iteration`) — báo nhưng không quyết định.

Khi fail cứng, driver đi ngược chuỗi và chỉ ra **chặng đầu tiên** sai, kèm cách
sửa — thay vì chỉ nói "final_train_rows differ":

```
  == why it did not reproduce ==
  * train: 4 rows after replaying the specs, original run had 6.
    The join found the right 6 rows, so the loss happened in cleaning.
    Check cleaning.row_ops in replay_manifest.json — a dedup or sample
    there behaves differently on this input.
```

Năm nguyên nhân nó phân biệt được: dòng input không có trong split đã ghi · key
join không unique (fan-out) · row op làm mất dòng lúc cleaning · feature spec
không dựng lại đủ cột · mọi thứ khớp nhưng fit khác (→ chỉ sang version thư
viện). Guard: `tests/test_replay_diagnosis.py`.

## Những chỗ dễ sai

**Key trùng được xử lý bằng occurrence rank.** `split_assignment.parquet` join
theo `composite_key_cols` (hoặc `entity_id_col`) **cộng cột `_key_occ_`** — số
thứ tự của dòng trong nhóm cùng key, đánh trên file raw theo thứ tự file. Key
unique thì `_key_occ_` toàn 0 và phép join thoái hoá về join key thuần, không
phụ thuộc thứ tự dòng.

Cột này bắt buộc phải có, vì một run có `drop_duplicates` thì **theo định nghĩa**
để lại nhiều dòng cùng key trong file raw — không có tiebreak thì join fan-out
và driver không biết bản sao nào là train. Với dòng trùng hoàn toàn thì chọn bản
nào cũng như nhau nên occurrence rank là đủ. Nếu các dòng cùng key mà **khác
nhau** thì đó là key sai: pipeline log cảnh báo lúc chạy, và bạn nên truyền
`--composite-key` đủ cột để unique. Nếu vẫn còn nhập nhằng sau tiebreak, driver
dừng và báo lỗi thay vì nhân bản dòng.

**Split mode (tự truyền `--valid`/`--oot`) chỉ replay được ở mức score.**
`replay_pipeline.py` nhận một file input, còn split mode có nhiều file; manifest
chỉ ghi `source_input` là file train. `--mode retrain` cho run kiểu này sẽ chỉ
dựng lại được phần train. Chế độ auto-split (single-file) mới là chế độ replay
toàn trình đầy đủ.

**Không có key thì rơi về vị trí dòng.** Lúc đó bundle đóng băng theo thứ tự
dòng và chỉ đúng nếu file input giữ nguyên số dòng và thứ tự. Driver in cảnh báo
và từ chối nếu số dòng lệch. Muốn chắc chắn thì truyền `--entity-id` /
`--composite-key` khi chạy pipeline.

**Thứ tự dòng trong partition có ý nghĩa.** `auto_split` trả về train đã bị
`train_test_split` xáo, và `valid` là valid_temporal nối valid_random — đều
không theo thứ tự file gốc. StratifiedKFold chia fold theo vị trí, nên nếu
replay dựng lại đúng tập dòng nhưng sai thứ tự thì `best_iteration` và metrics
sẽ lệch. Vì vậy `split_assignment.parquet` ghi cả cột `_split_pos_`, và driver
sort theo nó trước khi train.

**Row ops chỉ áp cho train.** `drop_duplicates`, `deduplicate_by_key`,
`stratified_sample` chạy trên train ở run gốc nhưng không nằm trong
`CleaningSpec.apply()` — áp chúng lên valid/oot sẽ xoá bớt dòng đánh giá và làm
đẹp metrics giả tạo. Chúng nằm riêng ở `spec.row_ops`, và driver chỉ gọi
`apply_row_ops` cho partition train.

**Script `pipeline_process_*.py` khác với bundle này.** Ba script đó replay
transform của từng agent một cách độc lập (dùng khi muốn áp transform lên dữ
liệu mới). `replay_pipeline.py` là thứ chạy lại *cả run*.
