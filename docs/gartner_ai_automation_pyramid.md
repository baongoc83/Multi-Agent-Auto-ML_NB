# Gartner AI Automation Pyramid — Analytics Maturity & Data Scientist Role

> Mapping của Gartner Analytics Ascendancy Model sang mức độ tự động hóa AI và sự dịch chuyển vai trò Data Scientist trong banking / fraud context.

---

## 1. Pyramid tổng quan


---

## 2. Đọc pyramid theo 3 chiều

| Chiều | Ý nghĩa |
|---|---|
| **Ngang (độ rộng)** | Số lượng use case / số org đang ở tầng đó. Càng rộng = càng phổ biến |
| **Dọc (độ cao)** | Độ phức tạp kỹ thuật + giá trị business. Càng cao = càng khó nhưng ROI lớn |
| **% AI bên phải** | Tỷ lệ decision được AI đưa ra mà không cần human-in-the-loop |

Cùng lúc, cột bên trái cho thấy **vai trò Data Scientist dịch chuyển** từ BI maker (đáy) → AI architect (đỉnh).

---

## 3. Chi tiết từng tầng

### 3.1. Descriptive — Báo cáo & Dashboard

**Câu hỏi**: Chuyện gì đã xảy ra?

**% AI**: ~10% — chủ yếu human đọc data và rút insight, AI chỉ tham gia ở mức báo cáo auto-generated hoặc anomaly highlighting.

**DS role**: BI / dashboard maker — viết SQL, build report, maintain data warehouse.

**Banking context**:
- Financial statements, regulatory reports (Basel, IFRS, báo cáo NHNN)
- FCRM alert volume dashboards, daily NPL monitoring
- Branch performance tracking, customer segment reports
- Real-time transaction volume, fraud rate trends

**Mức trưởng thành điển hình ở banking VN**: hầu hết banks đã ở đây cho 80%+ use case. Bottleneck thường là data quality và schema consistency giữa các source.

---

### 3.2. Diagnostic — Phân tích nguyên nhân

**Câu hỏi**: Tại sao xảy ra?

**% AI**: ~30% — AI tham gia ở mức anomaly detection, correlation discovery; nhưng causal interpretation vẫn là human.

**DS role**: Analyst — statistical literacy, hypothesis testing, root cause framework.

**Banking context**:
- Tại sao alert volume tăng tuần này? → drill xuống segment, branch, channel
- Cohort analysis cho NPL recovery (MOB3/MOB6, DPD30+ buckets)
- Vintage analysis cho BNPL/credit card portfolios
- Kitagawa decomposition để tách *mix shift* vs *behavior shift*
- Shapley để quantify contribution của từng risk factor

**Sự khác biệt giữa BI analyst và DS ở tầng này**: BI analyst dừng ở *correlation* (X tương quan với Y); DS đi tới *causation* (X gây ra Y qua mechanism Z).

---

### 3.3. Predictive — ML modeling

**Câu hỏi**: Sẽ xảy ra gì?

**% AI**: ~60% — ML model là decision maker chính, human verify và set threshold.

**DS role**: ML engineer — feature engineering, model training, evaluation, deployment.

**Banking context**:
- PD/LGD/EAD scoring (logistic regression vẫn là regulatory standard vì explainability)
- Customer churn prediction (XGBoost, Random Forest)
- **Fraud detection** (LightGBM + GraphFrame + Kafka streaming — sweet spot cho SME fraud ở MB Bank)
- Early-warning system cho NPL
- Recovery probability scoring

**Đặc điểm**: đây là tầng *hầu hết banks đang đầu tư mạnh nhất hiện nay*. Stack tiêu chuẩn:
- PySpark cho data prep + feature engineering
- LightGBM / XGBoost cho tabular fraud / risk
- GraphFrame / GNN cho network analysis
- MLflow + Airflow cho orchestration
- Feature Store (Feast hoặc custom) cho consistency train/serve

---

### 3.4. Prescriptive — Khuyến nghị hành động

**Câu hỏi**: Nên làm gì?

**% AI**: ~85% — AI propose action; human chỉ approve / override khi cần.

**DS role**: Optimizer / OR (Operations Research) — kết hợp ML score với business constraints để output decision.

**Banking context**:
- Threshold optimization: với capacity N reviewer/ngày, threshold nào maximize loss prevented?
- Capacity allocation: phân bổ alert priority giữa các tier analyst
- Uplift modeling: ai sẽ phản ứng thế nào nếu mình block / call / không làm gì?
- Counterfactual reasoning: did the policy change thực sự giảm fraud, hay chỉ là trend chung?
- Next-best-action recommendation cho collection (call, SMS, escalate, settle)

**Đây là gap lớn nhất ở Vietnamese banks hiện tại**. Predictive đã làm tốt, nhưng output model không được nối tiếp lên Prescriptive layer. Threshold thường cố định và không thay đổi theo segment / time / capacity.

**Stack điển hình**:
- LP/IP solvers (PuLP, CVXPY, OR-Tools)
- Uplift modeling (causalml, EconML)
- Bayesian optimization cho parameter tuning
- A/B test framework để validate prescription

---

### 3.5. Autonomous — AI tự xử lý

**Câu hỏi**: AI tự ra quyết định và hành động.

**% AI**: 100% — không có human-in-the-loop ngoài trường hợp exception.

**DS role**: AI architect — design hệ thống agent, define guardrails, monitor behavior, không build model thủ công nữa.

**Banking context**:
- Auto-close low-risk alerts (RPA + ML score threshold)
- Continuous learning models (online learning với feedback loop)
- LLM agent đọc case description → query data → produce investigation report
- Multi-agent fraud investigation (CrewAI / AutoGen): rule extractor + scenario matcher + pattern analyst + report writer làm việc trên cùng case
- Auto-retraining khi detect drift, không cần human trigger

**Constraint thực tế trong banking**:
- Regulatory thường cấm full autonomy với high-stakes decision (block account, freeze fund)
- Operational risk: model failure có thể gây mass mis-action
- Customer experience: auto-block sai → complaint storm
- Audit requirement: mọi decision phải log lý do, data used, confidence

**Realistic deployment**: agent ở "decision support" mode (suggest action với reasoning), human approve final action. Hoặc apply full autonomy chỉ cho narrow scope rủi ro thấp (low-amount + high-confidence transactions).

---

## 4. Sự dịch chuyển vai trò Data Scientist

Khi org leo lên pyramid, DS không còn build model đơn lẻ mà chuyển sang build *hệ thống ra quyết định*.

| Tầng | DS Role | Skill stack cần |
|---|---|---|
| L1 Descriptive | BI / dashboard maker | SQL, BI tools, data modeling |
| L2 Diagnostic | Analyst | Statistics, hypothesis testing, root cause framework |
| L3 Predictive | ML engineer | ML algorithms, feature engineering, MLOps |
| L4 Prescriptive | Optimizer / OR | LP/IP, causal inference, uplift modeling, decision theory |
| L5 Autonomous | AI architect | Agent orchestration, RL, system design, guardrails |

**Lưu ý quan trọng**: không phải DS nào cũng cần leo lên L5. Skill mỗi tầng khác nhau — một L3 ML engineer giỏi không nhất thiết là một L5 AI architect giỏi. Specialization theo tầng là chiến lược nghề nghiệp hợp lý.

---

## 5. Vị trí tham chiếu cho banking VN

Mức trưởng thành điển hình của các bank ở VN hiện tại:

| Tầng | Mức phổ biến | Ghi chú |
|---|---|---|
| L1 Descriptive | 90-100% | Hầu như universal |
| L2 Diagnostic | 60-80% | DS team có sẵn nhưng không phải bank nào cũng đầu tư đủ |
| L3 Predictive | 30-50% | Top-tier banks (MB, VPB, TCB, VCB) đầu tư mạnh; mid-tier chỉ ở early stage |
| L4 Prescriptive | 5-15% | Rất ít, thường chỉ ở pilot |
| L5 Autonomous | <5% | Chủ yếu R&D, chưa scale production cho high-stakes use case |

**Bước tiến hợp lý từ L3 → L4**: build optimization layer trên top của existing ML models, không cần đào lại từ đầu. ROI cao, rủi ro thấp, regulatory-friendly.

**Bước nhảy L3 → L5 (skip L4)**: anti-pattern thường gặp. Agent / LLM hệ thống wrap rule cứng + LLM mà không có optimization brain → fancy nhưng business impact thấp.

---

*Document version: 1.0 — Reference cho strategic planning ML/AI roadmap.*
