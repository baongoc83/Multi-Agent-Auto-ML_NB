# Prepare data
X = df.drop(columns=[target_col])
y = df[target_col]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    'roc_auc_score': float(roc_auc_score(y_test, y_pred_proba)),
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
    'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
    'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
}