X = df.drop(columns=[target_col])
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_results = {}

xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss', verbosity=0)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
model_results['XGBoost'] = {'roc_auc_score': float(roc_auc_score(y_test, y_pred)), 'accuracy': float(accuracy_score(y_test, y_pred)), 'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))}

rf = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
model_results['RandomForest'] = {'roc_auc_score': float(roc_auc_score(y_test, y_pred)), 'accuracy': float(accuracy_score(y_test, y_pred)), 'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))}

et = ExtraTreesClassifier(n_estimators=200, max_depth=7, random_state=42)
et.fit(X_train, y_train)
y_pred = et.predict(X_test)
model_results['ExtraTrees'] = {'roc_auc_score': float(roc_auc_score(y_test, y_pred)), 'accuracy': float(accuracy_score(y_test, y_pred)), 'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))}

if LGBMClassifier is not None:
    lgbm = LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    lgbm.fit(X_train, y_train)
    y_pred = lgbm.predict(X_test)
    model_results['LightGBM'] = {'roc_auc_score': float(roc_auc_score(y_test, y_pred)), 'accuracy': float(accuracy_score(y_test, y_pred)), 'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))}

best_model_name = max(model_results, key=lambda m: model_results[m].get('roc_auc_score', -999))
metrics = {'best_model': best_model_name, **model_results[best_model_name], 'model_comparison': model_results}