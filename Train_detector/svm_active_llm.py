import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import re
import os

def extract_semantic_features(text):
    """Extract semantic features from log messages"""
    features = {
        'has_error': int(any(word in text.lower() for word in ['error', 'fail', 'failed', 'fatal'])),
        'has_warning': int(any(word in text.lower() for word in ['warn', 'warning'])),
        'has_info': int(any(word in text.lower() for word in ['info', 'information'])),
        'message_length': len(text.split()),
        'has_numeric': int(bool(re.search(r'\d+', text))),
        'has_path': int('/' in text),
        'has_ip': int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', text))),
        'has_port': int(bool(re.search(r'port \d+', text.lower())))
    }
    return features

def preprocess_log(text):
    """Enhanced preprocessing"""
    text = str(text)
    

    semantic_features = extract_semantic_features(text)
    

    text = text.replace(',', ' ')
    text = re.sub(r'\d{10}', 'TIMESTAMP', text)
    text = re.sub(r'\d{4}\.\d{2}\.\d{2}', 'DATE', text)
    text = re.sub(r'\d{2}:\d{2}:\d{2}', 'TIME', text)
    text = re.sub(r'(?<=\[)\d+(?=\])', 'ID', text)

    text = re.sub(r'port \d+', 'PORT_NUM', text)
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', 'IP_ADDR', text)
    text = re.sub(r'\/[\w\-\.\/]+\/', 'PATH/', text)
    
    text = re.sub(r'\d+', 'NUM', text)

    text = ' '.join(text.split())
    
    return text.lower(), semantic_features

def load_and_process_data(train_file, old_test_file='testing_logs1.csv', new_test_file='testing_logs2.csv'):
    """Load and prepare all datasets with enhanced feature extraction"""
    print("Loading datasets...")
    

    train_df = pd.read_csv(train_file)
    
    old_test_df = pd.read_csv(old_test_file)
    new_test_df = pd.read_csv(new_test_file)
    
    print("\nPreprocessing logs...")
    
    train_processed = [preprocess_log(log) for log in train_df['remaining_log']]
    train_texts = [t[0] for t in train_processed]
    train_semantic = pd.DataFrame([t[1] for t in train_processed])
    
    old_test_processed = [preprocess_log(log) for log in old_test_df['log']]
    old_test_texts = [t[0] for t in old_test_processed]
    old_test_semantic = pd.DataFrame([t[1] for t in old_test_processed])
    
    new_test_processed = [preprocess_log(log) for log in new_test_df['log_content']]
    new_test_texts = [t[0] for t in new_test_processed]
    new_test_semantic = pd.DataFrame([t[1] for t in new_test_processed])
    

    train_labels = train_df['expanded_label']  
    old_test_labels = (old_test_df['Label'] != '-').astype(int) 
    new_test_labels = (new_test_df['label'] != '-').astype(int)  
    
    print("\nLabel distributions:")
    print("Training:", dict(train_labels.value_counts()))
    print("Old Test:", dict(old_test_labels.value_counts()))
    print("New Test:", dict(new_test_labels.value_counts()))
    
    # TF-IDF features
    print("\nExtracting text features...")
    vectorizer = TfidfVectorizer(
        max_features=200,
        ngram_range=(1, 1),
        stop_words='english',
        max_df=0.8,
        min_df=10
    )
    
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_old_test_tfidf = vectorizer.transform(old_test_texts)
    X_new_test_tfidf = vectorizer.transform(new_test_texts)
    
    print("\nScaling semantic features...")
    scaler = StandardScaler()
    train_semantic_scaled = scaler.fit_transform(train_semantic)
    old_test_semantic_scaled = scaler.transform(old_test_semantic)
    new_test_semantic_scaled = scaler.transform(new_test_semantic)
    
    X_train = np.hstack([X_train_tfidf.toarray(), train_semantic_scaled])
    X_old_test = np.hstack([X_old_test_tfidf.toarray(), old_test_semantic_scaled])
    X_new_test = np.hstack([X_new_test_tfidf.toarray(), new_test_semantic_scaled])
    
    return (X_train, X_old_test, X_new_test, 
            train_labels, old_test_labels, new_test_labels)



def evaluate_with_cv(X_train, y_train):
    """Evaluate SVM model using cross-validation (simplified version)"""
    print("\nPerforming 5-fold cross-validation...")

    model = SVC(
        kernel='rbf',
        class_weight='balanced',
        C=0.1,
        probability=True,
        random_state=42
    )
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)

    print(f"\nCross-validation mean accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores)*2:.4f})")

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X, y, dataset_name=""):
    """Evaluate model on a test set"""
    print(f"\nEvaluating on {dataset_name}:")
    y_pred = model.predict(X)
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    
    print(f"\nSummary Metrics for {dataset_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    return accuracy, precision, recall, f1


def main():
    folder_path = "clustered_data_LLM"
    results = []

    csv_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".csv")],
        key=lambda x: int(x.split("_")[-1].split(".")[0]) 
    )

    for csv_file in csv_files:
        train_file = os.path.join(folder_path, csv_file)
        print(f"\nProcessing file: {csv_file}")

        X_train, X_old_test, X_new_test, y_train, y_old_test, y_new_test = load_and_process_data(train_file)

        model = evaluate_with_cv(X_train, y_train)

        old_test_results = evaluate_model(model, X_old_test, y_old_test, "Original Test Set")
        new_test_results = evaluate_model(model, X_new_test, y_new_test, "New Test Set")

        results.append([
            csv_file, len(y_train), *old_test_results, *new_test_results
        ])

    results_df = pd.DataFrame(results, columns=[
        "filename", "train_size",
        "old_test_acc", "old_test_prec", "old_test_recall", "old_test_f1",
        "new_test_acc", "new_test_prec", "new_test_recall", "new_test_f1"
    ])

    results_df = results_df.sort_values(by="train_size", ascending=True)

    results_df.to_csv("svm_test_results_LLM_label.csv", index=False)
    print("\nAll results saved to svm_test_results_LLM_label.csv")


if __name__ == "__main__":
    model = main()