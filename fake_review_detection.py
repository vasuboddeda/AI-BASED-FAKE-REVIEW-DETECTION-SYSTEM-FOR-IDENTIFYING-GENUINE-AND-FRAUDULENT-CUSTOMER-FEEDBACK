"""
AI-Based Fake Review Detection System for Identifying Genuine and Fraudulent Customer Feedback
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

def generate_review_dataset(n_samples=1000):
    """Generate synthetic review dataset with genuine and fake reviews"""
    np.random.seed(42)
    
    # Genuine review templates
    genuine_templates = [
        "Great product! Works as expected. Highly recommended.",
        "Excellent quality and fast shipping. Very satisfied.",
        "Good value for money. Will buy again.",
        "Perfect! Exactly what I needed.",
        "Amazing product. Exceeded my expectations.",
        "Very good. Arrived on time and in perfect condition.",
        "Satisfied with my purchase. Good quality.",
        "Fantastic! Best purchase ever.",
        "Highly satisfied. Great customer service.",
        "Wonderful product. Worth every penny."
    ]
    
    # Fake review templates (suspicious patterns)
    fake_templates = [
        "BEST PRODUCT EVER!!! YOU MUST BUY NOW!!!",
        "5 stars 5 stars 5 stars 5 stars amazing amazing",
        "This is the best thing I have ever bought in my entire life forever",
        "Perfect perfect perfect perfect perfect perfect",
        "Buy now buy now buy now buy now",
        "AMAZING AMAZING AMAZING AMAZING AMAZING",
        "I love this product I love this product I love",
        "Best best best best best best best",
        "HIGHLY RECOMMEND HIGHLY RECOMMEND HIGHLY",
        "EXCELLENT EXCELLENT EXCELLENT EXCELLENT"
    ]
    
    reviews = []
    labels = []
    
    for _ in range(n_samples // 2):
        # Genuine reviews
        review = np.random.choice(genuine_templates)
        reviews.append(review)
        labels.append(0)  # 0 = Genuine
        
        # Fake reviews
        review = np.random.choice(fake_templates)
        reviews.append(review)
        labels.append(1)  # 1 = Fake
    
    df = pd.DataFrame({'review_text': reviews, 'label': labels})
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

def extract_linguistic_features(df):
    """Extract linguistic features from review text"""
    features = []
    
    for text in df['review_text']:
        # Basic features
        word_count = len(text.split())
        char_count = len(text)
        avg_word_length = char_count / word_count if word_count > 0 else 0
        
        # Capitalization features
        capital_count = sum(1 for c in text if c.isupper())
        capital_ratio = capital_count / char_count if char_count > 0 else 0
        
        # Punctuation features
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        # Repetition features
        words = text.split()
        unique_words = len(set(words))
        word_repetition_ratio = 1 - (unique_words / word_count) if word_count > 0 else 0
        
        # Sentiment indicators
        positive_words = sum(1 for w in words if w.lower() in ['excellent', 'amazing', 'perfect', 'best', 'great', 'wonderful', 'fantastic'])
        
        features.append({
            'word_count': word_count,
            'char_count': char_count,
            'avg_word_length': avg_word_length,
            'capital_ratio': capital_ratio,
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'word_repetition_ratio': word_repetition_ratio,
            'positive_word_count': positive_words
        })
    
    features_df = pd.DataFrame(features)
    return features_df

def preprocess_reviews(df):
    """Preprocess review text and extract features"""
    # Extract linguistic features
    linguistic_features = extract_linguistic_features(df)
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
    tfidf_features = vectorizer.fit_transform(df['review_text'])
    
    # Combine features
    X = np.hstack([tfidf_features.toarray(), linguistic_features.values])
    y = df['label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, vectorizer, linguistic_features

def train_classifiers(X_train, X_test, y_train, y_test):
    """Train classification models"""
    results = {}
    
    # Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)[:, 1]
    results['Logistic Regression'] = {
        'model': lr,
        'predictions': y_pred_lr,
        'probabilities': y_proba_lr,
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'precision': precision_score(y_test, y_pred_lr),
        'recall': recall_score(y_test, y_pred_lr),
        'f1': f1_score(y_test, y_pred_lr),
        'roc_auc': roc_auc_score(y_test, y_proba_lr)
    }
    
    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    results['Random Forest'] = {
        'model': rf,
        'predictions': y_pred_rf,
        'probabilities': y_proba_rf,
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf),
        'recall': recall_score(y_test, y_pred_rf),
        'f1': f1_score(y_test, y_pred_rf),
        'roc_auc': roc_auc_score(y_test, y_proba_rf)
    }
    
    # Gradient Boosting
    print("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    y_proba_gb = gb.predict_proba(X_test)[:, 1]
    results['Gradient Boosting'] = {
        'model': gb,
        'predictions': y_pred_gb,
        'probabilities': y_proba_gb,
        'accuracy': accuracy_score(y_test, y_pred_gb),
        'precision': precision_score(y_test, y_pred_gb),
        'recall': recall_score(y_test, y_pred_gb),
        'f1': f1_score(y_test, y_pred_gb),
        'roc_auc': roc_auc_score(y_test, y_proba_gb)
    }
    
    return results, y_test

def plot_label_distribution(df):
    """Plot distribution of genuine vs fake reviews"""
    label_counts = df['label'].value_counts()
    
    plt.figure(figsize=(10, 6))
    colors = ['#2ecc71', '#e74c3c']
    bars = plt.bar(['Genuine Reviews', 'Fake Reviews'], [label_counts[0], label_counts[1]], 
                    color=colors, edgecolor='black', alpha=0.7, width=0.6)
    plt.ylabel('Number of Reviews', fontsize=12)
    plt.title('Distribution of Genuine and Fake Reviews', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/review_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Review distribution plot saved")

def plot_model_comparison(results):
    """Compare model performance"""
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        values = [results[m][metric] for m in models]
        bars = ax.bar(models, values, color=['#3498db', '#e74c3c', '#f39c12'], alpha=0.7, edgecolor='black')
        ax.set_ylabel(metric.capitalize(), fontsize=11)
        ax.set_title(f'{metric.capitalize()} Comparison', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)
    
    # Remove extra subplot
    fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Model comparison plot saved")

def plot_confusion_matrices(results, y_test):
    """Plot confusion matrices for all models"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    for idx, (model_name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['predictions'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Genuine', 'Fake'], yticklabels=['Genuine', 'Fake'],
                   cbar_kws={'label': 'Count'})
        axes[idx].set_xlabel('Predicted', fontsize=11)
        axes[idx].set_ylabel('Actual', fontsize=11)
        axes[idx].set_title(f'Confusion Matrix - {model_name}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Confusion matrices plot saved")

def plot_roc_curves(results, y_test):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    
    colors = ['#3498db', '#e74c3c', '#f39c12']
    
    for idx, (model_name, result) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
        auc_score = result['roc_auc']
        
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})',
                linewidth=2.5, color=colors[idx])
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ ROC curves plot saved")

def plot_feature_importance(results):
    """Plot feature importance for tree-based models"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Random Forest
    rf_model = results['Random Forest']['model']
    rf_importance = rf_model.feature_importances_[:20]
    axes[0].barh(range(len(rf_importance)), rf_importance, color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Importance', fontsize=11)
    axes[0].set_title('Feature Importance - Random Forest', fontsize=12, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Gradient Boosting
    gb_model = results['Gradient Boosting']['model']
    gb_importance = gb_model.feature_importances_[:20]
    axes[1].barh(range(len(gb_importance)), gb_importance, color='#f39c12', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Importance', fontsize=11)
    axes[1].set_title('Feature Importance - Gradient Boosting', fontsize=12, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Feature importance plot saved")

def main():
    print("="*80)
    print("AI-BASED FAKE REVIEW DETECTION SYSTEM")
    print("="*80)
    
    # Step 1: Generate dataset
    print("\n[Step 1] Generating review dataset...")
    df = generate_review_dataset(1000)
    df.to_csv('/home/ubuntu/reviews.csv', index=False)
    print(f"✓ Dataset generated: {len(df)} reviews")
    
    # Step 2: Extract features
    print("\n[Step 2] Extracting linguistic features...")
    X_train, X_test, y_train, y_test, vectorizer, features = preprocess_reviews(df)
    print(f"✓ Features extracted: {X_train.shape[1]} total features")
    
    # Step 3: Train models
    print("\n[Step 3] Training classification models...")
    results, y_test = train_classifiers(X_train, X_test, y_train, y_test)
    
    # Step 4: Evaluate models
    print("\n[Step 4] Evaluating models...")
    print("\nModel Performance:")
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {result['accuracy']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall:    {result['recall']:.4f}")
        print(f"  F1-Score:  {result['f1']:.4f}")
        print(f"  ROC-AUC:   {result['roc_auc']:.4f}")
    
    # Step 5: Generate visualizations
    print("\n[Step 5] Generating visualizations...")
    plot_label_distribution(df)
    plot_model_comparison(results)
    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)
    plot_feature_importance(results)
    
    print("\n" + "="*80)
    print("FAKE REVIEW DETECTION SYSTEM ANALYSIS COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()
