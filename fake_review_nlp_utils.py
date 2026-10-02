"""
NLP Analysis Utilities for Fake Review Detection System
"""

import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def analyze_review_text_features(csv_file):
    """Analyze text features of reviews"""
    df = pd.read_csv(csv_file)
    
    features = []
    for idx, row in df.iterrows():
        text = row['review_text']
        label = row['label']
        
        words = text.split()
        chars = len(text)
        
        feature_dict = {
            'review_id': idx + 1,
            'label': 'Fake' if label == 1 else 'Genuine',
            'word_count': len(words),
            'char_count': chars,
            'avg_word_length': chars / len(words) if len(words) > 0 else 0,
            'capital_letters': sum(1 for c in text if c.isupper()),
            'capital_ratio': sum(1 for c in text if c.isupper()) / chars if chars > 0 else 0,
            'exclamation_marks': text.count('!'),
            'question_marks': text.count('?'),
            'unique_words': len(set(words)),
            'word_repetition': 1 - (len(set(words)) / len(words)) if len(words) > 0 else 0
        }
        features.append(feature_dict)
    
    features_df = pd.DataFrame(features)
    features_df.to_csv('/home/ubuntu/review_text_features.csv', index=False)
    print("✓ Review text features saved")
    
    return features_df

def analyze_by_label(csv_file):
    """Analyze reviews by genuine vs fake"""
    df = pd.read_csv(csv_file)
    features = analyze_review_text_features(csv_file)
    
    analysis = []
    for label in [0, 1]:
        label_name = 'Fake' if label == 1 else 'Genuine'
        label_features = features[features['label'] == label_name]
        
        analysis.append({
            'Review_Type': label_name,
            'Count': len(label_features),
            'Avg_Word_Count': label_features['word_count'].mean(),
            'Avg_Char_Count': label_features['char_count'].mean(),
            'Avg_Capital_Ratio': label_features['capital_ratio'].mean(),
            'Avg_Exclamation_Marks': label_features['exclamation_marks'].mean(),
            'Avg_Word_Repetition': label_features['word_repetition'].mean()
        })
    
    analysis_df = pd.DataFrame(analysis)
    analysis_df.to_csv('/home/ubuntu/review_analysis_by_type.csv', index=False)
    print("✓ Review analysis by type saved")
    
    return analysis_df

def extract_keywords_by_label(csv_file, top_n=10):
    """Extract top keywords for genuine and fake reviews"""
    df = pd.read_csv(csv_file)
    
    genuine_text = ' '.join(df[df['label'] == 0]['review_text'].values)
    fake_text = ' '.join(df[df['label'] == 1]['review_text'].values)
    
    # Extract keywords
    genuine_words = Counter(genuine_text.lower().split())
    fake_words = Counter(fake_text.lower().split())
    
    keywords_data = []
    
    # Top genuine keywords
    for word, count in genuine_words.most_common(top_n):
        keywords_data.append({
            'Review_Type': 'Genuine',
            'Keyword': word,
            'Frequency': count
        })
    
    # Top fake keywords
    for word, count in fake_words.most_common(top_n):
        keywords_data.append({
            'Review_Type': 'Fake',
            'Keyword': word,
            'Frequency': count
        })
    
    keywords_df = pd.DataFrame(keywords_data)
    keywords_df.to_csv('/home/ubuntu/review_keywords.csv', index=False)
    print("✓ Review keywords saved")
    
    return keywords_df

def analyze_linguistic_patterns(csv_file):
    """Analyze linguistic patterns in reviews"""
    df = pd.read_csv(csv_file)
    features = analyze_review_text_features(csv_file)
    
    patterns = []
    
    for label in [0, 1]:
        label_name = 'Fake' if label == 1 else 'Genuine'
        label_features = features[features['label'] == label_name]
        
        patterns.append({
            'Review_Type': label_name,
            'Avg_Sentence_Length': label_features['word_count'].mean(),
            'Max_Sentence_Length': label_features['word_count'].max(),
            'Min_Sentence_Length': label_features['word_count'].min(),
            'Avg_Capital_Ratio': label_features['capital_ratio'].mean(),
            'Max_Capital_Ratio': label_features['capital_ratio'].max(),
            'Avg_Punctuation': (label_features['exclamation_marks'] + label_features['question_marks']).mean(),
            'Avg_Word_Uniqueness': (1 - label_features['word_repetition']).mean()
        })
    
    patterns_df = pd.DataFrame(patterns)
    patterns_df.to_csv('/home/ubuntu/linguistic_patterns.csv', index=False)
    print("✓ Linguistic patterns saved")
    
    return patterns_df

def generate_detection_insights(csv_file):
    """Generate insights for fake review detection"""
    df = pd.read_csv(csv_file)
    features = analyze_review_text_features(csv_file)
    
    insights = []
    
    # Insight 1: Capital ratio difference
    genuine_cap = features[features['label'] == 'Genuine']['capital_ratio'].mean()
    fake_cap = features[features['label'] == 'Fake']['capital_ratio'].mean()
    
    insights.append({
        'Insight': 'Capital Letter Usage',
        'Genuine_Avg': f"{genuine_cap:.4f}",
        'Fake_Avg': f"{fake_cap:.4f}",
        'Difference': f"{abs(fake_cap - genuine_cap):.4f}",
        'Indicator': 'Fake reviews use more capital letters'
    })
    
    # Insight 2: Exclamation marks
    genuine_excl = features[features['label'] == 'Genuine']['exclamation_marks'].mean()
    fake_excl = features[features['label'] == 'Fake']['exclamation_marks'].mean()
    
    insights.append({
        'Insight': 'Exclamation Mark Usage',
        'Genuine_Avg': f"{genuine_excl:.4f}",
        'Fake_Avg': f"{fake_excl:.4f}",
        'Difference': f"{abs(fake_excl - genuine_excl):.4f}",
        'Indicator': 'Fake reviews use more exclamation marks'
    })
    
    # Insight 3: Word repetition
    genuine_rep = features[features['label'] == 'Genuine']['word_repetition'].mean()
    fake_rep = features[features['label'] == 'Fake']['word_repetition'].mean()
    
    insights.append({
        'Insight': 'Word Repetition',
        'Genuine_Avg': f"{genuine_rep:.4f}",
        'Fake_Avg': f"{fake_rep:.4f}",
        'Difference': f"{abs(fake_rep - genuine_rep):.4f}",
        'Indicator': 'Fake reviews have higher word repetition'
    })
    
    # Insight 4: Average word length
    genuine_wlen = features[features['label'] == 'Genuine']['avg_word_length'].mean()
    fake_wlen = features[features['label'] == 'Fake']['avg_word_length'].mean()
    
    insights.append({
        'Insight': 'Average Word Length',
        'Genuine_Avg': f"{genuine_wlen:.4f}",
        'Fake_Avg': f"{fake_wlen:.4f}",
        'Difference': f"{abs(fake_wlen - genuine_wlen):.4f}",
        'Indicator': 'Linguistic complexity differs'
    })
    
    insights_df = pd.DataFrame(insights)
    insights_df.to_csv('/home/ubuntu/detection_insights.csv', index=False)
    print("✓ Detection insights saved")
    
    return insights_df

def generate_authenticity_scores(csv_file):
    """Generate authenticity scores for reviews"""
    df = pd.read_csv(csv_file)
    features = analyze_review_text_features(csv_file)
    
    scores = []
    
    for idx, row in features.iterrows():
        # Calculate authenticity score (0-100, higher = more genuine)
        score = 100
        
        # Reduce score for high capital ratio
        if row['capital_ratio'] > 0.2:
            score -= (row['capital_ratio'] - 0.2) * 100
        
        # Reduce score for excessive exclamation marks
        if row['exclamation_marks'] > 3:
            score -= (row['exclamation_marks'] - 3) * 10
        
        # Reduce score for high word repetition
        if row['word_repetition'] > 0.3:
            score -= (row['word_repetition'] - 0.3) * 50
        
        # Ensure score is between 0 and 100
        score = max(0, min(100, score))
        
        # Determine authenticity level
        if score >= 70:
            authenticity = 'Likely Genuine'
        elif score >= 40:
            authenticity = 'Uncertain'
        else:
            authenticity = 'Likely Fake'
        
        scores.append({
            'Review_ID': idx + 1,
            'Authenticity_Score': f"{score:.2f}",
            'Authenticity_Level': authenticity,
            'Actual_Label': row['label']
        })
    
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv('/home/ubuntu/authenticity_scores.csv', index=False)
    print("✓ Authenticity scores saved")
    
    return scores_df

def main():
    print("="*80)
    print("FAKE REVIEW DETECTION - NLP ANALYSIS UTILITIES")
    print("="*80)
    
    csv_file = '/home/ubuntu/reviews.csv'
    
    print("\n[1] Analyzing review text features...")
    analyze_review_text_features(csv_file)
    
    print("\n[2] Analyzing reviews by type...")
    analyze_by_label(csv_file)
    
    print("\n[3] Extracting keywords...")
    extract_keywords_by_label(csv_file)
    
    print("\n[4] Analyzing linguistic patterns...")
    analyze_linguistic_patterns(csv_file)
    
    print("\n[5] Generating detection insights...")
    generate_detection_insights(csv_file)
    
    print("\n[6] Generating authenticity scores...")
    generate_authenticity_scores(csv_file)
    
    print("\n" + "="*80)
    print("NLP ANALYSIS COMPLETED - All datasets generated")
    print("="*80)

if __name__ == "__main__":
    main()
