import pandas as pd
import re

def clean_utterances(df):
    """Clean CHILDES utterances by removing annotations and standardizing text."""
    
    def clean_text(text):
        if pd.isna(text):
            return ""
        
        # Remove CHILDES annotations
        text = re.sub(r'\[.*?\]', '', text)  # Remove [brackets]
        text = re.sub(r'\(.*?\)', '', text)  # Remove (parentheses) 
        text = re.sub(r'<.*?>', '', text)   # Remove <angle brackets>
        text = re.sub(r'&-\w+', '', text)   # Remove &-uh, &-um etc
        text = re.sub(r'xxx', '', text)     # Remove xxx (unintelligible)
        text = re.sub(r'yyy', '', text)     # Remove yyy (phonological fragment)
        text = re.sub(r'\+\.\.\.', '', text) # Remove +... (incomplete)
        text = re.sub(r'@\w+', '', text)    # Remove @o, @b etc
        text = re.sub(r'\s+', ' ', text)    # Multiple spaces to single
        text = re.sub(r'[\x00-\x1F]*\d+_\d+[\x00-\x1F]*', '', text) # Remove random unicode items 
        
        return text.strip()
    
    # Clean utterances
    df['clean_utterance'] = df['utterance'].apply(clean_text)
    
    # Remove empty utterances
    df = df[df['clean_utterance'].str.len() > 0]
    
    # Add word count
    df['word_count'] = df['clean_utterance'].str.split().str.len()
    
    return df

def main():
    # Load data
    df = pd.read_csv('data/processed/utterances.csv')
    print(f"Original data: {len(df)} utterances")
    
    # Clean data
    df_clean = clean_utterances(df)
    print(f"After cleaning: {len(df_clean)} utterances")
    
    # Save cleaned data
    df_clean.to_csv('data/processed/utterances_clean.csv', index=False)
    print("Saved cleaned data to data/processed/utterances_clean.csv")
    
    # Show sample
    print("\nSample cleaned utterances:")
    print(df_clean[['clean_utterance', 'word_count', 'age_months']].head())

if __name__ == "__main__":
    main()