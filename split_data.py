import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed/utterances_clean.csv")

# Split data based on 20 percent test, 80 percent train
# And then split the training data to 10 percent dev, 90 percent train
train, test = train_test_split(df, test_size=0.2, random_state=42)
train, dev  = train_test_split(train, test_size=0.1, random_state=42)

# Place back into respective csv files
train.to_csv("data/split/train.csv", index=False)
dev.to_csv("data/split/dev.csv", index=False)
test.to_csv("data/split/test.csv", index=False)