import json
import matplotlib as pyplt
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

result_path = 'results_finetuned.json'

with open(result_path) as f:
    results = json.load(f)

# Convert to DataFrame for easier analysis
df = pd.DataFrame(results)

nltk.download('punkt_tab')
nltk.download('wordnet')
def calculate_metrics(row):
    # Tokenize the texts
    pred_tokens = nltk.word_tokenize(row['prediction'].lower())
    truth_tokens = nltk.word_tokenize(row['ground_truth'].lower())
    
    # Calculate BLEU score (using smoothing for short texts)
    smoothie = SmoothingFunction().method4
    bleu = corpus_bleu([[truth_tokens]], [pred_tokens], smoothing_function=smoothie)
    
    # Calculate METEOR score
    meteor = meteor_score([truth_tokens], pred_tokens)
    
    return pd.Series({'bleu_score': bleu, 'meteor_score': meteor})

# Apply the function to each row
df[['bleu_score', 'meteor_score']] = df.apply(calculate_metrics, axis=1)

# Descriptive statistics
stats = df[['rougel_score', 'bleu_score', 'meteor_score']].describe()
print("Descriptive Statistics:")
print(stats)

# Correlation between metrics
correlation = df[['rougel_score', 'bleu_score', 'meteor_score']].corr()
print("\nCorrelation Matrix:")
print(correlation)

# Set style
sns.set(style="whitegrid")

# Plot distributions
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(df['rougel_score'], bins=10, kde=True)
plt.title('ROUGE-L Score Distribution')

plt.subplot(1, 3, 2)
sns.histplot(df['bleu_score'], bins=10, kde=True)
plt.title('BLEU Score Distribution')

plt.subplot(1, 3, 3)
sns.histplot(df['meteor_score'], bins=10, kde=True)
plt.title('METEOR Score Distribution')

plt.tight_layout()
plt.show()

# Boxplot comparison
plt.figure(figsize=(10, 6))
df_melted = df.melt(value_vars=['rougel_score', 'bleu_score', 'meteor_score'], 
                    var_name='metric', value_name='score')
sns.boxplot(x='metric', y='score', data=df_melted)
plt.title('Comparison of Evaluation Metrics')
plt.show()

# Scatter plot matrix to show relationships
sns.pairplot(df[['rougel_score', 'bleu_score', 'meteor_score']])
plt.suptitle('Pairwise Relationships Between Metrics', y=1.02)
plt.show()
