import pandas as pd

# Load individual model responses
llama2_df = pd.read_csv("model_responses.csv")
mistral_df = pd.read_csv("model_responses(deepseek-llm).csv")
deepseek_df = pd.read_csv("model_responses(mistral).csv")

# Make sure each DataFrame has a 'model' column with the correct model name
# If they don't already have it, you can add it:
if 'model' not in llama2_df.columns:
    llama2_df['model'] = 'llama2'
if 'model' not in mistral_df.columns:
    mistral_df['model'] = 'mistral'
if 'model' not in deepseek_df.columns:
    deepseek_df['model'] = 'deepseek'

# Combine into a single DataFrame
combined_df = pd.concat([llama2_df, mistral_df, deepseek_df])

# Save combined responses
combined_df.to_csv("combined_model_responses.csv", index=False)

print("Combined model responses saved to 'combined_model_responses.csv'")