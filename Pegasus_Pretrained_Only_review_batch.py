import pandas as pd
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
from tqdm import tqdm

# Check if CUDA is available
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load pre-trained Pegasus model and tokenizer
model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)
model.to(device)

# Load original review data
df = pd.read_csv("E-Commerce Reviews.csv")

review_texts = df['Review Text'].tolist()

# Define batch size
batch_size = 32

# Generate fake review text in batches
generated_reviews = []
for i in tqdm(range(0, len(review_texts), batch_size), desc="Generating Fake Reviews"):
    batch_reviews = review_texts[i:i+batch_size]  # Ensure batch_reviews is a list of strings
    batch_reviews = [str(review) for review in batch_reviews]  # Convert each review to string
    input_ids = tokenizer(batch_reviews, return_tensors='pt', padding=True, truncation=True).input_ids.to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)  # Create attention mask
    max_length = max(len(review.split()) for review in batch_reviews) + 500  # Adjust max_length based on your needs
    outputs = model.generate(input_ids, max_length=max_length, num_beams=5, temperature=0.7, length_penalty=0.6, no_repeat_ngram_size=2, attention_mask=attention_mask)
    batch_fake_reviews = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    generated_reviews.extend(batch_fake_reviews)

# Print and save generated fake reviews
df_fake_reviews = pd.DataFrame({'Original Review': review_texts, 'Generated Fake Review': generated_reviews})
df_fake_reviews.to_csv("Pegasus_Generated_Reviews.csv", index=False)

print("Generated Fake Reviews saved to 'Pegasus_Generated_Reviews.csv'.")

