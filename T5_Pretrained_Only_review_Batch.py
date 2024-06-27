import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from tqdm import tqdm

# Load pre-trained T5 model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)
# Load original review data
df = pd.read_csv("E-Commerce Reviews.csv")
#df = df[:25000]
review_texts = df['Review Text'].tolist()

# Define batch size
batch_size = 32

# Generate fake review text in batches
generated_reviews = []
for i in tqdm(range(0, len(review_texts), batch_size), desc="Generating Fake Reviews"):
    batch_reviews = review_texts[i:i+batch_size]
    input_texts = [f"generate fake review: {review}" for review in batch_reviews]
    input_ids = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    max_length = 256  # Adjust the length as needed
    outputs = model.generate(input_ids, max_length=max_length, num_beams=5, temperature=0.7)
    batch_fake_reviews = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    generated_reviews.extend(batch_fake_reviews)

# Save generated fake reviews to a CSV file
df_fake_reviews = pd.DataFrame({'Original Review': review_texts, 'Generated Fake Review': generated_reviews})
df_fake_reviews.to_csv("T5_Pretrained_only_review_batch.csv", index=False)

