import json
import matplotlib.pyplot as plt

POSITIVE_WORDS = [
    "good",
    "great",
    "excellent",
    "amazing",
    "wonderful",
    "awesome",
    "fantastic",
    "love",
    "like",
]

NEGATIVE_WORDS = [
    "bad",
    "terrible",
    "awful",
    "horrible",
    "hate",
    "dislike",
    "worst",
    "boring",
]

def count_sentiment(text):
    lower_text = text.lower()
    pos_count = sum(lower_text.count(w) for w in POSITIVE_WORDS)
    neg_count = sum(lower_text.count(w) for w in NEGATIVE_WORDS)
    return pos_count, neg_count

def main():
    file_path = 'generated_texts.json'
    texts = []
    
    # Try reading as JSONL first
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    texts.append(data.get('generated_text', ''))
    except json.JSONDecodeError:
        # If that fails, try reading as a standard JSON array
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                texts = [item.get('generated_text', '') for item in data]
            else:
                print("Unknown JSON format")
                return

    pos_counts = []
    neg_counts = []
    
    for text in texts:
        p, n = count_sentiment(text)
        pos_counts.append(p)
        neg_counts.append(n)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(pos_counts, label='Positive Words', marker='o', linestyle='-', color='green', alpha=0.7)
    plt.plot(neg_counts, label='Negative Words', marker='x', linestyle='-', color='red', alpha=0.7)
    
    plt.title('Sentiment Word Counts per Generated Text')
    plt.xlabel('Sample Index')
    plt.ylabel('Word Count')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    output_file = 'sentiment_analysis.png'
    plt.savefig(output_file)
    print(f"Analysis complete. Chart saved to {output_file}")

if __name__ == "__main__":
    main()
