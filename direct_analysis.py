import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Load the results
with open("results/direct_results.json", "r") as f:
    results = json.load(f)

# Convert to DataFrame for easier analysis
df = pd.DataFrame(results)

# Basic statistics
total_questions = len(df)
correct_answers = df["is_correct"].sum()
accuracy = correct_answers / total_questions

print(f"Total questions: {total_questions}")
print(f"Correct answers: {correct_answers}")
print(f"Overall accuracy: {accuracy:.2%}")

# Accuracy by hop count
hop_accuracy = df.groupby("hop_count")["is_correct"].agg(["sum", "count"])
hop_accuracy["accuracy"] = hop_accuracy["sum"] / hop_accuracy["count"]

print("\nAccuracy by hop count:")
for hop, data in hop_accuracy.iterrows():
    print(f"{hop}-hop questions: {data['sum']}/{data['count']} = {data['accuracy']:.2%}")

# Analyze error patterns
# Check for questions where the model gave an answer but was incorrect
wrong_answers = df[df["is_correct"] == False]
answered_wrong = wrong_answers[wrong_answers["model_answer"].notnull()]
no_answer = wrong_answers[wrong_answers["model_answer"].isnull()]

print(f"\nQuestions with wrong answers: {len(answered_wrong)}/{total_questions} = {len(answered_wrong)/total_questions:.2%}")
print(f"Questions with no answers: {len(no_answer)}/{total_questions} = {len(no_answer)/total_questions:.2%}")

# Analyze common sources of errors
if "sources" in df.columns:
    # Flatten the list of sources
    all_sources = [source for sublist in df["sources"].tolist() for source in sublist]
    source_counts = Counter(all_sources)
    
    # Check which sources appear most in incorrect answers
    incorrect_sources = [source for sublist in wrong_answers["sources"].tolist() for source in sublist]
    incorrect_source_counts = Counter(incorrect_sources)
    
    # Calculate error rate by source
    source_error_rates = {}
    for source, count in source_counts.items():
        incorrect_count = incorrect_source_counts.get(source, 0)
        source_error_rates[source] = incorrect_count / count
    
    print("\nError rates by source question:")
    for source, rate in sorted(source_error_rates.items(), key=lambda x: x[1], reverse=True):
        print(f"{source}: {rate:.2%}")

# Print some examples of wrong answers
print("\nExamples of wrong answers:")
for i, row in answered_wrong.head(5).iterrows():
    print(f"Question: {row['question']}")
    print(f"Expected answer: {row['expected_answer']}")
    print(f"Model answer: {row['model_answer']}")
    print("---")

# Visualize results
plt.figure(figsize=(10, 6))
bars = plt.bar(hop_accuracy.index, hop_accuracy["accuracy"])
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f"{hop_accuracy['accuracy'].iloc[i]:.2%}", 
             ha='center', va='bottom')

plt.xlabel("Number of Hops")
plt.ylabel("Accuracy")
plt.title("Model Accuracy by Number of Hops (Direct Approach)")
plt.ylim(0, 1.1)
plt.savefig("results/direct_accuracy_by_hops.png")
plt.close()

print("\nAnalysis complete. Visualization saved to results/direct_accuracy_by_hops.png")
