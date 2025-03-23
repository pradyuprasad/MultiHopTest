import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Load both result files
try:
    with open("results/direct_results.json", "r") as f:
        direct_results = json.load(f)
    direct_df = pd.DataFrame(direct_results)
    print(f"Loaded {len(direct_df)} direct results")
except FileNotFoundError:
    print("Direct results file not found")
    direct_df = pd.DataFrame()

try:
    with open("results/reasoning_results.json", "r") as f:
        reasoning_results = json.load(f)
    reasoning_df = pd.DataFrame(reasoning_results)
    print(f"Loaded {len(reasoning_df)} reasoning results")
except FileNotFoundError:
    print("Reasoning results file not found")
    reasoning_df = pd.DataFrame()

# Analyze each dataset by hop count
print("\n=== Analysis by Hop Count ===")

# Get all hop counts from both datasets
all_hops = set()
if not direct_df.empty:
    all_hops.update(direct_df['hop_count'].unique())
if not reasoning_df.empty:
    all_hops.update(reasoning_df['hop_count'].unique())

# Process direct results
direct_by_hop = {}
if not direct_df.empty:
    direct_hop_df = direct_df.groupby('hop_count')['is_correct'].agg(['count', 'sum'])
    direct_hop_df['accuracy'] = direct_hop_df['sum'] / direct_hop_df['count']
    
    print("\nDirect Results by Hop Count:")
    for hop, data in direct_hop_df.iterrows():
        direct_by_hop[hop] = {
            'count': int(data['count']), 
            'sum': int(data['sum']), 
            'accuracy': data['accuracy']
        }
        print(f"{hop}-hop questions: {int(data['sum'])}/{int(data['count'])} = {data['accuracy']:.2%}")

# Process reasoning results
reasoning_by_hop = {}
if not reasoning_df.empty:
    reasoning_hop_df = reasoning_df.groupby('hop_count')['is_correct'].agg(['count', 'sum'])
    reasoning_hop_df['accuracy'] = reasoning_hop_df['sum'] / reasoning_hop_df['count']
    
    print("\nReasoning Results by Hop Count:")
    for hop, data in reasoning_hop_df.iterrows():
        reasoning_by_hop[hop] = {
            'count': int(data['count']), 
            'sum': int(data['sum']), 
            'accuracy': data['accuracy']
        }
        print(f"{hop}-hop questions: {int(data['sum'])}/{int(data['count'])} = {data['accuracy']:.2%}")

# Compare by hop count (for all hops, handling partial data)
if not direct_df.empty or not reasoning_df.empty:
    print("\nComparison by Hop Count (includes partial data):")
    
    for hop in sorted(all_hops):
        print(f"{hop}-hop questions:")
        
        # Direct data
        if hop in direct_by_hop:
            direct_data = direct_by_hop[hop]
            print(f"  Direct:    {direct_data['sum']}/{direct_data['count']} = {direct_data['accuracy']:.2%}")
        else:
            print(f"  Direct:    No data available")
        
        # Reasoning data
        if hop in reasoning_by_hop:
            reasoning_data = reasoning_by_hop[hop]
            print(f"  Reasoning: {reasoning_data['sum']}/{reasoning_data['count']} = {reasoning_data['accuracy']:.2%}")
        else:
            print(f"  Reasoning: No data available")
        
        # Delta (only if both datasets have this hop)
        if hop in direct_by_hop and hop in reasoning_by_hop:
            delta = reasoning_by_hop[hop]['accuracy'] - direct_by_hop[hop]['accuracy']
            print(f"  Delta:     {delta:.2%} {'(reasoning better)' if delta > 0 else '(direct better)'}")
        else:
            print(f"  Delta:     Cannot calculate (missing data)")
        print()

# Analyze questions that appear in both datasets
if not direct_df.empty and not reasoning_df.empty:
    print("\n=== Questions Present in Both Datasets ===")
    
    # Find common IDs
    direct_ids = set(direct_df['id'])
    reasoning_ids = set(reasoning_df['id'])
    common_ids = direct_ids.intersection(reasoning_ids)
    
    print(f"Questions in direct dataset: {len(direct_ids)}")
    print(f"Questions in reasoning dataset: {len(reasoning_ids)}")
    print(f"Questions in both datasets: {len(common_ids)}")
    
    # Filter to common questions
    direct_common = direct_df[direct_df['id'].isin(common_ids)]
    reasoning_common = reasoning_df[reasoning_df['id'].isin(common_ids)]
    
    # Calculate accuracy on common questions
    direct_correct = direct_common["is_correct"].sum()
    direct_accuracy = direct_correct / len(direct_common)
    
    reasoning_correct = reasoning_common["is_correct"].sum()
    reasoning_accuracy = reasoning_correct / len(reasoning_common)
    
    delta_accuracy = reasoning_accuracy - direct_accuracy
    
    print(f"\nOn common questions only:")
    print(f"Direct accuracy: {direct_correct}/{len(direct_common)} = {direct_accuracy:.2%}")
    print(f"Reasoning accuracy: {reasoning_correct}/{len(reasoning_common)} = {reasoning_accuracy:.2%}")
    print(f"Overall delta (reasoning - direct): {delta_accuracy:.2%}")
    
    # Analyze by hop count for common questions
    common_by_hop = {}
    for idx in common_ids:
        hop = direct_df[direct_df['id'] == idx]['hop_count'].iloc[0]
        if hop not in common_by_hop:
            common_by_hop[hop] = {
                'count': 0,
                'direct_correct': 0,
                'reasoning_correct': 0
            }
        
        common_by_hop[hop]['count'] += 1
        if direct_df[direct_df['id'] == idx]['is_correct'].iloc[0]:
            common_by_hop[hop]['direct_correct'] += 1
        if reasoning_df[reasoning_df['id'] == idx]['is_correct'].iloc[0]:
            common_by_hop[hop]['reasoning_correct'] += 1
    
    print("\nCommon Questions by Hop Count:")
    for hop, data in sorted(common_by_hop.items()):
        direct_acc = data['direct_correct'] / data['count']
        reasoning_acc = data['reasoning_correct'] / data['count']
        delta = reasoning_acc - direct_acc
        
        print(f"{hop}-hop questions ({data['count']} total):")
        print(f"  Direct:    {data['direct_correct']}/{data['count']} = {direct_acc:.2%}")
        print(f"  Reasoning: {data['reasoning_correct']}/{data['count']} = {reasoning_acc:.2%}")
        print(f"  Delta:     {delta:.2%} {'(reasoning better)' if delta > 0 else '(direct better)'}")
        print()

# Analyze differing outcomes (only for questions in both datasets)
if not direct_df.empty and not reasoning_df.empty and len(common_ids) > 0:
    print("\n=== Differing Outcomes Analysis ===")
    
    # Find questions where approaches differ
    differing_by_hop = {}
    direct_better_by_hop = {}
    reasoning_better_by_hop = {}
    
    direct_index = direct_df.set_index('id')
    reasoning_index = reasoning_df.set_index('id')
    
    for idx in common_ids:
        hop_count = direct_index.loc[idx, 'hop_count']
        
        direct_correct = direct_index.loc[idx, 'is_correct']
        reasoning_correct = reasoning_index.loc[idx, 'is_correct']
        
        if direct_correct != reasoning_correct:
            # Initialize counters if needed
            if hop_count not in differing_by_hop:
                differing_by_hop[hop_count] = 0
                direct_better_by_hop[hop_count] = 0
                reasoning_better_by_hop[hop_count] = 0
            
            differing_by_hop[hop_count] += 1
            
            if direct_correct and not reasoning_correct:
                direct_better_by_hop[hop_count] += 1
            elif reasoning_correct and not direct_correct:
                reasoning_better_by_hop[hop_count] += 1
    
    # Collect examples for each category
    direct_only_correct = []
    reasoning_only_correct = []
    
    for idx in common_ids:
        direct_correct = direct_index.loc[idx, 'is_correct']
        reasoning_correct = reasoning_index.loc[idx, 'is_correct']
        
        if direct_correct and not reasoning_correct:
            direct_only_correct.append({
                'id': idx,
                'question': direct_index.loc[idx, 'question'],
                'expected_answer': direct_index.loc[idx, 'expected_answer'],
                'direct_answer': direct_index.loc[idx, 'model_answer'],
                'reasoning_answer': reasoning_index.loc[idx, 'model_answer'],
                'hop_count': direct_index.loc[idx, 'hop_count']
            })
        elif reasoning_correct and not direct_correct:
            reasoning_only_correct.append({
                'id': idx,
                'question': direct_index.loc[idx, 'question'],
                'expected_answer': direct_index.loc[idx, 'expected_answer'],
                'direct_answer': direct_index.loc[idx, 'model_answer'],
                'reasoning_answer': reasoning_index.loc[idx, 'model_answer'],
                'hop_count': direct_index.loc[idx, 'hop_count']
            })
    
    print(f"Total questions with different outcomes: {sum(differing_by_hop.values())}")
    print(f"Direct correct but reasoning wrong: {len(direct_only_correct)}")
    print(f"Reasoning correct but direct wrong: {len(reasoning_only_correct)}")
    
    print("\nDifferences by Hop Count:")
    for hop in sorted(differing_by_hop.keys()):
        total_diff = differing_by_hop[hop]
        direct_better = direct_better_by_hop.get(hop, 0)
        reasoning_better = reasoning_better_by_hop.get(hop, 0)
        
        print(f"{hop}-hop questions: {total_diff} differences")
        print(f"  Direct better: {direct_better}")
        print(f"  Reasoning better: {reasoning_better}")
        print(f"  Net advantage: {reasoning_better - direct_better} to {'reasoning' if reasoning_better > direct_better else 'direct'}")
    
    # Show examples of questions where approaches differ
    if reasoning_only_correct:
        print("\nExamples where reasoning was correct but direct was wrong:")
        for i, q in enumerate(reasoning_only_correct[:3]):
            print(f"Example {i+1} ({q['hop_count']}-hop):")
            print(f"Question: {q['question']}")
            print(f"Expected: {q['expected_answer']}")
            print(f"Direct answer: {q['direct_answer']} (wrong)")
            print(f"Reasoning answer: {q['reasoning_answer']} (correct)")
            print("---")
    
    if direct_only_correct:
        print("\nExamples where direct was correct but reasoning was wrong:")
        for i, q in enumerate(direct_only_correct[:3]):
            print(f"Example {i+1} ({q['hop_count']}-hop):")
            print(f"Question: {q['question']}")
            print(f"Expected: {q['expected_answer']}")
            print(f"Direct answer: {q['direct_answer']} (correct)")
            print(f"Reasoning answer: {q['reasoning_answer']} (wrong)")
            print("---")

# === Create visualizations ===
print("\n=== Creating Visualizations ===")

plt.style.use('ggplot')
plt.figure(figsize=(12, 10))

# 1. Bar chart comparing accuracy by hop count
if not direct_df.empty or not reasoning_df.empty:
    plt.subplot(2, 2, 1)
    
    sorted_hops = sorted(all_hops)
    x = np.arange(len(sorted_hops))
    width = 0.35
    
    direct_accs = [direct_by_hop.get(hop, {}).get('accuracy', 0) for hop in sorted_hops]
    reasoning_accs = [reasoning_by_hop.get(hop, {}).get('accuracy', 0) for hop in sorted_hops]
    
    bars1 = plt.bar(x - width/2, direct_accs, width, label='Direct')
    bars2 = plt.bar(x + width/2, reasoning_accs, width, label='Reasoning')
    
    plt.xlabel('Hop Count')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Hop Count')
    plt.xticks(x, sorted_hops)
    plt.ylim(0, 1.0)
    plt.legend()
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            plt.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            plt.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)

# 2. Performance difference
if not direct_df.empty and not reasoning_df.empty and len(common_ids) > 0:
    plt.subplot(2, 2, 2)
    
    # Get common hops (those in both datasets)
    common_hops = sorted([hop for hop in all_hops 
                         if hop in direct_by_hop and hop in reasoning_by_hop])
    x = np.arange(len(common_hops))
    
    # Calculate deltas
    deltas = [reasoning_by_hop[hop]['accuracy'] - direct_by_hop[hop]['accuracy'] 
              for hop in common_hops]
    
    # Create bars with colors based on which is better
    colors = ['green' if delta > 0 else 'red' for delta in deltas]
    bars = plt.bar(x, deltas, color=colors)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Hop Count')
    plt.ylabel('Accuracy Difference (Reasoning - Direct)')
    plt.title('Performance Difference by Hop Count')
    plt.xticks(x, common_hops)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3 if height >= 0 else -12),
                     textcoords="offset points",
                     ha='center', va='bottom' if height >= 0 else 'top',
                     fontsize=9)

# 3. Differing outcomes analysis
if not direct_df.empty and not reasoning_df.empty and len(common_ids) > 0 and differing_by_hop:
    plt.subplot(2, 2, 3)
    
    # Get all hops with differences
    difference_hops = sorted(differing_by_hop.keys())
    x = np.arange(len(difference_hops))
    width = 0.35
    
    # Extract counts by hop
    direct_better_counts = [direct_better_by_hop.get(hop, 0) for hop in difference_hops]
    reasoning_better_counts = [reasoning_better_by_hop.get(hop, 0) for hop in difference_hops]
    
    plt.bar(x - width/2, direct_better_counts, width, label='Direct Better')
    plt.bar(x + width/2, reasoning_better_counts, width, label='Reasoning Better')
    
    plt.xlabel('Hop Count')
    plt.ylabel('Number of Questions')
    plt.title('Questions with Different Outcomes by Hop Count')
    plt.xticks(x, difference_hops)
    plt.legend()

# 4. Overall performance comparison pie chart
if not direct_df.empty and not reasoning_df.empty and len(common_ids) > 0:
    plt.subplot(2, 2, 4)
    
    # Calculate overall counts
    both_correct = sum(1 for idx in common_ids 
                      if direct_index.loc[idx, 'is_correct'] and reasoning_index.loc[idx, 'is_correct'])
    both_wrong = sum(1 for idx in common_ids 
                    if not direct_index.loc[idx, 'is_correct'] and not reasoning_index.loc[idx, 'is_correct'])
    direct_only = len(direct_only_correct)
    reasoning_only = len(reasoning_only_correct)
    
    labels = ['Both Correct', 'Both Wrong', 'Only Direct Correct', 'Only Reasoning Correct']
    sizes = [both_correct, both_wrong, direct_only, reasoning_only]
    colors = ['forestgreen', 'lightcoral', 'royalblue', 'gold']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Question Outcome Distribution')

plt.tight_layout()
plt.savefig('comparison_analysis.png', dpi=300)
plt.show()

print("Visualizations saved to 'comparison_analysis.png'")
print("\nAnalysis complete.")
