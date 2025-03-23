import os
from dotenv import load_dotenv
import requests
import json
import itertools
import re
import time
from tqdm import tqdm  # For progress bar

load_dotenv()

api_key = os.environ['OPENROUTER_API_KEY']

# Convert the string directly to a list of questions
question_list = [
    "What is the exponent in Avogadro's number?",
    "What is the value of Pi, truncated to the one's place?",
    "How many sides does a hexagon have?",
    "What is the atomic number of Helium?",
    "How many standard playing cards are there in one suit?",
    "What is the square root of nine?",
    "How many hydrogen atoms are in a single water molecule?",
    "What is the result of five factorial?",
    "How many vertices does a tetrahedron have?",
    "How many degrees are there in a right angle?"
]

# Function to generate prompt for different number of questions
def create_prompt(questions):
    facts_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    
    prompt = f"""Generate a multi-hop reasoning question that requires knowing the answers to these facts:
{facts_text}

Think through this step-by-step and show your complete reasoning process:

1. First, determine the correct answer to each fact.
2. Consider different mathematical operations (addition, subtraction, multiplication, division) that could meaningfully combine these answers.
3. Create a clear multi-hop question that requires finding all answers and then performing the chosen operations.
4. Calculate the final numerical answer to this multi-hop question.

For example, if I select "How many sides does a square have?" and "How many sides does a dodecagon have?", your reasoning might be:
- A square has 4 sides
- A dodecagon has 12 sides
- I can add these: 4 + 12 = 16
- Multi-hop question: "What is the sum of the number of sides on a square and the number of sides on a dodecagon?"
- Answer: 16

Show your complete thought process for solving this. After you've finished your reasoning, conclude your response by formatting the final multi-hop question and its answer using XML tags as follows:

<question>Your multi-hop question here</question>
<answer>The correct numerical answer to the multi-hop question</answer>
"""
    return prompt

# Function to generate multi-hop question from a tuple of questions
def generate_multi_hop_question(questions):
    prompt = create_prompt(questions)

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            data=json.dumps({
                "model": "google/gemini-2.0-flash-lite-001",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            }),
            timeout=(30, 120)  # 30 seconds to connect, 120 seconds to receive response
        )
        
        full_output = response.json()["choices"][0]["message"]["content"]
        
        # Extract question and answer using regex
        question_match = re.search(r'<question>(.*?)</question>', full_output, re.DOTALL)
        answer_match = re.search(r'<answer>(.*?)</answer>', full_output, re.DOTALL)
        
        if question_match and answer_match:
            question = question_match.group(1).strip()
            answer = answer_match.group(1).strip()
            extracted = {
                "question": question,
                "answer": answer,
                "full_response": full_output
            }
            print(f"Extracted Q: {question}")
            print(f"Extracted A: {answer}")
        else:
            extracted = {
                "error": "Could not extract question and answer",
                "full_response": full_output
            }
            print("Failed to extract Q&A from response")
        
    except Exception as e:
        extracted = {
            "error": f"Error: {str(e)}",
            "full_response": "Error occurred during API call"
        }
        print(f"Error: {str(e)}")
        print(response.json())
    
    # Add a small delay to avoid rate limiting
    time.sleep(1)
    
    return extracted

# Function to process combinations of a specific size
def process_combinations(size):
    print(f"\nGenerating multi-hop questions for {size}-way combinations")
    
    # Generate all possible combinations of the specified size
    question_combos = list(itertools.combinations(question_list, size))
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    filename = f"output/multi_hop_{size}_way.json"
    
    multi_hop_questions = []
    
    # Use tqdm for a progress bar
    for i, combo in enumerate(tqdm(question_combos, desc=f"{size}-way combos")):
        print(f"\nCombo {i+1}/{len(question_combos)}: {' + '.join(q.split('?')[0] + '?' for q in combo)}")
        
        try:
            result = generate_multi_hop_question(combo)
            multi_hop_questions.append({
                "combo": list(combo),
                "result": result
            })
        except Exception as e:
            print(f"Error processing combo: {str(e)}")
        
        # Save intermediate results periodically
        if (i + 1) % 5 == 0 or (i + 1) == len(question_combos):
            with open(filename, "w") as f:
                json.dump(multi_hop_questions, f, indent=2)
            print(f"Saved progress to {filename}")
    
    # Create a clean version with just questions and answers
    clean_qa_pairs = []
    for item in multi_hop_questions:
        if "result" in item and "question" in item["result"] and "answer" in item["result"]:
            clean_qa_pairs.append({
                "question": item["result"]["question"],
                "answer": item["result"]["answer"],
                "sources": item["combo"]
            })
    
    clean_filename = f"output/multi_hop_{size}_way_clean.json"
    with open(clean_filename, "w") as f:
        json.dump(clean_qa_pairs, f, indent=2)
    
    print(f"Generated {len(multi_hop_questions)} {size}-way multi-hop questions")
    print(f"Saved {len(clean_qa_pairs)} clean Q&A pairs to {clean_filename}")
    
    return multi_hop_questions

# Main execution
def main():
    # Process combinations of different sizes
    for size in [2, 3, 4, 5]:
        process_combinations(size)
    
    print("\nAll processing complete!")

if __name__ == "__main__":
    main()
