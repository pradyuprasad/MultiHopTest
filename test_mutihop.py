import os
from dotenv import load_dotenv
import requests
import json
import re
import time
from tqdm import tqdm

load_dotenv()

api_key = os.environ['OPENROUTER_API_KEY']

# Function to load the multi-hop questions from your generated files
def load_questions(sizes=[2, 3, 4, 5]):
    all_questions = []
    
    for size in sizes:
        try:
            filename = f"output/multi_hop_{size}_way_clean.json"
            with open(filename, "r") as f:
                questions = json.load(f)
                for q in questions:
                    q["hop_count"] = size
                all_questions.extend(questions)
            print(f"Loaded {len(questions)} {size}-way questions from {filename}")
        except FileNotFoundError:
            print(f"Warning: Could not find {filename}")
    
    return all_questions

# Function to solve a question with just the answer (no reasoning)
def solve_direct(question_data, index):
    question_text = question_data["question"]
    expected_answer = question_data["answer"]
    
    prompt = f"""You are given a multi-hop reasoning question. Provide ONLY the numerical answer with no explanation.
Format your response with an XML tag as follows:
<answer>Your numerical answer here</answer>

Question: "{question_text}"
"""

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
            timeout=(30, 120)
        )
        
        model_response = response.json()["choices"][0]["message"]["content"]
        print(model_response)
        
        # Extract answer using regex
        answer_match = re.search(r'<answer>(.*?)</answer>', model_response, re.DOTALL)
        extracted_answer = answer_match.group(1).strip() if answer_match else None
        
        # Check if the answer is correct
        is_correct = str(extracted_answer) == str(expected_answer) if extracted_answer is not None else False
        
        result = {
            "id": index,
            "question": question_text,
            "expected_answer": expected_answer,
            "model_answer": extracted_answer,
            "full_response": model_response,
            "is_correct": is_correct,
            "hop_count": question_data["hop_count"],
            "sources": question_data.get("sources", [])
        }
        
        print(f"Direct Q{index} (Hops: {question_data['hop_count']}): {'✓' if is_correct else '✗'}")
        return result
        
    except Exception as e:
        print(f"Error in direct solve Q{index}: {str(e)}")
        return {
            "id": index,
            "question": question_text,
            "expected_answer": expected_answer,
            "model_answer": None,
            "full_response": f"Error: {str(e)}",
            "is_correct": False,
            "hop_count": question_data["hop_count"],
            "sources": question_data.get("sources", [])
        }

# Function to solve with reasoning (showing work)
def solve_with_reasoning(question_data, index):
    question_text = question_data["question"]
    expected_answer = question_data["answer"]
    
    prompt = f"""You are given a multi-hop reasoning question. Solve it step-by-step, showing your full reasoning process.

Question: "{question_text}"

Think through the solution carefully and show all your calculations. Once you've determined the answer, format your response by ending with:
<answer>Your numerical answer here</answer>
"""

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
            timeout=(30, 180)  # Longer timeout for reasoning
        )
        
        model_response = response.json()["choices"][0]["message"]["content"]
        
        # Extract answer using regex
        answer_match = re.search(r'<answer>(.*?)</answer>', model_response, re.DOTALL)
        extracted_answer = answer_match.group(1).strip() if answer_match else None
        
        # Check if the answer is correct
        is_correct = str(extracted_answer) == str(expected_answer) if extracted_answer is not None else False
        
        result = {
            "id": index,
            "question": question_text,
            "expected_answer": expected_answer,
            "model_answer": extracted_answer,
            "full_response": model_response,
            "is_correct": is_correct,
            "hop_count": question_data["hop_count"],
            "sources": question_data.get("sources", [])
        }
        
        print(f"Reasoning Q{index} (Hops: {question_data['hop_count']}): {'✓' if is_correct else '✗'}")
        return result
        
    except Exception as e:
        print(f"Error in reasoning solve Q{index}: {str(e)}")
        return {
            "id": index,
            "question": question_text,
            "expected_answer": expected_answer,
            "model_answer": None,
            "full_response": f"Error: {str(e)}",
            "is_correct": False,
            "hop_count": question_data["hop_count"],
            "sources": question_data.get("sources", [])
        }

def main():
    # Load all questions
    all_questions = load_questions()
    print(f"Total questions loaded: {len(all_questions)}")
    
    # Create results directories
    os.makedirs("results", exist_ok=True)
    
    # Solve using direct approach
    print("\nTesting Direct Answer approach:")
    direct_results = []
    for i, q in enumerate(tqdm(all_questions)):
        result = solve_direct(q, i)
        direct_results.append(result)
        
        # Save progress periodically
        if (i + 1) % 10 == 0 or (i + 1) == len(all_questions):
            with open("results/direct_results.json", "w") as f:
                json.dump(direct_results, f, indent=2)
    
    # Calculate direct results accuracy
    direct_correct = sum(1 for r in direct_results if r["is_correct"])
    print(f"\nDirect Answer Accuracy: {direct_correct}/{len(direct_results)} = {direct_correct/len(direct_results):.2%}")
    
    # Save final direct results
    with open("results/direct_results.json", "w") as f:
        json.dump(direct_results, f, indent=2)
    
    # Add a pause to avoid rate limiting
    print("Pausing for 30 seconds before starting reasoning tests...")
    time.sleep(30)
    
    # Solve using reasoning approach
    print("\nTesting Reasoning approach:")
    reasoning_results = []
    for i, q in enumerate(tqdm(all_questions)):
        result = solve_with_reasoning(q, i)
        reasoning_results.append(result)
        
        # Save progress periodically
        if (i + 1) % 10 == 0 or (i + 1) == len(all_questions):
            with open("results/reasoning_results.json", "w") as f:
                json.dump(reasoning_results, f, indent=2)
    
    # Calculate reasoning results accuracy
    reasoning_correct = sum(1 for r in reasoning_results if r["is_correct"])
    print(f"\nReasoning Approach Accuracy: {reasoning_correct}/{len(reasoning_results)} = {reasoning_correct/len(reasoning_results):.2%}")
    
    # Save final reasoning results
    with open("results/reasoning_results.json", "w") as f:
        json.dump(reasoning_results, f, indent=2)
    
    # Generate hop-based accuracy report
    hop_accuracy = {}
    for hop_count in range(2, 6):
        direct_hop_results = [r for r in direct_results if r["hop_count"] == hop_count]
        direct_hop_correct = sum(1 for r in direct_hop_results if r["is_correct"])
        direct_hop_accuracy = direct_hop_correct / len(direct_hop_results) if direct_hop_results else 0
        
        reasoning_hop_results = [r for r in reasoning_results if r["hop_count"] == hop_count]
        reasoning_hop_correct = sum(1 for r in reasoning_hop_results if r["is_correct"])
        reasoning_hop_accuracy = reasoning_hop_correct / len(reasoning_hop_results) if reasoning_hop_results else 0
        
        hop_accuracy[hop_count] = {
            "direct": {
                "correct": direct_hop_correct,
                "total": len(direct_hop_results),
                "accuracy": direct_hop_accuracy
            },
            "reasoning": {
                "correct": reasoning_hop_correct,
                "total": len(reasoning_hop_results),
                "accuracy": reasoning_hop_accuracy
            }
        }
    
    # Save hop-based accuracy report
    with open("results/hop_accuracy.json", "w") as f:
        json.dump(hop_accuracy, f, indent=2)
    
    # Print hop-based accuracy report
    print("\nAccuracy by Hop Count:")
    for hop_count, stats in hop_accuracy.items():
        print(f"{hop_count}-hop questions:")
        print(f"  Direct:    {stats['direct']['correct']}/{stats['direct']['total']} = {stats['direct']['accuracy']:.2%}")
        print(f"  Reasoning: {stats['reasoning']['correct']}/{stats['reasoning']['total']} = {stats['reasoning']['accuracy']:.2%}")

if __name__ == "__main__":
    main()
