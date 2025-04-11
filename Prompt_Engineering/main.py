
import requests 

coherence_reward = []
anchor_word_reward = []

def call_model(prompt, model_name = 'llama3.2'):

    url = "http://localhost:11434/api/generate"
    
    headers = {
  "model": model_name,
  "prompt": prompt,
}
    response = requests.post(url, headers=headers)
    if response.status_code == 200:
        return response.json().get('generated_text', '')
    else:
        return "Error: Unable to generate answer"
    
def evaluate_answer_coherence(generated_answer):
    # Placeholder for answer evaluation
    return 1.0  # Dummy evaluation score

def evaluate_answer_anchor_words(generated_answer):
    # Placeholder for anchor words evaluation
    return 1.0  # Dummy evaluation score

def generate_prompt(datapoint):
    # Placeholder for prompt generation
    return f"Prompt based on {datapoint}"  


def main():

    # Import the dataset 
    dataset = None 

    for datapoint in dataset : 
        prompt = generate_prompt(datapoint)
        generated_answer = call_model(prompt)
        coherence_score = evaluate_answer_coherence(generated_answer)
        coherence_reward.append(coherence_score)
        anchor_word_score = evaluate_answer_anchor_words(generated_answer)
        anchor_word_reward.append(anchor_word_score)

        print(f"Prompt: {prompt}")
        print(f"Generated Answer: {generated_answer} /n-----------------------------------")
        print(f"Coherence Rewards: {coherence_reward}")
        print(f"Anchor Word Rewards: {anchor_word_reward}")
