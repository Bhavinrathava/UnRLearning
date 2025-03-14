

# Load the train dataset 

import pandas as pd 
from datasets import Dataset

training_df = pd.read_csv('src/train_with_anchors.csv', encoding_errors='ignore')
dataset = Dataset.from_pandas(training_df)

# Load the model 
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")


# Load the QLORA model 
def generate_r1_prompt(Question, target, anchor_words):
    r1_prefix = [{
        "role": "system",
        "content": """A conversation between User and Assistant. The user asks a question, and the Assistant answers it.
                    The assistant thinks about 4 things. 
                    First assistant figures out if the question is realated to Harry Potter. It will enclose this answer within <isRelated> </isRelated> tags.
                    Second, the assistant generates the normal answer to the question. It will enclose this answer within <normalAnswer> </normalAnswer> tags.
                    Third, the assistant indentifies the specific words and phrases, known as 'Anchor Words' in the normal answer that are related to Harry Potter. It will enclose this answer within <anchorWords> </anchorWords> tags.
                    Fourth, the assistant cleans up the answer so that it does not contain any of the 'Anchor Words' and presents a generic answer. It will enclose this answer within <cleanAnswer> </cleanAnswer> tags.
                    The reasoning process and answer are enclosed within <think> </think> and
                    <answer> </answer> tags, respectively, i.e., 

                    <think> 
                        <isRelated> 0 or 1 depending if answer is related to Harry Potter </isRelated>
                        <normalAnswer> generate normal answer </normalAnswer>
                        <anchorWords> generate anchor words from normal answer </anchorWords>
                        <cleanAnswer> generate generic answer </cleanAnswer>
                    </think>
                    <answer> answer here </answer>."""
      },
      { 
        "role": "user",
        "content": Question
      },
      {
        "role": "assistant",
        "content": "Let me solve this step by step.\n<think>"
      }]
    return {
        "prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True),
          "target": target,
          "anchor_words": anchor_words,
          "is_related": "0"}
 
# convert our dataset to the r1 prompt
dataset = dataset.map(lambda x: generate_r1_prompt(x["question"], x["answer"], x["anchor_words"]))
 
# split the dataset into train and test
train_test_split = dataset.train_test_split(test_size=0.1)
 
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# define the reward functions 

import re
 
def format_reward_func(completions, target, **kwargs):
    """
    Format: <think>
     <isRelated> 0 or 1 depending if answer is related to Harry Potter </isRelated>
     <normalAnswer> generate normal answer </normalAnswer>
     <anchorWords> generate anchor words from normal answer </anchorWords>
     <cleanAnswer> generate generic answer </cleanAnswer>
     </think>
     <answer> answer here </answer>
    
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
    Returns:
        list[float]: Reward scores
    """
    rewards = []
    for completion, gt in zip(completions, target):
        try:
            # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
            completion = "<think>" + completion
            # Check if the format is correct
            regex = r"^<think>\s*<isRelated>\s*(.*?)\s*<\/isRelated>\s*<normalAnswer>\s*(.*?)\s*<\/normalAnswer>\s*<anchorWords>\s*(.*?)\s*<\/anchorWords>\s*<cleanAnswer>\s*(.*?)\s*<\/cleanAnswer>\s*<\/think>\s*<answer>([\s\S]*?)<\/answer>$"
            match = re.search(regex, completion, re.DOTALL)
            # if the format is not correct, reward is 0
            if match is None or len(match.groups()) != 5:
                rewards.append(0.0)
            else:
                rewards.append(1.0)
        except Exception:
            rewards.append(0.0)
    return rewards

def check_anchor_words_in_answer(anchor_words, answer, Lambda = 0.1):
    """
    Check if any of the anchor words are present in the answer.
    Args:
        anchor_words (str): The anchor words.
        answer (str): The answer.
    Returns:
        bool: True if any of the anchor words are present in the answer, False otherwise.
    
    """
    reward = 1.0 
    for word in anchor_words:
        if word in answer:
            reward -= 1.0
    
    return Lambda * reward
    
def n_gram_similarity(model_output, target_output):
    """
    Compute the n-gram similarity between the model output and the target output.
    Args:
        model_output (str): The model output.
        target_output (str): The target output.
    Returns:
        float: The n-gram similarity score.
    """
    # Tokenize the outputs
    model_tokens = set(model_output.split())
    target_tokens = set(target_output.split())

    # Compute the n-gram similarity
    intersection = len(model_tokens.intersection(target_tokens))
    union = len(model_tokens.union(target_tokens))

    if union == 0:
        return 0.0

    return intersection / union

def answer_reward_function(completions, answer, anchor_words, is_related, **kwargs):

    rewards = []
    for output, target_answer, a_w, is_related_tag in zip(completions, answer, anchor_words, is_related):
        # do something
        reward = 0 
        # get the matching tag groups 
        try:
            # Check if the format is correct
            regex = r"^<think>\s*<isRelated>\s*(.*?)\s*<\/isRelated>\s*<normalAnswer>\s*(.*?)\s*<\/normalAnswer>\s*<anchorWords>\s*(.*?)\s*<\/anchorWords>\s*<cleanAnswer>\s*(.*?)\s*<\/cleanAnswer>\s*<\/think>\s*<answer>([\s\S]*?)<\/answer>$"
            match = re.search(regex, output, re.DOTALL)
            if match is None or len(match.groups()) != 5:
                rewards.append(0.0)
                continue
            else:
                generated_is_related, normal_answer, generated_anchor_words, clean_answer, final_answer = match.groups()
            
            # check if the answer is related to harry potter 
            if generated_is_related == is_related_tag:
                reward += 1

            
            # check the similarity of the normal answer and the target answer
            reward += n_gram_similarity(normal_answer, target_answer)

            # check the similarity of the anchor words and the target anchor words
            reward += n_gram_similarity(generated_anchor_words, a_w)

            # check if the clean answer contains anchor words 
            reward += check_anchor_words_in_answer(a_w, final_answer)

            rewards.append(reward)
        
        except Exception as e:
            print(f"Error: {e}")
            rewards.append(0.0)
            continue

# define the GRPO trainer 
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig
 
# our model we are going to use as policy 
model_config = ModelConfig(
    model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2",
    use_peft=True,
    load_in_4bit=True,
)
 
# Hyperparameters
training_args = GRPOConfig(
    output_dir="qwen-r1-aha-moment",
    learning_rate=5e-7,
    lr_scheduler_type="cosine",
    logging_steps=10,
    max_steps=100,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    # GRPO specific parameters
    max_prompt_length=256,
    max_completion_length=1024, # max length of the generated output for our solution
    num_generations=2,
    beta=0.001,
    
)
trainer = GRPOTrainer(
    model=model_config.model_name_or_path,
    reward_funcs=[format_reward_func, answer_reward_function],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=get_peft_config(model_config),
)
# train the fuckin model 

# Train and push the model to the Hub
trainer.train()
# Save model
trainer.save_model(training_args.output_dir)