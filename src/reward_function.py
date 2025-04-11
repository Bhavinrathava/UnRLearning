import re 

def answer_reward_function(completions, answer, anchor_words, is_related, **kwargs):

    rewards = []
    for output, target_answer, a_w, is_related_tag in zip(completions, answer, anchor_words, is_related):
        reward = 0
        # get the matching tag groups
        try:
            # Check if the format is correct
            regex = r"^<think>\s*<isRelated>\s*(.*?)\s*<\/isRelated>\s*<normalAnswer>\s*(.*?)\s*<\/normalAnswer>\s*<anchorWords>\s*(.*?)\s*<\/anchorWords>\s*<cleanAnswer>\s*(.*?)\s*<\/cleanAnswer>\s*<\/think>\s*<answer>([\s\S]*?)<\/answer>$"
            match = re.search(regex, output, re.DOTALL)
            if match is None:
                rewards.append(0.0)
                continue
            elif len(match.groups()) == 5:
                generated_is_related, normal_answer, generated_anchor_words, clean_answer, final_answer = match.groups()

                # check if the answer is related to harry potter
                if generated_is_related == is_related_tag:
                    reward += 10


                # check the similarity of the normal answer and the target answer
                reward += (n_gram_similarity(normal_answer, target_answer) * 50.0)

                # check the similarity of the anchor words and the target anchor words
                reward += (checkCommonAnchorWords(generated_anchor_words, a_w) * 50.0)

                # check if the clean answer contains anchor words
                reward += check_anchor_words_in_answer(a_w, final_answer)

                rewards.append(reward)
            else:
              rewards.append(len(match.groups()))
              
        except Exception as e:
          print(f"ERROR :: {e}")
          rewards.append(0.0)
          continue
    print(f'Answer Correctness Reward : {rewards}')
    return rewards





def answer_reward_function(completions, answer, anchor_words, is_related, **kwargs):

    rewards = []


    for output, target_answer, a_w, is_related_tag in zip(completions, answer, anchor_words, is_related):
        
        try :
        
            reward = 0.0

            # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
            completion = "<think>" + output

            patterns = {
            "isRelated": r"<isRelated>(.*?)<\/isRelated>",
            "normalAnswer": r"<normalAnswer>(.*?)<\/normalAnswer>",
            "anchorWords": r"<anchorWords>(.*?)<\/anchorWords>",
            "cleanAnswer": r"<cleanAnswer>(.*?)<\/cleanAnswer>",
            "answer": r"<answer>([\s\S]*?)<\/answer>"
            }

            # Check matches for each tag
            match_counts = {}
            for tag, pattern in patterns.items():
                matches = re.findall(pattern, completion, re.DOTALL)  # Find all occurrences
                match_counts[tag] = matches  # Store the count

            # Check for related tag : 
            if match_counts["isRelated"] and match_counts["isRelated"][0] == is_related_tag:
                reward += 10

            # check the similarity of the normal answer and the target answer
            if(match_counts["normalAnswer"] and match_counts["normalAnswer"][0] and target_answer):
                normal_answer = match_counts["normalAnswer"][0]
                reward += (n_gram_similarity(normal_answer, target_answer) * 50.0)

            # check the similarity of the anchor words and the target anchor words
            if(match_counts["anchorWords"] and match_counts["anchorWords"][0] and a_w):
                generated_anchor_words = match_counts["anchorWords"][0]
                reward += (checkCommonAnchorWords(generated_anchor_words, a_w) * 50.0)

            # check if the clean answer contains anchor words
            if(match_counts["cleanAnswer"] and match_counts["cleanAnswer"][0] and a_w):
                final_answer = match_counts["cleanAnswer"][0]
                reward += check_anchor_words_in_answer(a_w, final_answer)

            
            rewards.append(reward)

        except Exception as e:
            print(f"ERROR :: {e}")
            rewards.append(0.0)
    
    print(f'Answer Correctness Reward : {rewards}')
    return rewards 