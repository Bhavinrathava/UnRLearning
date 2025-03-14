
# Read the initial Conversation file dataset 

import pandas as pd


df = pd.read_csv('src/train.csv', encoding_errors='ignore')
df = df[["question", "answer"]]

# populate the anchor word column
anchor_word_list = []

# load achor word from csv 
anchor_df = pd.read_csv('src/Anchor_Words.csv', encoding_errors='ignore')
print(anchor_df.columns)
anchor_word_list = anchor_df[['Anchor Word']].values.tolist()
print(f'Anchor Word List Length: {len(anchor_word_list)}')

# flatten the list of list
anchor_word_list = [item for sublist in anchor_word_list for item in sublist]


def anchor_mapping(row):
    anchor_list = []

    for anchor in anchor_word_list:
        if anchor in row['question'] or anchor in row['answer']:
            anchor_list.append(anchor)
    
    return anchor_list

# Apply the function to each row in the DataFrame
df['Anchor Words'] = df.apply(anchor_mapping, axis=1)


print(df.head())

# Save the updated DataFrame to a new CSV file
df.to_csv('src/train_with_anchors.csv', index=False)




