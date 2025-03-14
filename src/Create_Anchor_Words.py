import pandas as pd

anchor_word_list = []

# Append values from Characters csv 

# Load CSV 
df = pd.read_csv('src/Characters.csv', encoding_errors='ignore')

df = df.drop(columns = 'Character ID')

# for each column, extract the dictinct values that are non null and non empty
for column in df.columns:
    # Get the distinct values in the column
    distinct_values = df[column].dropna().unique()
    
    # Filter out empty strings
    filtered_values = [value for value in distinct_values if value != '']
    
    # Append the filtered values to the anchor_word_list
    anchor_word_list.extend(filtered_values)

# Checkpoint for Anchor Word length 
print(f'Anchor Word List Length: {len(anchor_word_list)}')


# Append Values from Movies csv 

# Load CSV
df = pd.read_csv('src/Movies.csv', encoding_errors='ignore')
df = df['Movie Title']
distinct_values = df.dropna().unique()
anchor_word_list.extend(distinct_values)

print(f'Anchor Word List Length: {len(anchor_word_list)}')

# Append Values from Places csv 
df = pd.read_csv('src/Places.csv', encoding_errors='ignore')
df = df[['Place Name', 'Place Category']]

for column in df.columns:
    # Get the distinct values in the column
    distinct_values = df[column].dropna().unique()
    
    # Filter out empty strings
    filtered_values = [value for value in distinct_values if value != '']
    
    # Append the filtered values to the anchor_word_list
    anchor_word_list.extend(filtered_values)

print(f'Anchor Word List Length: {len(anchor_word_list)}')

# Append Values from Spells CSV 

df = pd.read_csv('src/Spells.csv', encoding_errors='ignore')

df = df[['Incantation','Spell Name']]

for column in df.columns:
    # Get the distinct values in the column
    distinct_values = df[column].dropna().unique()
    
    # Filter out empty strings
    filtered_values = [value for value in distinct_values if value != '']
    
    # Append the filtered values to the anchor_word_list
    anchor_word_list.extend(filtered_values)
print(f'Anchor Word List Length: {len(anchor_word_list)}')

# Add from chapters 
df = pd.read_csv('src/Chapters.csv', encoding_errors='ignore')

df = df[['Chapter Name']]

unique_values = df['Chapter Name'].dropna().unique()
filtered_values = [value for value in unique_values if value != '']
anchor_word_list.extend(filtered_values)
print(f'Anchor Word List Length: {len(anchor_word_list)}')

# Save to CSV
df = pd.DataFrame(anchor_word_list, columns=['Anchor Word'])
df.to_csv('src/Anchor_Words.csv', index=False)