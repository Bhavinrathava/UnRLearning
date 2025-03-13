from datasets import load_dataset
import pandas as pd 

'''
Loading a harry potter trivia dataset from huggingface and storing into csv Files 
'''
ds = load_dataset('saracandu/harry-potter-trivia-human')


df_train = pd.DataFrame(ds['train'])
df_test = pd.DataFrame(ds['test'])

# save to csv 
df_train.to_csv('train.csv', index=False)
df_test.to_csv('test.csv', index=False)


