import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.utils.utils import prepare_paths

from transformers import pipeline

## Arguments
parser = argparse.ArgumentParser(description='inference throgh zero-shot NLI')
parser.add_argument('--model_name', help='model_name', required=True, default="facebook/bart-large-mnli")
parser.add_argument('--sanity_check', help='test run samples', action='store_true')
parser.add_argument('--sanity_check_size', help='sanity check size, i.e. to test to 10 samples', type=int, default=10)
parser.add_argument('--dataset_type', help="whether it's an inference on time_t or time_t1", required=False, default='time_t1')
parser.add_argument('--data_folder_path', help='path to the dataset', required=False, default='dataset/')
parser.add_argument('--save_folder_path', help='path to the result dataset', required=False, default='save_non_api/')
args = vars(parser.parse_args())

model_name = args["model_name"]
data_folder_path = args["data_folder_path"]
dataset_type = args["dataset_type"]
print('='*20, model_name, '='*20)

# Load dataset
data_folder_path = args["data_folder_path"]
if dataset_type == 'time_t1':
    dataset_df = pd.read_csv(f'{data_folder_path}/belief_r/queries_time_t1.csv')
else:
    dataset_df = pd.read_csv(f'{data_folder_path}/belief_r/basic_time_t.csv')
sanity_check = '_sanity_check'+str(args['sanity_check_size']) if args["sanity_check"] else ''

save_filename = args["dataset_type"]+'_'+model_name.replace('/','_').replace('-','_')+'_nli_'+sanity_check+'.csv'
new_filepath = args["save_folder_path"] + save_filename
prepare_paths(new_filepath)


# Inference
if os.path.isfile(new_filepath):
    print('Skipping inference of', model_name, ': file exist.')
else:
    print('Now inferencing for', model_name)
    
    ## Model inference
    dataset_df = dataset_df.head(args['sanity_check_size']) if args["sanity_check"] else dataset_df
    dataset_df_copy = dataset_df.copy()

    classifier = pipeline("zero-shot-classification", model=model_name, device='cuda')
    
    inputs, answers, output_list = [], [], []
    for idx in tqdm(range(dataset_df.shape[0])):
        p1, p2, p3 = dataset_df.loc[idx,'questions'].split('\n')[:3]
        a, b, c = dataset_df.loc[idx,['a','b','c']]
        
        premises = f"Premise 1: {p1}\nPremise 2: {p2}"
        candidate_labels = [a, b, c]
        
        out = classifier(premises, candidate_labels=candidate_labels, hypothesis_template='Conclusion: {}')
        pred_idx = np.argmax(out['scores'])
        final_answer =  chr(ord('a') + pred_idx)
        input_texts = out['sequence']
        
        inputs.append(input_texts)
        answers.append(final_answer)
        output_list.append(candidate_labels[pred_idx])

    dataset_df['answer'] = answers
    dataset_df['inputs'] = inputs
    dataset_df['outputs'] = output_list
    dataset_df.to_csv(new_filepath, index=False)
    
    print('='*20, ' Done Inferencing for', model_name, '='*20)