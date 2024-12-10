import os
import argparse
import pandas as pd
from tqdm import tqdm

from src.dataset.load_dataset import load_dataset_fn
from src.utils.utils import is_model_on_api, prepare_paths
from src.prompts.one_time import prompt_one_time


## Arguments
parser = argparse.ArgumentParser(description='inference through API')
parser.add_argument('--dataset_name_or_path', help='path to the testing dataset', default="belief_r")
parser.add_argument('--prompt_mode', help='mode of prompting', required=True, default="interpret_consider")
parser.add_argument('--model_name', help='model_name', required=True, default="gpt-3.5-turbo")
parser.add_argument('--model_on_api', help='is model on api?', action='store_true')
parser.add_argument('--api', help='api to call', required=False, default="")
parser.add_argument('--batch_size', help='batch_size', required=False, default=1, type=int)
parser.add_argument('--max_new_token', help='max new token to generate', required=False, default=1024, type=int)
parser.add_argument('--sanity_check', help='test run samples', action='store_true')
parser.add_argument('--sanity_check_size', help='sanity check size, i.e. to test to 10 samples', type=int, default=10)
parser.add_argument('--dataset_type', help="whether it's an inference on time_t or time_t1", required=False, default='time_t1')
parser.add_argument('--data_folder_path', help='path to the dataset', required=False, default='dataset/')
parser.add_argument('--save_folder_path', help='path to the result dataset', required=False, default='save/')
args = vars(parser.parse_args())

model_name = args["model_name"]
api = args['api']
prompt_mode = args['prompt_mode']
is_model_on_api = args['model_on_api']
print('='*20, model_name, '='*20)


# Load dataset
data_folder_path = args["data_folder_path"]
dataset_name_or_path = args["dataset_name_or_path"]
dataset_type = args["dataset_type"]
if dataset_type == 'time_t1':
    dataset_df = pd.read_csv(f'{data_folder_path}/{dataset_name_or_path}/queries_time_t1.csv')
else:
    dataset_df = pd.read_csv(f'{data_folder_path}/{dataset_name_or_path}/basic_time_t.csv')

sanity_check = '_sanity_check'+str(args['sanity_check_size']) if args["sanity_check"] else ''
save_filename = args["dataset_type"]+'_'+\
                model_name.replace('/','_').replace('-','_')+'_'+\
                api+'_'+\
                prompt_mode+\
                sanity_check+'.csv'
new_filepath = args["save_folder_path"] + os.path.join(dataset_name_or_path, save_filename)
prepare_paths(new_filepath)


# Inference
if os.path.isfile(new_filepath):
    print('Skipping', model_name, prompt_mode, ': file exist.')

else:
    print('Now inferencing for', prompt_mode)
    
    temp_filepath = new_filepath[:-4]+'_temp.csv'
    if os.path.isfile(temp_filepath) and not args['sanity_check']:
        df_temp = pd.read_csv(temp_filepath)
        inputs = df_temp.inputs.tolist()
        answers = df_temp[model_name+'_'+prompt_mode].tolist()
        output_list = df_temp.outputs.tolist()
        temp_index = df_temp.shape[0]
        print('Continuing previous generation from checkpoint:', temp_filepath)
        
    else:
        inputs, answers, output_list = [], [], []
        temp_index=-1

    ## Model inference
    dataset_df = dataset_df.head(args['sanity_check_size']) if args["sanity_check"] else dataset_df
    dataset_df_copy = dataset_df.copy()
    if is_model_on_api:

        for index in tqdm(range(dataset_df.shape[0])):
            if index >= temp_index:
                question = dataset_df['questions'][index]

                if prompt_mode in ['ZS_vanilla', 'ZS_CoT', 'ZS_PS', 'ZS_RaR']:
                    input_texts, final_answer, outputs = prompt_one_time(prompt_mode, question, model_name, api)

                else:
                    raise NotImplementedError("Inference with "+prompt_mode+" method has not been set up.")

                inputs.append(input_texts)
                answers.append(final_answer)
                output_list.append(outputs)

                temp_df = dataset_df_copy.head(index+1)
                temp_df[model_name+'_'+prompt_mode] = answers
                temp_df['inputs'] = inputs
                temp_df['outputs'] = output_list
                temp_df.to_csv(temp_filepath, index=False)

    else:
        raise NotImplementedError("Inference to non API model has not been set up.")

    dataset_df[model_name+'_'+prompt_mode] = answers
    dataset_df['inputs'] = inputs
    dataset_df['outputs'] = output_list
    dataset_df.to_csv(new_filepath, index=False)
    
    print('='*20, ' Done Inferencing for', prompt_mode, '='*20)