import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.utils.utils import prepare_paths
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
import torch.nn.functional as F

@torch.inference_mode()
def get_logprobs(model, tokenizer, inputs, label_ids=None, label_attn=None):
    inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024).to('cuda')    
    if model.config.is_encoder_decoder:
        label_ids = label_ids.repeat((inputs['input_ids'].shape[0],1))
        label_attn = label_attn.repeat((inputs['input_ids'].shape[0],1))
        logits = model(**inputs, labels=label_ids).logits
        logprobs = torch.gather(F.log_softmax(logits, dim=-1), 2, label_ids.unsqueeze(2)).squeeze(dim=-1) * label_attn
        return logprobs.sum(dim=-1).cpu()
    else:
        logits = model(**inputs).logits
        output_ids = inputs["input_ids"][:, 1:]
        logprobs = torch.gather(F.log_softmax(logits, dim=-1), 2, output_ids.unsqueeze(2)).squeeze(dim=-1)
        logprobs[inputs["attention_mask"][:, :-1] == 0] = 0
        return logprobs.sum(dim=1).cpu()

@torch.inference_mode()
def predict_classification(model, tokenizer, prompts, labels):
    if model.config.is_encoder_decoder:
        labels_encoded = tokenizer(labels, add_special_tokens=False, padding=True, return_tensors='pt')
        list_label_ids = labels_encoded['input_ids'].to('cuda')
        list_label_attn = labels_encoded['attention_mask'].to('cuda')
        
        inputs = [prompt.replace('[LABELS_CHOICE]', '') for prompt in prompts]
        probs = []
        for (label_ids, label_attn) in zip(list_label_ids, list_label_attn):
            probs.append(
                get_logprobs(model, tokenizer, inputs, label_ids.view(1,-1), label_attn.view(1,-1))
            )
    else:
        probs = []
        for label in labels:
            inputs = [prompt.replace('[LABELS_CHOICE]', label) for prompt in prompts]
            probs.append(get_logprobs(model, tokenizer, inputs))
    return probs
    
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
sanity_check = '_sanity_check'+str(args['sanity_check_size']) if args["sanity_check"] else ''

save_filename = dataset_type+'_'+model_name.replace('/','_').replace('-','_')+'_non-api_'+sanity_check+'.csv'
new_filepath = args["save_folder_path"] + save_filename
prepare_paths(new_filepath)


# Inference
if os.path.isfile(new_filepath):
    print('Skipping inference of', model_name, ': file exist.')
else:
    print('Now inferencing for', model_name)
    
    ## Model inference
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if config.is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).cuda().eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).cuda().eval()

    if dataset_type == 'time_t1':
        dataset_df = pd.read_csv(f'{data_folder_path}/belief_r/queries_time_t1.csv')
    else:
        dataset_df = pd.read_csv(f'{data_folder_path}/belief_r/basic_time_t.csv')
    dataset_df = dataset_df.head(args['sanity_check_size']) if args["sanity_check"] else dataset_df
    
    inputs, answers, output_list = [], [], []
    for idx in tqdm(range(dataset_df.shape[0])):
        p1, p2, p3 = dataset_df.loc[idx,'questions'].split('\n')[:3]
        a, b, c = dataset_df.loc[idx,['a','b','c']]

        if dataset_type == 'batch123_3P':
            prompt = f"Premise 1: {p1}\nPremise 2: {p2}\nPremise 3: {p3}\nConclusion: [LABELS_CHOICE]"
        else:
            prompt = f"Premise 1: {p1}\nPremise 2: {p2}\nConclusion: [LABELS_CHOICE]"
        candidate_labels = [a, b, c]
        
        out = predict_classification(model, tokenizer, [prompt], candidate_labels)
        pred_idx = np.argmax(np.stack(out, axis=-1), axis=-1)[0]
        final_answer = chr(ord('a') + pred_idx)
        input_texts = prompt
        
        inputs.append(input_texts)
        answers.append(final_answer)
        output_list.append(candidate_labels[pred_idx])

    dataset_df['answer'] = answers
    dataset_df['inputs'] = inputs
    dataset_df['outputs'] = output_list
    dataset_df.to_csv(new_filepath, index=False)
    
    print('='*20, ' Done Inferencing for', model_name, '='*20)