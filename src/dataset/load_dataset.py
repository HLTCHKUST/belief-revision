import os
import pandas as pd
from datasets import load_dataset


def load_dataset_fn(args, question_colname = 'questions', data_folder_path = 'data_pool/', dataset_type = None):
    """
    Dataset should contain column_header: ["question", "answer", "GT answer"]
    """

    dataset_name_or_path = args["dataset_name_or_path"]

    if dataset_name_or_path == 'nmr_suppression':
        
        if dataset_type == 'batch1':
            filename = 'f2422226_agreement_full.csv'
        elif dataset_type == 'batch2':
            filename = 'nmr_suppression_batch2.csv'
        elif dataset_type == 'batch12':
            raise ValueError(dataset_type, "is now deprecated")
            # filename = 'nmr_suppression_b12.csv'
        elif dataset_type == 'batch12_onlyfirst2':
            raise ValueError(dataset_type, "is now deprecated")
            # filename = 'nmr_suppression_b12_onlyfirst2.csv'
        elif dataset_type == 'batch123':
            filename = 'nmr_suppression_b123_pt.csv'
        elif dataset_type == 'batch123_question_first':
            filename = 'nmr_suppression_b123_pt_question_first.csv'
        elif dataset_type == 'batch123_onlyfirst2':
            filename = 'nmr_suppression_b123_pt_onlyfirst2.csv'
        else:
            # default (and None) to batch123
            filename = 'nmr_suppression_b123_pt.csv'
        dataset_df = pd.read_csv(data_folder_path+os.path.join(dataset_name_or_path, filename))
        options = 'either a, b, or c'
        
    elif dataset_name_or_path == 'nmr_sarcasm':
        
        # filename = 'psarc_gpt-4-1106-preview_verbalirony_0to400_to_appen.csv'
        # dataset_df = pd.read_csv(data_folder_path+os.path.join(dataset_name_or_path, filename))
        # options = 'either a, b, or c'
        raise NotImplementedError("Dataset has not been setup.")

    elif dataset_name_or_path == 'anli':

        if not os.path.isfile('data_pool/anli/test.jsonl'):
            os.system('wget https://storage.googleapis.com/ai2-mosaic/public/abductive-commonsense-reasoning-iclr2020/anli.zip')
            os.system('/usr/bin/unzip anli.zip')
            os.system('rm -rf anli.zip')
            os.system('mv anli data_pool/')

        test_data = load_dataset('json', data_files='data_pool/anli/test.jsonl')
        label_filename = 'data_pool/anli/test-labels.lst'
        with open(label_filename) as file:
            labels = [int(line.rstrip()) for line in file]
        test_data['train'] = test_data['train'].add_column("label", labels)
        dataset_df = pd.DataFrame(test_data['train'])

        dataset_df['questions'] = dataset_df.apply(lambda x: "Observation 1: "+x['obs1']+
                                                             "\nObservation 2: "+x['obs2']+\
                                                             "\n\nGiven observation 1 happens before observation 2,\n"+\
                                                             "Which of the explanation below is the most plausible to happen between them?"+\
                                                             "\na) "+x['hyp1']+\
                                                             "\nb) "+x['hyp2'], axis=1)
        options = 'either a or b'


    elif dataset_name_or_path == 'hendrycks/competition_math':

        dataset = load_dataset(dataset_name_or_path)
        dataset_df = dataset['test'].to_pandas()
        dataset_df = dataset_df.rename(columns={'problem':'questions'})
        options = 'your final answer'

    else:
        raise NotImplementedError("Dataset has not been setup.")
        
    return dataset_df, options