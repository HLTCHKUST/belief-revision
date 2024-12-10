from src.utils.api import hit_api
from src.prompts.utils import get_final_answer

default_mode = 'ZS_vanilla'
prompts = {'ZS_vanilla':None,
           'ZS_CoT':'Let’s think step by step.',
           'ZS_PS':"Let's first understand the problem and devise a plan to solve the problem."+\
                   "\nThen, let's carry out the plan and solve the problem step by step.",
           'ZS_RaR':"Rephrase and expand the question, and respond."
           }

# following https://github.com/williamcotton/empirical-philosophy/blob/main/articles/how-react-prompting-works.md#1st-call-to-llm
formatting = 'Begin! Reminder to write your final answer as "Final Answer [X]." and fill [X] with either a, b, or c.'

    
def default_prompt_format(text, formatting):
    
    text = text + '\n\n' + formatting
    return text


def custom_prompt_format(text, prompt, formatting):
    
    text = text + '\n\n' + formatting + '\n\n' + prompt
    return text


def prompt_one_time(prompt_mode, question, model_name, api=None):
    
    input_texts = []
    
    ### baseline prompts
    input_text = default_prompt_format(question, formatting) \
                    if prompt_mode == default_mode \
                    else custom_prompt_format(question, prompts[prompt_mode], formatting)
    outputs = hit_api(input_text, model_name=model_name, api=api)
    final_answer = get_final_answer(outputs)

    input_texts.append(input_text)
    
    return input_texts, final_answer, outputs