from openai import OpenAI
from src.utils.apikeys import openai_apikey, azure_apikey

SEARCH_COUNT = 0
def hit_api(input_text, system_text = '', model_name = 'gpt-4-1106-preview', api = 'Together', seed=3407):
    
    if 'command' in model_name:
        import cohere
        co = cohere.Client(cohere_key)
        
        if '@rag' in model_name:
            response = co.chat(
              model=model_name.replace('@rag',''),
              message=input_text,
              connectors=[{"id": "web-search"}]
            )
        else:
            response = co.chat(
              model=model_name,
              message=input_text,
            )
        global SEARCH_COUNT
        if response.is_search_required:
            SEARCH_COUNT += 1 
            print(f'SEARCH_COUNT: {SEARCH_COUNT}')
        output = response.text
        
    elif 'claude' in model_name:
        
        import anthropic
        client = anthropic.Anthropic(
            # defaults to os.environ.get("ANTHROPIC_API_KEY")
            api_key=anthropic_key,
        )

        message = client.messages.create(
            model=model_name,
            max_tokens=512,
            # temperature=0.0,
            # system="Respond only in Yoda-speak.",
            messages=[
                {
                    "role": "user", 
                    "content": input_text,
                }
            ]
        )
        output = message.content[0].text
        
    else:
            
        if api == 'openai':
            import openai as client
            client.api_key = chatgpt_key
            
        elif api == 'azure':
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=azure_apikey,
                api_version="2024-02-01",
                azure_endpoint="https://hkust.azure-api.net"
            )

        # seed=seed,
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    # "role": "system",
                    #  "content": system_text,
                    "role": "user",
                    "content": input_text,
                },
            ],
        )
        output = completion.choices[0].message.content
    
    return output


