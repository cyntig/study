import json

ENV_JSON = "./env.json"

def get_api_conf(api_type): 
    with open(ENV_JSON, 'r') as f: 
        dict = json.load(f)
    return dict[api_type]
    
if __name__ == "__main__":
    print(get_api_conf('open_ai'))