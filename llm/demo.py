from openai import OpenAI

base_url = 'https://api.siliconflow.cn/v1'
api_key = 'sk-kledjhymnmmqdrfxpjorunnvdtgbgmhccwzvlifxhvpelyob'

model_name = 'Qwen/Qwen3-8B'
client = OpenAI(
    base_url=base_url,
    api_key=api_key
)
resp = client.chat.completions.create(
    model=model_name,
    messages=[
        {'role': 'user', 'content': '请用一句话介绍一下LLM'}
    ]
)
print(resp.choices[0].message.content)