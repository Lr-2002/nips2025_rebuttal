import openai

client = openai.OpenAI(
    api_key="sk-iCmbBZPKTncpR_UY7vhTNQ",
    base_url="http://litellm.mybigai.ac.cn:4000",  # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
)

response = client.chat.completions.create(
    model="gpt-4o-2024-11-20",  # model to send to the proxy
    messages=[
        {"role": "user", "content": "this is a test request, write a short poem"}
    ],
)

print(response)
