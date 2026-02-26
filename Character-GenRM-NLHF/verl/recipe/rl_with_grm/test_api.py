from openai import OpenAI

GRM_HOST="http://todo:8011/v1"

client = OpenAI(
    base_url=GRM_HOST,
    api_key="any-key",
)

# call API
chat_completion = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": "Who are you?"
    }],
    model="grm_rmnlhf",
    max_tokens=10,
)

print(chat_completion)