import os
from langchain_openai import ChatOpenAI

api_key = os.getenv("HUNYUAN_API_KEY")

model = ChatOpenAI(
    model="hunyuan-lite",
    base_url="https://api.hunyuan.cloud.tencent.com/v1",
    api_key=api_key
)

response = model.invoke("简单解释一下什么是量子纠缠")
print(response.content)