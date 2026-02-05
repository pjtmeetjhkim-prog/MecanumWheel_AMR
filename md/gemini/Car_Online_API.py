import os,sys,base64
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from API_KEY import *
from openai import OpenAI


# OpenAI 호환 클라이언트를 사용하여 모델 응답을 가져오는 함수 (실행 층 지능체 모델)
def get_response(messages): #执行层智能体模型
    MODEL='qwen-vl-max-2025-04-08'
    client = OpenAI(
        api_key=TONYI_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(model=MODEL, messages=messages)
    return completion

# 이미지 파일을 Base64 문자열로 인코딩하는 함수
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 대화 기록을 저장하는 리스트 초기화 (시스템 프롬프트 포함)
messages = [
        {
            "role": "system",
            "content": """你是一个麦轮小车助手，如果有听到再见、退出的意思，你要说：好的再见，有需要再找我喔。""",
        }
    ]

assistant_output = "xxx"

# 새로운 세션을 초기화하는 함수 (대화 기록 리셋)
def New_session_init():
    global messages,assistant_output
    # 初始化一个 messages 数组
    messages = [
        {
            "role": "system",
            "content": """你是一个麦轮小车助手，如果有听到再见、退出的意思，你要说：好的再见，有需要再找我喔。""",
        }
    ]

    assistant_output = "xxx"



# Qwen-VL(Vision Language) 모델을 사용하여 이미지와 텍스트 프롬프트를 처리하는 함수
def QwenVL_api_picture(PROMPT='执行智能体', image_path=None):
    global messages,assistant_output,new_speak
    if image_path is None:
        image_path = "./AI_CarAgent/rec.jpg"
    base64_image = encode_image(image_path) #AI_CarAgent/
    
    # 이전 대화에서 종료 신호가 없었을 경우에만 실행
    if "有需要再找我喔" not in assistant_output:
        # 将用户问题信息添加到messages列表中
        # 사용자 질문과 이미지를 messages 리스트에 추가
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}, 
                    },
                    {"type": "text", 
                     "text": PROMPT
                    },
                ],
            })
        # 모델 호출 및 응답 획득
        assistant_output = get_response(messages).choices[0].message.content
        
        # 将大模型的回复信息添加到messages列表中
        # 모델의 응답을 messages 리스트에 추가 (대화 맥락 유지)
        messages.append({"role": "assistant", "content": assistant_output})
        #print(f"模型输出：{assistant_output}")
        #print("\n")
        
        # 종료 문구가 포함되어 있으면 세션 초기화
        if "有需要再找我喔" in assistant_output: #输出结果会在下一个问题前更新
            #print("清空对话")
            exit = assistant_output  
            New_session_init()  
            return exit
        
        return assistant_output
      
      

# 텍스트 기반의 의사결정 지능체 모델 호출 함수 (Qwen-Turbo 사용)
def QwenVL_api_decision(PROMPT='决策智能体'):
    client = OpenAI(
        api_key=TONYI_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": PROMPT},
        ],
    )
    #print(completion.model_dump_json())
    result = completion.choices[0].message.content
    #print(result)
    return result