#是否使用本地数据或在线数据-智能体 True:在线     False:本地 
#Whether to use local data or online data - Agent True: Online False: Local
AI_Agent = False #The English version needs no attention

#语音和播报是否使用通义千问的api接口
#Is the voice and broadcast using the Tongyi Qianwen API interface
#False:讯飞识别和合成方式（需要有api）  #False: iFlytek recognition and synthesis method (requires API)
TTS_IAT_Tongyi = True   #The English version needs no attention


#智能体id,AI_Agnet =True时 生效
#Effective when the agent ID is AI_Sgnet=True
#The English version needs no attention
Online_Action_ID = 'xxxxxxxxxxxx' #动作编排智能体id号 Action choreography agent ID number
Online_Iamge_ID = 'xxxxxxxxxxxxxxxxxxxx' #带图像的智能体id号 Intelligent agent ID with image

 
####国内key相关的 Domestic key related
#TTS_IAT_Tongyi == Flase 生效  #TTS_IAT_Tongyi==Flase take effect
XINGHOU_APPID = 'xxxxxxxxxxxxxxx' #填写星火大模型的APPID   The English version needs no attention
XINGHOU_APISecret = 'xxxxxxxxxxxxxxxx' #填写星火大模型的APISecret The English version needs no attention
XINGHOU_KEY = "xxxxxxxxxxxxxxxxx" #填写星火大模型的APIKEY The English version needs no attention



#通义千问 Tongyi Qianwen
TONYI_key='xxxxx' #填写通义千问的APIKEY  Fill in the APIKEY of Tongyi Qianwen

TONYI_API_IAT_MODEL='paraformer-realtime-v2' #通义语音识别模型 General Speech Recognition Model
TONYI_API_TTS_MODEL='qwen-tts' #通义语音合成模型 General Speech Synthesis Model

# 国外dify的开关 Foreign DIY switches
DIFY_SWITCH = False

#####国外key相关的  Foreign key related
#注册的地址：https://openrouter.ai/   Registered address：https://openrouter.ai/
openAI_KEY = 'xxxx' #填写openrouter的APIKEY  Fill in the APIKEY of openrouter

