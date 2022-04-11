import requests
import torch
import streamlit as st
from transformers import pipeline, set_seed, GPT2Model, GPT2Tokenizer, GPT2LMHeadModel, MT5ForConditionalGeneration, T5Tokenizer
from transformers.tokenization_utils_base import AddedToken
from fengshen import T5ForConditionalGeneration as fengshenT5ForConditionalGeneration
from fengshen import T5Tokenizer as fengshenT5Tokenizer


st.set_page_config(
     page_title="明星信息问答",
     page_icon=":shark:",
     layout="wide",
     initial_sidebar_state="expanded",
     #  menu_items={
     #      'Get Help': 'https://www.extremelycoolapp.com/help',
     #      'Report a bug': "https://www.extremelycoolapp.com/bug",
     #      'About': "# This is a header. This is an *extremely* cool app!"
     #  }
 )
st.title('Demo for CelebrityQA')

form = st.form("参数设置")

st.sidebar.header("参数配置")
sbform = st.sidebar.form("固定参数设置")
n_sample = sbform.slider("设置返回条数",min_value=1,max_value=10,value=3)
text_length = sbform.slider('生成长度:',min_value=32,max_value=1024,value=64,step=32)
beam_search = sbform.checkbox('Use beam search')
greedy = sbform.checkbox('Use greedy search')
#  text_level = sbform.slider('文本多样性:',min_value=0.1,max_value=1.0,value=0.9,step=0.1)
#  model_id = sbform.number_input('选择模型号:',min_value=0,max_value=12,value=12,step=1)
#  trans = sbform.selectbox('选择翻译内核',['百度通用','医疗生物'])
trans = sbform.selectbox('选择模型',['GPT','T5'])
sbform.form_submit_button("配置")


input_text = form.text_input('请输入你的问题:',value='',placeholder='例如：赵丽颖的家乡是哪里？')

form.form_submit_button("提交")

@st.cache
def load_gpt_model():
    #  set_seed(55)
    model = GPT2LMHeadModel.from_pretrained('/cognitive_comp/zhuxinyu/task/wenzhong-v2-knowledge/merged_wenzhong-v2-cmrc-v1-celebrity/iter_0002000/huggingface_model/')
    model = model.cuda(4)
    model.eval()

    return model

@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda _: None})
def load_t5_model():
    #  set_seed(55)
    #  model = fengshenT5ForConditionalGeneration.from_pretrained('/cognitive_comp/zhuxinyu/codes/t5_mrc_zxy/outputs/Randeng-770M-celebrity-cmrc-separated-04-01_02-41-span-corruption-objective/')
    model = MT5ForConditionalGeneration.from_pretrained("/cognitive_comp/zhuxinyu/codes/t5_mrc_zxy/outputs/mt5-large-celebrity-cmrc-separated-04-07_01-52-prefix-lm-objective/")
    model = model.cuda(6)
    model.eval()

    return model

# @st.cache(suppress_st_warning=True)
#  def generate_qa(input_text,n_sample,model_id='7',length=64,translator='baidu',level=0.7):
#      # st.write('调用了generate函数')
#      URL = 'http://192.168.190.2:6666/qa'
#      data = {"text":input_text,"n_sample":n_sample,"model_id":model_id,"length":length,'translator':translator,'level':level}
#      r = requests.get(URL,params=data)
#      return r.text
# my_bar = st.progress(80)

gpt_tokenizer = GPT2Tokenizer.from_pretrained('/cognitive_comp/zhuxinyu/task/wenzhong-v2-knowledge/merged_wenzhong-v2-cmrc-v1-celebrity/iter_0002000/huggingface_model/')
t5_tokenizer = T5Tokenizer.from_pretrained("/cognitive_comp/zhuxinyu/codes/t5_mrc_zxy/outputs/mt5-large-celebrity-cmrc-separated-04-07_01-52-prefix-lm-objective/")
#  t5_tokenizer = fengshenT5Tokenizer.from_pretrained('/cognitive_comp/zhuxinyu/codes/t5_mrc_zxy/outputs/Randeng-770M-celebrity-cmrc-separated-04-01_02-41-span-corruption-objective/')

gpt_model = load_gpt_model()
t5_model = load_t5_model()

with st.spinner('老夫正在思考中🤔...'):
    if input_text:
        if trans == "T5":
            model = t5_model
            tokenizer = t5_tokenizer
        else:
            model = gpt_model
            tokenizer = gpt_tokenizer
        input_ids = tokenizer(input_text, padding=False, add_special_tokens=True, return_tensors="pt").input_ids.to(model.device)
        #  if trans == "T5":
        #      generated_sequence = model.generate(input_ids,
        #                                     max_length=text_length,
        #                                     greedy=False,
        #      )
        #      generated_sequence = [generated_sequence]
        #  else:
        generated_sequence = model.generate(input_ids=input_ids, 
                                            max_length=text_length, 
                                            num_beams=n_sample if beam_search else None,
                                            num_return_sequences=n_sample if not greedy else 1, 
                                            top_k=100,
                                            top_p=0.9,
                                            do_sample=not greedy,
                                            use_cache=True,
                                            early_stopping=True,
                                            )

        if not isinstance(generated_sequence, list):
            generated_sequence = generated_sequence.cpu()
            generated_sequence = generated_sequence.numpy().tolist()
        results = []
        for sequence in generated_sequence:
            text = tokenizer.decode(sequence,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True,
                                    )
            results.append(text)
        for idx,item in enumerate(results,start=1):
            st.markdown(f"""
            **候选回答「{idx}」:**\n
            """)
            st.info(item)

# st.info('This is a purely informational message')
# st.success('Done!')
# # st.balloons()
# with st.echo():
#     st.write('This code will be printed')
# import pandas as pd
# # st.help(pd.DataFrame)

# dataframe = pd.DataFrame({
#      'first column': [1, 2, 3, 4],
#      'second column': [10, 20, 30, 40],})
# st.experimental_show(dataframe)
