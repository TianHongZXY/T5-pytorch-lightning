# ====================================================
#   Copyright (C) 2022  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : fastapi_gpt_server.py
#   Last Modified : 2022-04-12 12:23
#   Describe      : 
#
# ====================================================

from fastapi import FastAPI
import uvicorn
from transformers import set_seed, GPT2Tokenizer, GPT2LMHeadModel


#  根据具体情况写—个模型加裁方法
def load_model() :
    #  set_seed(55)
    model = GPT2LMHeadModel.from_pretrained('/cognitive_comp/zhuxinyu/task/wenzhong-v2-knowledge/merged_wenzhong-v2-cmrc-v1-celebrity/iter_0002000/huggingface_model/')
    tokenizer = GPT2Tokenizer.from_pretrained('/cognitive_comp/zhuxinyu/task/wenzhong-v2-knowledge/merged_wenzhong-v2-cmrc-v1-celebrity/iter_0002000/huggingface_model/')
    #  model = GPT2LMHeadModel.from_pretrained('/cognitive_comp/zhuxinyu/pretrained_models/distilgpt2')
    #  tokenizer = GPT2Tokenizer.from_pretrained('/cognitive_comp/zhuxinyu/pretrained_models/distilgpt2')
    model = model.cuda(1)
    model.eval()

    return model, tokenizer

model, tokenizer = load_model()

max_length = 128
beam_search = False
greedy = False
n_sample = 3
top_k = 100
top_p = 0.9
#  创建一个fastapi app
app = FastAPI()

@app.get("/")
async def main():
    return {"message": "Hello，FastAPI"}

#  绑定路由
@app.get("/predict")
async def predict(input_text,
        max_length=128, 
        num_beams=None,
        n_sample=3,
        top_k=100,
        top_p=0.9,
        greedy=False,
        ):
    max_length = int(max_length)
    n_sample = int(n_sample)
    top_k = int(top_k)
    top_p = float(top_p)
    if num_beams:
        num_beams = int(num_beams)

    input_ids = tokenizer(input_text, padding=False, add_special_tokens=True, return_tensors="pt").input_ids.to(model.device)
    generated_sequence = model.generate(input_ids=input_ids, 
                                        max_length=max_length, 
                                        num_beams=n_sample if beam_search else None,
                                        num_return_sequences=n_sample if not greedy else 1, 
                                        top_k=top_k,
                                        top_p=top_p,
                                        do_sample=not greedy,
                                        use_cache=True,
                                        early_stopping=True,
                                        )
    results = []
    for sequence in generated_sequence:
        text = tokenizer.decode(sequence,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=True,
                                )
        results.append(text.strip())
    for idx, item in enumerate(results,start=1):
        print(f"""**候选回答「{idx}」:**\n""")
        print(item)

    return_message = {}
    for idx, item in enumerate(results,start=1):
        return_message[idx] = item

    return return_message


if __name__ == "__main__":
    #  启动服务,host是地址，port是端口
    uvicorn.run(app, host="0.0.0.0", port=6690, log_level="info")

