import torch
from transformers import BertModel, BertTokenizer,BertForMaskedLM
# 这里我们调用bert-base模型，同时模型的词典经过小写处理



def example_encode(model_name):

    #model_name = 'pytorch_model.bin'
    # 读取模型对应的tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # 载入模型
   # model = BertModel.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    #print(model)
    # 输入文本
    input_text = "Here is some text to encode"
    # 通过tokenizer把文本变成 token_id
    input_ids = tokenizer.encode(input_text, add_special_tokens=True)
    # input_ids: [101, 2182, 2003, 2070, 3793, 2000, 4372, 16044, 102]
    input_ids = torch.tensor([input_ids])
    # 获得BERT模型最后一个隐层结果
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
        print(last_hidden_states)
        print(last_hidden_states.shape)


if __name__ == "__main__":
    model_name = './bert-base-uncased'
    example_encode(model_name)
