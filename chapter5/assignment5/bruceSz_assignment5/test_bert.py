import torch
from transformers import BertModel, BertTokenizer,BertForMaskedLM
from torch.nn import functional as F
#import onnxruntime
import time
import numpy as np
# 这里我们调用bert-base模型，同时模型的词典经过小写处理



def model_test(model_name):

    #model_name = 'pytorch_model.bin'
    # 读取模型对应的tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # 载入模型
    #model = BertModel.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    #print(model)
    # 输入文本
    input_text = "Here is some text to encode"
    # 通过tokenizer把文本变成 token_id
    input_ids = tokenizer.encode(input_text)
    #print(input_ids)
    # input_ids: [101, 2182, 2003, 2070, 3793, 2000, 4372, 16044, 102]
    
    # output form tokenizer.encode should be converted to torch.tensor here.
    input_ids = torch.tensor([input_ids])
    
    
    # 获得BERT模型最后一个隐层结果
    with torch.no_grad():
        begin = time.time()
        output = model(input_ids)  # Models outputs are now tuples
        end = time.time()
        print("normal model infer cost:",end-begin)
        #print(output[0].shape)
        #print(output)
        #logits = output.logits
        softmax = F.softmax(output[0], dim = -1)
        #print(softmax)
        #mask_word = softmax[0, mask_index, :]
        #top_10 = torch.topk(mask_word, 10, dim = 1)[1][0]
        #print("model test topk10 output:")
        #for token in top_10:
        #    word = tokenizer.decode([token])
        #    new_sentence = text.replace(tokenizer.mask_token, word)
        #    print(new_sentence)

        # save inputs and output
        print("Saving inputs and output to case_data.npz ...")
        position_ids = torch.arange(0, input_ids.shape[1]).int().view(1, -1)
        print(position_ids)
        input_ids=input_ids.int().detach().numpy()
        #token_type_ids=encoded_input['token_type_ids'].int().detach().numpy()
        
        #print(last_hidden_states)
        #print(last_hidden_states.shape)

def model_transform(model_dir):
   # BERT_PATH ='./bert-base-uncased'
    #model_dir = 'pytorch_model.bin'
    # 读取模型对应的tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_dir)

    # 载入模型
    #model = BertModel.from_pretrained(model_dir)
    model = BertForMaskedLM.from_pretrained(model_dir)
    text = "The capital of France, contains the Eiffel Tower."
    print("===================model2onnx=======================")
    encoded_input = tokenizer.encode(text, return_tensors = "pt")
    print(type(encoded_input))
    print(encoded_input)

    model.eval()
    export_model_path = model_dir + "/model.onnx"
    opset_version = 12
    symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
    torch.onnx.export(model,                                            # model being run
                      args=encoded_input,                      # model input (or a tuple for multiple inputs)
                      f=export_model_path,                              # where to save the model (can be a file or file-like object)
                      opset_version=opset_version,                      # the ONNX version to export the model to
                      do_constant_folding=False,                         # whether to execute constant folding for optimization
                      input_names=['input_ids',                         # the model's input names
                                   #'attention_mask',
                                   'token_type_ids'],
                    output_names=['logits'],                    # the model's output names
                    dynamic_axes={'input_ids': symbolic_names,        # variable length axes
                                  #'attention_mask' : symbolic_names,
                                  'token_type_ids' : symbolic_names,
                                  'logits' : symbolic_names})
    print("Model exported at ", export_model_path)


#def test_speed(model_name):
#    tokenizer = BertTokenizer.from_pretrained(model_name)
#    input_text = "Here is some text to encode"
    # 通过tokenizer把文本变成 token_id
#    input_ids = tokenizer.encode(input_text)
#    print(type(input_ids))
#    input_ids = np.array([input_ids])
#
#    path = "./bert-base-uncased//model.onnx"
#    sess = onnxruntime.InferenceSession(path,None)
#    input_n = sess.get_inputs()[0].name
#    output_n = sess.get_inputs()[0].name
#    print("input nmae is: ", input_n)
#    begin = time.time()
#    res = sess.run([], {input_n: input_ids})
#    end = time.time()
#    print("onnx-runtime cost: ", (end-begin))
    #print(input_ids)
    # input_ids: [101, 2182, 2003, 2070, 3793, 2000, 4372, 16044, 102]
    
    # output form tokenizer.encode should be converted to torch.tensor here.
    

if __name__ == "__main__":
    model_dir = './bert-base-uncased'
    #model_transform(model_dir)
    #test_speed(model_dir)
    model_test(model_dir)

    
