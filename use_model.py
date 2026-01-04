import torch
from transformers import AutoTokenizer
from modeling_mplus import MPlus
from modeling_memoryllm import MemoryLLM

# # load the model mplus-8b (currently we only have the pretrained version)
model = MPlus.from_pretrained("YuWangX/mplus-8b", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("YuWangX/mplus-8b")
model = model.to(torch.bfloat16) # need to call it again to cast the `inv_freq` in rotary_emb to bfloat16 as well
model.put_ltm_to_numpy() # We include ltm as modules so that it can be uploaded to huggingface, but for inference we need to put ltm on CPU and cast ltm_ags to numpy. 
model = model.cuda()
# # After this, the usage of MPlus is the same as MemoryLLM-8B, please check "How to use the model" below. 

# # load pretrained model
# model = MemoryLLM.from_pretrained("YuWangX/memoryllm-8b", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
# tokenizer = AutoTokenizer.from_pretrained("YuWangX/memoryllm-8b")
# model = model.cuda()

# # load chat model
# model = MemoryLLM.from_pretrained("YuWangX/memoryllm-8b-chat", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
# tokenizer = AutoTokenizer.from_pretrained("YuWangX/memoryllm-8b-chat")
# model = model.cuda()


# #use model


# Self-Update with the new context
ctx = "Last week, John had a wonderful picnic with David. During their conversation, David mentioned multiple times that he likes eating apples. Though he didn't mention any other fruits, John says he can infer that David also like bananas."

# please make sure the context to inject into the memory is larger than 16 tokens, this is the hard minimum when training the model. The memory will be disturbed when less than 16 tokens are injected into the memory. 
model.inject_memory(tokenizer(ctx, return_tensors='pt', add_special_tokens=False).input_ids.cuda(), update_memory=True)



#Then for chat model, use the following template:
# Generation
# messages = [{
#     'role': 'user', "content": "What fruits does David like?",
# }]

# inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)[:, 1:] # remove bos tokens as the model has its own trained bos embeddings.
# terminators = [
#     tokenizer.eos_token_id,
#     tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]

# outputs = model.generate(input_ids=inputs.cuda(),
#                          max_new_tokens=20,
#                          eos_token_id=terminators)

# response = tokenizer.decode(outputs[0])
print(response)


# #for the pretrained model, use the following template:
inputs = tokenizer("Question: What fruits does David like? Answer: David likes", return_tensors='pt', add_special_tokens=False).input_ids.cuda()
outputs = model.generate(input_ids=inputs, max_new_tokens=20)
response = tokenizer.decode(outputs[0][inputs.shape[1]:])
print(response)
