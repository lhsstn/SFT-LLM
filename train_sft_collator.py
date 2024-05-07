import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from peft import get_peft_model,LoraConfig,TaskType,PeftModel
from datasets import load_dataset,Dataset,concatenate_datasets
from transformers import Trainer
from transformers import TrainingArguments
from transformers import DataCollatorWithPadding
from transformers import AutoModelForCausalLM,AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--training_rag', action='store_true')
parser.add_argument('--base_model_id', type=str)
parser.add_argument('--pt_lora_id', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--epochs', type=int)
parser.add_argument('--gradient_accumulation_steps', type=int)
parser.add_argument('--evaluation_strategy', type=str)
parser.add_argument('--save_checkpoint_limit', type=int)
parser.add_argument('--wandb_run_name', type=str)

args = parser.parse_args()

# PROMPT 설정
if args.training_rag is True:
    print("(1) RAG 프롬프트로 학습합니다.")
    PROMPT = \
    '''당신은 유용한 AI 어시스턴트로, 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다. 

    Non-parametric memory는 주어진 외부 Context를 활용하여 답변을 하는 것을 말하며, Parametric memory는 외부 Context 없이 모델 자체의 저장된 지식을 이용하는 것을 의미합니다.

    # 질문에 따라 필요한 Context가 제공될 경우, Non-parametric memory를 기반으로 답변을 제공해야 합니다. 제공된 Context가 없을 경우에는, 당신의 내장된 지식(Parametric memory)을 활용하여 답변을 생성합니다.'''

else:        
    print("(1) LIMA 프롬프트로 학습합니다.")
    PROMPT = \
    '''당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.
    You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner.'''

def formatting_func(examples):
    input_ids=[]
    labels = []
        
    for ins,inp,ou in zip(examples['instruction'],examples['input'],examples['output']):
        instruction = ins
        response = ou
        context =inp 
        if args.training_rag is True:
            # print("(3) RAG 프롬프트로 학습합니다.")
            messages = [{"role": "system", "content": f"{PROMPT}\n\ncontext : {context}"},
                        {'role':'user', 'content':instruction}]
        else:
            # print("(3) LIMA 프롬프트로 학습합니다.")
            messages = [{'role':'system', 'content':f"{PROMPT}"},
                        {'role':'user', 'content':instruction}]
        
        instruction_chat= tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
        response_chat = tokenizer(response,return_attention_mask=False,add_special_tokens=False)['input_ids']
        
        chat_messages = instruction_chat+response_chat+[tokenizer.convert_tokens_to_ids('<|eot_id|>')]
        label = ([-100]*len(instruction_chat))+response_chat+[tokenizer.convert_tokens_to_ids('<|eot_id|>')]
        
        input_ids.append(chat_messages)
        labels.append(label)
        
    return {'input_ids':input_ids,'labels':labels}

class CustomDataCollator(object):
    def __init__(self,tokenizer,prompt,padding_value,batch_first):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.padding_value=padding_value
        self.batch_first=batch_first

    def __call__(self, examples):
        # [{},{},{}]
        input_ids=[]
        labels = []
        
        for i in range(len(examples)):
            input_ids.append(torch.tensor(examples[i]['input_ids'],dtype=torch.long))
            labels.append(torch.tensor(examples[i]['labels'],dtype=torch.long))
            
        padded_input_ids = pad_sequence(input_ids,padding_value=self.padding_value,batch_first=self.batch_first)
        padded_labels = pad_sequence(labels,padding_value=self.padding_value,batch_first=self.batch_first)
        attention_mask = padded_input_ids.ne(self.padding_value)
    
        return {'input_ids': padded_input_ids, 'labels': padded_labels,'attention_mask':attention_mask}


# model and tokenizer load
base_model = AutoModelForCausalLM.from_pretrained(args.base_model_id,torch_dtype=torch.bfloat16,device_map={"":int(os.environ.get('LOCAL_RANK') or 0)})
tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side='right'

# added lora adapter 
# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     r=2,
#     target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
#     lora_alpha = 128,
#     lora_dropout=0.05,
#     modules_to_save=['embed_tokens','lm_head']
# )
#model = get_peft_model(base_model,lora_config)

model = PeftModel.from_pretrained(base_model,args.pt_lora_id)

if args.training_rag is True:
    print("(2) RAG 데이터 셋을 포함하여 학습합니다.")
    alpaca_dataset = load_dataset('MLP-KTLim/koalpaca_for_sft',split='train')
    lima_en_dataset = load_dataset('MLP-KTLim/cleaned_lima_en',split='train')
    lima_ko_dataset = load_dataset('MLP-KTLim/korquad_lima_rv517',split='train').select(range(len(lima_en_dataset)))
    lima_dataset = concatenate_datasets([lima_en_dataset,lima_ko_dataset])
    ethics_datset = load_dataset('MLP-KTLim/kosafe_non_sorry',split='train')
    wiki_gpt_dataset = load_dataset('MLP-KTLim/wiki_gpt_rag_ko_en',split='train')
    merged_dataset = concatenate_datasets([alpaca_dataset, lima_dataset, ethics_datset, wiki_gpt_dataset])
else:        
    print("(2) RAG 데이터 셋을 제외하고 학습합니다.")
    alpaca_dataset = load_dataset('MLP-KTLim/koalpaca_for_sft',split='train')
    lima_en_dataset = load_dataset('MLP-KTLim/cleaned_lima_en',split='train')
    lima_ko_dataset = load_dataset('MLP-KTLim/korquad_lima_rv517',split='train').select(range(len(lima_en_dataset)))
    lima_dataset = concatenate_datasets([lima_en_dataset,lima_ko_dataset])
    ethics_datset = load_dataset('MLP-KTLim/kosafe_non_sorry',split='train')
    platypus_ko_dataset = load_dataset('kyujinpy/KOR-OpenOrca-Platypus-v3',split='train').select(range(5000))
    platypus_en_dataset = load_dataset('garage-bAInd/Open-Platypus',split='train').select(range(5000))
    platypus_dataset = concatenate_datasets([platypus_en_dataset,platypus_ko_dataset])
    merged_dataset = concatenate_datasets([alpaca_dataset, lima_dataset, ethics_datset, platypus_dataset])

train_dataset = merged_dataset.shuffle()
print('데이터셋 길이: ',len(train_dataset))
train_dataset = train_dataset.map(formatting_func,
                                  num_proc=8,
                                  batched=True)

our_dataset=train_dataset.train_test_split(test_size=int(len(train_dataset)*0.1),seed=42)

# Training Arguments
training_args = TrainingArguments(
    dataloader_num_workers=12,
    output_dir = args.output_dir,
    num_train_epochs = args.epochs,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    save_strategy=args.evaluation_strategy,
    evaluation_strategy=args.evaluation_strategy,
    save_total_limit=args.save_checkpoint_limit,
    optim='adamw_torch_fused',
    load_best_model_at_end=True,
    logging_strategy='steps',
    logging_steps=30,
    label_names=['labels'],
    run_name = args.wandb_run_name,
    report_to = 'wandb',
    #torch_compile=True # 모델 로드 오래걸리는 반면, vram이 적게 먹음
)

data_collator = CustomDataCollator(tokenizer=tokenizer,
                                   prompt=PROMPT,
                                   padding_value=tokenizer.pad_token_id,
                                   batch_first=True)
#train_dataloader = DataLoader(rag_dataset['train'],batch_size=1,collate_fn=data_collator,shuffle=True,num_workers=4)
#eval_dataloader = DataLoader(rag_dataset['test'],batch_size=1,collate_fn=data_collator,num_workers=4)

trainer = Trainer(
    model=model,
    train_dataset=our_dataset['train'],
    eval_dataset=our_dataset['test'],
    args=training_args,
    data_collator=data_collator,
    # ignore_keys_for_eval=[''] 이렇게 쓰는거같은데...어디서 쓰이는지 찾아가보는중
)

trainer.train()