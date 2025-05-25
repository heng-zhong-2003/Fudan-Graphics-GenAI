#!/usr/bin/env python3  
"""  
稳定的训练版本 - 解决NaN问题  
"""  

import torch  
import json  
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling  
import os  
from torch.utils.data import Dataset  

class ChairDataset(Dataset):  
    def __init__(self, data, tokenizer, max_length=512):  
        self.data = data  
        self.tokenizer = tokenizer  
        self.max_length = max_length  
    
    def __len__(self):  
        return len(self.data)  
    
    def __getitem__(self, idx):  
        item = self.data[idx]  
        text = f"Human: {item['input']}\n\nAssistant: {item['output']}"  
        
        # Tokenize  
        encoding = self.tokenizer(  
            text,  
            truncation=True,  
            padding='max_length',  
            max_length=self.max_length,  
            return_tensors='pt'  
        )  
        
        return {  
            'input_ids': encoding['input_ids'].flatten(),  
            'attention_mask': encoding['attention_mask'].flatten(),  
            'labels': encoding['input_ids'].flatten()  
        }  

def stable_train():  
    print("🪑 稳定训练椅子设计模型...")  
    
    # 设置环境  
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 调试CUDA错误  
    torch.cuda.empty_cache()  
    
    # 加载数据  
    data_file = "./output/new_training_data/chair_training_data.json"  
    with open(data_file, 'r', encoding='utf-8') as f:  
        training_data = json.load(f)  
    
    # 只用少量数据进行稳定训练  
    training_data = training_data[:20]  
    print(f"📊 使用 {len(training_data)} 个样本进行稳定训练")  
    
    # 加载模型和tokenizer  
    print("🔄 加载模型...")  
    tokenizer = AutoTokenizer.from_pretrained("../models/BlenderLLM", trust_remote_code=True)  
    model = AutoModelForCausalLM.from_pretrained(  
        "../models/BlenderLLM",  
        trust_remote_code=True,  
        torch_dtype=torch.float32,  # 使用float32而不是float16  
        device_map="auto",  
        low_cpu_mem_usage=True  
    )  
    
    # 设置tokenizer  
    if tokenizer.pad_token is None:  
        tokenizer.pad_token = tokenizer.eos_token  
        tokenizer.pad_token_id = tokenizer.eos_token_id  
    
    # 只训练最后的语言模型头  
    for param in model.parameters():  
        param.requires_grad = False  
    
    # 只训练lm_head  
    if hasattr(model, 'lm_head'):  
        for param in model.lm_head.parameters():  
            param.requires_grad = True  
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  
    total_params = sum(p.numel() for p in model.parameters())  
    print(f"🎯 可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")  
    
    # 创建数据集  
    dataset = ChairDataset(training_data, tokenizer, max_length=512)  
    
    # 训练参数 - 更保守的设置  
    training_args = TrainingArguments(  
        output_dir="./output/stable_chair_model",  
        overwrite_output_dir=True,  
        num_train_epochs=1,  
        per_device_train_batch_size=1,  # 小batch size  
        gradient_accumulation_steps=4,  
        learning_rate=1e-6,  # 更小的学习率  
        weight_decay=0.01,  
        logging_steps=5,  
        save_steps=50,  
        save_total_limit=2,  
        prediction_loss_only=True,  
        remove_unused_columns=False,  
        dataloader_drop_last=True,  
        fp16=False,  # 不使用混合精度  
        max_grad_norm=0.5,  # 严格的梯度裁剪  
        warmup_steps=2,  
    )  
    
    # 数据收集器  
    data_collator = DataCollatorForLanguageModeling(  
        tokenizer=tokenizer,  
        mlm=False,  
        return_tensors="pt"  
    )  
    
    # 创建Trainer  
    trainer = Trainer(  
        model=model,  
        args=training_args,  
        train_dataset=dataset,  
        data_collator=data_collator,  
        tokenizer=tokenizer,  
    )  
    
    print("🏋️ 开始稳定训练...")  
    
    try:  
        # 训练  
        trainer.train()  
        
        print("✅ 训练完成!")  
        
        # 保存模型  
        trainer.save_model("./output/stable_chair_model")  
        tokenizer.save_pretrained("./output/stable_chair_model")  
        
        print("💾 模型已保存")  
        
        # 测试模型  
        print("\n🧪 测试训练后的模型...")  
        model.eval()  
        
        test_cases = [  
            "Generate chair design: simple wooden chair",  
            "Generate chair design: modern office chair"  
        ]  
        
        for i, test_input in enumerate(test_cases):  
            print(f"\n--- 测试 {i+1} ---")  
            prompt = f"Human: {test_input}\n\nAssistant:"  
            
            try:  
                inputs = tokenizer(prompt, return_tensors="pt")  
                inputs = {k: v.to(model.device) for k, v in inputs.items()}  
                
                with torch.no_grad():  
                    outputs = model.generate(  
                        **inputs,  
                        max_length=len(inputs['input_ids'][0]) + 150,  
                        temperature=0.8,  
                        do_sample=True,  
                        top_p=0.9,  
                        pad_token_id=tokenizer.pad_token_id,  
                        eos_token_id=tokenizer.eos_token_id,  
                        repetition_penalty=1.1,  
                        no_repeat_ngram_size=3  
                    )  
                
                result = tokenizer.decode(outputs[0], skip_special_tokens=True)  
                generated = result[len(prompt):].strip()  
                
                print(f"📝 输入: {test_input}")  
                print(f"🤖 生成: {generated[:200]}...")  
                
            except Exception as e:  
                print(f"❌ 生成失败: {e}")  
        
    except Exception as e:  
        print(f"❌ 训练失败: {e}")  
        import traceback  
        traceback.print_exc()  

if __name__ == "__main__":  
    stable_train()  
