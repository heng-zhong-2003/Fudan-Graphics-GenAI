#!/usr/bin/env python3  
"""  
简化的BlenderLLM微调脚本 - 不依赖datasets模块  
"""  

import json  
import logging  
import os  
from pathlib import Path  
from typing import Dict, List  

import torch  
from torch.utils.data import Dataset  
from transformers import (  
    AutoTokenizer,   
    AutoModelForCausalLM,   
    Trainer,   
    TrainingArguments,  
    DataCollatorForLanguageModeling  
)  

# 设置日志  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  

class ChairDataset(Dataset):  
    """椅子设计数据集"""  
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):  
        self.data = data  
        self.tokenizer = tokenizer  
        self.max_length = max_length  
        
        # 确保tokenizer有pad_token  
        if tokenizer.pad_token is None:  
            tokenizer.pad_token = tokenizer.eos_token  
    
    def __len__(self):  
        return len(self.data)  
    
    def __getitem__(self, idx):  
        item = self.data[idx]  
        
        # 构建输入文本  
        input_text = item.get('input', '')  
        output_text = item.get('output', '')  
        
        # 组合为训练格式  
        if output_text:  
            full_text = f"{input_text}\n\n{output_text}"  
        else:  
            full_text = input_text  
        
        # Tokenize  
        encoding = self.tokenizer(  
            full_text,  
            truncation=True,  
            padding='max_length',  
            max_length=self.max_length,  
            return_tensors='pt'  
        )  
        
        return {  
            'input_ids': encoding['input_ids'].squeeze(),  
            'attention_mask': encoding['attention_mask'].squeeze(),  
            'labels': encoding['input_ids'].squeeze().clone()  
        }  

class SimpleBlenderLLMFineTuner:  
    """简化的BlenderLLM微调器"""  
    
    def __init__(self, model_path: str, device: str = "cuda:0"):  
        self.model_path = model_path  
        self.model = None  
        self.tokenizer = None  
        self.device = torch.device(device)  
        
        # 强制使用指定GPU  
        if "cuda" in device:  
            gpu_id = int(device.split(":")[-1])  
            torch.cuda.set_device(gpu_id)  
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  
        
        logger.info(f"Using device: {self.device}")  
    
    def load_model(self):  
        """加载模型和tokenizer"""  
        logger.info(f"Loading model from {self.model_path}")  
        
        try:  
            # 加载tokenizer  
            self.tokenizer = AutoTokenizer.from_pretrained(  
                self.model_path,  
                trust_remote_code=True,  
                use_fast=False  
            )  
            
            # 确保有pad_token  
            if self.tokenizer.pad_token is None:  
                self.tokenizer.pad_token = self.tokenizer.eos_token  
                logger.info("Set pad_token to eos_token")  
            
            # 加载模型 - 强制到指定设备  
            self.model = AutoModelForCausalLM.from_pretrained(  
                self.model_path,  
                trust_remote_code=True,  
                torch_dtype=torch.float16,  
                device_map={"": self.device},  
                low_cpu_mem_usage=True  
            )  
            
            logger.info("Model and tokenizer loaded successfully")  
            logger.info(f"Model device: {next(self.model.parameters()).device}")  
            
        except Exception as e:  
            logger.error(f"Failed to load model: {e}")  
            raise  
    
    def create_dataset(self, data_path: str, max_length: int = 512):  
        """创建数据集"""  
        logger.info(f"Creating dataset from {data_path}")  
        
        # 加载JSON数据  
        with open(data_path, 'r', encoding='utf-8') as f:  
            data = json.load(f)  
        
        if not data:  
            raise ValueError("No data loaded")  
        
        dataset = ChairDataset(data, self.tokenizer, max_length)  
        logger.info(f"Dataset created with {len(dataset)} samples")  
        
        return dataset  
    
    def train(self, dataset, output_dir: str, epochs: int = 1, batch_size: int = 1,   
              learning_rate: float = 2e-5, **kwargs):  
        """训练模型"""  
        
        if self.model is None or self.tokenizer is None:  
            raise ValueError("Model not loaded. Call load_model() first.")  
        
        # 确保输出目录存在  
        os.makedirs(output_dir, exist_ok=True)  
        
        # 训练参数  
        training_args = TrainingArguments(  
            output_dir=output_dir,  
            num_train_epochs=epochs,  
            per_device_train_batch_size=batch_size,  
            gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 1),  
            warmup_steps=kwargs.get('warmup_steps', 50),  
            learning_rate=learning_rate,  
            fp16=kwargs.get('fp16', True),  
            logging_steps=kwargs.get('logging_steps', 10),  
            save_steps=kwargs.get('save_steps', 200),  
            save_total_limit=2,  
            prediction_loss_only=True,  
            remove_unused_columns=False,  
            dataloader_pin_memory=False,  
            dataloader_num_workers=0,  
            report_to=None,  # 禁用wandb等报告  
        )  
        
        # 数据整理器  
        data_collator = DataCollatorForLanguageModeling(  
            tokenizer=self.tokenizer,  
            mlm=False,  
        )  
        
        # 创建Trainer  
        trainer = Trainer(  
            model=self.model,  
            args=training_args,  
            train_dataset=dataset,  
            data_collator=data_collator,  
            processing_class=self.tokenizer,  
        )  
        
        try:  
            logger.info(f"Starting training for {epochs} epochs...")  
            
            # 开始训练  
            trainer.train()  
            
            # 保存模型  
            model_save_path = os.path.join(output_dir, "final_model")  
            trainer.save_model(model_save_path)  
            self.tokenizer.save_pretrained(model_save_path)  
            
            logger.info(f"Training completed. Model saved to {model_save_path}")  
            return model_save_path  
            
        except Exception as e:  
            logger.error(f"Training failed: {e}")  
            raise e  
    
    def generate(self, prompt: str, max_length: int = 256, **kwargs):  
        """生成文本"""  
        if self.model is None or self.tokenizer is None:  
            raise ValueError("Model not loaded")  
        
        inputs = self.tokenizer(prompt, return_tensors="pt")  
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  
        
        with torch.no_grad():  
            outputs = self.model.generate(  
                **inputs,  
                max_length=max_length,  
                num_return_sequences=1,  
                temperature=kwargs.get('temperature', 0.7),  
                do_sample=kwargs.get('do_sample', True),  
                pad_token_id=self.tokenizer.eos_token_id  
            )  
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  
        return generated_text[len(prompt):].strip()  
