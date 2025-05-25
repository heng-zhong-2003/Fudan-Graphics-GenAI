"""  
内存优化的BlenderLLM微调脚本  
"""  

import json  
import logging  
import os  
from pathlib import Path  
from typing import Dict, List  
import gc  

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
    """内存优化的椅子设计数据集"""  
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 256):  
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
        
        # 截断过长的文本以节省内存  
        if len(input_text) > 200:  
            input_text = input_text[:200]  
        if len(output_text) > 300:  
            output_text = output_text[:300]  
        
        # 组合为训练格式  
        if output_text:  
            full_text = f"{input_text}\n{output_text}"  
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

class MemoryOptimizedBlenderLLMFineTuner:  
    """内存优化的BlenderLLM微调器"""  
    
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
    
    def clear_memory(self):  
        """清理内存"""  
        gc.collect()  
        torch.cuda.empty_cache()  
    
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
            
            # 内存优化的模型加载  
            self.model = AutoModelForCausalLM.from_pretrained(  
                self.model_path,  
                trust_remote_code=True,  
                torch_dtype=torch.float16,  # 使用半精度  
                device_map={"": self.device},  
                low_cpu_mem_usage=True,  
                # 使用8bit量化以节省内存（如果可用）  
                # load_in_8bit=True,  # 需要bitsandbytes库  
            )  
            
            # 启用梯度检查点以节省内存  
            self.model.gradient_checkpointing_enable()  
            
            logger.info("Model and tokenizer loaded successfully")  
            logger.info(f"Model device: {next(self.model.parameters()).device}")  
            
            # 显示内存使用  
            if torch.cuda.is_available():  
                allocated = torch.cuda.memory_allocated() / 1e9  
                reserved = torch.cuda.memory_reserved() / 1e9  
                logger.info(f"GPU memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")  
            
        except Exception as e:  
            logger.error(f"Failed to load model: {e}")  
            raise  
    
    def create_dataset(self, data_path: str, max_length: int = 256):  
        """创建数据集"""  
        logger.info(f"Creating dataset from {data_path}")  
        
        # 加载JSON数据  
        with open(data_path, 'r', encoding='utf-8') as f:  
            data = json.load(f)  
        
        if not data:  
            raise ValueError("No data loaded")  
        
        # 如果数据太多，进行采样以节省内存  
        if len(data) > 500:  
            logger.info(f"Sampling {500} examples from {len(data)} total")  
            import random  
            data = random.sample(data, 500)  
        
        dataset = ChairDataset(data, self.tokenizer, max_length)  
        logger.info(f"Dataset created with {len(dataset)} samples")  
        
        return dataset  
    
    def train(self, dataset, output_dir: str, epochs: int = 1, batch_size: int = 1,   
              learning_rate: float = 1e-5, **kwargs):  
        """内存优化的训练"""  
        
        if self.model is None or self.tokenizer is None:  
            raise ValueError("Model not loaded. Call load_model() first.")  
        
        # 确保输出目录存在  
        os.makedirs(output_dir, exist_ok=True)  
        
        # 内存优化的训练参数  
        training_args = TrainingArguments(  
            output_dir=output_dir,  
            num_train_epochs=epochs,  
            per_device_train_batch_size=batch_size,  
            gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 4),  
            warmup_steps=kwargs.get('warmup_steps', 20),  
            learning_rate=learning_rate,  
            fp16=True,  # 半精度训练  
            logging_steps=kwargs.get('logging_steps', 50),  
            save_steps=kwargs.get('save_steps', 300),  
            save_total_limit=1,  # 只保存最新的checkpoint  
            prediction_loss_only=True,  
            remove_unused_columns=False,  
            dataloader_pin_memory=False,  
            dataloader_num_workers=0,  
            report_to=None,  
            # 额外的内存优化  
            max_grad_norm=kwargs.get('max_grad_norm', 1.0),  
            optim="adamw_torch",  # 使用内存效率更高的优化器  
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
            logger.info(f"Starting memory optimized training for {epochs} epochs...")  
            
            # 清理内存  
            self.clear_memory()  
            
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
            self.clear_memory()  
            raise e  
    
    def generate(self, prompt: str, max_length: int = 150, **kwargs):  
        """生成文本"""  
        if self.model is None or self.tokenizer is None:  
            raise ValueError("Model not loaded")  
        
        inputs = self.tokenizer(prompt, return_tensors="pt")  
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  
        
        # 清理内存  
        self.clear_memory()  
        
        with torch.no_grad():  
            outputs = self.model.generate(  
                **inputs,  
                max_length=max_length,  
                num_return_sequences=1,  
                temperature=kwargs.get('temperature', 0.7),  
                do_sample=kwargs.get('do_sample', True),  
                pad_token_id=self.tokenizer.eos_token_id,  
                eos_token_id=self.tokenizer.eos_token_id,  
            )  
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  
        return generated_text[len(prompt):].strip()  
