import torch  
import torch.nn as nn  
from transformers import (  
    AutoTokenizer, AutoModelForCausalLM,   
    TrainingArguments, Trainer, DataCollatorForLanguageModeling  
)  
from torch.utils.data import DataLoader  
import json  
import os  
from pathlib import Path  
from tqdm import tqdm  
import logging  
from fixed_dataset import ChairStyleDataset, custom_collate_fn  

# 设置日志  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  

class BlenderLLMFineTuner:  
    def __init__(self, base_model_path, device='auto'):  
        self.base_model_path = base_model_path  
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')  
        self.model = None  
        self.tokenizer = None  
        
    def load_model(self):  
        """加载预训练模型和tokenizer"""  
        logger.info(f"Loading model from {self.base_model_path}")  
        
        # 加载tokenizer  
        self.tokenizer = AutoTokenizer.from_pretrained(  
            self.base_model_path,  
            trust_remote_code=True,  
            padding_side='left'  
        )  
        
        # 添加pad token如果不存在  
        if self.tokenizer.pad_token is None:  
            self.tokenizer.pad_token = self.tokenizer.eos_token  
            
        # 加载模型  
        self.model = AutoModelForCausalLM.from_pretrained(  
            self.base_model_path,  
            device_map='auto' if self.device == 'cuda' else None,  
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,  
            trust_remote_code=True  
        )  
        
        logger.info("Model and tokenizer loaded successfully")  
        
    def create_dataset(self, data_path, max_length=1024):  
        """创建数据集"""  
        logger.info(f"Creating dataset from {data_path}")  
        dataset = ChairStyleDataset(data_path, self.tokenizer, max_length)  
        logger.info(f"Dataset created with {len(dataset)} samples")  
        return dataset  
        
    def train(self, dataset, output_dir, epochs=3, batch_size=4, learning_rate=2e-5, **kwargs):  
        """训练模型"""  
        if self.model is None or self.tokenizer is None:  
            self.load_model()  
            
        # 创建输出目录  
        output_path = Path(output_dir)  
        output_path.mkdir(parents=True, exist_ok=True)  
        
        # 设置训练参数  
        training_args = TrainingArguments(  
            output_dir=str(output_path),  
            num_train_epochs=epochs,  
            per_device_train_batch_size=batch_size,  
            learning_rate=learning_rate,  
            warmup_steps=kwargs.get('warmup_steps', 100),  
            logging_steps=kwargs.get('logging_steps', 50),  
            save_steps=kwargs.get('save_steps', 500),  
            eval_steps=kwargs.get('eval_steps', 500),  
            save_total_limit=2,  
            prediction_loss_only=True,  
            remove_unused_columns=False,  # 重要：防止删除自定义列  
            dataloader_pin_memory=False,  # 避免内存问题  
            fp16=kwargs.get('fp16', True) and self.device == 'cuda',  
            gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 1),  
            logging_dir=str(output_path / 'logs'),  
            report_to=None,  # 禁用wandb等  
        )  
        
        # 创建data collator  
        data_collator = DataCollatorForLanguageModeling(  
            tokenizer=self.tokenizer,  
            mlm=False,  # 不使用MLM，使用CLM  
        )  
        
        # 创建trainer  
        trainer = Trainer(  
            model=self.model,  
            args=training_args,  
            train_dataset=dataset,  
            data_collator=data_collator,  
            tokenizer=self.tokenizer,  
        )  
        
        # 开始训练  
        logger.info(f"Starting training for {epochs} epochs...")  
        try:  
            trainer.train()  
            
            # 保存模型  
            final_model_path = output_path / 'final_model'  
            trainer.save_model(str(final_model_path))  
            self.tokenizer.save_pretrained(str(final_model_path))  
            
            logger.info(f"Training completed! Model saved to {final_model_path}")  
            return str(final_model_path)  
            
        except Exception as e:  
            logger.error(f"Training failed: {e}")  
            raise e  
    
    def generate(self, prompt, max_length=512, temperature=0.7, top_p=0.9):  
        """生成文本"""  
        if self.model is None or self.tokenizer is None:  
            self.load_model()  
            
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)  
        
        with torch.no_grad():  
            outputs = self.model.generate(  
                **inputs,  
                max_length=max_length,  
                temperature=temperature,  
                top_p=top_p,  
                do_sample=True,  
                pad_token_id=self.tokenizer.eos_token_id  
            )  
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  
        return generated_text[len(prompt):]  # 返回生成的部分  