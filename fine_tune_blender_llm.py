# fine_tune_blender_llm.py  
import torch  
import torch.nn as nn  
from transformers import AutoModelForCausalLM, AutoTokenizer  
from torch.utils.data import Dataset, DataLoader  
import json  
import os  
from pathlib import Path  

class ChairStyleDataset(Dataset):  
    """椅子风格数据集"""  
    
    def __init__(self, data_path, tokenizer, max_length=512):  
        self.data_path = Path(data_path)  
        self.tokenizer = tokenizer  
        self.max_length = max_length  
        self.samples = self._load_data()  
    
    def _load_data(self):  
        samples = []  
        for chair_dir in self.data_path.iterdir():  
            if chair_dir.is_dir():  
                # 读取标签文件  
                tags_file = chair_dir / "tags.txt"  
                if tags_file.exists():  
                    with open(tags_file, 'r', encoding='utf-8') as f:  
                        tags_content = f.read()  
                    
                    # 检查三视图文件是否存在  
                    svg_files = {  
                        'front': chair_dir / f"{chair_dir.name}_front.svg",  
                        'side': chair_dir / f"{chair_dir.name}_side.svg",   
                        'top': chair_dir / f"{chair_dir.name}_top.svg"  
                    }  
                    
                    if all(f.exists() for f in svg_files.values()):  
                        samples.append({  
                            'chair_id': chair_dir.name,  
                            'tags': tags_content,  
                            'svg_files': svg_files  
                        })  
        
        return samples  
    
    def __len__(self):  
        return len(self.samples)  
    
    def __getitem__(self, idx):  
        sample = self.samples[idx]  
        
        # 构建训练提示  
        prompt = self._build_prompt(sample['tags'])  
        
        # Tokenize  
        encoding = self.tokenizer(  
            prompt,  
            truncation=True,  
            padding='max_length',  
            max_length=self.max_length,  
            return_tensors='pt'  
        )  
        
        return {  
            'input_ids': encoding['input_ids'].squeeze(),  
            'attention_mask': encoding['attention_mask'].squeeze(),  
            'chair_id': sample['chair_id'],  
            'svg_files': sample['svg_files']  
        }  
    
    def _build_prompt(self, tags_content):  
        """根据标签构建训练提示"""  
        return f"""根据以下椅子风格描述，生成对应的Blender脚本：  

{tags_content}  

请生成相应的Blender Python脚本来创建这种风格的椅子：  

```python  
"""  

class BlenderLLMFineTuner:  
    """BlenderLLM微调器"""  
    
    def __init__(self, base_model_name, output_dir):  
        self.base_model_name = base_model_name  
        self.output_dir = Path(output_dir)  
        self.output_dir.mkdir(exist_ok=True)  
        
        # 加载模型和tokenizer  
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)  
        self.model = AutoModelForCausalLM.from_pretrained(  
            base_model_name,  
            torch_dtype=torch.float16,  
            device_map="auto"  
        )  
        
        # 添加特殊token  
        self.tokenizer.pad_token = self.tokenizer.eos_token  
    
    def prepare_data(self, data_path):  
        """准备训练数据"""  
        dataset = ChairStyleDataset(data_path, self.tokenizer)  
        return dataset  
    
    def train(self, dataset, epochs=3, batch_size=4, learning_rate=2e-5, max_length=512,  
          save_steps=100, eval_steps=50, warmup_steps=100, logging_steps=10,  
          gradient_accumulation_steps=1, resume_from_checkpoint=None):  
        """训练模型 - 增强版"""  
        from torch.utils.data import DataLoader  
        from transformers import get_linear_schedule_with_warmup  
        import torch.nn.functional as F  
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  
        
        # 计算总训练步数  
        total_steps = len(dataloader) * epochs // gradient_accumulation_steps  
        
        # 优化器  
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)  
        
        # 学习率调度器  
        scheduler = get_linear_schedule_with_warmup(  
            optimizer,  
            num_warmup_steps=warmup_steps,  
            num_training_steps=total_steps  
        )  
        
        # 恢复检查点  
        start_epoch = 0  
        global_step = 0  
        if resume_from_checkpoint:  
            checkpoint = torch.load(resume_from_checkpoint)  
            self.model.load_state_dict(checkpoint['model_state_dict'])  
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  
            start_epoch = checkpoint['epoch']  
            global_step = checkpoint['global_step']  
            print(f"Resumed from checkpoint: epoch {start_epoch}, step {global_step}")  
        
        self.model.train()  
        
        print(f"Training for {epochs} epochs, {total_steps} total steps")  
        
        for epoch in range(start_epoch, epochs):  
            epoch_loss = 0  
            num_batches = 0  
            
            for batch_idx, batch in enumerate(dataloader):  
                input_ids = batch['input_ids'].to(self.model.device)  
                attention_mask = batch['attention_mask'].to(self.model.device)  
                
                # 前向传播  
                outputs = self.model(  
                    input_ids=input_ids,  
                    attention_mask=attention_mask,  
                    labels=input_ids  
                )  
                
                loss = outputs.loss / gradient_accumulation_steps  
                epoch_loss += loss.item()  
                num_batches += 1  
                
                # 反向传播  
                loss.backward()  
                
                # 梯度累积  
                if (batch_idx + 1) % gradient_accumulation_steps == 0:  
                    # 梯度裁剪  
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  
                    
                    optimizer.step()  
                    scheduler.step()  
                    optimizer.zero_grad()  
                    
                    global_step += 1  
                    
                    # 日志记录  
                    if global_step % logging_steps == 0:  
                        current_lr = scheduler.get_last_lr()[0]  
                        print(f"Epoch {epoch}, Step {global_step}, Loss: {loss.item():.6f}, LR: {current_lr:.2e}")  
                    
                    # 保存检查点  
                    if global_step % save_steps == 0:  
                        self.save_checkpoint_with_optimizer(epoch, global_step, optimizer, scheduler)  
                    
                    # 评估  
                    if global_step % eval_steps == 0:  
                        self.evaluate_model(dataset)  
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0  
            print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.6f}")  
            
            # 每个epoch结束保存检查点  
            self.save_checkpoint_with_optimizer(epoch, global_step, optimizer, scheduler)  
        
        print("Training completed!")  

    def save_checkpoint_with_optimizer(self, epoch, global_step, optimizer, scheduler):  
        """保存包含优化器状态的检查点"""  
        checkpoint_dir = self.output_dir / f"checkpoint-step-{global_step}"  
        checkpoint_dir.mkdir(exist_ok=True)  
        
        # 保存模型和tokenizer  
        self.model.save_pretrained(checkpoint_dir)  
        self.tokenizer.save_pretrained(checkpoint_dir)  
        
        # 保存训练状态  
        checkpoint_path = checkpoint_dir / "training_state.pt"  
        torch.save({  
            'epoch': epoch,  
            'global_step': global_step,  
            'model_state_dict': self.model.state_dict(),  
            'optimizer_state_dict': optimizer.state_dict(),  
            'scheduler_state_dict': scheduler.state_dict(),  
        }, checkpoint_path)  
        
        print(f"Checkpoint saved: {checkpoint_dir}")  

    def evaluate_model(self, dataset):  
        """模型评估"""  
        self.model.eval()  
        
        # 简单评估：生成一个样本  
        sample = dataset[0]  
        test_prompt = sample['tags'] if hasattr(sample, 'tags') else "生成一个现代风格的椅子"  
        
        try:  
            generated_script = self.generate_blender_script(test_prompt, max_length=512)  
            print(f"Generated script preview: {generated_script[:200]}...")  
        except Exception as e:  
            print(f"Evaluation failed: {str(e)}")  
        
        self.model.train()  
        
    def generate_blender_script(self, style_description, max_length=1024):  
        """根据风格描述生成Blender脚本"""  
        prompt = f"""根据以下椅子风格描述，生成对应的Blender脚本：  

{style_description}  

请生成相应的Blender Python脚本来创建这种风格的椅子：  

```python  
"""  
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)  
        
        with torch.no_grad():  
            outputs = self.model.generate(  
                **inputs,  
                max_length=max_length,  
                temperature=0.7,  
                do_sample=True,  
                pad_token_id=self.tokenizer.eos_token_id  
            )  
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  
        return generated_text