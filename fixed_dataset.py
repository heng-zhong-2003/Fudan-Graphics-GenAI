import torch  
from torch.utils.data import Dataset  
import json  
import os  
from pathlib import Path  

class ChairStyleDataset(Dataset):  
    def __init__(self, data_path, tokenizer, max_length=1024):  
        self.data_path = Path(data_path)  
        self.tokenizer = tokenizer  
        self.max_length = max_length  
        self.data = self._load_data()  
        
    def _load_data(self):  
        """加载数据，确保所有路径都转换为字符串"""  
        data = []  
        
        # 支持多种数据格式  
        if self.data_path.is_file():  
            # 单个文件  
            if self.data_path.suffix == '.json':  
                with open(self.data_path, 'r', encoding='utf-8') as f:  
                    raw_data = json.load(f)  
                    if isinstance(raw_data, list):  
                        data.extend(raw_data)  
                    else:  
                        data.append(raw_data)  
        else:  
            # 目录  
            for file_path in self.data_path.rglob('*.json'):  
                try:  
                    with open(file_path, 'r', encoding='utf-8') as f:  
                        file_data = json.load(f)  
                        if isinstance(file_data, list):  
                            data.extend(file_data)  
                        else:  
                            data.append(file_data)  
                except Exception as e:  
                    print(f"Warning: Failed to load {file_path}: {e}")  
                    
            # 也尝试加载txt文件  
            for file_path in self.data_path.rglob('*.txt'):  
                try:  
                    with open(file_path, 'r', encoding='utf-8') as f:  
                        lines = f.readlines()  
                        for line in lines:  
                            line = line.strip()  
                            if line:  
                                data.append({"text": line})  
                except Exception as e:  
                    print(f"Warning: Failed to load {file_path}: {e}")  
        
        # 清理数据，确保所有路径都是字符串  
        cleaned_data = []  
        for item in data:  
            cleaned_item = self._clean_item(item)  
            if cleaned_item:  
                cleaned_data.append(cleaned_item)  
                
        return cleaned_data  
    
    def _clean_item(self, item):  
        """清理数据项，将Path对象转换为字符串"""  
        if isinstance(item, dict):  
            cleaned = {}  
            for key, value in item.items():  
                if isinstance(value, (Path, str)) and key in ['file_path', 'image_path', 'model_path']:  
                    cleaned[key] = str(value)  # 转换Path为字符串  
                elif isinstance(value, dict):  
                    cleaned[key] = self._clean_item(value)  
                elif isinstance(value, list):  
                    cleaned[key] = [self._clean_item(v) if isinstance(v, dict) else str(v) if isinstance(v, Path) else v for v in value]  
                else:  
                    cleaned[key] = value  
            return cleaned  
        elif isinstance(item, str):  
            return {"text": item}  
        return None  
    
    def __len__(self):  
        return len(self.data)  
    
    def __getitem__(self, idx):  
        item = self.data[idx]  
        
        # 构建输入文本  
        if isinstance(item, dict):  
            if 'input' in item and 'output' in item:  
                input_text = item['input']  
                target_text = item['output']  
            elif 'style_description' in item:  
                input_text = f"Generate chair design: {item['style_description']}"  
                target_text = item.get('blender_code', '')  
            elif 'text' in item:  
                input_text = item['text']  
                target_text = ""  
            else:  
                input_text = str(item)  
                target_text = ""  
        else:  
            input_text = str(item)  
            target_text = ""  
        
        # Tokenize  
        inputs = self.tokenizer(  
            input_text,  
            max_length=self.max_length,  
            padding='max_length',  
            truncation=True,  
            return_tensors='pt'  
        )  
        
        targets = self.tokenizer(  
            target_text,  
            max_length=self.max_length,  
            padding='max_length',  
            truncation=True,  
            return_tensors='pt'  
        )  
        
        return {  
            'input_ids': inputs['input_ids'].squeeze(),  
            'attention_mask': inputs['attention_mask'].squeeze(),  
            'labels': targets['input_ids'].squeeze(),  
            'target_attention_mask': targets['attention_mask'].squeeze(),  
        }  

def custom_collate_fn(batch):  
    """自定义collate函数，处理字符串和Path对象"""  
    keys = batch[0].keys()  
    result = {}  
    
    for key in keys:  
        values = [item[key] for item in batch]  
        
        # 确保都是tensor  
        if all(isinstance(v, torch.Tensor) for v in values):  
            result[key] = torch.stack(values)  
        else:  
            # 如果不是tensor，转换为tensor  
            result[key] = torch.stack([torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in values])  
    
    return result  