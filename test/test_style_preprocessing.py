
## 创建测试文件  

```python  
# tests/test_style_preprocessing.py  
import pytest  
import json  
from pathlib import Path  
import tempfile  
import sys  
sys.path.append('..')  

from preprocess_styles import StylePreprocessor  

class TestStylePreprocessor:  
    
    def setup_method(self):  
        self.preprocessor = StylePreprocessor()  
    
    def test_structured_format_parsing(self):  
        """测试结构化格式解析"""  
        description = """传统/古典风格: 维多利亚风格 (Victorian)  
现代风格: null  
其他特色风格: 奢华古典 (Luxury Classical)  
材质相关描述: 天鹅绒布艺 (Velvet Fabric)  
功能型椅子: null  
主要功能: 装饰展示  
人体工学符合性: 中  
高度可调节性: 无  
角度可调节性: 无  
折叠性: 无"""  
        
        result = self.preprocessor.process_description(description)  
        
        assert result is not None  
        assert result['metadata']['traditional_style'] == '维多利亚风格 (Victorian)'  
        assert result['metadata']['modern_style'] == 'null'  
        assert result['metadata']['main_function'] == '装饰展示'  
        assert result['metadata']['ergonomics'] == '中'  
    
    def test_free_text_format(self):  
        """测试自由文本格式"""  
        description = "设计一把现代简约风格的办公椅，采用人体工学设计。"  
        
        result = self.preprocessor.process_description(description)  
        
        assert result is not None  
        assert 'formatted_description' in result  
        assert len(result['formatted_description']) > 0  
    
    def test_formatted_description_generation(self):  
        """测试格式化描述生成"""  
        metadata = {  
            'traditional_style': 'null',  
            'modern_style': '极简主义',  
            'material': '实木原生态',  
            'main_function': '就座',  
            'ergonomics': '高'  
        }  
        
        from preprocess_styles import generate_formatted_description  
        desc = generate_formatted_description(metadata)  
        
        assert '现代极简主义风格' in desc  
        assert '实木原生态材质' in desc  
        assert '就座' in desc  
    
    def test_batch_processing(self):  
        """测试批量处理"""  
        test_input = """### 测试样例1  
传统/古典风格: null  
现代风格: 极简主义  
材质相关描述: 实木  

### 测试样例2  
传统/古典风格: 古典  
现代风格: null  
材质相关描述: 金属"""  
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:  
            f.write(test_input)  
            input_file = f.name  
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:  
            output_file = f.name  
        
        try:  
            self.preprocessor.process_file(input_file, output_file)  
            
            with open(output_file) as f:  
                results = json.load(f)  
            
            assert len(results) == 2  
            assert results[0]['metadata']['modern_style'] == '极简主义'  
            assert results[1]['metadata']['traditional_style'] == '古典'  
            
        finally:  
            Path(input_file).unlink()  
            Path(output_file).unlink()  

# tests/test_batch_processing.py  
import pytest  
import json  
import tempfile  
from pathlib import Path  
from unittest.mock import Mock, patch  
import sys  
sys.path.append('..')  

from batch_process import BatchProcessor  

class TestBatchProcessor:  
    
    @patch('batch_process.BlenderLLMGenerator')  
    def test_initialization(self, mock_generator_class):  
        """测试初始化"""  
        mock_generator = Mock()  
        mock_generator_class.return_value = mock_generator  
        
        processor = BatchProcessor(  
            model_path="/fake/model/path",  
            max_workers=2  
        )  
        
        assert processor.max_workers == 2  
        mock_generator_class.assert_called_once_with("/fake/model/path")  
    
    def test_style_loading(self):  
        """测试风格加载"""  
        test_styles = [  
            {  
                "id": "chair_1",  
                "formatted_description": "设计一把现代椅子",  
                "metadata": {"style": "modern"}  
            }  
        ]  
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:  
            json.dump(test_styles, f)  
            styles_file = f.name  
        
        try:  
            processor = BatchProcessor("/fake/model", max_workers=1)  
            styles = processor.load_styles(styles_file)  
            
            assert len(styles) == 1  
            assert styles[0]['id'] == 'chair_1'  
            
        finally:  
            Path(styles_file).unlink()  
    
    @patch('batch_process.BlenderLLMGenerator')  
    def test_single_style_processing_success(self, mock_generator_class):  
        """测试单个风格处理成功"""  
        mock_generator = Mock()  
        mock_generator.generate.return_value = {  
            'status': 'success',  
            'generated_views': {  
                'front': '/path/to/front.png',  
                'side': '/path/to/side.png'  
            }  
        }  
        mock_generator_class.return_value = mock_generator  
        
        processor = BatchProcessor("/fake/model", max_workers=1)  
        
        style = {  
            "id": "chair_1",  
            "formatted_description": "设计一把现代椅子"  
        }  
        
        result = processor.process_single_style(style, "/fake/output", 0)  
        
        assert result['status'] == 'success'  
        assert result['style_id'] == 'chair_1'  
        mock_generator.generate.assert_called_once()  
    
    @patch('batch_process.BlenderLLMGenerator')  
    def test_single_style_processing_failure(self, mock_generator_class):  
        """测试单个风格处理失败"""  
        mock_generator = Mock()  
        mock_generator.generate.side_effect = Exception("Generation failed")  
        mock_generator_class.return_value = mock_generator  
        
        processor = BatchProcessor("/fake/model", max_workers=1)  
        
        style = {  
            "id": "chair_1",   
            "formatted_description": "设计一把现代椅子"  
        }  
        
        result = processor.process_single_style(style, "/fake/output", 0)  
        
        assert result['status'] == 'failed'  
        assert 'Generation failed' in result['error']  

if __name__ == "__main__":  
    pytest.main([__file__])