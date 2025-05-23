import os
import openai
from PIL import Image
import base64
import cairosvg
import io

# 设置你的 OpenAI API Key
openai.api_key = "your-api-key"

# 所有可能的属性
attribute_prompt = """
请根据以下提供的椅子的三视图和文字描述，在每组属性中选择最符合的一项，若没有符合的则返回 null。

### 传统/古典风格
- 巴洛克风格 (Baroque)
- 洛可可风格 (Rococo)
- 维多利亚风格 (Victorian)
- 乔治亚风格 (Georgian)
- 路易十四/十五/十六风格 (Louis XIV/XV/XVI)
- 温莎风格 (Windsor)
- 柴郡风格 (Chippendale)

### 现代风格
- 现代主义 (Modernist)
- 极简主义 (Minimalist)
- 斯堪的纳维亚/北欧风格 (Scandinavian/Nordic)
- 工业风格 (Industrial)
- 中世纪现代 (Mid-century Modern)
- 后现代主义 (Postmodern)
- 功能主义 (Functionalist)
- 有机现代 (Organic Modern)

### 其他特色风格
- 乡村风格 (Rustic/Country)
- 艺术装饰风格 (Art Deco)
- 艺术nouveau风格 (Art Nouveau)
- 波西米亚风格 (Bohemian)
- 东方/亚洲风格 (Oriental/Asian)
- 地中海风格 (Mediterranean)
- 复古风格 (Retro/Vintage)
- 折衷主义 (Eclectic)

### 材质相关描述
- 藤编风格 (Rattan/Wicker)
- 皮革风格 (Leather)
- 金属工业风 (Metal Industrial)
- 实木原生态 (Solid Wood Natural)
- 透明亚克力 (Transparent Acrylic)

### 功能型椅子
- 人体工学 (Ergonomic)
- 游戏椅风格 (Gaming Chair)
- 可调节办公 (Adjustable Office)
- 多功能折叠 (Multifunctional Folding)

### 主要功能
- 就座
- 休闲
- 工作
- 特殊功能

### 人体工学符合性
- 高
- 中
- 低

### 高度可调节性
- 有
- 无

### 角度可调节性
- 有
- 无

### 折叠性
- 有
- 无

请按以下格式输出：
传统/古典风格: xxx
现代风格: xxx
其他特色风格: xxx
材质相关描述: xxx
功能型椅子: xxx
主要功能: xxx
人体工学符合性: xxx
高度可调节性: xxx
角度可调节性: xxx
折叠性: xxx
"""

# 将SVG转换为PNG
def convert_svg_to_png(svg_path):
    png_data = cairosvg.svg2png(url=svg_path)
    return png_data

# 编码图片为base64
def encode_image(image_data):
    return base64.b64encode(image_data).decode('utf-8')

# 编码 svg 文件为 base64
def encode_svg(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# 主路径
root_path = r"D:\edu\CAD\github_chair\Fudan-Graphics-GenAI\data_grouped"

# 遍历所有子文件夹
for chair_id in os.listdir(root_path):
    folder_path = os.path.join(root_path, chair_id)
    if not os.path.isdir(folder_path):
        print("!!!fault")
        continue

    # 获取三张 svg 图像
    image_paths = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".svg")
    ])[:3]

    if len(image_paths) < 3:
        print(f"[跳过] {chair_id}：三视图不足三张")
        continue

    # 获取文字描述
    desc_path = os.path.join(folder_path, "description.txt")
    description = ""
    if os.path.exists(desc_path):
        with open(desc_path, "r", encoding="utf-8") as f:
            description = f.read()

    # 构造 image 输入
    image_inputs = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encode_image(convert_svg_to_png(img_path))}"
            }
        }
        for img_path in image_paths
    ]

    # 构造消息
    messages = [
        {"role": "system", "content": "你是一个家具风格专家。"},
        {"role": "user", "content": image_inputs + [
            {"type": "text", "text": description + "\n\n" + attribute_prompt}
        ]}
    ]

    # 调用模型
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            temperature=0
        )
        output = response.choices[0].message.content.strip()

        # 写入标签结果
        output_file = os.path.join(folder_path, "tags.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output)

        print(f"[完成] {chair_id}")
    except Exception as e:
        print(f"[失败] {chair_id}：{e}")
