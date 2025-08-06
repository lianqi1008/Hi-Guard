import json
import re

PROMPT="""
### 风险点:
...

### 任务要求

请我提供内容后，按照以下步骤进行分析：
1. 仔细阅读内容，识别其中可能的风险点或违规元素
2. 如果存在上面任一风险点，则判定为有风险，如果上面的风险点都不存在，判定为无风险。

### 输出格式：

有风险或者无风险

现在，请提供需要分类的内容：
"""

input_file = 'json_path'
output_file = 'jsonl_path'

# Read the original JSON file
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Process each item and convert to the new format
with open(output_file, 'w', encoding='utf-8') as out_f:
    for item in data:
        # Extract the image path
        images = item.get('images', [])
        
        # Create the <image> token
        image_tokens = '<image>'
        
        # Extract text
        text = item.get('text', '')
        
        # Extract solution - getting content between <answer> tags
        solution_text = item.get('solution', '')
        answer_match = re.search(r'<answer>(.*?)</answer>', solution_text)
        if answer_match:
            answer = answer_match.group(1)
        else:
            answer = solution_text

        # if answer != "无风险":
        #     answer = "有风险"

        new_item = {
            "conversations": [
                {"from": "human", "value": f"{PROMPT}{image_tokens}{text}"},
                {"from": "gpt", "value": answer}
            ],
            "system": "你是一名资深风险分类专家，你的任务是对所给的内容深入理解并判定内容涉及的风险类别。",
            "images": [images[0]] if len(images)>0 else []
        }
        
        # Write each item as a JSON line
        out_f.write(json.dumps(new_item, ensure_ascii=False) + '\n')
    
data_list = []
with open(output_file, 'r', encoding='utf-8') as f:
    for line in f:
        data_list.append(json.loads(line.strip()))
        
print(len(data_list))
print(output_file)

print("Conversion completed.")

