import os
import re
from html.parser import HTMLParser

class PugConverter(HTMLParser):
    def __init__(self):
        super().__init__()
        self.pug_output = ""
        self.indent_level = 0
        self.skip_depth = 0
        self.skip_tags = ["script", "style", "link", "noscript", "iframe"]
        self.always_include = ["src", "href", "type", "name", "value", "placeholder"]

    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags or self.skip_depth > 0:
            self.skip_depth += 1
            return

        indent = "    " * self.indent_level
        attr_parts = []
        id_val = None
        class_val = None
        
        other_attrs = []
        for name, val in attrs:
            if name == "id": id_val = val
            elif name == "class": class_val = val
            elif name.startswith("data-") or name in self.always_include:
                if val:
                    safe_v = val.replace('"', "'")
                    other_attrs.append(f'{name}="{safe_v}"')

        attr_str = ""
        if id_val: attr_str += f"#{id_val}"
        if class_val: attr_str += "." + class_val.replace(" ", ".")
        if other_attrs: attr_str += f"({ ' '.join(other_attrs) })"

        self.pug_output += f"{indent}{tag}{attr_str}\n"
        self.indent_level += 1

    def handle_endtag(self, tag):
        if self.skip_depth > 0:
            self.skip_depth -= 1
            return
        self.indent_level = max(0, self.indent_level - 1)

    def handle_data(self, data):
        if self.skip_depth > 0: return
        text = data.strip()
        if text:
            indent = "    " * self.indent_level
            safe_text = text.replace('"', "'")
            self.pug_output += f"{indent}| {safe_text}\n"

def pre_clean_html(html):
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
    # Remove tags that the parser should skip anyway to be safe
    html = re.sub(r"<(script|style|link|noscript|iframe)\b[^>]*>.*?</\1>", "", html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r"<(meta|br|hr|source)\b[^>]*>", "", html, flags=re.IGNORECASE | re.DOTALL)
    return html.strip()

def run_diagnosis(task_id):
    path = f"src-tauri/tmp/task_data/{task_id}/raw_html.txt"
    if not os.path.exists(path):
        path = f"tmp/task_data/{task_id}/raw_html.txt"
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return
        
    with open(path, "r", encoding="utf-8") as f:
        raw_html = f.read()
        
    print(f"--- Diagnosis for {task_id} ---")
    print(f"Raw HTML length: {len(raw_html)}")
    
    clean_html = pre_clean_html(raw_html)
    print(f"Clean HTML length: {len(clean_html)}")
    
    converter = PugConverter()
    converter.feed(clean_html)
    
    pug_output = converter.pug_output.replace("<|", "< |").replace("|>", "| >")
    
    print(f"PUG Output length: {len(pug_output)}")
    print(f"PUG Output preview:\n{pug_output[:500]}")
    
    with open("debug_qwen3_logic.py", "w", encoding="utf-8") as f:
        f.write(f"# Diagnosed PUG for {task_id}\npug = \"\"\"{pug_output}\"\"\"")

if __name__ == "__main__":
    task_dir = "src-tauri/tmp/task_data"
    if not os.path.exists(task_dir): task_dir = "tmp/task_data"
    
    if os.path.exists(task_dir):
        folders = [f for f in os.listdir(task_dir) if f.startswith("task_")]
        if folders:
            folders.sort(reverse=True)
            run_diagnosis(folders[0])
        else:
            print("No task folders found.")
    else:
        print("Task directory not found.")