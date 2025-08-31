import os
with open('config/analyzer_config.yaml', 'r', encoding='utf-8') as f:
    config = f.read()
    print(config)