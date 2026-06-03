"""

This sub folder has some function iclude:
    1. fetch data asyncrounously from data/bronze/...
    2. process raw html using beatifoulsoup asyncronously.
    3. and write output into jsonl file streamly using append.

considering to give a thread pool for process so we can run fast.
Also we will parse it by YAML config

three class here have some function:
    1. call YAML config to load
    2. class to perform single batch parsing
    3. extraction id from data

"""
import json
import yaml
from bs4 import BeautifulSoup
from typing import Dict,List,Any
from urllib.parse import parse_qs, unquote
import re

def _load_config(config_path: str = "config/selector.yaml") -> Dict:
    """ Load safely """
    with open(config_path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)
    
def load_bronze(file_path: str, config: Dict) -> List[Dict]:
    """ 
    We do preparation here to call json file from bronze-level.
    Also make sure _extract_html() and _transform_derived() do their things
    also we return List[Dict] here to return on records 
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            bronze_data = json.load(f)
    except Exception as e:
        print("[Parser] Error reading file {file_path}: {e}")
        return []
    
    if isinstance(bronze_data, Dict):
        bronze_data = [bronze_data]

    return bronze_data

def _extract_html(bronze_data, config: Dict ) -> List[Dict]:
    """
    we process config here to be more readable, and do the extraction here.
    keep it clean simple.
    """
    container_fields    = config.get('container_fields', 'raw_cards')
    html_extractions    = config.get('html_extraction', {})
    derived_fields = config.get('derived_fields', {})
    
    for record in bronze_data:
        raw_html = record.get(container_fields)
        if not raw_html:
            print(f"    [Parser] Field '{container_fields}' Not found in the record."
                  f"Keys appear: {list(record.keys())}")
            continue

        soup = BeautifulSoup(raw_html, 'html.parser')
        extracted = {}

        # do extraction here
        for field, rules in html_extractions.items():
            css         = rules.get('css')
            fall_css    = rules.get('fallback_css')
            attr        = rules.get('attribute', 'text')
            fall_attr   = rules.get('fallback_attribute', 'alt') 

            #We only take one element from css cause the list only contain one at the time
            element = soup.select_one(css)
            val = None

            if element:
                val = element.get_text(strip=True) if attr == 'text' else element.get(attr)

            if not val and fall_css:
                element = soup.select_one(fall_css)
                attr = fall_attr or attr
                if element:
                    val = element.get_text(strip=True) if attr == 'text' else element.get(attr)
            
            if not val:
                extracted[field] = rules.get('default_value')
                continue

            extracted[field] = val

    return extracted

def _transform_derived(record, extracted, derived_fields) -> Dict:
    """
    Perform cleaning transformation here.
    """

    silver = {}
    for field, rules in derived_fields.items():
        source = rules.get('source')

        if source == 'bronze_root_field':
            silver[field] = record.get(rules.get('key'))
        
        elif source == 'url_array_index_0':
            url_array = record.get('url', [])
            silver[field] = url_array[0] if url_array else None 

        # price product after discount and before discont if there a discount
        elif source == 'transformation_clean_digits' and rules.get('depends_on'):
            val = extracted.get(rules.get('depends_on'))
            if val:
                silver[field] = int(re.sub(rules.get('clean_regex'), '', str(val)))
            else:
                silver[field] = rules.get('default_value')
        
        elif source == 'transformation_extract_number' and rules.get('depends_on'):
            val = extracted.get(rules.get('depends_on'))
            if val:
                match = re.search(rules.get('clean_regex'), str(val))
                silver[field] = int(match.group(1) if match else rules.get('default_value', 0))

        elif source == 'url_query_param':
            silver[field] = _extract_query_param(record, rules)
        
        else: silver[field] = extracted.get(field)
    
    # Merge result from _extract_html
    for k, v  in extracted.items():
        if k not in silver:
            silver[k] = v

    return silver

def _validate_record(record: Dict,silver: Dict,data_quality_rules: List) -> Any:
    """ validate record data by rules from data_quality_rules on selector.yaml"""
    # rules from required_fields yaml
    req = data_quality_rules.get('required_fields', [])
    # rules from drop_if_null yaml
    null = data_quality_rules.get('drop_if_null', [])
    for field in req:
        if source:
            val = silver.get(rules.get('required_fields'))
            if val:
                


def _extract_query_param(record: Dict, rules: Dict) -> Any:
    """Helper to parse nested query parameter, much more clean if we make it here"""
    try:
        url_array = record.get("url", [])
        if len(url_array) < 2:
            return None
        ext_param = parse_qs(url_array[1]).get (rules.get('parent_key'), [None]) [0]
        if ext_param:
            inner = parse_qs(unquote(ext_param))
            return inner.get(rules.get('child_key'), [None])[0]
    except:
        pass
    return None