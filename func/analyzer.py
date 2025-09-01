import yaml
import os

class ConfigurableAnalyzer:
    """
    A simple, rule-based analyzer that loads scraping selectors from a YAML file.
    """
    
    def __init__(self, config_path="config/analyzer_config.yaml"):
        """
        Initializes the analyzer by loading the configuration file.
        
        Args:
            config_path (str): The path to the YAML configuration file.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        if self.config is None or 'sites' not in self.config:
            raise ValueError("Configuration file is invalid or missing 'sites' section.")
            
        print(f"✅ Selector configuration loaded from {config_path}")

    def get_site_selectors(self, site_name):
        """
        Retrieves the selectors for a specific site.
        
        Args:
            site_name (str): The name of the site (e.g., 'tokopedia').
            
        Returns:
            dict: A dictionary of selectors for the site, or None if not found.
        """
        return self.config.get('sites', {}).get(site_name)