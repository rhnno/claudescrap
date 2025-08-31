#!/usr/bin/env python3
"""
Continuous Bug & Logic Scanner System
Real-time monitoring and error detection for Enhanced ML-Powered Scraper
"""

import os
import sys
import ast
import json
import yaml
import time
import pickle
import inspect
import importlib
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

# Watchdog for file monitoring
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    print("⚠️ Watchdog not available. Install with: pip install watchdog")
    WATCHDOG_AVAILABLE = False

@dataclass
class BugReport:
    """Represents a bug or issue found by the scanner"""
    category: str  # 'critical', 'warning', 'config', 'performance', 'integration'
    severity: int  # 1-5 (5 = critical)
    title: str
    description: str
    file_path: str
    line_number: Optional[int]
    suggestion: str
    timestamp: datetime
    
    def to_dict(self):
        return {
            'category': self.category,
            'severity': self.severity,
            'title': self.title,
            'description': self.description,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'suggestion': self.suggestion,
            'timestamp': self.timestamp.isoformat()
        }

class BugScanner:
    """Comprehensive bug and logic scanner"""
    
    def __init__(self, project_path="./", watch_mode=True, focus_modules=None):
        """Initialize the bug scanner"""
        self.project_path = Path(project_path)
        self.watch_mode = watch_mode
        self.focus_modules = focus_modules or [
            "func.browser",
            "func.analyzer", 
            "func.storage",
            "func.utils"
        ]
        
        # Bug tracking
        self.bugs = []
        self.last_scan_time = None
        self.scan_count = 0
        self.health_score = 100
        
        # File monitoring
        self.observer = None
        self.file_handler = None
        
        # Performance tracking
        self.performance_metrics = {
            'scan_times': [],
            'file_changes': 0,
            'bugs_found': 0,
            'bugs_fixed': 0
        }
        
        print("🔍 Continuous Bug Scanner initialized")
        print(f"📁 Project path: {self.project_path}")
        print(f"👀 Watch mode: {watch_mode}")
        print(f"🎯 Focus modules: {self.focus_modules}")
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        print("\n🚀 Starting continuous bug monitoring...")
        
        # Initial scan
        self.run_comprehensive_scan()
        
        # Start file watching if available
        if self.watch_mode and WATCHDOG_AVAILABLE:
            self._start_file_watcher()
        
        # Start monitoring loop
        self._monitoring_loop()
    
    def _start_file_watcher(self):
        """Start file system watcher"""
        try:
            self.file_handler = FileChangeHandler(self)
            self.observer = Observer()
            self.observer.schedule(self.file_handler, str(self.project_path), recursive=True)
            self.observer.start()
            print("👀 File watcher started")
        except Exception as e:
            print(f"⚠️ File watcher failed: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while True:
                time.sleep(30)  # Scan every 30 seconds
                self.run_comprehensive_scan()
                self._display_dashboard()
        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped by user")
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
        finally:
            if self.observer:
                self.observer.stop()
                self.observer.join()
    
    def run_comprehensive_scan(self):
        """Run comprehensive bug scan"""
        start_time = time.time()
        self.scan_count += 1
        
        print(f"\n🔍 Running comprehensive scan #{self.scan_count}...")
        
        # Clear previous bugs
        old_bug_count = len(self.bugs)
        self.bugs = []
        
        # A. Method Signature Analysis
        self._scan_method_signatures()
        
        # B. Configuration Validation
        self._scan_configurations()
        
        # C. ML Pipeline Integrity
        self._scan_ml_pipeline()
        
        # D. Import Dependency Analysis
        self._scan_import_dependencies()
        
        # E. Performance Analysis
        self._scan_performance_issues()
        
        # Update metrics
        scan_time = time.time() - start_time
        self.performance_metrics['scan_times'].append(scan_time)
        self.last_scan_time = datetime.now()
        
        # Calculate health score
        self._calculate_health_score()
        
        # Report changes
        new_bug_count = len(self.bugs)
        if new_bug_count != old_bug_count:
            print(f"📊 Bug count changed: {old_bug_count} → {new_bug_count}")
        
        print(f"✅ Scan completed in {scan_time:.2f}s")
    
    def _scan_method_signatures(self):
        """A. Method Signature Analysis"""
        print("🔍 Scanning method signatures...")
        
        try:
            # Check BrowserManager for parameter issues
            browser_file = self.project_path / "func" / "browser.py"
            if browser_file.exists():
                self._analyze_browser_manager(browser_file)
            
            # Check other modules for signature mismatches
            for module_name in self.focus_modules:
                self._analyze_module_signatures(module_name)
                
        except Exception as e:
            self._add_bug("critical", 5, "Method Signature Scan Failed", 
                         f"Could not scan method signatures: {e}", 
                         "func/", None, "Check file permissions and syntax")
    
    def _analyze_browser_manager(self, file_path):
        """Analyze BrowserManager class specifically"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == "BrowserManager":
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                            # Check __init__ parameters
                            args = [arg.arg for arg in item.args.args[1:]]  # Skip 'self'
                            
                            expected_params = ['use_profile', 'profile_name', 'headless']
                            if 'headless' not in args:
                                self._add_bug("critical", 5, "Missing headless parameter",
                                            "BrowserManager.__init__ missing headless parameter",
                                            str(file_path), item.lineno,
                                            "Add headless=False parameter to __init__ method")
                            else:
                                print("✅ BrowserManager headless parameter found")
                                
        except Exception as e:
            self._add_bug("warning", 3, "BrowserManager Analysis Failed",
                         f"Could not analyze BrowserManager: {e}",
                         str(file_path), None, "Check file syntax")
    
    def _analyze_module_signatures(self, module_name):
        """Analyze module for signature issues"""
        try:
            # Import module dynamically
            # Using proper package imports
            module = importlib.import_module(module_name)
            
            # Check for common signature issues
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj):
                    self._check_class_methods(obj, module_name)
                    
        except ImportError as e:
            self._add_bug("integration", 4, f"Module Import Failed: {module_name}",
                         f"Could not import {module_name}: {e}",
                         f"{module_name.replace('.', '/')}.py", None,
                         "Check module dependencies and syntax")
        except Exception as e:
            self._add_bug("warning", 2, f"Module Analysis Failed: {module_name}",
                         f"Could not analyze {module_name}: {e}",
                         f"{module_name.replace('.', '/')}.py", None,
                         "Check module structure")
    
    def _check_class_methods(self, cls, module_name):
        """Check class methods for common issues"""
        try:
            for method_name, method in inspect.getmembers(cls, inspect.ismethod):
                sig = inspect.signature(method)
                
                # Check for common parameter issues
                if method_name == "__init__":
                    params = list(sig.parameters.keys())
                    
                    # Specific checks for known classes
                    if cls.__name__ == "BrowserManager":
                        if 'headless' not in params:
                            self._add_bug("critical", 5, "Missing headless parameter",
                                        f"{cls.__name__}.__init__ missing headless parameter",
                                        f"{module_name.replace('.', '/')}.py", None,
                                        "Add headless parameter to __init__ method")
                        
        except Exception as e:
            pass  # Skip method analysis errors
    
    def _scan_configurations(self):
        """B. Configuration Validation"""
        print("🔍 Scanning configurations...")
        
        # Check YAML files
        yaml_files = [
            "config/analyzer_config.yaml",
            "config/enhanced_training_templates.yaml"
        ]
        
        for yaml_file in yaml_files:
            file_path = self.project_path / yaml_file
            if file_path.exists():
                self._validate_yaml_file(file_path)
            else:
                self._add_bug("config", 3, f"Missing config file: {yaml_file}",
                             f"Required configuration file not found: {yaml_file}",
                             yaml_file, None, f"Create {yaml_file} file")
        
        # Check JSON files
        json_files = [
            "config/login_credentials.json",
            "scraping_config.json"
        ]
        
        for json_file in json_files:
            file_path = self.project_path / json_file
            if file_path.exists():
                self._validate_json_file(file_path)
    
    def _validate_yaml_file(self, file_path):
        """Validate YAML file structure"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if not data:
                self._add_bug("config", 3, f"Empty YAML file: {file_path.name}",
                             f"YAML file is empty or invalid: {file_path}",
                             str(file_path), None, "Add valid YAML content")
                return
            
            # Specific validation for known files
            if "enhanced_training_templates" in file_path.name:
                self._validate_training_templates(data, file_path)
            elif "analyzer_config" in file_path.name:
                self._validate_analyzer_config(data, file_path)
                
            print(f"✅ YAML valid: {file_path.name}")
            
        except yaml.YAMLError as e:
            self._add_bug("config", 4, f"YAML syntax error: {file_path.name}",
                         f"YAML syntax error in {file_path}: {e}",
                         str(file_path), None, "Fix YAML syntax")
        except Exception as e:
            self._add_bug("config", 3, f"YAML validation failed: {file_path.name}",
                         f"Could not validate {file_path}: {e}",
                         str(file_path), None, "Check file permissions")
    
    def _validate_training_templates(self, data, file_path):
        """Validate training templates structure"""
        required_sections = ['training_templates']
        
        for section in required_sections:
            if section not in data:
                self._add_bug("config", 3, f"Missing section: {section}",
                             f"Required section '{section}' missing in {file_path.name}",
                             str(file_path), None, f"Add {section} section")
        
        # Check ecommerce sites
        if 'training_templates' in data:
            templates = data['training_templates']
            if 'ecommerce_sites' in templates:
                sites = templates['ecommerce_sites']
                if len(sites) == 0:
                    self._add_bug("config", 2, "No e-commerce sites configured",
                                 "No e-commerce sites found in training templates",
                                 str(file_path), None, "Add site configurations")
                else:
                    print(f"✅ Found {len(sites)} e-commerce site configurations")
    
    def _validate_analyzer_config(self, data, file_path):
        """Validate analyzer config structure"""
        required_sections = ['paths', 'features', 'training']
        
        for section in required_sections:
            if section not in data:
                self._add_bug("config", 3, f"Missing config section: {section}",
                             f"Required section '{section}' missing in analyzer config",
                             str(file_path), None, f"Add {section} section")
    
    def _validate_json_file(self, file_path):
        """Validate JSON file structure"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                self._add_bug("config", 3, f"Empty JSON file: {file_path.name}",
                             f"JSON file is empty: {file_path}",
                             str(file_path), None, "Add valid JSON content")
                return
            
            print(f"✅ JSON valid: {file_path.name}")
            
        except json.JSONDecodeError as e:
            self._add_bug("config", 4, f"JSON syntax error: {file_path.name}",
                         f"JSON syntax error in {file_path}: {e}",
                         str(file_path), None, "Fix JSON syntax")
        except Exception as e:
            self._add_bug("config", 3, f"JSON validation failed: {file_path.name}",
                         f"Could not validate {file_path}: {e}",
                         str(file_path), None, "Check file permissions")
    
    def _scan_ml_pipeline(self):
        """C. ML Pipeline Integrity"""
        print("🔍 Scanning ML pipeline...")
        
        # Check model files
        model_files = [
            "models/enhanced_pagination_model.pkl",
            "models/enhanced_pagination_vectorizer.pkl"
        ]
        
        for model_file in model_files:
            file_path = self.project_path / model_file
            if file_path.exists():
                self._validate_model_file(file_path)
            else:
                self._add_bug("warning", 2, f"Missing model file: {model_file}",
                             f"ML model file not found: {model_file}",
                             model_file, None, "Train model or check file path")
        
        # Check analyzer integration
        self._check_analyzer_integration()
    
    def _validate_model_file(self, file_path):
        """Validate ML model file"""
        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            
            # Basic model validation
            if hasattr(model, 'predict'):
                print(f"✅ Model valid: {file_path.name}")
            else:
                self._add_bug("warning", 3, f"Invalid model: {file_path.name}",
                             f"Model file does not have predict method: {file_path}",
                             str(file_path), None, "Retrain model")
                
        except Exception as e:
            self._add_bug("warning", 3, f"Model validation failed: {file_path.name}",
                         f"Could not validate model {file_path}: {e}",
                         str(file_path), None, "Check model file integrity")
    
    def _check_analyzer_integration(self):
        """Check EnhancedConfigurableAnalyzer integration"""
        try:
            # Using proper package imports
            from func import EnhancedConfigurableAnalyzer
            
            # Test initialization
            analyzer = EnhancedConfigurableAnalyzer()
            print("✅ EnhancedConfigurableAnalyzer integration OK")
            
        except ImportError as e:
            self._add_bug("integration", 4, "Analyzer import failed",
                         f"Could not import EnhancedConfigurableAnalyzer: {e}",
                         "func/analyzer.py", None, "Check analyzer module")
        except Exception as e:
            self._add_bug("integration", 3, "Analyzer initialization failed",
                         f"Could not initialize analyzer: {e}",
                         "func/analyzer.py", None, "Check analyzer configuration")
    
    def _scan_import_dependencies(self):
        """D. Import Dependency Analysis"""
        print("🔍 Scanning import dependencies...")
        
        # Check critical imports
        critical_modules = [
            ('selenium', 'Selenium WebDriver'),
            ('pandas', 'Data processing'),
            ('numpy', 'Numerical computing'),
            ('sklearn', 'Machine learning'),
            ('yaml', 'YAML processing')
        ]
        
        for module_name, description in critical_modules:
            try:
                importlib.import_module(module_name)
                print(f"✅ {description} available")
            except ImportError:
                self._add_bug("integration", 4, f"Missing dependency: {module_name}",
                             f"Required module not available: {module_name} ({description})",
                             "requirements.txt", None, f"Install {module_name}")
    
    def _scan_performance_issues(self):
        """E. Performance Analysis"""
        print("🔍 Scanning performance issues...")
        
        # Check for large files that might slow down processing
        large_files = []
        for file_path in self.project_path.rglob("*.py"):
            try:
                size = file_path.stat().st_size
                if size > 100000:  # 100KB
                    large_files.append((file_path, size))
            except:
                continue
        
        if large_files:
            for file_path, size in large_files:
                if size > 500000:  # 500KB
                    self._add_bug("performance", 2, f"Large Python file: {file_path.name}",
                                 f"File is {size/1024:.1f}KB, may impact performance",
                                 str(file_path), None, "Consider splitting into smaller modules")
        
        # Check scan performance
        if self.performance_metrics['scan_times']:
            avg_scan_time = sum(self.performance_metrics['scan_times']) / len(self.performance_metrics['scan_times'])
            if avg_scan_time > 10:  # 10 seconds
                self._add_bug("performance", 2, "Slow bug scanning",
                             f"Average scan time is {avg_scan_time:.1f}s",
                             "continuous_bug_scanner.py", None, "Optimize scanning algorithms")
    
    def _add_bug(self, category, severity, title, description, file_path, line_number, suggestion):
        """Add a bug to the list"""
        bug = BugReport(
            category=category,
            severity=severity,
            title=title,
            description=description,
            file_path=file_path,
            line_number=line_number,
            suggestion=suggestion,
            timestamp=datetime.now()
        )
        self.bugs.append(bug)
    
    def _calculate_health_score(self):
        """Calculate overall health score"""
        if not self.bugs:
            self.health_score = 100
            return
        
        # Weight bugs by severity
        total_penalty = 0
        for bug in self.bugs:
            if bug.category == 'critical':
                total_penalty += bug.severity * 10
            elif bug.category == 'warning':
                total_penalty += bug.severity * 5
            elif bug.category == 'config':
                total_penalty += bug.severity * 3
            elif bug.category == 'performance':
                total_penalty += bug.severity * 2
            elif bug.category == 'integration':
                total_penalty += bug.severity * 4
        
        self.health_score = max(0, 100 - total_penalty)
    
    def _display_dashboard(self):
        """Display bug scanner dashboard"""
        print("\n" + "=" * 60)
        print("🧠 Enhanced Analyzer Bug Scanner Dashboard")
        print("=" * 60)
        
        # Categorize bugs
        bug_categories = defaultdict(list)
        for bug in self.bugs:
            bug_categories[bug.category].append(bug)
        
        # Display by category with emojis
        category_emojis = {
            'critical': '🔴',
            'warning': '🟡',
            'config': '🔵',
            'performance': '🟢',
            'integration': '🟣'
        }
        
        for category in ['critical', 'warning', 'config', 'performance', 'integration']:
            bugs = bug_categories[category]
            emoji = category_emojis[category]
            print(f"{emoji} {category.upper()} ({len(bugs)}): ", end="")
            
            if bugs:
                for bug in bugs[:3]:  # Show first 3
                    print(f"{bug.title}", end="")
                    if bug != bugs[-1] and len(bugs) > 1:
                        print(", ", end="")
                if len(bugs) > 3:
                    print(f" and {len(bugs)-3} more...")
                else:
                    print()
            else:
                print("All clear")
        
        print()
        print(f"📈 Overall Health Score: {self.health_score}/100")
        print(f"🔄 Last Scan: {self.last_scan_time.strftime('%H:%M:%S') if self.last_scan_time else 'Never'}")
        print(f"⚡ Real-time monitoring: {'ACTIVE' if self.watch_mode else 'DISABLED'}")
        print(f"📊 Total scans: {self.scan_count}")
        
        # Show critical issues in detail
        critical_bugs = [bug for bug in self.bugs if bug.category == 'critical']
        if critical_bugs:
            print("\n🚨 CRITICAL ISSUES:")
            for bug in critical_bugs:
                print(f"   • {bug.title}")
                print(f"     📁 {bug.file_path}")
                print(f"     💡 {bug.suggestion}")
    
    def generate_report(self):
        """Generate detailed bug report"""
        report_data = {
            'scan_info': {
                'timestamp': datetime.now().isoformat(),
                'scan_count': self.scan_count,
                'health_score': self.health_score,
                'project_path': str(self.project_path)
            },
            'bugs': [bug.to_dict() for bug in self.bugs],
            'performance_metrics': self.performance_metrics
        }
        
        # Save report
        report_file = self.project_path / f"bug_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"📄 Bug report saved: {report_file}")
        return report_file


class FileChangeHandler(FileSystemEventHandler):
    """Handle file system changes"""
    
    def __init__(self, scanner):
        self.scanner = scanner
        self.last_scan = time.time()
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        # Only scan Python and config files
        if event.src_path.endswith(('.py', '.yaml', '.yml', '.json')):
            # Debounce rapid changes
            if time.time() - self.last_scan > 5:
                print(f"📝 File changed: {event.src_path}")
                self.scanner.run_comprehensive_scan()
                self.last_scan = time.time()


def main():
    """Main function"""
    print("🔍 Continuous Bug & Logic Scanner System")
    print("=" * 60)
    
    # Check if watchdog is available
    if not WATCHDOG_AVAILABLE:
        print("⚠️ File watching disabled (install watchdog for real-time monitoring)")
    
    try:
        # Initialize scanner
        scanner = BugScanner(
            project_path="./",
            watch_mode=WATCHDOG_AVAILABLE,
            focus_modules=[
                "func.browser",
                "func.analyzer", 
                "func.storage",
                "func.utils"
            ]
        )
        
        print("\n📋 Scanner options:")
        print("1. Single comprehensive scan")
        print("2. Start continuous monitoring")
        print("3. Generate detailed report")
        print("4. Exit")
        
        choice = input("\n🔢 Select option (1-4): ").strip()
        
        if choice == "1":
            print("\n🔍 Running single scan...")
            scanner.run_comprehensive_scan()
            scanner._display_dashboard()
        
        elif choice == "2":
            print("\n🚀 Starting continuous monitoring...")
            print("Press Ctrl+C to stop")
            scanner.start_monitoring()
        
        elif choice == "3":
            print("\n📄 Generating detailed report...")
            scanner.run_comprehensive_scan()
            report_file = scanner.generate_report()
            scanner._display_dashboard()
            print(f"\n📄 Report saved: {report_file}")
        
        elif choice == "4":
            print("👋 Goodbye!")
        
        else:
            print("❌ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n⏹️ Scanner stopped by user")
    except Exception as e:
        print(f"❌ Scanner error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()