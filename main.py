from typing import Any, Dict
from mcp.server.fastmcp import FastMCP
import sys
import logging
import asyncio
import numpy as np
import json
import re
import openai
import os
import tempfile
import time
import yaml
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
from dataclasses import dataclass, asdict
from typing import List, Optional, Union
import shutil

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    # timestamp, logger name, level, message
)

logger = logging.getLogger("MatlabMCP")

import matlab.engine
mcp = FastMCP("MatlabMCP")

# MATLAB Script Discovery and Intelligent Selection System
class MatlabProgramDiscovery:
    """
    Intelligent MATLAB script discovery and selection system
    Implements the "search first, then call" approach for automated script selection
    """
    
    def __init__(self, search_directories: List[str] = None):
        self.search_directories = search_directories or ['./design_plate', './']
        self.program_index = {}
        self.cache_file = ".ea_mcp_matlab_index.json"
        self.last_scan_time = 0
        
    def discover_programs(self, force_refresh: bool = False) -> Dict[str, List[Dict]]:
        """
        Discover and index all MATLAB programs with metadata
        
        Args:
            force_refresh: Force rescan even if cache is fresh
            
        Returns:
            Dictionary of categorized programs
        """
        # Check if we need to refresh the cache
        if not force_refresh and self._is_cache_fresh():
            return self.program_index
            
        logger.info("Discovering MATLAB programs...")
        
        programs = {
            'process': [],
            'plot': [],
            'utility': [],
            'analysis': [],
            'merge': [],
            'save': [],
            'filter': []
        }
        
        for directory in self.search_directories:
            if not os.path.exists(directory):
                continue
                
            # Find all .m and .mlx files
            matlab_files = []
            for ext in ['*.m', '*.mlx']:
                matlab_files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
            
            for file_path in matlab_files:
                try:
                    metadata = self._extract_metadata(file_path)
                    if metadata:
                        # Categorize based on filename and metadata
                        category = self._categorize_program(file_path, metadata)
                        programs[category].append({
                            'name': os.path.basename(file_path),
                            'path': os.path.abspath(file_path),
                            'category': category,
                            'tags': metadata.get('tags', []),
                            'description': metadata.get('description', ''),
                            'signature': metadata.get('signature', ''),
                            'modified': os.path.getmtime(file_path)
                        })
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
        
        self.program_index = programs
        self.last_scan_time = time.time()
        
        # Cache the results
        self._save_cache()
        
        total_programs = sum(len(progs) for progs in programs.values())
        logger.info(f"Discovered {total_programs} MATLAB programs across {len(programs)} categories")
        
        return programs
    
    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from MATLAB file comments"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            metadata = {
                'tags': [],
                'description': '',
                'signature': ''
            }
            
            # Process first 20 lines for metadata
            for i, line in enumerate(lines[:20]):
                line = line.strip()
                
                # Skip non-comment lines
                if not line.startswith('%'):
                    if i > 0:  # Stop after first non-comment block
                        break
                    continue
                
                comment = line[1:].strip()
                
                # Extract tags: % @process arduino temperature smoothing
                if comment.startswith('@'):
                    parts = comment[1:].split()
                    if parts:
                        category = parts[0]
                        tags = parts[1:] if len(parts) > 1 else []
                        metadata['tags'].extend(tags)
                        metadata['category_hint'] = category
                
                # Extract function signature
                elif comment.startswith('function') or 'function' in comment:
                    metadata['signature'] = comment
                
                # Extract description (first meaningful comment line)
                elif not metadata['description'] and len(comment) > 10:
                    metadata['description'] = comment[:100]  # Limit description length
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Could not extract metadata from {file_path}: {e}")
            return {}
    
    def _categorize_program(self, file_path: str, metadata: Dict) -> str:
        """Categorize program based on filename and metadata"""
        filename = os.path.basename(file_path).lower()
        tags = [tag.lower() for tag in metadata.get('tags', [])]
        description = metadata.get('description', '').lower()
        
        # Check metadata category hint first
        if 'category_hint' in metadata:
            hint = metadata['category_hint'].lower()
            if hint in ['process', 'plot', 'utility', 'analysis', 'merge', 'save']:
                return hint
        
        # Categorize based on filename patterns
        if any(word in filename for word in ['process', 'filter', 'enhance', 'smooth']):
            return 'process'
        elif any(word in filename for word in ['plot', 'chart', 'graph', 'visualize']):
            return 'plot'
        elif any(word in filename for word in ['merge', 'combine', 'join']):
            return 'merge'
        elif any(word in filename for word in ['save', 'export', 'write']):
            return 'save'
        elif any(word in filename for word in ['analyze', 'analysis', 'stats', 'summary']):
            return 'analysis'
        else:
            return 'utility'
    
    def find_best_script(self, task_type: str, keywords: List[str], 
                        user_intent: str = "") -> Optional[Dict]:
        """
        Find the best matching script for a given task
        
        Args:
            task_type: Expected task type ('process', 'plot', etc.)
            keywords: List of keywords to match against
            user_intent: Original user intent for additional context
            
        Returns:
            Best matching script metadata or None
        """
        if not self.program_index:
            self.discover_programs()
        
        candidates = self.program_index.get(task_type, [])
        
        if not candidates:
            return None
        
        def calculate_score(program: Dict) -> float:
            score = 0.0
            
            # Keyword matching in name, tags, and description
            searchable_text = " ".join([
                program['name'],
                " ".join(program['tags']),
                program['description']
            ]).lower()
            
            for keyword in keywords:
                if keyword.lower() in searchable_text:
                    score += 1.0
            
            # Bonus for exact filename matches
            if any(kw.lower() in program['name'].lower() for kw in keywords):
                score += 2.0
            
            # Bonus for recent modifications (prefer newer scripts)
            days_old = (time.time() - program['modified']) / (24 * 3600)
            if days_old < 30:  # Scripts modified in last 30 days
                score += 0.5
            
            return score
        
        # Rank candidates by score
        scored_candidates = [(prog, calculate_score(prog)) for prog in candidates]
        ranked = sorted([item for item in scored_candidates if item[1] > 0],
                       key=lambda x: x[1], reverse=True)
        
        return ranked[0][0] if ranked else None
    
    def _is_cache_fresh(self) -> bool:
        """Check if cached index is still fresh"""
        if not os.path.exists(self.cache_file):
            return False
            
        try:
            cache_mtime = os.path.getmtime(self.cache_file)
            
            # Check if any source directory is newer than cache
            for directory in self.search_directories:
                if os.path.exists(directory):
                    if os.path.getmtime(directory) > cache_mtime:
                        return False
            
            # Load cached index
            with open(self.cache_file, 'r') as f:
                cached_data = json.load(f)
                self.program_index = cached_data.get('program_index', {})
                self.last_scan_time = cached_data.get('last_scan_time', 0)
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking cache freshness: {e}")
            return False
    
    def _save_cache(self):
        """Save program index to cache file"""
        try:
            cache_data = {
                'program_index': self.program_index,
                'last_scan_time': self.last_scan_time,
                'version': '1.0'
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Error saving cache: {e}")

class IntelligentTaskHandler:
    """
    Handles intelligent task processing using discovered MATLAB scripts
    """
    
    def __init__(self, matlab_engine, program_discovery: MatlabProgramDiscovery):
        self.eng = matlab_engine
        self.discovery = program_discovery
        
    def extract_keywords(self, user_intent: str) -> List[str]:
        """Extract relevant keywords from user intent"""
        # Common technical keywords in data processing context
        technical_keywords = [
            'arduino', 'temperature', 'pressure', 'sensor', 'data',
            'filter', 'smooth', 'process', 'plot', 'chart', 'graph',
            'merge', 'combine', 'save', 'export', 'analyze', 'stats',
            'contour', 'thermal', 'time', 'series', 'frame'
        ]
        
        # Simple keyword extraction (can be enhanced with NLP)
        words = re.findall(r'\b\w+\b', user_intent.lower())
        keywords = [word for word in words if word in technical_keywords or len(word) > 4]
        
        return list(set(keywords))  # Remove duplicates
    
    async def handle_data_request(self, user_intent: str, data_file: str = None, **kwargs) -> Dict:
        """
        Intelligently handle data processing requests
        
        Args:
            user_intent: Natural language description of what user wants
            data_file: Input data file path (if applicable)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Handling intelligent data request: {user_intent}")
        
        # 1. Analyze user intent to determine task types
        intent_lower = user_intent.lower()
        wants_plot = any(k in intent_lower for k in ["plot", "绘图", "chart", "graph", "visualize", "contour", "曲线"])
        wants_process = any(k in intent_lower for k in ["process", "处理", "filter", "滤波", "smooth", "平滑", "enhance"])
        wants_merge = any(k in intent_lower for k in ["merge", "合并", "combine", "join"])
        wants_save = any(k in intent_lower for k in ["save", "保存", "export", "输出"])
        wants_analyze = any(k in intent_lower for k in ["analyze", "分析", "stats", "统计", "summary"])
        
        # 2. Extract keywords
        keywords = self.extract_keywords(user_intent)
        
        # 3. Find appropriate scripts
        results = []
        
        if wants_process:
            script = self.discovery.find_best_script("process", keywords, user_intent)
            if script:
                result = await self._execute_script(script, data_file, **kwargs)
                results.append(result)
        
        if wants_plot:
            script = self.discovery.find_best_script("plot", keywords, user_intent)
            if script:
                result = await self._execute_script(script, data_file, **kwargs)
                results.append(result)
        
        if wants_merge:
            script = self.discovery.find_best_script("merge", keywords, user_intent)
            if script:
                result = await self._execute_script(script, data_file, **kwargs)
                results.append(result)
        
        if wants_analyze:
            script = self.discovery.find_best_script("analysis", keywords, user_intent)
            if script:
                result = await self._execute_script(script, data_file, **kwargs)
                results.append(result)
        
        # 4. Fallback to default processing if no specific scripts found
        if not results:
            result = await self._fallback_processing(user_intent, data_file, **kwargs)
            results.append(result)
        
        return {
            "status": "success",
            "user_intent": user_intent,
            "keywords_extracted": keywords,
            "processing_results": results,
            "total_operations": len(results)
        }
    
    async def _execute_script(self, script: Dict, data_file: str = None, **kwargs) -> Dict:
        """Execute a discovered MATLAB script"""
        try:
            script_path = script['path']
            script_name = script['name']
            
            logger.info(f"Executing discovered script: {script_name}")
            
            # Prepare execution command
            if data_file:
                # If script expects a filename parameter, inject it
                if 'filename' in script.get('signature', '').lower():
                    run_command = f"run('{script_path}'); % with filename='{data_file}'"
                else:
                    # Set global filename variable
                    run_command = f"filename = '{data_file}'; run('{script_path}');"
            else:
                run_command = f"run('{script_path}');"
            
            # Execute the script
            output = await asyncio.to_thread(self.eng.evalc, run_command)
            
            return {
                "script_used": script_name,
                "script_path": script_path,
                "execution_status": "success",
                "output": sanitize_matlab_output(output),
                "script_category": script['category']
            }
            
        except Exception as e:
            logger.error(f"Error executing script {script['name']}: {e}")
            return {
                "script_used": script['name'],
                "execution_status": "error",
                "error": str(e)
            }
    
    async def _fallback_processing(self, user_intent: str, data_file: str = None, **kwargs) -> Dict:
        """Fallback processing when no suitable script is found"""
        logger.info("No suitable script found, using fallback processing")
        
        # Use existing experiment assistant capabilities as fallback
        try:
            if "process" in user_intent.lower() and data_file:
                # Use offline processor
                if experiment_assistant:
                    result = await experiment_assistant.integrator.offline_processor.process_file(data_file)
                    return {
                        "processing_method": "fallback_offline_processor",
                        "execution_status": "success",
                        "result": asdict(result)
                    }
            
            # Generic MATLAB execution
            fallback_code = f"""
                fprintf('Fallback processing for: {user_intent}\\n');
                if exist('{data_file or ""}', 'file')
                    fprintf('Processing file: {data_file}\\n');
                else
                    fprintf('No input file specified\\n');
                end
            """
            
            output = await asyncio.to_thread(self.eng.evalc, fallback_code)
            
            return {
                "processing_method": "fallback_generic",
                "execution_status": "success",
                "output": sanitize_matlab_output(output),
                "message": "Used generic fallback processing"
            }
            
        except Exception as e:
            return {
                "processing_method": "fallback_failed",
                "execution_status": "error",
                "error": str(e)
            }

# Global intelligent processing instances
program_discovery = None
intelligent_handler = None

def initialize_intelligent_processing():
    """Initialize intelligent script discovery and processing"""
    global program_discovery, intelligent_handler
    
    if eng and not program_discovery:
        program_discovery = MatlabProgramDiscovery()
        intelligent_handler = IntelligentTaskHandler(eng, program_discovery)
        logger.info("Intelligent processing system initialized")

# Configuration settings
class Config:
    def __init__(self):
        self.temp_filter_threshold = 0.15  # Temperature std threshold (°C)
        self.precheck_steps = 20           # Number of precheck steps
        self.precheck_interval = 0.5       # Seconds between precheck readings
        self.auto_filter_enabled = True   # Enable automatic filter decision
        
        # Batch processing settings
        self.batch_workers = 4             # Number of parallel workers for batch processing
        self.supported_extensions = ['.txt', '.mat', '.xlsx']  # Supported file types
        self.output_dir = 'processed_data'  # Default output directory
        self.enable_kafka = False          # Enable Kafka messaging
        self.kafka_topic = 'offline_metrics'  # Kafka topic for metrics
        
    def load_from_file(self, config_path="config.yaml"):
        """Load configuration from YAML file if it exists"""
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    for key, value in config_data.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
                logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}, using defaults")
    
    def save_to_file(self, config_path="config.yaml"):
        """Save current configuration to YAML file"""
        try:
            config_data = {
                'temp_filter_threshold': self.temp_filter_threshold,
                'precheck_steps': self.precheck_steps,
                'precheck_interval': self.precheck_interval,
                'auto_filter_enabled': self.auto_filter_enabled,
                'batch_workers': self.batch_workers,
                'supported_extensions': self.supported_extensions,
                'output_dir': self.output_dir,
                'enable_kafka': self.enable_kafka,
                'kafka_topic': self.kafka_topic
            }
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config file: {e}")

# Data classes for structured processing
@dataclass
class ProcessingResult:
    """Result of processing a single file"""
    input_file: str
    output_files: List[str]
    summary_file: str
    status: str
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    frame_count: Optional[int] = None
    dp_mean: Optional[float] = None
    dp_std: Optional[float] = None

@dataclass
class BatchSummary:
    """Summary of batch processing results"""
    directory: str
    total_files: int
    processed_files: int
    failed_files: int
    processing_time: float
    results: List[ProcessingResult]
    timestamp: str

# Global configuration instance
config = Config()
config.load_from_file()

# File Discovery utilities
class FileDiscovery:
    """Utility class for discovering and filtering data files"""
    
    @staticmethod
    def discover_files(directory: str, recursive: bool = True, extensions: List[str] = None) -> List[str]:
        """
        Discover processable files in directory
        
        Args:
            directory: Directory to scan
            recursive: Whether to scan subdirectories
            extensions: List of file extensions to include
            
        Returns:
            List of discovered file paths
        """
        if extensions is None:
            extensions = config.supported_extensions
            
        discovered_files = []
        search_pattern = "**/*" if recursive else "*"
        
        for ext in extensions:
            pattern = os.path.join(directory, f"{search_pattern}{ext}")
            files = glob.glob(pattern, recursive=recursive)
            discovered_files.extend(files)
        
        # Filter out already processed files
        filtered_files = []
        for file_path in discovered_files:
            if not FileDiscovery.is_already_processed(file_path):
                filtered_files.append(file_path)
        
        logger.info(f"Discovered {len(filtered_files)} unprocessed files in {directory}")
        return sorted(filtered_files)
    
    @staticmethod
    def is_already_processed(file_path: str) -> bool:
        """Check if file has already been processed"""
        base_name = os.path.splitext(file_path)[0]
        summary_file = f"{base_name}_summary.json"
        return os.path.exists(summary_file)
    
    @staticmethod
    def get_output_paths(input_file: str, output_dir: str = None) -> Dict[str, str]:
        """Generate output file paths for a given input file"""
        if output_dir is None:
            output_dir = os.path.dirname(input_file)
        
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        return {
            'processed_txt': os.path.join(output_dir, f"{base_name}_processed.txt"),
            'excel': os.path.join(output_dir, f"{base_name}_processed.xlsx"),
            'report': os.path.join(output_dir, f"{base_name}_report.json"),
            'summary': os.path.join(output_dir, f"{base_name}_summary.json")
        }

# Offline Processing Engine
class OfflineProcessor:
    """Handles offline processing of individual files using MATLAB"""
    
    def __init__(self, matlab_engine):
        self.eng = matlab_engine
        
    async def process_file(self, input_file: str, filter_options: Dict = None) -> ProcessingResult:
        """
        Process a single data file using MATLAB ProcessArduinoDataEnhanced
        
        Args:
            input_file: Path to input data file
            filter_options: Dictionary of filter options for MATLAB processing
            
        Returns:
            ProcessingResult object with processing details
        """
        start_time = time.time()
        
        try:
            # Generate output paths
            output_paths = FileDiscovery.get_output_paths(input_file)
            
            # Prepare filter options with all required parameters
            if filter_options is None:
                filter_options = {}
            
            # Set default values for all required MATLAB parameters
            default_options = {
                'pressureMaxValue': 1000,
                'tempMinValue': 20,
                'tempFilterMethod': 'movmean',
                'tempWindowSize': 5,
                'tempAlpha': 0.3,
                'tempMaxChange': 1.5,
                'pressureDensity': 1000,
                'pressureGravity': 9.81,
                'pressureHeight': 0
            }
            
            # Merge provided options with defaults
            for key, default_value in default_options.items():
                if key not in filter_options:
                    filter_options[key] = default_value
            
            # Convert filter options to MATLAB struct
            matlab_options = await self._create_matlab_struct(filter_options)
            
            # Call MATLAB processing function
            matlab_code = f"""
                try
                    % Get absolute path and set working directory
                    current_dir = pwd;
                    project_dir = '{os.path.abspath(".").replace(os.sep, "/")}';
                    design_plate_dir = fullfile(project_dir, 'design_plate');
                    
                    % Add design_plate to MATLAB path if not already there
                    if ~contains(path, design_plate_dir)
                        addpath(design_plate_dir);
                    end
                    
                    inputFile = '{input_file.replace(os.sep, '/')}';
                    outputFile = '{output_paths['processed_txt'].replace(os.sep, '/')}';
                    
                    % Create filter options struct
                    filterOptions = struct();
                    filterOptions.pressureMaxValue = {filter_options['pressureMaxValue']};
                    filterOptions.tempMinValue = {filter_options['tempMinValue']};
                    filterOptions.tempFilterMethod = '{filter_options['tempFilterMethod']}';
                    filterOptions.tempWindowSize = {filter_options['tempWindowSize']};
                    filterOptions.tempAlpha = {filter_options['tempAlpha']};
                    filterOptions.tempMaxChange = {filter_options['tempMaxChange']};
                    filterOptions.pressureDensity = {filter_options['pressureDensity']};
                    filterOptions.pressureGravity = {filter_options['pressureGravity']};
                    filterOptions.pressureHeight = {filter_options['pressureHeight']};
                    
                    % Process the file
                    ProcessArduinoDataEnhanced(inputFile, outputFile, filterOptions);
                    
                    fprintf('Processing completed successfully\\n');
                catch ME
                    fprintf('Error: %s\\n', ME.message);
                    rethrow(ME);
                end
            """
            
            result_output = await asyncio.to_thread(self.eng.evalc, matlab_code)
            
            # Extract processing information from MATLAB report if available
            frame_count, dp_mean, dp_std = await self._extract_matlab_stats(output_paths['report'])
            
            processing_time = time.time() - start_time
            
            # Generate summary
            await self._generate_summary(input_file, output_paths, processing_time, frame_count, dp_mean, dp_std)
            
            return ProcessingResult(
                input_file=input_file,
                output_files=[output_paths['processed_txt'], output_paths['excel']],
                summary_file=output_paths['summary'],
                status="success",
                processing_time=processing_time,
                frame_count=frame_count,
                dp_mean=dp_mean,
                dp_std=dp_std
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing {input_file}: {e}")
            
            return ProcessingResult(
                input_file=input_file,
                output_files=[],
                summary_file="",
                status="error",
                error_message=str(e),
                processing_time=processing_time
            )
    
    async def _create_matlab_struct(self, options: Dict) -> str:
        """Create MATLAB struct string from Python dictionary"""
        # This is handled directly in the matlab_code string above
        return ""
    
    async def _extract_matlab_stats(self, report_file: str) -> tuple:
        """Extract statistics from MATLAB-generated report JSON"""
        try:
            if os.path.exists(report_file):
                with open(report_file, 'r') as f:
                    report_data = json.load(f)
                    
                frame_count = report_data.get('metadata', {}).get('total_frames', None)
                dp_mean = report_data.get('dp_mean', None)
                dp_std = report_data.get('dp_std', None)
                
                return frame_count, dp_mean, dp_std
        except Exception as e:
            logger.warning(f"Could not extract stats from {report_file}: {e}")
        
        return None, None, None
    
    async def _generate_summary(self, input_file: str, output_paths: Dict, 
                              processing_time: float, frame_count: int, 
                              dp_mean: float, dp_std: float):
        """Generate Python summary JSON file"""
        try:
            summary = {
                'input_file': input_file,
                'output_files': output_paths,
                'processing_timestamp': datetime.now().isoformat(),
                'processing_time_seconds': processing_time,
                'frame_count': frame_count,
                'dp_statistics': {
                    'mean': dp_mean,
                    'std': dp_std
                },
                'file_sizes': {},
                'status': 'completed'
            }
            
            # Add file sizes if files exist
            for key, path in output_paths.items():
                if os.path.exists(path):
                    summary['file_sizes'][key] = os.path.getsize(path)
            
            # Save summary
            with open(output_paths['summary'], 'w') as f:
                json.dump(summary, f, indent=2)
                
            logger.info(f"Summary generated: {output_paths['summary']}")
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")

# Batch Processing Engine
class BatchProcessor:
    """Handles batch processing of multiple files"""
    
    def __init__(self, matlab_engine, max_workers: int = None):
        self.eng = matlab_engine
        self.max_workers = max_workers or config.batch_workers
        self.offline_processor = OfflineProcessor(matlab_engine)
        
    async def process_directory(self, directory: str, recursive: bool = True, 
                              filter_options: Dict = None) -> BatchSummary:
        """
        Process all files in a directory
        
        Args:
            directory: Directory to process
            recursive: Whether to process subdirectories
            filter_options: MATLAB filter options
            
        Returns:
            BatchSummary with processing results
        """
        start_time = time.time()
        
        # Discover files
        files_to_process = FileDiscovery.discover_files(directory, recursive)
        
        if not files_to_process:
            logger.warning(f"No files to process in {directory}")
            return BatchSummary(
                directory=directory,
                total_files=0,
                processed_files=0,
                failed_files=0,
                processing_time=0,
                results=[],
                timestamp=datetime.now().isoformat()
            )
        
        logger.info(f"Starting batch processing of {len(files_to_process)} files")
        
        # Process files (sequential for now to avoid MATLAB engine conflicts)
        results = []
        processed_count = 0
        failed_count = 0
        
        for file_path in files_to_process:
            try:
                logger.info(f"Processing {file_path}...")
                result = await self.offline_processor.process_file(file_path, filter_options)
                results.append(result)
                
                if result.status == "success":
                    processed_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"Unexpected error processing {file_path}: {e}")
                failed_count += 1
                results.append(ProcessingResult(
                    input_file=file_path,
                    output_files=[],
                    summary_file="",
                    status="error",
                    error_message=str(e)
                ))
        
        processing_time = time.time() - start_time
        
        # Generate batch summary
        batch_summary = BatchSummary(
            directory=directory,
            total_files=len(files_to_process),
            processed_files=processed_count,
            failed_files=failed_count,
            processing_time=processing_time,
            results=results,
            timestamp=datetime.now().isoformat()
        )
        
        # Save batch report
        await self._save_batch_report(batch_summary)
        
        logger.info(f"Batch processing completed: {processed_count} success, {failed_count} failed")
        return batch_summary
    
    async def _save_batch_report(self, batch_summary: BatchSummary):
        """Save batch processing report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(batch_summary.directory, f"batch_report_{timestamp}.json")
            
            with open(report_file, 'w') as f:
                json.dump(asdict(batch_summary), f, indent=2, default=str)
            
            logger.info(f"Batch report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"Error saving batch report: {e}")

def try_auto_start_matlab():
    """
    Try to automatically start MATLAB and share engine if no shared sessions found.
    """
    try:
        import subprocess
        import time
        
        logger.info("Attempting to auto-start MATLAB...")
        
        # Create temporary MATLAB script for sharing engine
        startup_script = "temp_auto_startup.m"
        with open(startup_script, 'w') as f:
            f.write("matlab.engine.shareEngine;\n")
            f.write("fprintf('MATLAB engine auto-shared successfully!\\n');\n")
            f.write("fprintf('MCP Server can now connect.\\n');\n")
        
        # Start MATLAB with the startup script
        subprocess.Popen(['matlab', '-r', f"run('{startup_script}')"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        
        # Wait for MATLAB to start and share engine
        logger.info("Waiting for MATLAB to start (30 seconds)...")
        for i in range(30):
            time.sleep(1)
            names = matlab.engine.find_matlab()
            if names:
                logger.info("MATLAB engine auto-started successfully!")
                # Clean up temporary file
                try:
                    os.remove(startup_script)
                except:
                    pass
                return names
            if i % 5 == 0:
                logger.info(f"Still waiting... ({i+1}/30 seconds)")
        
        # Clean up temporary file if MATLAB didn't start
        try:
            os.remove(startup_script)
        except:
            pass
            
        logger.warning("Auto-start timeout. MATLAB may need manual startup.")
        return []
        
    except Exception as e:
        logger.error(f"Auto-start failed: {e}")
        return []

logger.info("Finding shared MATLAB sessions...")
names = matlab.engine.find_matlab()
logger.info(f"Found sessions: {names}")

if not names:
    logger.warning("No shared MATLAB sessions found.")
    logger.info("Attempting to auto-start MATLAB...")
    names = try_auto_start_matlab()

if not names:
    logger.error("No shared MATLAB sessions found after auto-start attempt.")
    logger.error("Please start MATLAB manually and run 'matlab.engine.shareEngine' in its Command Window.")
    logger.error("Or use the provided batch file: start_matlab_mcp.bat")
    sys.exit(0)
else:
    session_name = names[0] 
    logger.info(f"Connecting to session: {session_name}")
    try:
        eng = matlab.engine.connect_matlab(session_name)
        logger.info("Successfully connected to shared MATLAB session.")
    except matlab.engine.EngineError as e:
        logger.error(f"Error connecting or communicating with MATLAB: {e}")
        sys.exit(0)

# Helper Function
def matlab_to_python(data : Any) -> Any:
    """
    Converts common MATLAB data types returned by the engine into JSON-Serializable Python types.
    """
    if isinstance(data, (str, int, float, bool, type(None))):
        # already JSON-serializable
        return data
    elif isinstance(data, matlab.double):
        # convert MATLAB double array to Python list (handles scalars, vectors, matrices)
        # using squeeze to remove singleton dimensions for simpler representation
        np_array = np.array(data).squeeze()
        if np_array.ndim == 0:
            return float(np_array)
        else:
            return np_array.tolist()
    elif isinstance(data, matlab.logical):
        np_array = np.array(data).squeeze()
        if np_array.ndim == 0:
            return bool(np_array)
        else:
            return np_array.tolist()
    elif isinstance(data, matlab.char):
        return str(data)
    else:
        logger.warning(f"Unsupported MATLAB type encountered: {type(data)}. Returning string representation.")
        try:
            return str(data)
        except Exception as e:
            return f"Unserializable MATLAB Type: {type(data)}"
    
    # --- TODO: Add more MATLAB types ---

async def get_ai_response(prompt: str, context: str = "") -> str:
    """
    Get AI response for MATLAB input prompt.
    """
    try:
        # Prepare the prompt for the AI
        system_prompt = """You are an AI assistant helping to control a MATLAB Arduino data collection system.
        You need to provide appropriate responses to MATLAB input prompts.
        Consider the context and provide the most suitable response."""
        
        user_prompt = f"""Context: {context}
        MATLAB Input Prompt: {prompt}
        Please provide an appropriate response. For yes/no questions, use 'y' or 'n'.
        For numeric inputs, provide a number. For file paths, provide a valid path.
        Keep responses concise and appropriate for the context."""
        
        # Call OpenAI API (you'll need to set up your API key)
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=50
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Error getting AI response: {e}")
        return "1"  # Fallback to default value

def preprocess_matlab_commands(code: str) -> str:
    """
    Preprocess MATLAB commands to handle special cases that might cause JSON parsing issues.
    """
    # List of special MATLAB commands that need to be handled
    special_commands = {
        'clear all': 'clear("all")',
        'close all': 'close("all")',
        'clc': 'clc()',
        'MergeDataF': 'MergeDataF',
        'MergingDat': 'MergingDat',
        'PlotingArd': 'PlotingArd',
        'autoSaveDa': 'autoSaveDa',
        'auto_input': 'auto_input',
        'displayDat': 'displayDat'
    }
    
    # Replace special commands
    processed_code = code
    for cmd, replacement in special_commands.items():
        processed_code = processed_code.replace(cmd, replacement)
    
    return processed_code

def verify_matlab_path(path: str) -> str:
    """
    Verify and normalize MATLAB file paths.
    """
    # Convert any single backslashes to double backslashes
    normalized_path = path.replace('\\', '\\\\')
    return normalized_path

def sanitize_matlab_output(output: str) -> str:
    """
    Sanitize MATLAB output to make it JSON-safe.
    """
    if not isinstance(output, str):
        return str(output)
    
    # Replace problematic characters and escape sequences
    output = output.replace('\\', '\\\\')  # Escape backslashes
    output = output.replace('"', '\\"')    # Escape quotes
    output = output.replace('\n', '\\n')   # Handle newlines
    output = output.replace('\r', '\\r')   # Handle carriage returns
    output = output.replace('\t', '\\t')   # Handle tabs
    
    # Remove or replace other problematic characters
    output = ''.join(char if ord(char) >= 32 else ' ' for char in output)
    
    return output

def extract_filename_from_code(code: str) -> str:
    """
    从 Claude 的指令或代码中提取 filename（如有）。
    支持 @xxx.m 文件名 或 filename='xxx.txt' 格式。
    """
    # 匹配 @xxx.m 文件名
    match = re.match(r'@\w+\.m\s+([^\s]+)', code.strip())
    if match:
        return match.group(1)
    # 匹配 filename='xxx.txt' 或 filename="xxx.txt"
    match2 = re.search(r'filename\s*=\s*[\'"]([^\'"]+)[\'"]', code)
    if match2:
        return match2.group(1)
    return None

def inject_filename_parameter(code: str, filename: str) -> str:
    """
    如果 MATLAB 代码中有 filename 变量或 input filename，则自动注入 filename 参数。
    """
    # 1. 替换函数调用中的 filename 参数
    code = re.sub(
        r'renewPlotArduinoData\s*\(\s*filename\s*\)',
        f"renewPlotArduinoData('{filename}')",
        code
    )
    # 2. 替换 input('filename') 或 input("filename")
    code = re.sub(
        r"input\s*\(\s*['\"]filename['\"]\s*\)",
        f"'{filename}'",
        code
    )
    # 3. 替换 filename=xxx 赋值
    code = re.sub(
        r"filename\s*=\s*['\"].*?['\"]",
        f"filename='{filename}'",
        code
    )
    return code

@mcp.tool()
async def runMatlabCode(code: str) -> dict:
    """
    Run MATLAB code in a shared MATLAB session with AI-controlled input handling.
    """
    logger.info(f"Running MATLAB code request: {code[:100]}...")
    
    # 新增：自动提取并注入 filename 参数（如有）
    filename = extract_filename_from_code(code)
    if filename:
        code = inject_filename_parameter(code, filename)
    
    try:
        # Preprocess the code to handle special MATLAB commands
        processed_code = preprocess_matlab_commands(code)
        
        # First, check if the code contains input statements
        input_patterns = [
            r'input\s*\([^)]*\)',
            r'input\s*\([^)]*,\s*[\'"]s[\'"]\)',
            r'getUserConfirmation\s*\([^)]*\)',
            r'getNumericInput\s*\([^)]*\)',
            r'getBooleanInput\s*\([^)]*\)'
        ]
        
        has_input = any(re.search(pattern, processed_code) for pattern in input_patterns)
        
        if has_input:
            logger.info("Code contains input statements, using AI-controlled method...")
            
            # Replace input statements with auto_input
            modified_code = processed_code
            for pattern in input_patterns:
                modified_code = re.sub(
                    pattern,
                    lambda m: f"auto_input({m.group(0)}, 'auto')",
                    modified_code
                )
            
            # Run the modified code and sanitize output
            result = await asyncio.to_thread(eng.evalc, modified_code)
            sanitized_result = sanitize_matlab_output(result)
            logger.info("Code executed successfully using AI-controlled method.")
            return {"status": "success", "output": sanitized_result}
        else:
            # For code without input statements, try to use the eval approach first
            # This avoids the "Too many output parameters" error
            try:
                # Try with evalc first for capturing output
                result = await asyncio.to_thread(eng.evalc, processed_code)
                sanitized_result = sanitize_matlab_output(result)
                logger.info("Code executed successfully using direct evaluation.")
                return {"status": "success", "output": sanitized_result}
            except Exception as eval_error:
                logger.info(f"Direct evaluation failed with error: {eval_error}")
                logger.info("Falling back to simplified execution without capturing output...")
                
                # Try with eval instead of evalc (doesn't capture output but may avoid parameter issues)
                try:
                    await asyncio.to_thread(eng.eval, processed_code)
                    logger.info("Code executed successfully using simplified evaluation.")
                    return {"status": "success", "output": "Code executed successfully (output not captured)."}
                except Exception as simple_eval_error:
                    logger.info(f"Simplified evaluation failed with error: {simple_eval_error}")
                    logger.info("Falling back to temp file approach...")
                    
                    # Last resort: temp file approach but with careful execution
                    import os
                    import tempfile
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_filename = os.path.join(temp_dir, "temp_script.m")
                        
                        # Write the code to the temporary file with UTF-8 encoding
                        with open(temp_filename, "w", encoding='utf-8') as f:
                            f.write(processed_code)
                        
                        # Get the absolute path of the temporary file
                        abs_temp_path = os.path.abspath(temp_filename)
                        
                        try:
                            # Create a diary file to capture output
                            diary_file = os.path.join(temp_dir, "output.txt")
                            await asyncio.to_thread(eng.eval, f"diary('{verify_matlab_path(diary_file)}')")
                            await asyncio.to_thread(eng.eval, "diary on")
                            
                            # Execute the file but be careful about output parameters
                            # Use eval with run instead of direct run to avoid parameter issues
                            await asyncio.to_thread(eng.eval, f"run('{verify_matlab_path(abs_temp_path)}')")
                            
                            # Get the output from the diary
                            await asyncio.to_thread(eng.eval, "diary off")
                            output = ""
                            if os.path.exists(diary_file):
                                with open(diary_file, 'r', encoding='utf-8') as f:
                                    output = f.read()
                            
                            sanitized_output = sanitize_matlab_output(output)
                            logger.info("Code executed successfully using temp file method.")
                            return {"status": "success", "output": sanitized_output}
                        except Exception as run_error:
                            error_msg = str(run_error)
                            logger.error(f"All execution methods failed. Final error: {error_msg}")
                            return {
                                "status": "error",
                                "error_type": "MatlabExecutionError",
                                "message": f"All execution methods failed. Final error: {error_msg}"
                            }

    except matlab.engine.MatlabExecutionError as e:
        error_msg = sanitize_matlab_output(str(e))
        logger.error(f"MATLAB execution error: {error_msg}", exc_info=True)
        return {
            "status": "error",
            "error_type": "MatlabExecutionError",
            "message": f"Execution failed: {error_msg}"
        }
    except matlab.engine.EngineError as e:
        error_msg = sanitize_matlab_output(str(e))
        logger.error(f"MATLAB Engine communication error: {error_msg}", exc_info=True)
        return {
            "status": "error",
            "error_type": "EngineError",
            "message": f"MATLAB Engine error: {error_msg}"
        }
    except Exception as e:
        error_msg = sanitize_matlab_output(str(e))
        logger.error(f"Unexpected error executing MATLAB code: {error_msg}", exc_info=True)
        return {
            "status": "error",
            "error_type": e.__class__.__name__,
            "message": f"Unexpected error: {error_msg}"
        }

@mcp.tool()
async def getVariable(variable_name: str) -> dict:
    """
    Gets the value of a variable from the MATLAB workspace.

    Args:
        variable_name: The name of the variable to retrieve.

    Returns:
        A dictionary with status and either the variable's value (JSON serializable)
        or an error message, including error_type.
    """
    logger.info(f"Attempting to get variable: '{variable_name}'")
    try:
        if not eng:
            logger.error("No active MATLAB session found for getVariable.")
            return {"status": "error", "error_type": "RuntimeError", "message": "No active MATLAB session found."}

        # using asyncio.to_thread for the potentially blocking workspace access
        # directly accessing eng.workspace[variable_name] is blocking
        def get_var_sync():
             var_str = str(variable_name)
             if var_str not in eng.workspace:
                 raise KeyError(f"Variable '{var_str}' not found in MATLAB workspace.")
             return eng.workspace[var_str]

        matlab_value = await asyncio.to_thread(get_var_sync)

        # convert matlab value to a JSON-serializable Python type
        python_value = matlab_to_python(matlab_value)

        # test serialization before returning
        try:
            json.dumps({"value": python_value}) # test within dummy "dict"
            logger.info(f"Successfully retrieved and converted variable '{variable_name}'.")
            return {"status": "success", "variable": variable_name, "value": python_value}
        except TypeError as json_err:
            logger.error(f"Failed to serialize MATLAB value for '{variable_name}' after conversion: {json_err}", exc_info=True)
            return {
                "status": "error",
                "error_type": "TypeError",
                "message": f"Could not serialize value for variable '{variable_name}'. Original MATLAB type: {type(matlab_value)}"
            }

    except KeyError as ke:
        logger.warning(f"Variable '{variable_name}' not found in workspace: {ke}")
        return {"status": "error", "error_type": "KeyError", "message": str(ke)}
    except matlab.engine.EngineError as e_eng:
        logger.error(f"MATLAB Engine communication error during getVariable: {e_eng}", exc_info=True)
        return {"status": "error", "error_type": "EngineError", "message": f"MATLAB Engine error: {str(e_eng)}"}
    except Exception as e:
        logger.error(f"Unexpected error getting variable '{variable_name}': {e}", exc_info=True)
        return {
            "status": "error",
            "error_type": e.__class__.__name__,
            "message": f"Failed to get variable '{variable_name}': {str(e)}"
        }

def get_default_input(prompt: str) -> Any:
    """
    Generate appropriate default responses based on the input prompt.
    """
    # Convert prompt to lowercase for easier matching
    prompt_lower = prompt.lower()
    
    # Handle Arduino system specific prompts
    if 'select mode (1-5)' in prompt_lower:
        return '1'  # Default to real-time data collection
    
    # Handle settings modification prompts
    if 'modify these settings?' in prompt_lower:
        return 'n'  # Always return 'n' for settings modification
    
    # Handle numeric inputs with default values
    if '[' in prompt and ']' in prompt:
        # Extract default value from prompt if available
        try:
            default_value = re.search(r'\[(.*?)\]', prompt).group(1)
            return default_value.strip()
        except:
            pass
    
    # Handle specific settings prompts
    if 'auto-save interval' in prompt_lower:
        return '1000'
    elif 'frames per file' in prompt_lower:
        return '100'
    elif 'enable excel auto-save?' in prompt_lower:
        return 'y'
    elif 'excel save interval' in prompt_lower:
        return '100'
    elif 'excel frames per file' in prompt_lower:
        return '20'
    elif 'enable txt auto-save?' in prompt_lower:
        return 'y'
    elif 'txt save interval' in prompt_lower:
        return '100'
    elif 'enable temperature filtering?' in prompt_lower or 'temperature filtering?' in prompt_lower:
        return 'y'  # Always return 'y' for temperature filtering
    elif 'select method (1-3)' in prompt_lower:
        return '1'  # Default to moving average
    elif 'window size' in prompt_lower:
        return '5'
    
    # Default handlers for common input types
    if any(word in prompt_lower for word in ['yes', 'no', 'continue', '(y/n)']):
        return 'y'  # Default to yes for confirmation prompts
    elif 'file' in prompt_lower or 'path' in prompt_lower:
        return 'default.txt'  # Default filename
    elif 'number' in prompt_lower:
        return '1'  # Default number
    elif any(word in prompt_lower for word in ['name', 'string']):
        return 'default'  # Default string
    else:
        return '1'  # Generic default response

@mcp.tool()
async def handleMatlabInput(prompt: str = None) -> dict:
    """
    Automatically handle MATLAB input requests with predefined or generated responses.
    
    Args:
        prompt: The input prompt from MATLAB (if available)
        
    Returns:
        A dictionary with status and the provided input value
    """
    try:
        if not prompt:
            return {
                "status": "error",
                "error_type": "ValueError",
                "message": "No input prompt provided"
            }

        logger.info(f"Handling MATLAB input request: {prompt}")
        
        # Generate appropriate response based on the prompt
        response = get_default_input(prompt)
        
        logger.info(f"Providing automatic response: {response}")
        
        # Set the response in MATLAB's global variable
        try:
            # Clear previous response if any
            await asyncio.to_thread(eng.eval, "global AUTO_INPUT_RESPONSE; AUTO_INPUT_RESPONSE = [];")
            # Set new response
            await asyncio.to_thread(eng.eval, f"AUTO_INPUT_RESPONSE = '{response}';")
            
            return {
                "status": "success",
                "prompt": prompt,
                "provided_input": response
            }
        except matlab.engine.MatlabExecutionError as e:
            logger.error(f"Failed to send response to MATLAB: {e}")
            return {
                "status": "error",
                "error_type": "MatlabExecutionError",
                "message": f"Failed to handle input: {str(e)}"
            }
            
    except Exception as e:
        logger.error(f"Unexpected error handling MATLAB input: {e}", exc_info=True)
        return {
            "status": "error",
            "error_type": e.__class__.__name__,
            "message": f"Failed to handle input request: {str(e)}"
        }

# Integration utilities
class SystemIntegrator:
    """Handles integration between real-time and offline processing systems"""
    
    def __init__(self, matlab_engine):
        self.eng = matlab_engine
        self.batch_processor = BatchProcessor(matlab_engine)
        self.offline_processor = OfflineProcessor(matlab_engine)
        
    async def trigger_post_experiment_processing(self, experiment_dir: str, 
                                               filter_options: Dict = None) -> BatchSummary:
        """
        Automatically triggered after real-time experiment completion
        
        Args:
            experiment_dir: Directory containing experiment data files
            filter_options: Processing options for the batch
            
        Returns:
            BatchSummary of the processing results
        """
        logger.info(f"Post-experiment processing triggered for {experiment_dir}")
        
        try:
            # Process all files in the experiment directory
            batch_result = await self.batch_processor.process_directory(
                experiment_dir, recursive=True, filter_options=filter_options
            )
            
            # Optionally send to messaging system
            if config.enable_kafka:
                await self._send_batch_metrics(batch_result)
            
            return batch_result
            
        except Exception as e:
            logger.error(f"Error in post-experiment processing: {e}")
            raise
    
    async def _send_batch_metrics(self, batch_result: BatchSummary):
        """Send batch processing metrics to messaging system (Kafka/MQTT)"""
        try:
            # This would integrate with actual Kafka/MQTT client
            # For now, just log the metrics
            metrics = {
                'timestamp': batch_result.timestamp,
                'directory': batch_result.directory,
                'total_files': batch_result.total_files,
                'processed_files': batch_result.processed_files,
                'failed_files': batch_result.failed_files,
                'processing_time': batch_result.processing_time,
                'success_rate': batch_result.processed_files / batch_result.total_files if batch_result.total_files > 0 else 0
            }
            
            logger.info(f"Batch metrics: {json.dumps(metrics, indent=2)}")
            
            # TODO: Implement actual Kafka/MQTT publishing
            # await kafka_producer.send(config.kafka_topic, metrics)
            
        except Exception as e:
            logger.error(f"Error sending batch metrics: {e}")

# Experiment Assistant Core Functions
class ExperimentAssistant:
    """Core experiment assistant for Arduino data collection system"""
    
    def __init__(self, matlab_engine):
        self.eng = matlab_engine
        self.config = config
        self.precheck_data = []
        self.filter_enabled = False
        self.integrator = SystemIntegrator(matlab_engine)
        
    async def sensor_precheck(self, n_steps=None):
        """
        Perform sensor health check by reading temperature data for n_steps
        and determine if temperature filtering should be enabled
        """
        if n_steps is None:
            n_steps = self.config.precheck_steps
            
        logger.info(f"Starting sensor precheck with {n_steps} steps...")
        
        temps = []
        
        try:
            # Prepare MATLAB for sensor precheck
            await asyncio.to_thread(self.eng.eval, """
                % Initialize precheck mode
                global PRECHECK_MODE;
                PRECHECK_MODE = true;
                
                % Try to setup serial connection for precheck
                if exist('setupSerialConnection.m', 'file')
                    try
                        s = setupSerialConnection();
                        fprintf('Serial connection established for precheck\\n');
                    catch ME
                        fprintf('Warning: Could not establish serial connection: %s\\n', ME.message);
                        s = [];
                    end
                else
                    fprintf('Warning: setupSerialConnection.m not found, using simulated data\\n');
                    s = [];
                end
            """)
            
            for step in range(n_steps):
                try:
                    # Read temperature data from Arduino or use simulated data
                    result = await asyncio.to_thread(self.eng.evalc, """
                        if ~isempty(s)
                            % Try to read from actual Arduino
                            try
                                [validFrame, parsedData, rawTemps] = readDataFrame(s, 11);
                                if validFrame
                                    temp_data = rawTemps(1);  % Get first temperature sensor
                                else
                                    temp_data = 25 + randn()*0.2;  % Fallback to simulated
                                end
                            catch
                                temp_data = 25 + randn()*0.2;  % Fallback to simulated
                            end
                        else
                            % Use simulated temperature data for testing
                            temp_data = 25 + randn()*0.2;
                        end
                        fprintf('Step %d: T = %.3f°C\\n', %d, temp_data);
                    """ % (step + 1))
                    
                    # Extract temperature value from MATLAB output
                    temp_match = re.search(r'T = ([\d\.-]+)', result)
                    if temp_match:
                        temp_value = float(temp_match.group(1))
                        temps.append(temp_value)
                    else:
                        # Fallback to simulated data
                        temp_value = 25.0 + np.random.randn() * 0.2
                        temps.append(temp_value)
                    
                    # Wait between readings
                    await asyncio.sleep(self.config.precheck_interval)
                    
                except Exception as e:
                    logger.warning(f"Error in precheck step {step + 1}: {e}")
                    # Use simulated data for this step
                    temp_value = 25.0 + np.random.randn() * 0.2
                    temps.append(temp_value)
            
            # Calculate temperature noise statistics
            temps_array = np.array(temps)
            noise_std = float(np.std(temps_array))
            noise_mean = float(np.mean(temps_array))
            
            # Determine if filtering should be enabled
            use_filter = noise_std > self.config.temp_filter_threshold
            self.filter_enabled = use_filter
            
            # Set filter flag in MATLAB
            await self.set_matlab_filter(use_filter)
            
            # Clean up precheck mode
            await asyncio.to_thread(self.eng.eval, """
                global PRECHECK_MODE;
                PRECHECK_MODE = false;
                
                % Close serial connection if it was opened for precheck
                if exist('s', 'var') && ~isempty(s)
                    try
                        delete(s);
                        clear s;
                    catch
                        % Ignore cleanup errors
                    end
                end
            """)
            
            precheck_result = {
                "noise_std": noise_std,
                "noise_mean": noise_mean,
                "use_filter": use_filter,
                "threshold": self.config.temp_filter_threshold,
                "steps_completed": len(temps),
                "temperature_data": temps
            }
            
            self.precheck_data = precheck_result
            
            logger.info(f"Precheck completed: σ={noise_std:.4f}°C → Filter: {'ON' if use_filter else 'OFF'}")
            
            return precheck_result
            
        except Exception as e:
            logger.error(f"Error during sensor precheck: {e}")
            return {
                "error": str(e),
                "use_filter": False,
                "steps_completed": len(temps)
            }
    
    async def set_matlab_filter(self, enable_filter):
        """Set temperature filter flag in MATLAB global variables"""
        try:
            filter_flag = 'true' if enable_filter else 'false'
            
            matlab_code = f"""
                global tempFilterEnabled;
                global tempFilterMethod;
                global tempFilterWindowSize;
                global tempFilterThreshold;
                
                tempFilterEnabled = {filter_flag};
                
                if tempFilterEnabled
                    tempFilterMethod = 'movmean';      % Default to moving average
                    tempFilterWindowSize = 5;          % 5-point window
                    tempFilterThreshold = 1.5;         % 1.5°C max change threshold
                    fprintf('Temperature filtering ENABLED with method: %s\\n', tempFilterMethod);
                else
                    fprintf('Temperature filtering DISABLED\\n');
                end
            """
            
            await asyncio.to_thread(self.eng.eval, matlab_code)
            logger.info(f"MATLAB filter setting updated: {'ENABLED' if enable_filter else 'DISABLED'}")
            
        except Exception as e:
            logger.error(f"Error setting MATLAB filter: {e}")
    
    async def run_full_experiment(self, mode=1, max_frames=None, auto_process=True):
        """
        Run complete experiment workflow:
        1. Sensor precheck
        2. Configure system based on precheck results
        3. Execute selected mode
        4. (Optional) Trigger post-experiment processing
        """
        try:
            # Step 1: Sensor precheck
            logger.info("=== Starting Full Experiment Workflow ===")
            precheck_result = await self.sensor_precheck()
            
            if "error" in precheck_result:
                return {"status": "error", "message": "Precheck failed", "details": precheck_result}
            
            # Step 2: Configure system
            await self.configure_experiment_settings(mode, max_frames)
            
            # Step 3: Execute experiment
            result = await self.execute_experiment_mode(mode)
            
            # Step 4: Post-experiment processing (if enabled and mode 1)
            batch_result = None
            if auto_process and mode == 1:
                try:
                    # Get current experiment directory
                    experiment_dir = await self.get_current_experiment_directory()
                    if experiment_dir:
                        logger.info("Starting post-experiment batch processing...")
                        
                        # Use filter settings from precheck
                        filter_options = {
                            'tempFilterMethod': 'movmean' if self.filter_enabled else 'none',
                            'tempWindowSize': 5,
                            'tempMaxChange': 1.5,
                            'pressureMaxValue': 1000,
                            'tempMinValue': 20
                        }
                        
                        batch_result = await self.integrator.trigger_post_experiment_processing(
                            experiment_dir, filter_options
                        )
                        logger.info("Post-experiment processing completed successfully")
                except Exception as pe:
                    logger.warning(f"Post-experiment processing failed: {pe}")
            
            return {
                "status": "success",
                "precheck_result": precheck_result,
                "experiment_result": result,
                "batch_processing_result": batch_result
            }
            
        except Exception as e:
            logger.error(f"Error in full experiment workflow: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_current_experiment_directory(self) -> Optional[str]:
        """Get the current experiment data directory"""
        try:
            # Try to get from MATLAB workspace
            result = await asyncio.to_thread(self.eng.evalc, """
                if exist('outputDirectory', 'var')
                    fprintf('Current directory: %s\\n', outputDirectory);
                else
                    fprintf('Current directory: ./design_plate\\n');
                end
            """)
            
            # Extract directory from output
            dir_match = re.search(r'Current directory: (.+)', result)
            if dir_match:
                return dir_match.group(1).strip()
            else:
                # Default to design_plate directory
                return './design_plate'
                
        except Exception as e:
            logger.warning(f"Could not get experiment directory: {e}")
            return './design_plate'
    
    async def configure_experiment_settings(self, mode, max_frames):
        """Configure MATLAB experiment settings based on mode and parameters"""
        try:
            settings_code = f"""
                % Configure experiment settings
                global mode;
                global maxFrameCount;
                global tempFilterEnabled;
                
                mode = {mode};
                
                if ~isempty({max_frames if max_frames else 'NaN'}) && ~isnan({max_frames if max_frames else 'NaN'})
                    maxFrameCount = {max_frames if max_frames else 20};
                else
                    maxFrameCount = 20;  % Default value
                end
                
                fprintf('Experiment configured: Mode=%d, MaxFrames=%d, Filter=%s\\n', ...
                    mode, maxFrameCount, tempFilterEnabled ? 'ON' : 'OFF');
            """
            
            await asyncio.to_thread(self.eng.eval, settings_code)
            logger.info(f"Experiment configured: Mode={mode}, MaxFrames={max_frames}")
            
        except Exception as e:
            logger.error(f"Error configuring experiment settings: {e}")
    
    async def execute_experiment_mode(self, mode):
        """Execute the selected experiment mode"""
        try:
            mode_map = {
                1: "Real-time data collection",
                2: "Offline data processing", 
                3: "Data file merge",
                4: "Report generation",
                5: "Exit"
            }
            
            logger.info(f"Executing experiment mode {mode}: {mode_map.get(mode, 'Unknown')}")
            
            # Execute run_arduino_system.m with the configured mode
            result = await asyncio.to_thread(self.eng.evalc, "run_arduino_system")
            
            return {
                "mode": mode,
                "mode_description": mode_map.get(mode, "Unknown"),
                "output": sanitize_matlab_output(result)
            }
            
        except Exception as e:
            logger.error(f"Error executing experiment mode: {e}")
            return {"error": str(e)}

# Global experiment assistant instance
experiment_assistant = None

# Initialize experiment assistant after MATLAB connection
def initialize_experiment_assistant():
    global experiment_assistant
    if eng and not experiment_assistant:
        experiment_assistant = ExperimentAssistant(eng)
        logger.info("Experiment Assistant initialized")

@mcp.tool()
async def sensorPrecheck(steps: int = 20) -> dict:
    """
    Perform sensor health check by reading temperature data for specified steps
    and automatically determine if temperature filtering should be enabled.
    
    Args:
        steps: Number of temperature readings to take for noise analysis (default: 20)
        
    Returns:
        Dictionary with precheck results including noise statistics and filter decision
    """
    global experiment_assistant
    
    if not experiment_assistant:
        initialize_experiment_assistant()
    
    if not experiment_assistant:
        return {
            "status": "error",
            "error_type": "RuntimeError",
            "message": "Experiment Assistant not available - MATLAB connection required"
        }
    
    try:
        logger.info(f"Starting sensor precheck with {steps} steps...")
        result = await experiment_assistant.sensor_precheck(steps)
        
        if "error" in result:
            return {
                "status": "error",
                "error_type": "PrecheckError",
                "message": f"Sensor precheck failed: {result['error']}"
            }
        
        return {
            "status": "success",
            "precheck_result": result
        }
        
    except Exception as e:
        logger.error(f"Error in sensor precheck: {e}", exc_info=True)
        return {
            "status": "error",
            "error_type": e.__class__.__name__,
            "message": f"Sensor precheck failed: {str(e)}"
        }

@mcp.tool()
async def runFullExperiment(mode: int = 1, max_frames: int = 20, filename: str = None) -> dict:
    """
    Run complete experiment workflow including sensor precheck, configuration, and execution.
    
    Args:
        mode: Experiment mode (1=Real-time, 2=Offline, 3=Merge, 4=Report, 5=Exit)
        max_frames: Maximum number of frames to collect (for real-time mode)
        filename: Input filename for offline processing mode
        
    Returns:
        Dictionary with complete experiment results
    """
    global experiment_assistant
    
    if not experiment_assistant:
        initialize_experiment_assistant()
    
    if not experiment_assistant:
        return {
            "status": "error",
            "error_type": "RuntimeError",
            "message": "Experiment Assistant not available - MATLAB connection required"
        }
    
    try:
        logger.info(f"Starting full experiment: mode={mode}, max_frames={max_frames}")
        
        # Set filename for offline processing if provided
        if filename and mode == 2:
            await asyncio.to_thread(experiment_assistant.eng.eval, f"inputFilename = '{filename}';")
        
        result = await experiment_assistant.run_full_experiment(mode, max_frames)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in full experiment: {e}", exc_info=True)
        return {
            "status": "error",
            "error_type": e.__class__.__name__,
            "message": f"Full experiment failed: {str(e)}"
        }

@mcp.tool()
async def configureTemperatureFilter(enabled: bool, method: str = "movmean", window_size: int = 5, threshold: float = 1.5) -> dict:
    """
    Configure temperature filtering settings in the MATLAB system.
    
    Args:
        enabled: Whether to enable temperature filtering
        method: Filtering method ('movmean', 'expsmooth', 'kalman')
        window_size: Window size for moving average filter
        threshold: Maximum allowed temperature change threshold (°C)
        
    Returns:
        Dictionary with configuration status
    """
    global experiment_assistant
    
    if not experiment_assistant:
        initialize_experiment_assistant()
    
    if not experiment_assistant:
        return {
            "status": "error",
            "error_type": "RuntimeError",
            "message": "Experiment Assistant not available - MATLAB connection required"
        }
    
    try:
        # Set filter configuration in MATLAB
        matlab_code = f"""
            global tempFilterEnabled;
            global tempFilterMethod;
            global tempFilterWindowSize;
            global tempFilterThreshold;
            
            tempFilterEnabled = {'true' if enabled else 'false'};
            tempFilterMethod = '{method}';
            tempFilterWindowSize = {window_size};
            tempFilterThreshold = {threshold};
            
            fprintf('Temperature filter configured:\\n');
            fprintf('  Enabled: %s\\n', tempFilterEnabled ? 'true' : 'false');
            fprintf('  Method: %s\\n', tempFilterMethod);
            fprintf('  Window Size: %d\\n', tempFilterWindowSize);
            fprintf('  Threshold: %.2f°C\\n', tempFilterThreshold);
        """
        
        result = await asyncio.to_thread(experiment_assistant.eng.evalc, matlab_code)
        
        return {
            "status": "success",
            "configuration": {
                "enabled": enabled,
                "method": method,
                "window_size": window_size,
                "threshold": threshold
            },
            "matlab_output": sanitize_matlab_output(result)
        }
        
    except Exception as e:
        logger.error(f"Error configuring temperature filter: {e}", exc_info=True)
        return {
            "status": "error",
            "error_type": e.__class__.__name__,
            "message": f"Filter configuration failed: {str(e)}"
        }

@mcp.tool()
async def getExperimentStatus() -> dict:
    """
    Get current experiment status and configuration.
    
    Returns:
        Dictionary with current experiment status and settings
    """
    global experiment_assistant
    
    if not experiment_assistant:
        initialize_experiment_assistant()
    
    if not experiment_assistant:
        return {
            "status": "error",
            "error_type": "RuntimeError",
            "message": "Experiment Assistant not available"
        }
    
    try:
        # Get current MATLAB global variables
        result = await asyncio.to_thread(experiment_assistant.eng.evalc, """
            global mode maxFrameCount tempFilterEnabled tempFilterMethod;
            global tempFilterWindowSize tempFilterThreshold;
            
            fprintf('Current Experiment Status:\\n');
            fprintf('  Mode: %d\\n', mode);
            fprintf('  Max Frames: %d\\n', maxFrameCount);
            fprintf('  Filter Enabled: %s\\n', tempFilterEnabled ? 'true' : 'false');
            fprintf('  Filter Method: %s\\n', tempFilterMethod);
            fprintf('  Window Size: %d\\n', tempFilterWindowSize);
            fprintf('  Threshold: %.2f°C\\n', tempFilterThreshold);
        """)
        
        # Get precheck data if available
        precheck_data = experiment_assistant.precheck_data if hasattr(experiment_assistant, 'precheck_data') else None
        
        return {
            "status": "success",
            "experiment_assistant_available": True,
            "filter_enabled": experiment_assistant.filter_enabled,
            "precheck_data": precheck_data,
            "matlab_status": sanitize_matlab_output(result),
            "config": {
                "temp_filter_threshold": experiment_assistant.config.temp_filter_threshold,
                "precheck_steps": experiment_assistant.config.precheck_steps,
                "precheck_interval": experiment_assistant.config.precheck_interval
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting experiment status: {e}", exc_info=True)
        return {
            "status": "error",
            "error_type": e.__class__.__name__,
            "message": f"Failed to get experiment status: {str(e)}"
        }

# Batch Processing MCP Tools
@mcp.tool()
async def discoverFiles(directory: str, recursive: bool = True, extensions: List[str] = None) -> dict:
    """
    Discover processable data files in a directory.
    
    Args:
        directory: Directory path to scan
        recursive: Whether to scan subdirectories recursively
        extensions: List of file extensions to include (e.g., ['.txt', '.mat'])
        
    Returns:
        Dictionary with discovered files and summary information
    """
    try:
        if not os.path.exists(directory):
            return {
                "status": "error",
                "error_type": "DirectoryNotFound",
                "message": f"Directory not found: {directory}"
            }
        
        discovered_files = FileDiscovery.discover_files(directory, recursive, extensions)
        
        # Get file statistics
        file_stats = []
        for file_path in discovered_files:
            try:
                stat = os.stat(file_path)
                file_stats.append({
                    "path": file_path,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "extension": os.path.splitext(file_path)[1]
                })
            except Exception as e:
                logger.warning(f"Could not get stats for {file_path}: {e}")
        
        return {
            "status": "success",
            "directory": directory,
            "total_files": len(discovered_files),
            "files": file_stats,
            "extensions_found": list(set(f["extension"] for f in file_stats))
        }
        
    except Exception as e:
        logger.error(f"Error discovering files: {e}", exc_info=True)
        return {
            "status": "error",
            "error_type": e.__class__.__name__,
            "message": f"File discovery failed: {str(e)}"
        }

@mcp.tool()
async def processOfflineFile(input_file: str, filter_options: Dict = None) -> dict:
    """
    Process a single data file offline using MATLAB ProcessArduinoDataEnhanced.
    
    Args:
        input_file: Path to the input data file (.txt, .mat, or .xlsx)
        filter_options: Dictionary of processing options for MATLAB
        
    Returns:
        Dictionary with processing results and generated files
    """
    global experiment_assistant
    
    if not experiment_assistant:
        initialize_experiment_assistant()
    
    if not experiment_assistant:
        return {
            "status": "error",
            "error_type": "RuntimeError",
            "message": "Experiment Assistant not available - MATLAB connection required"
        }
    
    try:
        if not os.path.exists(input_file):
            return {
                "status": "error",
                "error_type": "FileNotFound",
                "message": f"Input file not found: {input_file}"
            }
        
        # Process the file
        result = await experiment_assistant.integrator.offline_processor.process_file(
            input_file, filter_options
        )
        
        return {
            "status": "success",
            "processing_result": asdict(result)
        }
        
    except Exception as e:
        logger.error(f"Error processing offline file: {e}", exc_info=True)
        return {
            "status": "error",
            "error_type": e.__class__.__name__,
            "message": f"Offline file processing failed: {str(e)}"
        }

@mcp.tool()
async def processBatchDirectory(directory: str, recursive: bool = True, filter_options: Dict = None) -> dict:
    """
    Process all data files in a directory using batch processing.
    
    Args:
        directory: Directory containing data files to process
        recursive: Whether to process subdirectories recursively
        filter_options: Dictionary of processing options for MATLAB
        
    Returns:
        Dictionary with batch processing results and summary
    """
    global experiment_assistant
    
    if not experiment_assistant:
        initialize_experiment_assistant()
    
    if not experiment_assistant:
        return {
            "status": "error",
            "error_type": "RuntimeError",
            "message": "Experiment Assistant not available - MATLAB connection required"
        }
    
    try:
        if not os.path.exists(directory):
            return {
                "status": "error",
                "error_type": "DirectoryNotFound",
                "message": f"Directory not found: {directory}"
            }
        
        logger.info(f"Starting batch processing of directory: {directory}")
        
        # Process the directory
        batch_result = await experiment_assistant.integrator.batch_processor.process_directory(
            directory, recursive, filter_options
        )
        
        return {
            "status": "success",
            "batch_summary": asdict(batch_result)
        }
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}", exc_info=True)
        return {
            "status": "error",
            "error_type": e.__class__.__name__,
            "message": f"Batch processing failed: {str(e)}"
        }

@mcp.tool()
async def getBatchProcessingStatus(directory: str) -> dict:
    """
    Get status of batch processing for a directory, including existing summaries.
    
    Args:
        directory: Directory to check for processing status
        
    Returns:
        Dictionary with processing status and summary information
    """
    try:
        if not os.path.exists(directory):
            return {
                "status": "error",
                "error_type": "DirectoryNotFound",
                "message": f"Directory not found: {directory}"
            }
        
        # Find all data files and summaries
        all_files = FileDiscovery.discover_files(directory, recursive=True)
        processed_files = []
        unprocessed_files = []
        
        for file_path in glob.glob(os.path.join(directory, "**/*"), recursive=True):
            if any(file_path.endswith(ext) for ext in config.supported_extensions):
                if FileDiscovery.is_already_processed(file_path):
                    processed_files.append(file_path)
                else:
                    unprocessed_files.append(file_path)
        
        # Find batch reports
        batch_reports = glob.glob(os.path.join(directory, "batch_report_*.json"))
        
        # Find summary files
        summary_files = glob.glob(os.path.join(directory, "**/*_summary.json"), recursive=True)
        
        return {
            "status": "success",
            "directory": directory,
            "processing_status": {
                "total_data_files": len(processed_files) + len(unprocessed_files),
                "processed_files": len(processed_files),
                "unprocessed_files": len(unprocessed_files),
                "processing_rate": len(processed_files) / (len(processed_files) + len(unprocessed_files)) if (len(processed_files) + len(unprocessed_files)) > 0 else 0
            },
            "files": {
                "processed": processed_files,
                "unprocessed": unprocessed_files,
                "batch_reports": batch_reports,
                "summaries": summary_files
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting batch processing status: {e}", exc_info=True)
        return {
            "status": "error",
            "error_type": e.__class__.__name__,
            "message": f"Failed to get processing status: {str(e)}"
        }

@mcp.tool()
async def triggerPostExperimentProcessing(experiment_dir: str = None, filter_options: Dict = None) -> dict:
    """
    Manually trigger post-experiment batch processing for a directory.
    
    Args:
        experiment_dir: Directory containing experiment data (defaults to current experiment directory)
        filter_options: Processing options for the batch
        
    Returns:
        Dictionary with batch processing results
    """
    global experiment_assistant
    
    if not experiment_assistant:
        initialize_experiment_assistant()
    
    if not experiment_assistant:
        return {
            "status": "error",
            "error_type": "RuntimeError",
            "message": "Experiment Assistant not available - MATLAB connection required"
        }
    
    try:
        # Use provided directory or get current experiment directory
        if not experiment_dir:
            experiment_dir = await experiment_assistant.get_current_experiment_directory()
        
        if not experiment_dir or not os.path.exists(experiment_dir):
            return {
                "status": "error",
                "error_type": "DirectoryNotFound",
                "message": f"Experiment directory not found: {experiment_dir}"
            }
        
        logger.info(f"Triggering post-experiment processing for: {experiment_dir}")
        
        # Trigger processing
        batch_result = await experiment_assistant.integrator.trigger_post_experiment_processing(
            experiment_dir, filter_options
        )
        
        return {
            "status": "success",
            "experiment_directory": experiment_dir,
            "batch_result": asdict(batch_result)
        }
        
    except Exception as e:
        logger.error(f"Error triggering post-experiment processing: {e}", exc_info=True)
        return {
            "status": "error",
            "error_type": e.__class__.__name__,
            "message": f"Post-experiment processing failed: {str(e)}"
        }

# Global instances
batch_processor = None
system_integrator = None

def initialize_batch_processing():
    """Initialize batch processing components"""
    global batch_processor, system_integrator
    
    if eng and not batch_processor:
        batch_processor = BatchProcessor(eng)
        system_integrator = SystemIntegrator(eng)
        logger.info("Batch processing components initialized")

if __name__ == "__main__":
    logger.info("Starting Enhanced MATLAB MCP server with Intelligent Processing...")
    
    # Initialize all components after MATLAB connection is established
    if 'eng' in globals() and eng:
        initialize_experiment_assistant()
        initialize_batch_processing()
        initialize_intelligent_processing()
    
    mcp.run(transport='stdio')
    logger.info("Enhanced MATLAB MCP server is running...")

# New Intelligent Processing MCP Tools
@mcp.tool()
async def discoverMatlabPrograms(force_refresh: bool = False) -> dict:
    """
    Discover and index all available MATLAB programs in the project.
    
    Args:
        force_refresh: Force rescan even if cache is fresh
        
    Returns:
        Dictionary with categorized MATLAB programs and their metadata
    """
    global program_discovery
    
    if not program_discovery:
        initialize_intelligent_processing()
    
    if not program_discovery:
        return {
            "status": "error",
            "error_type": "RuntimeError",
            "message": "Program discovery not available - MATLAB connection required"
        }
    
    try:
        programs = program_discovery.discover_programs(force_refresh)
        
        # Generate summary statistics
        stats = {}
        total_programs = 0
        for category, prog_list in programs.items():
            stats[category] = len(prog_list)
            total_programs += len(prog_list)
        
        return {
            "status": "success",
            "total_programs": total_programs,
            "programs_by_category": programs,
            "category_stats": stats,
            "cache_file": program_discovery.cache_file,
            "last_scan_time": program_discovery.last_scan_time
        }
        
    except Exception as e:
        logger.error(f"Error discovering MATLAB programs: {e}", exc_info=True)
        return {
            "status": "error",
            "error_type": e.__class__.__name__,
            "message": f"Program discovery failed: {str(e)}"
        }

@mcp.tool()
async def searchMatlabPrograms(query: str, task_type: str = None) -> dict:
    """
    Search for MATLAB programs matching a query and task type.
    
    Args:
        query: Search query with keywords
        task_type: Optional task type filter ('process', 'plot', 'merge', 'analysis', 'save', 'utility')
        
    Returns:
        Dictionary with matching programs ranked by relevance
    """
    global program_discovery
    
    if not program_discovery:
        initialize_intelligent_processing()
    
    if not program_discovery:
        return {
            "status": "error",
            "error_type": "RuntimeError",
            "message": "Program discovery not available"
        }
    
    try:
        # Extract keywords from query
        keywords = re.findall(r'\b\w+\b', query.lower())
        keywords = [kw for kw in keywords if len(kw) > 2]  # Filter short words
        
        if task_type:
            # Search specific category
            script = program_discovery.find_best_script(task_type, keywords, query)
            results = [script] if script else []
        else:
            # Search all categories
            results = []
            for category in ['process', 'plot', 'merge', 'analysis', 'save', 'utility']:
                script = program_discovery.find_best_script(category, keywords, query)
                if script:
                    results.append(script)
        
        return {
            "status": "success",
            "query": query,
            "keywords": keywords,
            "task_type_filter": task_type,
            "matching_programs": results,
            "total_matches": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error searching MATLAB programs: {e}", exc_info=True)
        return {
            "status": "error",
            "error_type": e.__class__.__name__,
            "message": f"Program search failed: {str(e)}"
        }

@mcp.tool()
async def intelligentDataProcessing(user_intent: str, data_file: str = None, **kwargs) -> dict:
    """
    Intelligently process data based on natural language intent.
    This is the main entry point for the "search first, then call" approach.
    
    Args:
        user_intent: Natural language description of what you want to do
        data_file: Optional input data file path
        **kwargs: Additional parameters for processing
        
    Returns:
        Dictionary with intelligent processing results
    """
    global intelligent_handler
    
    if not intelligent_handler:
        initialize_intelligent_processing()
    
    if not intelligent_handler:
        return {
            "status": "error",
            "error_type": "RuntimeError",
            "message": "Intelligent handler not available - MATLAB connection required"
        }
    
    try:
        result = await intelligent_handler.handle_data_request(user_intent, data_file, **kwargs)
        return result
        
    except Exception as e:
        logger.error(f"Error in intelligent data processing: {e}", exc_info=True)
        return {
            "status": "error",
            "error_type": e.__class__.__name__,
            "message": f"Intelligent processing failed: {str(e)}"
        }

@mcp.tool()
async def runSpecificMatlabScript(script_name: str, data_file: str = None, **kwargs) -> dict:
    """
    Run a specific MATLAB script by name, with intelligent parameter injection.
    
    Args:
        script_name: Name of the MATLAB script to run (e.g., "ProcessArduinoDataEnhanced.m")
        data_file: Optional input data file path
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with script execution results
    """
    global program_discovery, intelligent_handler
    
    if not program_discovery or not intelligent_handler:
        initialize_intelligent_processing()
    
    if not program_discovery or not intelligent_handler:
        return {
            "status": "error",
            "error_type": "RuntimeError",
            "message": "Intelligent processing not available"
        }
    
    try:
        # Find the script in our index
        program_discovery.discover_programs()  # Ensure index is fresh
        
        script = None
        for category, programs in program_discovery.program_index.items():
            for prog in programs:
                if prog['name'].lower() == script_name.lower():
                    script = prog
                    break
            if script:
                break
        
        if not script:
            return {
                "status": "error",
                "error_type": "ScriptNotFound",
                "message": f"Script '{script_name}' not found in program index"
            }
        
        # Execute the script
        result = await intelligent_handler._execute_script(script, data_file, **kwargs)
        
        return {
            "status": "success",
            "script_info": script,
            "execution_result": result
        }
        
    except Exception as e:
        logger.error(f"Error running specific MATLAB script: {e}", exc_info=True)
        return {
            "status": "error",
            "error_type": e.__class__.__name__,
            "message": f"Script execution failed: {str(e)}"
        }

@mcp.tool()
async def getIntelligentProcessingStatus() -> dict:
    """
    Get status of the intelligent processing system.
    
    Returns:
        Dictionary with system status and configuration
    """
    global program_discovery, intelligent_handler
    
    try:
        return {
            "status": "success",
            "system_initialized": program_discovery is not None and intelligent_handler is not None,
            "matlab_connected": eng is not None,
            "program_discovery": {
                "available": program_discovery is not None,
                "cache_file": program_discovery.cache_file if program_discovery else None,
                "search_directories": program_discovery.search_directories if program_discovery else None,
                "last_scan_time": program_discovery.last_scan_time if program_discovery else None,
                "total_indexed": sum(len(progs) for progs in program_discovery.program_index.values()) if program_discovery and program_discovery.program_index else 0
            },
            "intelligent_handler": {
                "available": intelligent_handler is not None
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting intelligent processing status: {e}", exc_info=True)
        return {
            "status": "error",
            "error_type": e.__class__.__name__,
            "message": f"Status check failed: {str(e)}"
        }