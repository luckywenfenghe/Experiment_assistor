# Experiment Assistant MCP - Intelligent Processing System Guide

## Overview

The Enhanced MATLAB MCP server now includes an intelligent "search first, then call" processing system that automatically discovers, indexes, and selects the best MATLAB scripts for your data processing tasks.

## Key Features

### 🔍 **Automatic Script Discovery**
- Scans your project directories for `.m` and `.mlx` files
- Extracts metadata from comments and function signatures
- Categorizes scripts by purpose (process, plot, merge, analysis, save, utility)
- Caches results for fast subsequent access

### 🧠 **Intelligent Script Selection**
- Analyzes natural language user intent
- Extracts relevant keywords automatically
- Ranks scripts by relevance and recency
- Falls back to default processing if no suitable script found

### ⚡ **Smart Execution**
- Automatically injects filename parameters
- Handles different script signatures intelligently
- Provides detailed execution feedback
- Integrates with existing experiment assistant capabilities

## How to Add Metadata to Your MATLAB Scripts

To make your scripts discoverable by the intelligent system, add metadata comments at the top:

```matlab
% @process arduino temperature filtering smoothing
% Brief description of what this script does
% More detailed explanation if needed
%
% function myFunction(param1, param2)
%   Brief function signature description

function myFunction(param1, param2)
    % Your code here
end
```

### Metadata Format

1. **Category Tag**: `% @category keyword1 keyword2 ...`
   - Categories: `process`, `plot`, `merge`, `analysis`, `save`, `utility`
   - Keywords: relevant terms for matching

2. **Description**: Following comment lines describe functionality

3. **Function Signature**: Automatic extraction from function declarations

### Example Categories

- `% @process arduino temperature filtering` - Data processing scripts
- `% @plot thermal contour visualization` - Plotting and visualization
- `% @merge combine data files` - File merging utilities
- `% @analysis statistics summary` - Data analysis functions
- `% @save export excel csv` - Data export utilities
- `% @utility helper configuration` - General utility functions

## MCP Tools Reference

### Core Intelligent Processing Tools

#### `intelligentDataProcessing`
**Main entry point for natural language data processing**

```python
result = await intelligentDataProcessing(
    user_intent="Process Arduino temperature data with smoothing and create a plot",
    data_file="arduino_data_20241201.txt"
)
```

**Parameters:**
- `user_intent` (str): Natural language description of what you want to do
- `data_file` (str, optional): Input data file path
- `**kwargs`: Additional parameters

#### `discoverMatlabPrograms`
**Discover and index all available MATLAB programs**

```python
result = await discoverMatlabPrograms(force_refresh=False)
```

**Parameters:**
- `force_refresh` (bool): Force rescan even if cache is fresh

#### `searchMatlabPrograms`
**Search for programs matching specific criteria**

```python
result = await searchMatlabPrograms(
    query="arduino temperature processing",
    task_type="process"
)
```

**Parameters:**
- `query` (str): Search query with keywords
- `task_type` (str, optional): Filter by category

#### `runSpecificMatlabScript`
**Run a specific script by name with intelligent parameter handling**

```python
result = await runSpecificMatlabScript(
    script_name="ProcessArduinoDataEnhanced.m",
    data_file="data.txt"
)
```

#### `getIntelligentProcessingStatus`
**Get system status and configuration**

```python
status = await getIntelligentProcessingStatus()
```

## Usage Examples

### Example 1: Natural Language Processing
```python
# User says: "I want to smooth the temperature data and create a contour plot"
result = await intelligentDataProcessing(
    user_intent="smooth temperature data and create contour plot",
    data_file="sensor_data.txt"
)

# System will:
# 1. Extract keywords: ["smooth", "temperature", "contour", "plot"]
# 2. Find best "process" script for smoothing
# 3. Find best "plot" script for contour plotting
# 4. Execute both in sequence
# 5. Return detailed results
```

### Example 2: Discover Available Scripts
```python
# Get all available scripts
programs = await discoverMatlabPrograms()

# View by category
print(f"Processing scripts: {len(programs['programs_by_category']['process'])}")
print(f"Plotting scripts: {len(programs['programs_by_category']['plot'])}")
```

### Example 3: Search for Specific Functionality
```python
# Search for Arduino-related processing scripts
results = await searchMatlabPrograms(
    query="arduino sensor data processing",
    task_type="process"
)

# Results are ranked by relevance
best_script = results['matching_programs'][0] if results['matching_programs'] else None
```

### Example 4: Integration with Experiment Assistant
```python
# Run full experiment with intelligent post-processing
experiment_result = await runFullExperiment(
    mode=1,  # Real-time data collection
    max_frames=100
)

# Then intelligently process the results
processing_result = await intelligentDataProcessing(
    user_intent="analyze the experiment data for temperature trends and create summary plots",
    data_file=experiment_result.get('output_file')
)
```

## Directory Structure

The system automatically scans these directories:
- `./design_plate/` - Main MATLAB scripts directory
- `./` - Root project directory
- Any additional directories specified in configuration

## Caching System

- **Cache File**: `.ea_mcp_matlab_index.json`
- **Auto-refresh**: When source directories are modified
- **Manual refresh**: Use `force_refresh=True` parameter

## Best Practices

### 1. **Consistent Metadata**
Always add category tags and descriptions to your MATLAB scripts:
```matlab
% @process arduino temperature filtering smoothing
% Process Arduino sensor data with advanced temperature filtering
```

### 2. **Descriptive Function Names**
Use clear, descriptive names for your functions:
```matlab
function ProcessArduinoDataWithFiltering(inputFile, outputFile, options)
```

### 3. **Natural Language Queries**
Be specific but natural in your intent descriptions:
- ✅ "Process Arduino temperature data with moving average filter"
- ✅ "Create thermal contour plot from processed data"
- ❌ "Do stuff with data"

### 4. **Parameter Consistency**
Use consistent parameter names across scripts:
- `inputFile` or `filename` for input files
- `outputFile` for output files
- `options` or `filterOptions` for configuration

## Integration with Existing Features

The intelligent processing system seamlessly integrates with:

- **Experiment Assistant**: Automatic post-experiment processing
- **Batch Processing**: Intelligent script selection for batch operations
- **Temperature Filtering**: Smart filter configuration based on script capabilities
- **File Discovery**: Enhanced file type detection and processing

## Troubleshooting

### Common Issues

1. **No Scripts Found**
   - Check that scripts have proper metadata tags
   - Verify scripts are in scanned directories
   - Use `force_refresh=True` to rebuild index

2. **Script Not Executing**
   - Ensure MATLAB path includes script directory
   - Check script syntax and dependencies
   - Verify parameter names match expected format

3. **Poor Search Results**
   - Add more descriptive keywords to script metadata
   - Use more specific search queries
   - Check script categorization

### Debug Information

Use `getIntelligentProcessingStatus()` to check system status:
```python
status = await getIntelligentProcessingStatus()
print(f"System initialized: {status['system_initialized']}")
print(f"Total indexed scripts: {status['program_discovery']['total_indexed']}")
```

## Future Enhancements

Planned improvements include:
- **Semantic Search**: Advanced NLP-based script matching
- **Auto-tagging**: Automatic metadata extraction from code analysis
- **Performance Metrics**: Script execution time tracking and optimization
- **Version Control**: Integration with git for script versioning
- **Collaborative Features**: Shared script libraries and ratings

---

*This intelligent processing system transforms your MATLAB MCP server into a self-organizing, AI-driven data processing powerhouse that learns and adapts to your workflow patterns.* 