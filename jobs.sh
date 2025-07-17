#!/bin/bash

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mt

# Set the project root directory
PROJECT_ROOT="/Users/jasonlam/Desktop/github/FYP"

# Function to run a job and check its status
run_job() {
    local job_name=$1
    local job_file=$2
    
    echo "Starting $job_name..."
    python "$PROJECT_ROOT/src/jobs/$job_file"
    
    if [ $? -eq 0 ]; then
        echo "$job_name completed successfully"
    else
        echo "$job_name failed with exit code $?"
        exit 1
    fi
}

# # Run FinBERT vs LLM comparison
echo "Running FinBERT vs LLM comparison..."
python "$PROJECT_ROOT/src/jobs/finbert_vs_llm.py"

# Run LLM with Technical Indicators
echo "Running LLM with Technical Indicators..."
python "$PROJECT_ROOT/src/jobs/llm_ti.py"

# Run LLM with Technical Indicators and Directional Change
echo "Running LLM with Technical Indicators and Directional Change..."
python "$PROJECT_ROOT/src/jobs/llm_dc_mi.py"

# Run Sentiment Comparison (FinBERT vs LLM sentiment only)
echo "Running Sentiment Comparison..."
python "$PROJECT_ROOT/src/jobs/sentiment_comparison.py"

# Run Sentiment Comparison (full features)
echo "Running Sentiment Comparison (full features)..."
python "$PROJECT_ROOT/src/jobs/llm_full_features.py"

echo "All jobs completed!"
