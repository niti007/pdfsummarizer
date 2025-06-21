import streamlit as st
import pandas as pd
from typing import Dict, List
import json
import time
import os
from dotenv import load_dotenv

def display_processing_status(current_step: int, total_steps: int, step_name: str):
    """Display processing progress"""
    progress = current_step / total_steps
    st.progress(progress)
    st.write(f"Step {current_step}/{total_steps}: {step_name}")

def format_summary_display(summary_data: Dict):
    """Format summary data for display"""
    st.subheader("📊 Summary Statistics")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Chunks", summary_data['total_chunks'])
    with col2:
        successful_summaries = len([s for s in summary_data['individual_summaries'] if 'Error' not in s['summary']])
        st.metric("Successful Summaries", successful_summaries)
    with col3:
        st.metric("Summary Type", summary_data['summary_type'].title())

    st.subheader("📝 Final Summary")
    st.write(summary_data['combined_summary'])

    with st.expander("View Individual Chunk Summaries"):
        for chunk_summary in summary_data['individual_summaries']:
            st.write(f"**Chunk {chunk_summary['chunk_number']}**")
            st.write(chunk_summary['summary'])
            st.write(f"*Original length: {chunk_summary['original_length']} chars, "
                     f"Summary length: {chunk_summary['summary_length']} chars*")
            st.divider()

def export_summary_to_text(summary_data: Dict, filename: str = "summary.txt") -> str:
    """Export summary to text format"""
    content = f"PDF Summary Report\n"
    content += f"==================\n\n"
    content += f"Summary Type: {summary_data['summary_type'].title()}\n"
    content += f"Total Chunks Processed: {summary_data['total_chunks']}\n"
    content += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    content += f"FINAL SUMMARY\n"
    content += f"=============\n\n"
    content += summary_data['combined_summary']

    content += f"\n\nINDIVIDUAL CHUNK SUMMARIES\n"
    content += f"==========================\n\n"
    for chunk_summary in summary_data['individual_summaries']:
        content += f"Chunk {chunk_summary['chunk_number']}:\n"
        content += f"{chunk_summary['summary']}\n\n"

    return content

def create_summary_dataframe(summary_data: Dict) -> pd.DataFrame:
    """Create a DataFrame from summary data for analysis"""
    data = []
    for chunk_summary in summary_data['individual_summaries']:
        data.append({
            'Chunk': chunk_summary['chunk_number'],
            'Original Length': chunk_summary['original_length'],
            'Summary Length': chunk_summary['summary_length'],
            'Compression Ratio': round(chunk_summary['summary_length'] / chunk_summary['original_length'], 2)
            if chunk_summary['original_length'] != 0 else 0,
            'Status': 'Success' if 'Error' not in chunk_summary['summary'] else 'Error'
        })
    return pd.DataFrame(data)

def estimate_processing_time(num_chunks: int) -> str:
    """Estimate processing time based on number of chunks"""
    estimated_seconds = num_chunks ** 2.5

    if estimated_seconds < 60:
        return f"~{int(estimated_seconds)} seconds"
    elif estimated_seconds < 3600:
        minutes = int(estimated_seconds // 60)
        return f"~{minutes} minutes"
    else:
        hours = int(estimated_seconds // 3600)
        minutes = int((estimated_seconds % 3600) // 60)
        return f"~{hours}h {minutes}m"

def validate_api_key() -> bool:
    """Validate that the API key is set"""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        st.error("❌ Google API key not found! Please check your .env file.")
        st.info("1. Create a .env file in your project directory")
        st.info("2. Add: GOOGLE_API_KEY=your_api_key_here")
        st.info("3. Get your API key from: https://aistudio.google.com/")
        return False
    elif len(api_key) < 30:
        st.error("❌ API key seems invalid. Please check if it's complete.")
        return False
    else:
        st.success("✅ API key loaded successfully!")
        return True
