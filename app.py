import streamlit as st
import os
from pdf_processor import PDFProcessor
from summarizer import PDFSummarizer
from utils import (
    validate_api_key,
    estimate_processing_time,
    display_processing_status,
    format_summary_display,
    export_summary_to_text,
    create_summary_dataframe,
)

import time

# Streamlit page configuration
st.set_page_config(
    page_title="PDF AI Summarizer",
    page_icon="📄",
    layout="wide"
)

def main():
    st.title("📄 PDF Reader + AI Summarizer")
    st.markdown("Upload a PDF document and get intelligent summaries powered by Google Gemini API.")

    # Validate API key
    if not validate_api_key():
        st.stop()

    # Initialize processors in session state
    if 'pdf_processor' not in st.session_state:
        st.session_state.pdf_processor = PDFProcessor()

    if 'summarizer' not in st.session_state:
        st.session_state.summarizer = PDFSummarizer()

    # Sidebar settings
    st.sidebar.header("⚙ Settings")

    summary_type = st.sidebar.selectbox(
        "Summary Type",
        ["brief", "detailed", "bullet_points", "executive"],
        index=1,
        help="Choose the type of summary you want"
    )

    max_tokens = st.sidebar.slider(
        "Max Tokens per Chunk",
        4000, 10000, 8000,
        help="Maximum tokens per text chunk (affects processing speed)"
    )

    show_analysis = st.sidebar.checkbox(
        "Show Document Analysis",
        value=True,
        help="Analyze document structure and content"
    )

    show_quotes = st.sidebar.checkbox(
        "Extract Key Quotes",
        value=False,
        help="Extract important quotes from the document"
    )

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to summarize"
    )

    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")

        # Display PDF metadata
        with st.spinner("Analyzing PDF metadata..."):
            metadata = st.session_state.pdf_processor.get_pdf_metadata(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📄 Pages: {metadata.get('num_pages', 'Unknown')}")
            st.info(f"📝 Title: {metadata.get('title', 'Unknown')}")
        with col2:
            st.info(f"👤 Author: {metadata.get('author', 'Unknown')}")
            st.info(f"📋 Subject: {metadata.get('subject', 'Unknown')}")

        # Process PDF button
        if st.button("🚀 Process PDF", type="primary"):
            process_pdf(uploaded_file, summary_type, max_tokens, show_analysis, show_quotes)

def process_pdf(uploaded_file, summary_type, max_tokens, show_analysis, show_quotes):
    """Process the PDF and generate summaries"""

    # Step 1: Extract text
    with st.spinner("📖 Extracting text from PDF..."):
        try:
            raw_text = st.session_state.pdf_processor.extract_text_from_pdf(uploaded_file)
            st.success(f"✅ Extracted {len(raw_text)} characters")
        except Exception as e:
            st.error(f"❌ Error extracting text: {str(e)}")
            return

    # Step 2: Clean text
    with st.spinner("🧹 Cleaning and processing text..."):
        clean_text = st.session_state.pdf_processor.clean_text(raw_text)
        token_count = st.session_state.pdf_processor.count_tokens(clean_text)
        st.success(f"✅ Cleaned text: {len(clean_text)} characters, ~{token_count} tokens")

    # Step 3: Chunk text
    with st.spinner("✂ Splitting text into chunks..."):
        chunks = st.session_state.pdf_processor.chunk_text(clean_text, max_tokens)
        st.success(f"✅ Created {len(chunks)} chunks")

        estimated_time = estimate_processing_time(len(chunks))
        st.info(f"⏱ Estimated processing time: {estimated_time}")

    # Step 4: Document analysis (optional)
    if show_analysis:
        with st.spinner("🔍 Analyzing document structure..."):
            analysis = st.session_state.summarizer.analyze_document_structure(clean_text)
            if analysis['status'] == 'success':
                st.subheader("📊 Document Analysis")
                st.write(analysis['analysis'])
            else:
                st.warning("⚠ Could not analyze document structure")

    # Step 5: Generate summaries
    with st.spinner(f"🤖 Generating {summary_type} summaries..."):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            summary_data = st.session_state.summarizer.summarize_chunks(chunks, summary_type)
            progress_bar.progress(1.0)
            status_text.success("✅ Summaries generated successfully!")
        except Exception as e:
            st.error(f"❌ Error generating summaries: {str(e)}")
            return

    # Step 6: Extract key quotes (optional)
    if show_quotes:
        with st.spinner("💬 Extracting key quotes..."):
            quotes = st.session_state.summarizer.extract_key_quotes(clean_text[:5000])
            st.subheader("💬 Key Quotes")
            for i, quote in enumerate(quotes, 1):
                st.write(f"{i}. *\"{quote}\"*")

    # Step 7: Display results
    st.header("📋 Summary Results")
    format_summary_display(summary_data)

    # Step 8: Export options
    st.subheader("📤 Export Options")
    col1, col2 = st.columns(2)

    with col1:
        summary_text = export_summary_to_text(summary_data, uploaded_file.name)
        st.download_button(
            label="📄 Download as Text",
            data=summary_text,
            file_name=f"{uploaded_file.name}_summary.txt",
            mime="text/plain"
        )

    with col2:
        summary_df = create_summary_dataframe(summary_data)
        csv_data = summary_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Statistics as CSV",
            data=csv_data,
            file_name=f"{uploaded_file.name}_stats.csv",
            mime="text/csv"
        )

    with st.expander("📈 Processing Statistics"):
        st.dataframe(summary_df)
        avg_compression = summary_df['Compression Ratio'].mean()
        st.metric("Average Compression Ratio", f"{avg_compression:.2f}")
        success_rate = (summary_df['Status'] == 'Success').mean() * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")

if __name__ == "__main__":
    main()
