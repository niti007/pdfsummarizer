# 📄 PDF AI Summarizer

A powerful Python application that uses **OpenRouter** (access to 300+ LLMs via an OpenAI-compatible API) to automatically summarize PDF documents. Upload any PDF and get intelligent, customizable summaries in multiple formats!

## ✨ What This Project Does

**PDF AI Summarizer** is an intelligent tool that:

- **Reads PDF Files**: Extracts all text from PDF documents automatically
- **Cleans & Processes Text**: Removes unnecessary characters and formats text for analysis
- **Splits into Chunks**: Breaks large documents into manageable pieces for the AI
- **Generates Smart Summaries**: Uses OpenRouter (OpenAI-compatible API) to create summaries in different styles:
  - **Brief**: Quick 2-3 sentence overview
  - **Detailed**: Comprehensive summary with key points, arguments, and conclusions
  - **Bullet Points**: Organized, easy-to-scan summary format
  - **Executive**: Business-focused summary with insights and recommendations
- **Analyzes Document Structure**: Identifies document type, main topics, and target audience
- **Extracts Key Quotes**: Pulls out important statements and phrases from the document
- **Exports Results**: Download summaries as text files or statistics as CSV

Perfect for students, professionals, researchers, and anyone who needs to quickly understand the content of large PDF documents!

---

## 🚀 Getting Started

### Prerequisites

Before you start, make sure you have:
- **Python 3.8 or higher** installed on your computer
- An **OpenRouter API Key** (get one at [openrouter.ai/keys](https://openrouter.ai/keys))

### Step 1: Clone the Repository

```bash
git clone https://github.com/niti007/pdfsummarizer.git
cd pdfsummarizer
```

### Step 2: Create a Virtual Environment (Optional but Recommended)

This keeps your project dependencies isolated:

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

**What gets installed:**
- `streamlit` - Interactive web interface
- `langchain` & `langchain-openai` - AI framework
- `PyPDF2` - PDF text extraction
- `python-dotenv` - Environment variable management
- `pandas` - Data handling
- `tiktoken` - Token counting

### Step 4: Set Up Your OpenRouter API Key

1. Go to [OpenRouter Keys](https://openrouter.ai/keys)
2. Click "Create Key"
3. Copy your API key

Create a `.env` file in the project root directory:

```bash
echo "OPENROUTER_API_KEY=your_api_key_here" > .env
```

Replace `your_api_key_here` with your actual API key.

**⚠️ Important:** Never share or commit your `.env` file with your API key!

### Step 5: Run the Application

Start the Streamlit web application:

```bash
streamlit run app.py
```

This will open a browser window at `http://localhost:8501` where you can use the application.

---

## 📖 How to Use

### Using the Web Interface

1. **Upload a PDF**: Click "Choose a PDF file" and select your document
2. **Configure Settings** (optional):
   - **Summary Type**: Choose from brief, detailed, bullet points, or executive summaries
   - **Max Tokens per Chunk**: Adjust processing speed and depth (default: 8000)
   - **Show Document Analysis**: Enable to analyze document structure
   - **Extract Key Quotes**: Enable to extract important quotes
3. **Process**: Click "🚀 Process PDF" button
4. **View Results**: See the summary, analysis, and statistics
5. **Export**: Download results as text or CSV file

### Example Workflow

```
1. Upload a 20-page research paper
2. Select "Executive" summary type
3. Click Process
4. Read the AI-generated summary in 2 minutes instead of 20 minutes
5. Download the summary for sharing or reference
```

---

## 🛠️ Project Structure

```
pdfsummarizer/
├── app.py                 # Main Streamlit web application
├── pdf_processor.py       # Handles PDF reading and text cleaning
├── summarizer.py          # AI summarization logic using OpenRouter
├── utils.py              # Helper functions (validation, formatting, export)
├── requirements.txt      # Python dependencies
├── .env.example          # Example environment file
└── README.md             # This file
```

### Key Files Explained

- **app.py**: The main web interface. Handles file upload, displays UI, and coordinates the workflow
- **pdf_processor.py**: Reads PDFs, extracts text, cleans it, and splits it into chunks
- **summarizer.py**: Uses the OpenRouter API to generate different types of summaries
- **utils.py**: Helper functions for API validation, time estimation, and result formatting

---

## 💻 Technical Details

### How It Works

```
User uploads PDF
     ↓
Extract text from all pages
     ↓
Clean and normalize text
     ↓
Split into manageable chunks (respecting token limits)
     ↓
Send each chunk to OpenRouter
     ↓
Get individual summaries
     ↓
Combine summaries into one cohesive final summary
     ↓
Display results and export options
```

### Token Management

The app uses `tiktoken` to count tokens before sending text to the AI. This ensures:
- Text stays within API limits
- Processing is efficient
- Accurate time estimation

---

## ⚙️ Configuration Options

### Summary Types Explained

| Type | Best For | Length |
|------|----------|--------|
| **Brief** | Quick overviews | 2-3 sentences |
| **Detailed** | Complete understanding | Full coverage of topics |
| **Bullet Points** | Quick scanning | Organized key points |
| **Executive** | Business decisions | Key insights & recommendations |

### Max Tokens Setting

- **Lower (4000-5000)**: Faster, less detail
- **Default (8000)**: Balanced speed and quality
- **Higher (9000-10000)**: More detail, slower processing

---

## 🔐 Security & Privacy

- Your API key is stored locally in `.env` (never committed to git)
- PDFs are processed locally; nothing is permanently stored
- Summaries are generated server-side by OpenRouter

---

## 🐛 Troubleshooting

### "OPENROUTER_API_KEY not found"
- Make sure `.env` file exists in the project root
- Verify the key is correctly set: `OPENROUTER_API_KEY=your_actual_key`
- Don't use quotes around the key in `.env`

### "Error extracting text from PDF"
- Ensure the PDF is not corrupted
- Try opening the PDF in Adobe Reader to verify it's valid
- Some scanned PDFs (images) may not have extractable text

### "Rate limit exceeded"
- Google API has daily limits on free tier
- Wait a few hours and try again, or upgrade your API plan
- Consider reducing the number of chunks

### Application runs but interface doesn't load
- Make sure port 8501 is available
- Try: `streamlit run app.py --server.port 8502`

---

## 📦 Requirements

All dependencies are listed in `requirements.txt`:

```
streamlit>=1.28.0
langchain>=0.0.350
langchain-google-genai>=0.0.5
python-dotenv>=1.0.0
PyPDF2>=3.0.1
pandas>=2.0.0
tiktoken>=0.5.0
```

---

## 🤝 Contributing

Found a bug or want to suggest a feature? Feel free to:
1. Create an issue describing the problem
2. Fork the repository and submit a pull request

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🎯 Future Enhancements

- Support for more document formats (DOCX, TXT, etc.)
- Multiple language support
- Summary comparison between different AI models
- Batch processing for multiple PDFs
- Custom summary templates
- Cloud storage integration

---

## 📞 Support

For issues or questions:
- Check the **Troubleshooting** section above
- Open an issue on GitHub
- Review the code comments for technical details

---

## 🌟 Acknowledgments

- **OpenRouter** for powerful AI summarization
- **Streamlit** for the beautiful web interface
- **LangChain** for AI orchestration
- **PyPDF2** for PDF processing

---

**Happy summarizing! 📚**
