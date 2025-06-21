# Import the PyPDF2 library to read and extract text and metadata from PDF files
import PyPDF2

# Import regular expressions for text pattern matching and cleaning
import re

# Import typing hints to specify expected types for function arguments and return values
from typing import List, Dict

# Import the tiktoken library for counting tokens based on a specific tokenizer model (useful for AI models like GPT)
import tiktoken


class PDFProcessor:
    def __init__(self):
        """
        Initialize the PDFProcessor class.
        Sets up the tokenizer encoding using tiktoken.
        This encoding helps determine how many tokens a text contains,
        which is important when working with models like OpenAI GPT.
        """
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def extract_text_from_pdf(self, pdf_file) -> str:
        """
        Extracts all the text content from a PDF file.

        Args:
            pdf_file: The uploaded PDF file object.

        Returns:
            A single string containing the text from all pages of the PDF.
        """
        try:
            # Read the PDF file
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""

            # Loop through each page to extract text
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    # Extract text from the current page
                    page_text = page.extract_text()
                    # Append the extracted text with a page marker
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                except Exception as e:
                    # If a page fails to extract, log and skip it
                    print(f"Error extracting page {page_num + 1}: {e}")
                    continue

            return text
        except Exception as e:
            # Raise an error if the PDF could not be read at all
            raise Exception(f"Error reading PDF: {str(e)}")

    def clean_text(self, text: str) -> str:
        """
        Cleans and normalizes the extracted text.

        Steps:
        - Removes excessive whitespace
        - Strips headers like "--- Page X ---"
        - Removes special characters (retains useful punctuation)

        Args:
            text: The raw extracted text.

        Returns:
            A cleaned string with normalized formatting.
        """
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n+', '\n', text)

        # Replace multiple spaces with a single space
        text = re.sub(r' +', ' ', text)

        # Remove page markers like "--- Page 1 ---"
        text = re.sub(r'--- Page \d+ ---', '', text)

        # Remove special characters except common punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)

        # Collapse any additional spaces
        text = ' '.join(text.split())

        return text.strip()

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in the text using the tokenizer.

        This is important for ensuring chunks of text stay within the model's limit.

        Args:
            text: The input text.

        Returns:
            The number of tokens in the input text.
        """
        return len(self.encoding.encode(text))

    def chunk_text(self, text: str, max_tokens: int = 8000) -> List[str]:
        """
        Splits long text into smaller chunks that stay under a token limit.

        Useful when sending text to models that have a maximum token limit (e.g., GPT-4 has ~8k or ~32k limits).

        Args:
            text: The cleaned input text.
            max_tokens: The maximum number of tokens allowed per chunk.

        Returns:
            A list of text chunks (strings), each within the token limit.
        """
        # Split the text into sentences using period + space as delimiter
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Try adding the sentence to the current chunk
            test_chunk = current_chunk + sentence + ". "

            # If this new chunk exceeds the token limit
            if self.count_tokens(test_chunk) > max_tokens and current_chunk:
                # Save the current chunk and start a new one
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
            else:
                # Otherwise, keep adding to the current chunk
                current_chunk = test_chunk

        # Add the final chunk if any content is left
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def get_pdf_metadata(self, pdf_file) -> Dict:
        """
        Extracts metadata from the PDF such as number of pages, title, author, and subject.

        Args:
            pdf_file: The uploaded PDF file object.

        Returns:
            A dictionary containing metadata fields or error info.
        """
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            # Extract standard metadata fields if available
            metadata = {
                'num_pages': len(pdf_reader.pages),
                'title': pdf_reader.metadata.get('/Title', 'Unknown') if pdf_reader.metadata else 'Unknown',
                'author': pdf_reader.metadata.get('/Author', 'Unknown') if pdf_reader.metadata else 'Unknown',
                'subject': pdf_reader.metadata.get('/Subject', 'Unknown') if pdf_reader.metadata else 'Unknown'
            }

            return metadata
        except Exception as e:
            # If something goes wrong, return an error dict
            return {'error': str(e)}
