from bs4 import BeautifulSoup 
import logfire

def parse_html(file_path: str):
    """
    Parses HTML content using BeautifulSoup.
    it make proper tree so you can say things like:
    "give me all <p> tags" 
    "remove all <script> tags"
    "get me only the visible text"
    Cleans scripts, styles, and extracts readable text for RAG.
    """
    """Logfire.  
    A observability/monitoring tool — think of it as a smart logger built for production AI apps.
    logfire.span(...) → creates a trace around a block of code. It records:
    How long that block took
    Any errors that happened inside
    Custom metadata"""
    with logfire.span("📄 HTML Parsing", filename=file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, "html.parser")
            ### still thi soup is html file but we can now remove all junk


            # 1. Remove Junk (Scripts, Styles, Metadata)
            for script in soup(["script", "style", "meta", "noscript"]):
                script.decompose()
                
            # 2. Extract Text
            text = soup.get_text(separator="\n")
            
            # this much imp to white space if not given then unneceesary chunking 
            # 3. Clean Whitespace (Collapse multiple newlines)
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_clean = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text_clean
        except Exception as e:
            logfire.error(f"❌ HTML Parse Failed: {e}")
            raise e