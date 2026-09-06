import logfire
from unstructured.partition.auto import partition

def parse_office(file_path: str):
    """
    Parses Office documents (.docx, .pptx) using the Unstructured library.
    """
    #you don't need to say "this is docx" or "this is pptx". It figures out itself
    # structured objects with type info.

    
    with logfire.span("📄 Office Document Parsing", filename=file_path):
        try:
            # Unstructured automatically detects if it's docx or pptx
            #It doesn't split by sentence or word count — 
            # it splits by what that block of content actually is (heading, paragraph, list, table) based on the file's internal structure tags.
            elements = partition(filename=file_path)
            full_text = "\n".join([str(el) for el in elements])
            #Split = extract text from objects.
            #Join = combine them. Both needed.
            if not full_text.strip():
                logfire.warning(f"⚠️ Unstructured returned empty text for {file_path}")
            else:
                logfire.info(f"✅ Successfully parsed {len(full_text)} characters")

            return full_text
        except Exception as e:
            logfire.error(f"❌ Office Parse Failed: {e}")
            raise e
            # give clean data to chunk 
            #Title("Intro")NarrativeText("RAG...")ListItem("Point 1") to 
            #"Intro"
            #"RAG..."
            #"Point 1"