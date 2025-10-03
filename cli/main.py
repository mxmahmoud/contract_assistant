# cli/main.py
import typer
from rich.console import Console
from pathlib import Path
import time

from ca_core import extraction, chunking, vectorstore, qa, registry, ner
from utility.utility import get_hash_of_file
from utility.config import settings
from utility.model_loader import check_ollama_status, check_tei_status, load_local_llm_model


app = typer.Typer()
console = Console()


def ensure_services_ready():
    """Check for local services and wait for them to be ready."""
    if not settings.is_local_mode:
        return

    console.print("[yellow]Checking local service readiness...[/yellow]")
    
    with console.status("[bold green]Waiting for services...") as status:
        while not (ollama_ready := check_ollama_status()):
            status.update("[bold yellow]Ollama service not ready, waiting...[/bold yellow] The model may be downloading.")
            time.sleep(5)
        
        console.print("[bold green]✅ Ollama service is ready.[/bold green]")
        
        while not (tei_ready := check_tei_status()):
            status.update("[bold yellow]TEI service not ready, waiting...[/bold yellow]")
            time.sleep(2)
            
        console.print("[bold green]✅ TEI service is ready.[/bold green]")

        status.update(f"[bold green]Loading local LLM model: {settings.LOCAL_LLM_MODEL}...[/bold green]")
        load_local_llm_model()
        console.print(f"[bold green]✅ LLM model '{settings.LOCAL_LLM_MODEL}' is ready.[/bold green]")

    console.print("[bold green]All local services are ready.[/bold green]")


@app.callback()
def main_callback(ctx: typer.Context):
    """
    Main callback to run before any command.
    Ensures local services are ready before proceeding.
    """
    # This check will run before any command in the CLI app
    ensure_services_ready()


@app.command()
def ingest(
    pdf_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True, help="Path to the PDF contract to ingest."),
    ocr: bool = typer.Option(False, "--ocr", help="Force OCR extraction for the PDF.")
):
    """
    Ingest, chunk, and embed a PDF contract into the vector store.
    """
    console.print(f"[bold green]Starting extraction for: {pdf_path.name}[/bold green]")
    
    contract_id = get_hash_of_file(pdf_path.read_bytes())
    console.print(f"Generated Contract ID: [cyan]{contract_id}[/cyan]")
    
    extraction_strategy = "paddleocr" if ocr else "pypdf2"
    console.print(f"Extracting text from PDF using strategy: [bold]{extraction_strategy}[/bold]...")
    doc_pages = extraction.extract_text_from_pdf(str(pdf_path), strategy=extraction_strategy)
    
    console.print("Chunking document...")
    chunks = chunking.chunk_document(doc_pages, contract_id)
    
    console.print("Adding chunks to vector store...")
    vectorstore.add_chunks_to_vector_store(chunks)

    console.print("Extracting entities and registering contract metadata...")
    entities = []
    if doc_pages:
        try:
            entities = ner.extract_entities(doc_pages)
            console.print(f"[green]Extracted {len(entities)} entities using '{settings.NER_STRATEGY.value}' strategy.[/green]")
        except Exception as e:
            console.print(f"[red]Error:[/red] Entity extraction failed: {e}")

    registry.save_contract(
        contract_id=contract_id,
        original_filename=pdf_path.name,
        uploaded_bytes=pdf_path.read_bytes(),
        num_pages=len(doc_pages),
        entities=entities,
    )
    
    console.print(f"[bold green]✅ Ingestion complete for contract ID: {contract_id}[/bold green]")

@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask about the contract."),
    contract_id: str = typer.Option(None, "--id", help="The specific contract ID to query against.")
):
    """
    Ask a question about an ingested contract.
    """
    console.print(f"[bold blue]Question:[/bold blue] {question}")
    if contract_id:
        console.print(f"[bold blue]Contract ID:[/bold blue] {contract_id}")

    # Initialize retriever (with optional filtering by contract_id)
    console.print("Initializing retriever...")
    retriever = vectorstore.get_vector_store_retriever(contract_id=contract_id)
    
    console.print("Initializing QA chain...")
    qa_chain = qa.get_qa_chain(retriever)
    
    console.print("Querying the model...")
    result = qa_chain.invoke({"query": question})
    
    console.print("\n[bold green]------ Assistant's Response ------[/bold green]\n")
    console.print(result['result'])
    console.print("\n[bold green]----------------------------------[/bold green]\n")


if __name__ == "__main__":
    app()