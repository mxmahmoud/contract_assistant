# cli/main.py
import typer
from rich.console import Console
from pathlib import Path

from ca_core import extraction, chunking, vectorstore, qa, registry
from utility.utility import get_hash_of_file

app = typer.Typer()
console = Console()


@app.command()
def ingest(
    pdf_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True, help="Path to the PDF contract to ingest.")
):
    """
    Ingest, chunk, and embed a PDF contract into the vector store.
    """
    console.print(f"[bold green]Starting extraciton for: {pdf_path.name}[/bold green]")
    
    contract_id = get_hash_of_file(pdf_path.read_bytes())
    console.print(f"Generated Contract ID: [cyan]{contract_id}[/cyan]")
    
    console.print("Extracting text from PDF...")
    doc_pages = extraction.extract_text_from_pdf(str(pdf_path))
    
    console.print("Chunking document...")
    chunks = chunking.chunk_document(doc_pages, contract_id)
    
    console.print("Adding chunks to vector store...")
    vectorstore.add_chunks_to_vector_store(chunks)

    console.print("Extracting entities and registering contract metadata...")
    entities = []
    try:
        if doc_pages:
            first_page_text = doc_pages[0]["text"]
            key_entities_data = qa.extract_key_entities(first_page_text)
            
            # Transform for storage
            entity_mapping = {
                "parties": "Party",
                "agreement_date": "Agreement Date", 
                "jurisdiction": "Jurisdiction",
                "contract_type": "Contract Type",
                "termination_date": "Termination Date",
                "monetary_amounts": "Monetary Amount",
                "key_obligations": "Key Obligation"
            }
            
            for key, label in entity_mapping.items():
                values = key_entities_data.get(key)
                if values and values != "Not Found" and values != ["Extraction Failed"]:
                    if isinstance(values, list):
                        for value in values:
                            if value != "Extraction Failed":
                                entities.append({"label": label, "value": value, "page": 1})
                    else:
                        entities.append({"label": label, "value": values, "page": 1})
                        
        console.print(f"[green]Extracted {len(entities)} entities using LLM[/green]")
    except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] Entity extraction failed: {e}")
        # Fallback to regex-based extraction
        try:
            from ca_core import ner as _ner
            entities = _ner.extract_entities(doc_pages)
            console.print(f"[green]Fallback: Extracted {len(entities)} entities using regex[/green]")
        except Exception as e2:
            console.print(f"[red]Error:[/red] Both LLM and regex extraction failed: {e2}")

    registry.save_contract(
        contract_id=contract_id,
        original_filename=pdf_path.name,
        source_pdf_path=pdf_path,
        num_pages=len(doc_pages),
        entities=entities,
    )
    
    console.print(f"[bold green]✅ extraction complete for contract ID: {contract_id}[/bold green]")

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