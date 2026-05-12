import fitz  # PyMuPDF
from pathlib import Path

import os
import sys

# Questo aiuta Linux a trovare Ghostscript se installato via Conda
os.environ["PATH"] += os.pathsep + sys.prefix + "/bin"

def batch_convert_eps_to_pdf(input_folder_name):
    # Definisce i percorsi
    input_dir = Path(input_folder_name)
    output_dir = input_dir.parent / "Images_pdf"

    # Crea la cartella di destinazione se non esiste
    output_dir.mkdir(exist_ok=True)

    # Conta i file convertiti
    count = 0

    # Cerca tutti i file .eps (case-insensitive)
    for eps_file in input_dir.glob("*.[eE][pP][sS]"):
        try:
            # Definisce il percorso del file PDF in uscita
            pdf_file = output_dir / (eps_file.stem + ".pdf")

            # Conversione
            doc = fitz.open(str(eps_file.absolute()))
            pdf_bytes = doc.convert_to_pdf()
            
            with open(pdf_file, "wb") as f:
                f.write(pdf_bytes)
            
            doc.close()
            print(f"✅ Convertito: {eps_file.name} -> {pdf_file.name}")
            count += 1

        except Exception as e:
            print(f"❌ Errore durante la conversione di {eps_file.name}: {e}")

    print(f"\nOperazione completata. {count} file salvati in: {output_dir}")

# Esecuzione
if __name__ == "__main__":
    batch_convert_eps_to_pdf("Images_eps")