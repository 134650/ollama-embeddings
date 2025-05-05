import requests
import json

def gerar_embeddings(texto: str, modelo: str = "nomic-embed-text"):
    url = "http://localhost:11434/api/embeddings"
    payload = {
        "model": modelo,
        "prompt": texto
    }

    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        vetor = data.get("embedding", [])
        print(f"\nTexto: {texto}")
        print(f"Tamanho do vetor de embeddings: {len(vetor)}")
        print(f"Primeiros valores: {vetor[:10]}")
    else:
        print("Erro ao gerar embeddings:", response.text)

if __name__ == "__main__":
    texto_exemplo = "Exemplo de texto em português para gerar embeddings."
    gerar_embeddings(texto_exemplo)
