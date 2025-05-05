import requests
 import json
 
 # Texto de exemplo
 texto = "A inteligência artificial está transformando o mundo."
 
 # Requisição para o Ollama local (o modelo 'nomic-embed-text' deve estar ativo)
 response = requests.post(
     'http://localhost:11434/api/embeddings',
     json={
         "model": "nomic-embed-text",
         "prompt": texto
     }
 )
 
 # Verificando resposta
 if response.status_code == 200:
     data = response.json()
     embeddings = data.get("embedding", [])
     print(f"Tamanho do vetor: {len(embeddings)}")
     print("Primeiros valores:", embeddings[:10])
 else:
     print("Erro:", response.text)

