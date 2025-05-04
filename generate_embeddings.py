import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def generate_embeddings(data, method='pca', n_components=2):
    """
    Gera embeddings para o dados fornecidos.

    Args:
        data (numpy.ndarray): Dados a serem transformados.
        method (str, optional): Método de transformação. Pode ser 'pca' ou 't-sne'. Defaults to 'pca'.
        n_components (int, optional): Número de componentes para o PCA. Defaults to 2.

    Returns:
        numpy.ndarray: Embeddings gerados.
    """
    if method == 'pca':
        pca = PCA(n_components=n_components)
        embeddings = pca.fit_transform(data)
    elif method == 't-sne':
        tsne = TSNE(n_components=n_components, random_state=42)
        embeddings = tsne.fit_transform(data)
    else:
        raise ValueError("Método de transformação inválido. Pode ser 'pca' ou 't-sne'.")

    return embeddings

if __name__ == '__main__':
    # Geração de dados amostra
    np.random.seed(42)
    data = np.random.rand(100, 3)

    # Geração de embeddings
    embeddings_pca = generate_embeddings(data, method='pca')
    embeddings_t_sne = generate_embeddings(data, method='t-sne')

    # Impressão dos resultados
    print("Embeddings PCA:")
    print(embeddings_pca.shape)
    print("Embeddings t-SNE:")
    print(embeddings_t_sne.shape)
