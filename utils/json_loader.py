# utils/json_loader.py
import os
import json

def load_instagram_json(directory_path: str):
    """
    Procura e carrega o arquivo JSON mais recente na pasta especificada.
    
    Args:
        directory_path (str): Caminho da pasta onde os JSONs estão salvos.
    
    Returns:
        dict: Dados do JSON carregado.
    """
    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    if not json_files:
        raise FileNotFoundError("Nenhum arquivo JSON encontrado em " + directory_path)
    
    # Ordena os arquivos pela data de modificação (descendente)
    json_files.sort(key=lambda f: os.path.getmtime(os.path.join(directory_path, f)), reverse=True)
    latest_file = os.path.join(directory_path, json_files[0])
    with open(latest_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Para teste
if __name__ == "__main__":
    try:
        data = load_instagram_json("data")
        print("Arquivo carregado:", data.get("title", "Título não encontrado"))
    except Exception as e:
        print("Erro:", e)
