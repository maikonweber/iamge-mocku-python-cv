import cv2
import numpy as np
import os

# Diretórios
IMAGEM_DIR = "./imagem/"
ESTAMPAS_DIR = "./estampas/"
RESULTADOS_DIR = "./resultados/"

especificacoes = {
    "BABY_LONG": {"altura_logo": 150, "posicao": "centro"},
    "CAMISA_FEMININA": {"altura_logo": 150, "posicao": "centro"},
    "CAMISA_MANGA_LONGA": {"altura_logo": 150, "posicao": "centro"},
    "CAMISA_SPORT": {"altura_logo": 150, "posicao": "centro"},
    "CROPPED": {"altura_logo": 150, "posicao": "centro"},
    "INFANTIL": {"altura_logo": 150, "posicao": "centro"},
}


def carregar_imagens(diretorio):
    """Carrega todas as imagens de um diretório."""
    print(f"Carregando imagens do diretório: {diretorio}")
    imagens = [cv2.imread(os.path.join(diretorio, img)) for img in os.listdir(diretorio) if img.endswith(('.png', '.jpg', '.jpeg'))]
    if not imagens:
        print("Nenhuma imagem encontrada no diretório.")
    return imagens

def identificar_area_camiseta(imagem):
    """Usa segmentação para identificar a área da camiseta."""
    print("Identificando a área da camiseta...")
    # Converter para escala de cinza
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # Aplicar detecção de bordas
    bordas = cv2.Canny(cinza, 50, 150)
    cv2.imshow("Bordas detectadas", bordas)
    cv2.waitKey(0)

    # Encontrar contornos
    contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        raise ValueError("Nenhum contorno detectado.")

    # Selecionar o maior contorno (presumidamente a camiseta)
    contorno_camiseta = max(contornos, key=cv2.contourArea)
    print(f"Contorno maior encontrado com área: {cv2.contourArea(contorno_camiseta)}")

    # Criar máscara da área da camiseta
    mascara = np.zeros_like(cinza)
    cv2.drawContours(mascara, [contorno_camiseta], -1, 255, thickness=cv2.FILLED)
    cv2.imshow("Máscara da camiseta", mascara)
    cv2.waitKey(0)

    return mascara
def aplicar_estampa_personalizada(imagem, estampa, config):
    """Aplica o logo com dimensões e posicionamento personalizados."""
    print(f"Aplicando estampa com configurações: {config}")
    
    # Extrair dimensões e posição da configuração
    largura_logo = config["largura_logo"]
    altura_logo = config["altura_logo"]
    x_offset = config["x_offset"]
    y_offset = config["y_offset"]

    # Redimensionar o logo para as dimensões especificadas
    estampa_redimensionada = cv2.resize(estampa, (largura_logo, altura_logo))
    print(f"Logo redimensionado para: {largura_logo}x{altura_logo} pixels")

    # Garantir que o deslocamento não ultrapasse os limites da imagem
    altura_imagem, largura_imagem = imagem.shape[:2]
    if x_offset + largura_logo > largura_imagem or y_offset + altura_logo > altura_imagem:
        raise ValueError("O logo ultrapassa os limites da imagem. Verifique as configurações.")

    # Aplicar o logo na posição especificada
    resultado = imagem.copy()
    for i in range(estampa_redimensionada.shape[0]):
        for j in range(estampa_redimensionada.shape[1]):
            if len(estampa_redimensionada[i, j]) == 4 and estampa_redimensionada[i, j][3] > 0:  # Transparência BGRA
                resultado[y_offset + i, x_offset + j] = estampa_redimensionada[i, j][:3]

    cv2.imshow(f"Resultado - {config.get('nome', 'Desconhecido')}", resultado)
    cv2.waitKey(0)
    return resultado

def processar_camisetas_personalizadas():
    """Processa as imagens aplicando o logo com dimensões e posicionamento personalizados."""
    print("Processando camisetas com configurações personalizadas...")
    imagens = carregar_imagens(IMAGEM_DIR)
    if not imagens:
        print("Nenhuma imagem para processar.")
        return

    estampa = cv2.imread(f"{ESTAMPAS_DIR}/estampa.png", cv2.IMREAD_UNCHANGED)
    if estampa is None:
        raise FileNotFoundError("Estampa não encontrada. Verifique o caminho.")

    # Configurações personalizadas para cada tipo de peça
    especificacoes = {
        "BABY_LONG": {"largura_logo": 180, "altura_logo": 200, "x_offset": 165, "y_offset": 325, "nome": "Baby Long"},
        "CAMISA_FEMININA": {"largura_logo": 180, "altura_logo": 200, "x_offset": 165, "y_offset": 300, "nome": "Camisa Feminina"},
        "CAMISA_MANGA_LONGA": {"largura_logo": 180, "altura_logo": 200, "x_offset": 165, "y_offset": 305, "nome": "Camisa Manga Longa"},
        "CAMISA_SPORT": {"largura_logo": 180, "altura_logo": 200, "x_offset": 248, "y_offset": 360, "nome": "Camisa Sport"},
        "CROPPED": {"largura_logo": 160, "altura_logo": 180, "x_offset": 147, "y_offset": 346, "nome": "Cropped"},
        "INFANTIL": {"largura_logo": 180, "altura_logo": 200, "x_offset": 190, "y_offset": 305, "nome": "Infantil"},
        "CANECA": {"largura_logo": 180, "altura_logo": 200, "x_offset": 230, "y_offset": 365, "nome": "CANECA"},
    }

    for img_path in os.listdir(IMAGEM_DIR):
        # Identificar tipo de camiseta pelo nome do arquivo
        nome_base = os.path.splitext(img_path)[0].upper()
        config = especificacoes.get(nome_base)
        if not config:
            print(f"Configurações não encontradas para {nome_base}. Ignorando.")
            continue

        img = cv2.imread(os.path.join(IMAGEM_DIR, img_path))
        print(f"Processando {img_path}...")

        # Aplicar o logo com configurações personalizadas
        try:
            resultado = aplicar_estampa_personalizada(img, estampa, config)
        except Exception as e:
            print(f"Erro ao aplicar estampa em {img_path}: {e}")
            continue

        # Salvar o resultado
        resultado_caminho = f"{RESULTADOS_DIR}/{nome_base}_resultado.png"
        os.makedirs(RESULTADOS_DIR, exist_ok=True)
        cv2.imwrite(resultado_caminho, resultado)
        print(f"Imagem processada e salva em: {resultado_caminho}")

if __name__ == "__main__":
    processar_camisetas_personalizadas()