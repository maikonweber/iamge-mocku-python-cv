# Usar imagem base do Python
FROM python:3.10-slim

# Setar diretório de trabalho no contêiner
WORKDIR /app

# Copiar os arquivos do projeto para o contêiner
COPY . /app

# Instalar dependências do Python (redis, opencv, requests)
RUN pip install --no-cache-dir -r requirements.txt

# Expor a porta (se necessário, por exemplo, se quiser expor algum serviço)
# EXPOSE 5000

# Comando para rodar o script
CMD ["python", "transformeClass.py"]
