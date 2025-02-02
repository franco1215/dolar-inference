#!/bin/bash

echo "Criando ambiente virtual Python..."
python3 -m venv .venv

echo "Ativando ambiente virtual..."
source .venv/bin/activate

echo "Instalando dependências..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Ambiente virtual configurado com sucesso!"
echo "Para ativar o ambiente virtual, use: source .venv/bin/activate" 