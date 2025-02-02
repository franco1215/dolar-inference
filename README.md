# Previsão do Dólar

Projeto para previsão do preço do dólar no mercado brasileiro.

## Configuração do Ambiente

### Pré-requisitos
- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Configuração

1. Clone o repositório: 

## Variáveis Base do Projeto

O projeto utiliza diversas variáveis econômicas e financeiras para realizar a previsão do dólar. Abaixo estão as principais variáveis base:

### Variáveis Macroeconômicas
- **SELIC**: Taxa básica de juros da economia brasileira
- **DIVIDA_PUBLICA**: Dívida pública federal
- **BALANCA_COMERCIAL**: Saldo da balança comercial brasileira
- **RESERVAS_INTERNACIONAIS**: Volume de reservas internacionais do Brasil
- **INVESTIMENTO_ESTRANGEIRO**: Fluxo de investimento estrangeiro
- **PIB_BR**: Produto Interno Bruto do Brasil
- **PIB_GLOBAL**: Indicador de atividade econômica global

### Variáveis de Commodities
- **SOJA_BR**: Preço da soja no mercado brasileiro
- **SOJA_US**: Preço da soja no mercado americano
- **MILHO_BR**: Preço do milho no mercado brasileiro
- **MILHO_US**: Preço do milho no mercado americano
- **ENERGIA**: Índice de preços de energia

### Variáveis de Mercado
- **DOLAR**: Cotação do dólar (variável alvo)
- **CDS**: Credit Default Swap do Brasil
- **DI_FUTURO**: Taxa DI Futuro

### Variáveis Logísticas
- **BALTIC_DRY**: Índice Baltic Dry (frete marítimo)
- **CUSTO_LOGISTICA**: Índice de custo logístico
- **FRETES**: Índice de fretes

### Variáveis de Análise de Notícias
- **news_impact_mean**: Média do impacto das notícias
- **news_prob_mean**: Média da probabilidade de impacto das notícias
- **news_count**: Contagem de notícias relevantes
- **news_volatility**: Volatilidade baseada em notícias

### Transformações das Variáveis
Para cada variável base, são calculadas as seguintes transformações:
- Retorno (return)
- Volatilidade (volatility)
- Médias Móveis (MA5, MA10, MA20)
- RSI (Índice de Força Relativa)
- Defasagens (lag1, lag2, lag3)

### Fontes de Dados
- Banco Central do Brasil (BCB)
- Yahoo Finance
- Federal Reserve
- Agências de notícias financeiras