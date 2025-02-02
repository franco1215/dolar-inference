import pandas as pd
import requests
from datetime import datetime, timedelta
from newspaper import Article
from typing import List, Dict, Optional, Union
import os
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
import json
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import feedparser
import time
import re
import random
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import sys
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewsAnalyzer:
    """
    Classe unificada para coleta e análise de notícias financeiras.
    Combina funcionalidades de coleta, processamento e análise de impacto.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa o analisador de notícias com configurações personalizáveis
        
        Args:
            config_path: Caminho para arquivo de configuração JSON (opcional)
        """
        load_dotenv()
        self.logger = logger
        
        # Inicializar atributos
        self.config: Dict = {}
        self.collection_config: Dict = {}
        self.model_config: Dict = {}
        self.news_sources: Dict = {}
        
        # Lista de User-Agents comuns
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36'
        ]
        
        # Setup inicial
        self.setup_config(config_path)
        self.setup_directories()
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def setup_config(self, config_path: Optional[str] = None) -> None:
        """
        Configura parâmetros do analisador
        
        Args:
            config_path: Caminho para arquivo de configuração JSON
        """
        default_config = {
            "news_sources": {
                "official_sources": {
                    "federal_reserve": {
                        "endpoints": {
                            "press_releases": "https://www.federalreserve.gov/feeds/press_all.xml",
                            "speeches": "https://www.federalreserve.gov/feeds/speeches.xml",
                            "testimony": "https://www.federalreserve.gov/feeds/testimony.xml"
                        },
                        "topics": ["monetary policy", "dollar", "exchange rate", "interest rates", "inflation", "economic outlook"]
                    },
                    "banco_central": {
                        "endpoints": {
                            "noticias": "https://www.bcb.gov.br/noticias/busca",
                            "comunicados": "https://www.bcb.gov.br/detalhenoticia/",
                            "notas_imprensa": "https://www.bcb.gov.br/notas-imprensa-comunicados"
                        },
                        "rss_fallback": "https://www.bcb.gov.br/rss/noticias",
                        "topics": [
                            "câmbio", "dólar", "política monetária", "taxa selic", "inflação", 
                            "mercado cambial", "fluxo cambial", "balanço de pagamentos",
                            "reservas internacionais", "swap cambial", "intervenção cambial",
                            "copom", "ptax", "taxa de juros", "economia internacional",
                            "balança comercial", "investimento estrangeiro"
                        ]
                    }
                },
                "rss_feeds": {
                    "primary": [
                        "https://www.moneytimes.com.br/feed/",
                        "https://www.infomoney.com.br/feed/",
                        "https://rss.cnnbrasil.com.br/rss/economia",
                        "https://br.investing.com/rss/news_1.rss",
                        "https://br.investing.com/rss/news_25.rss",
                        "https://br.investing.com/rss/news_95.rss",
                        "https://br.investing.com/rss/news_1.rss",
                        "https://www.valor.com.br/financas/rss"
                    ],
                    "secondary": [
                        "https://www.federalreserve.gov/feeds/press_all.xml",
                        "https://www.imf.org/en/News/rss",
                        "https://www.ecb.europa.eu/rss/press.html"
                    ]
                }
            },
            "request_config": {
                "timeout": 60,
                "max_retries": 3,
                "retry_delay": 2,
                "verify_ssl": True,
                "headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection": "keep-alive",
                    "Cache-Control": "no-cache"
                }
            },
            "model_config": {
                "model": "mixtral-8x7b-32768",
                "temperature": 0,
                "max_tokens": 2000,
                "top_p": 0.1
            },
            "data_paths": {
                "raw_news": "data/raw/news",
                "processed_news": "data/processed/news",
                "impact_analysis": "data/processed/impact"
            },
            "collection_config": {
                "max_articles_per_source": 50,
                "max_threads": 4,
                "retry_attempts": 3,
                "request_timeout": 30,
                "min_article_length": 100,
                "max_article_age_days": 30,
                "verify_ssl": False,
                "follow_redirects": True,
                "required_keywords": [
                    "dólar", "dollar", "real", "câmbio", "USD", "BRL",
                    "exchange rate", "forex", "currency", "FX"
                ]
            }
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                self.config = {**default_config, **user_config}
        else:
            self.config = default_config

        self.news_sources = self.config["news_sources"]
        self.model_config = self.config["model_config"]
        self.collection_config = self.config["collection_config"]
        
    def setup_directories(self) -> None:
        """Cria estrutura de diretórios necessária"""
        for path in self.config["data_paths"].values():
            Path(path).mkdir(parents=True, exist_ok=True)
            
    def _get_browser_headers(self) -> Dict[str, str]:
        """
        Gera headers simulando um browser comum
        
        Returns:
            Dict[str, str]: Headers para requisição
        """
        # Escolhe um User-Agent aleatório
        user_agent = random.choice(self.user_agents)
        
        # Headers comuns de browser
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
        
        return headers

    def _collect_source_articles(self, source: str, days_back: int) -> List[Dict]:
        """
        Coleta artigos de uma fonte específica
        
        Args:
            source: Nome ou URL da fonte
            days_back: Número de dias para coletar
            
        Returns:
            List[Dict]: Lista de artigos coletados
        """
        try:
            articles = []
            
            # Coletar de fontes oficiais
            if source in self.config["news_sources"]["official_sources"]:
                source_config = self.config["news_sources"]["official_sources"][source]
                if source == "banco_central":
                    articles.extend(self._collect_from_bcb(source_config, days_back))
                elif source == "federal_reserve":
                    articles.extend(self._collect_from_fed(source_config, days_back))
            
            # Coletar de RSS feeds
            elif source in self.config["news_sources"]["rss_feeds"]["primary"] or \
                 source in self.config["news_sources"]["rss_feeds"]["secondary"]:
                articles.extend(self._collect_from_rss(source, days_back))
            
            return self._filter_relevant_articles(articles)
            
        except Exception as e:
            _, _, tb = sys.exc_info()
            lineno = tb.tb_lineno if tb is not None else "unknown"
            self.logger.error(f"Erro ao coletar de {source} (linha {lineno}): {str(e)}")
            return []
            
    def _collect_from_bcb(self, source_config: Dict, days_back: int) -> List[Dict]:
        """
        Coleta notícias do Banco Central usando LLM para extrair dados
        
        Args:
            source_config: Configuração da fonte
            days_back: Número de dias para coletar
            
        Returns:
            List[Dict]: Lista de artigos coletados
        """
        articles = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Criar sessão com headers de browser
        session = requests.Session()
        session.headers.update(self.config["request_config"]["headers"])
        
        # Verificar se a configuração possui 'endpoints'; se não, usa fallback para RSS
        endpoints = source_config.get("endpoints")
        if not endpoints:
            self.logger.warning("Configuração da fonte BCB sem endpoints, utilizando método RSS como fallback")
            return self._collect_from_rss("banco_central", days_back)
        
        # Coletar de cada endpoint
        for endpoint_name, endpoint_url in endpoints.items():
            try:
                self.logger.info(f"Coletando do endpoint BCB {endpoint_name}: {endpoint_url}")
                
                # Fazer requisição
                response = session.get(
                    endpoint_url,
                    timeout=self.config["request_config"]["timeout"],
                    verify=self.config["request_config"]["verify_ssl"],
                    allow_redirects=True
                )
                
                if response.status_code == 200:
                    # Extrair notícias usando LLM
                    extracted_news = self._extract_news_with_llm(
                        response.text,
                        f"Banco Central do Brasil - {endpoint_name.title()}"
                    )
                    
                    # Filtrar por data e relevância
                    for news in extracted_news:
                        try:
                            # Converter data
                            news_date = datetime.fromisoformat(news['date'].replace('Z', '+00:00'))
                            
                            if news_date >= cutoff_date:
                                # Verificar relevância
                                if any(topic.lower() in news['title'].lower() or 
                                      topic.lower() in news['content'].lower() 
                                      for topic in source_config["topics"]):
                                    
                                    # Normalizar conteúdo
                                    news['content'] = self._normalize_content(news['content'])
                                    news['title'] = self._normalize_content(news['title'])
                                    
                                    if len(news['content']) >= self.collection_config["min_article_length"]:
                                        news['tipo'] = endpoint_name
                                        articles.append(news)
                                        
                        except Exception as e:
                            self.logger.debug(f"Erro ao processar notícia do BCB: {str(e)}")
                            continue
                            
                else:
                    self.logger.warning(f"Endpoint do BCB {endpoint_name} retornou status {response.status_code}")
                    
            except Exception as e:
                _, _, tb = sys.exc_info()
                lineno = tb.tb_lineno if tb is not None else "unknown"
                self.logger.error(f"Erro ao acessar endpoint {endpoint_name} do BCB (linha {lineno}): {str(e)}")
                continue
                
            # Delay entre endpoints
            time.sleep(random.uniform(2, 3))
        
        if articles:
            self.logger.info(f"Coletadas {len(articles)} notícias do BCB")
            # Log detalhado por tipo
            for tipo in source_config["endpoints"].keys():
                count = len([a for a in articles if a['tipo'] == tipo])
                if count > 0:
                    self.logger.info(f"- {tipo.title()}: {count} itens")
        else:
            self.logger.warning("Nenhuma notícia coletada do BCB")
            
        return articles
        
    def _collect_from_fed(self, source_config: Dict, days_back: int) -> List[Dict]:
        """
        Coleta notícias do Federal Reserve usando LLM para extrair dados
        
        Args:
            source_config: Configuração da fonte
            days_back: Número de dias para coletar
            
        Returns:
            List[Dict]: Lista de artigos coletados
        """
        articles = []
        try:
            endpoints = source_config.get("endpoints")
            if not endpoints or "press_releases" not in endpoints:
                self.logger.warning("Configuração da fonte FED sem endpoints adequados, utilizando método RSS como fallback")
                return self._collect_from_rss("federal_reserve", days_back)
            feed_url = endpoints["press_releases"]
            
            # Criar sessão com headers de browser
            session = requests.Session()
            session.headers.update(self.config["request_config"]["headers"])
            
            # Fazer requisição
            response = session.get(
                feed_url,
                timeout=self.config["request_config"]["timeout"],
                verify=self.config["request_config"]["verify_ssl"],
                allow_redirects=True
            )
            
            if response.status_code == 200:
                # Extrair notícias usando LLM
                extracted_news = self._extract_news_with_llm(
                    response.text,
                    "Federal Reserve"
                )
                
                # Filtrar por relevância
                for news in extracted_news:
                    if any(topic.lower() in news['title'].lower() or 
                          topic.lower() in news['content'].lower() 
                          for topic in source_config["topics"]):
                        
                        # Normalizar conteúdo
                        news['content'] = self._normalize_content(news['content'])
                        news['title'] = self._normalize_content(news['title'])
                        
                        if len(news['content']) >= self.collection_config["min_article_length"]:
                            articles.append(news)
                            
            else:
                self.logger.warning(f"Feed do FED retornou status {response.status_code}")
                    
        except Exception as e:
            _, _, tb = sys.exc_info()
            lineno = tb.tb_lineno if tb is not None else "unknown"
            self.logger.error(f"Erro na coleta do FED (linha {lineno}): {str(e)}")
            return []
        
        if articles:
            self.logger.info(f"Coletadas {len(articles)} notícias do FED")
        else:
            self.logger.warning("Nenhuma notícia coletada do FED")
            
        return articles

    def _collect_from_rss(self, source: str, days_back: int) -> List[Dict]:
        """
        Coleta notícias de um feed RSS usando LLM para extrair dados
        
        Args:
            source: URL do feed RSS
            days_back: Número de dias para coletar
            
        Returns:
            List[Dict]: Lista de artigos coletados
        """
        articles = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        try:
            # Se for chamada de fallback, usar URL correta do feed
            if source == "banco_central":
                if "banco_central" in self.config["news_sources"]["official_sources"]:
                    source = self.config["news_sources"]["official_sources"]["banco_central"].get("rss_fallback", source)
            elif source == "federal_reserve":
                if "federal_reserve" in self.config["news_sources"]["official_sources"]:
                    source = self.config["news_sources"]["official_sources"]["federal_reserve"]["endpoints"]["press_releases"]

            # Configurar sessão com certificados SSL
            session = requests.Session()
            session.verify = True  # Habilitar verificação SSL
            
            try:
                import certifi
                session.verify = certifi.where()
            except ImportError:
                self.logger.warning("Pacote certifi não encontrado, usando certificados padrão")
            
            # Configurar headers simulando browser
            headers = self._get_browser_headers()
            headers.update({
                'Accept': '*/*'
            })
            
            # Atualizar headers da sessão
            session.headers.update(headers)
            
            # Configurar adaptadores com retry
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET", "HEAD"]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            # Tentar diferentes variações da URL
            urls_to_try = [
                source,
                source.replace('http://', 'https://'),
                source.replace('https://', 'http://')
            ]
            
            success = False
            for url in urls_to_try:
                try:
                    # Adicionar referrer baseado no domínio
                    parsed_url = urlparse(url)
                    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                    session.headers.update({'Referer': base_url})
                    
                    # Primeiro tentar com feedparser
                    feed = feedparser.parse(url)
                    
                    if feed.bozo == 0 and feed.entries:  # Feed válido
                        for entry in feed.entries:
                            try:
                                # Extrair data
                                if hasattr(entry, 'published_parsed'):
                                    news_date = datetime(*entry.published_parsed[:6])
                                elif hasattr(entry, 'updated_parsed'):
                                    news_date = datetime(*entry.updated_parsed[:6])
                                else:
                                    continue

                                if news_date >= cutoff_date:
                                    # Extrair conteúdo
                                    content = entry.get('description', '') or entry.get('summary', '') or ''
                                    if hasattr(entry, 'content'):
                                        content = entry.content[0].value if isinstance(entry.content, list) else entry.content

                                    # Limpar HTML
                                    if content:
                                        soup = BeautifulSoup(content, 'html.parser')
                                        content = soup.get_text(separator=' ', strip=True)

                                    if len(content) >= self.collection_config["min_article_length"]:
                                        news = {
                                            'title': entry.get('title', ''),
                                            'content': content,
                                            'date': news_date.isoformat(),
                                            'url': entry.get('link', ''),
                                            'source': parsed_url.netloc
                                        }
                                        articles.append(news)

                            except Exception as e:
                                self.logger.debug(f"Erro ao processar entrada RSS de {url}: {str(e)}")
                                continue

                        if articles:
                            success = True
                            self.logger.info(f"Coletadas {len(articles)} notícias via RSS de {url}")
                            break
                     
                    # Se feedparser falhar, tentar com requests + BeautifulSoup
                    try:
                        response = session.get(url, timeout=self.collection_config["request_timeout"])
                        response.raise_for_status()
                        
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Tentar extrair notícias da página
                        articles.extend(self._extract_news_from_html(soup, url, cutoff_date))
                         
                        if articles:
                            success = True
                            self.logger.info(f"Coletadas {len(articles)} notícias via HTML de {url}")
                            break
                         
                    except Exception as e:
                        self.logger.error(f"Erro ao acessar {url}: {str(e)}")
                        continue
                
                except Exception as e:
                    self.logger.warning(f"Erro ao tentar {url}: {str(e)}")
                    continue
                
                # Adicionar delay aleatório entre tentativas
                time.sleep(random.uniform(1, 3))
            
            if not success:
                self.logger.error(f"Todas as tentativas falharam para {source}")
                
        except Exception as e:
            _, _, tb = sys.exc_info()
            lineno = tb.tb_lineno if tb is not None else "unknown"
            self.logger.error(f"Erro na coleta do RSS {source} (linha {lineno}): {str(e)}")
            return []
        
        return articles

    def _extract_date(self, entry) -> Optional[datetime]:
        """Extrai e valida data de uma entrada RSS"""
        try:
            if hasattr(entry, 'published_parsed'):
                return datetime(*entry.published_parsed[:6])
            elif hasattr(entry, 'updated_parsed'):
                return datetime(*entry.updated_parsed[:6])
            return None
        except Exception:
            return None

    def _extract_content(self, entry) -> Optional[str]:
        """Extrai e valida conteúdo de uma entrada RSS"""
        try:
            if hasattr(entry, 'description'):
                return entry.description
            elif hasattr(entry, 'summary'):
                return entry.summary
            elif hasattr(entry, 'content'):
                return entry.content[0].value if isinstance(entry.content, list) else entry.content
            return None
        except Exception:
            return None

    def _extract_title(self, entry) -> Optional[str]:
        """Extrai e valida título de uma entrada RSS"""
        try:
            return entry.title if hasattr(entry, 'title') else None
        except Exception:
            return None

    def _normalize_content(self, text: str) -> str:
        """
        Normaliza o conteúdo do texto removendo HTML e caracteres especiais
        
        Args:
            text: Texto para normalizar
            
        Returns:
            str: Texto normalizado
        """
        try:
            # Remover tags HTML
            text = re.sub(r'<[^>]+>', ' ', text)
            
            # Normalizar espaços
            text = ' '.join(text.split())
            
            # Remover caracteres especiais mantendo pontuação básica
            text = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', ' ', text)
            
            # Normalizar aspas
            text = text.replace('"', '"').replace('"', '"')
            
            # Escapar caracteres especiais para JSON
            text = json.dumps(text)[1:-1]
            
            return text.strip()
        except Exception as e:
            _, _, tb = sys.exc_info()
            lineno = tb.tb_lineno if tb is not None else "unknown"
            self.logger.warning(f"Erro ao normalizar texto (linha {lineno}): {str(e)}")
            return ""

    def _filter_relevant_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        Filtra artigos por relevância
        
        Args:
            articles: Lista de artigos para filtrar
            
        Returns:
            List[Dict]: Lista de artigos filtrados
        """
        filtered = []
        
        for article in articles:
            try:
                if not all(k in article for k in ['title', 'content', 'date', 'source']):
                    continue
                    
                content = article['content'].lower()
                title = article['title'].lower()
                
                if any(keyword.lower() in content or keyword.lower() in title 
                      for keyword in self.collection_config["required_keywords"]):
                    filtered.append(article)
                    
            except Exception as e:
                _, _, tb = sys.exc_info()
                lineno = tb.tb_lineno if tb is not None else "unknown"
                self.logger.warning(f"Erro ao filtrar artigo (linha {lineno}): {str(e)}")
                continue
                
        return filtered
        
    def _analyze_impact(self, title: str, content: str) -> Dict:
        """
        Analisa o impacto de uma notícia com conteúdo normalizado
        
        Args:
            title: Título da notícia
            content: Conteúdo da notícia
            
        Returns:
            Dict: Análise de impacto
        """
        try:
            # Garantir que título e conteúdo estão normalizados
            title = self._normalize_content(title)
            content = self._normalize_content(content)
            
            prompt = f'''Você é um analisador financeiro especializado em forex. Analise o impacto desta notícia no par USD/BRL (dólar/real):

TÍTULO: {title}
CONTEÚDO: {content}

INSTRUÇÕES IMPORTANTES:
1. Responda APENAS com um JSON válido
2. Não inclua explicações adicionais
3. Siga exatamente a estrutura abaixo
4. Use apenas números para scores e probabilidades
5. Use apenas strings para textos explicativos
6. Mantenha respostas concisas e objetivas

ESTRUTURA OBRIGATÓRIA:
{{
    "impact_score": <número entre 1 e 5>,
    "impact_probability": <número entre 0 e 100>,
    "reasoning": {{
        "macro_context": <string com análise do contexto macroeconômico>,
        "key_variables": <string listando principais variáveis afetadas>,
        "usd_correlation": <string explicando correlação com USD>,
        "uncertainty_factors": <string listando fatores de incerteza>,
        "final_reasoning": <string com conclusão final>
    }}
}}

EXEMPLO DE FORMATO VÁLIDO:
{{
    "impact_score": 3,
    "impact_probability": 75,
    "reasoning": {{
        "macro_context": "Cenário de alta inflação global",
        "key_variables": "Taxa Selic, inflação, balança comercial",
        "usd_correlation": "Correlação positiva moderada",
        "uncertainty_factors": "Tensões geopolíticas, decisões do Fed",
        "final_reasoning": "Tendência de alta no curto prazo"
    }}
}}'''
            
            completion = self.groq_client.chat.completions.create(
                model=self.model_config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.model_config["temperature"],
                max_tokens=self.model_config["max_tokens"],
                top_p=self.model_config["top_p"]
            )
            
            if not completion.choices or not completion.choices[0].message:
                raise ValueError("Resposta vazia do modelo")
            
            result_text = completion.choices[0].message.content
            if not isinstance(result_text, str):
                raise ValueError("Resposta do modelo não é uma string")
            
            # Tentar parsear JSON com tratamento de erro
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError as e:
                self.logger.warning(f"Erro ao decodificar JSON: {str(e)}")
                # Tentar limpar o JSON antes de parsear novamente
                result_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', result_text)
                result = json.loads(result_text)
            
            # Validar resultado
            if not isinstance(result, dict):
                raise ValueError("Resultado inválido")
            
            required_fields = ['impact_score', 'impact_probability', 'reasoning']
            if not all(field in result for field in required_fields):
                raise ValueError("Campos obrigatórios ausentes")
            
            return result
            
        except Exception as e:
            _, _, tb = sys.exc_info()
            lineno = tb.tb_lineno if tb is not None else "unknown"
            self.logger.error(f"Erro na análise de impacto (linha {lineno}): {str(e)}")
            return {
                "impact_score": 3,
                "impact_probability": 50,
                "reasoning": {
                    "macro_context": "Análise indisponível",
                    "key_variables": "Análise indisponível",
                    "usd_correlation": "Análise indisponível",
                    "uncertainty_factors": "Análise indisponível",
                    "final_reasoning": "Análise indisponível"
                }
            }

    def analyze_news(
        self,
        days_back: int = 30,
        force_reload: bool = False
    ) -> pd.DataFrame:
        """
        Método principal para análise de notícias. Combina coleta e análise.
        
        Args:
            days_back: Número de dias para buscar histórico
            force_reload: Se True, força recarga mesmo se dados existirem
            
        Returns:
            pd.DataFrame: DataFrame com notícias processadas e seus impactos
        """
        try:
            output_path = f"{self.config['data_paths']['impact_analysis']}/news_impact_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            
            if not force_reload and os.path.exists(output_path):
                self.logger.info(f"Carregando dados existentes de {output_path}")
                df = pd.read_csv(output_path)
                df['date'] = pd.to_datetime(df['date'])
                return df

            self.logger.info(f"Iniciando análise de {days_back} dias de notícias...")
            
            # Coleta paralela de notícias
            all_news = self._collect_all_news(days_back)
            
            if not all_news:
                self.logger.error("Nenhuma notícia coletada")
                return pd.DataFrame()

            # Análise de impacto em lotes
            processed_news = self._process_news_batch(all_news)
            
            if not processed_news:
                self.logger.error("Nenhuma notícia processada com sucesso")
                return pd.DataFrame()

            # Criar DataFrame e selecionar top notícias
            df = pd.DataFrame(processed_news)
            df['date'] = pd.to_datetime(df['date'])
            
            # Ordenar e filtrar top 20 por dia
            df = df.sort_values(
                by=['date', 'impact_score', 'impact_probability'], 
                ascending=[True, False, False]
            )
            top_news = df.groupby(df['date'].dt.date).head(20)
            
            # Salvar resultados
            top_news.to_csv(output_path, index=False)
            
            # Exibir resumo
            self._display_analysis_summary(top_news)
            
            return top_news

        except Exception as e:
            _, _, tb = sys.exc_info()
            lineno = tb.tb_lineno if tb is not None else "unknown"
            self.logger.error(f"Erro na análise de notícias (linha {lineno}): {str(e)}")
            return pd.DataFrame()
            
    def _validate_url(self, url: str) -> bool:
        """
        Valida se uma URL está acessível
        
        Args:
            url: URL para validar
            
        Returns:
            bool: True se a URL está acessível, False caso contrário
        """
        try:
            session = requests.Session()
            session.headers.update(self.config["request_config"]["headers"])
            
            response = session.head(
                url,
                timeout=self.config["request_config"]["timeout"],
                verify=self.config["request_config"]["verify_ssl"],
                allow_redirects=True
            )
            
            return response.status_code == 200
            
        except Exception as e:
            _, _, tb = sys.exc_info()
            lineno = tb.tb_lineno if tb is not None else "unknown"
            self.logger.debug(f"URL {url} inválida (linha {lineno}): {str(e)}")
            return False

    def _get_valid_sources(self) -> Dict[str, List[str]]:
        """
        Filtra e retorna apenas as fontes válidas
        
        Returns:
            Dict[str, List[str]]: Dicionário com fontes válidas
        """
        valid_sources = {
            "primary": [],
            "secondary": []
        }
        
        # Validar feeds RSS
        for category in ["primary", "secondary"]:
            for url in self.news_sources["rss_feeds"][category]:
                if self._validate_url(url):
                    valid_sources[category].append(url)
                else:
                    self.logger.warning(f"Feed RSS inválido removido: {url}")
        
        return valid_sources

    def _collect_all_news(self, days_back: int) -> List[Dict]:
        """
        Coleta notícias apenas de fontes válidas com tratamento robusto de erros
        
        Args:
            days_back: Número de dias para coletar
            
        Returns:
            List[Dict]: Lista de notícias coletadas
        """
        all_news = []
        successful_sources = set()
        failed_sources = {}
        source_attempts = {}
        
        # Obter apenas fontes válidas
        valid_sources = self._get_valid_sources()
        
        with ThreadPoolExecutor(max_workers=self.collection_config["max_threads"]) as executor:
            futures = {}
            
            # Coletar de fontes oficiais
            for source, config in self.config["news_sources"]["official_sources"].items():
                source_attempts[source] = 0
                futures[source] = executor.submit(self._collect_source_articles, source, days_back)
            
            # Coletar apenas de feeds RSS válidos
            for category in ["primary", "secondary"]:
                for source in valid_sources[category]:
                    source_attempts[source] = 0
                    futures[source] = executor.submit(self._collect_from_rss, source, days_back)
            
            # Processar resultados com retry
            while futures:
                for source, future in list(futures.items()):
                    if future.done():
                        try:
                            articles = future.result()
                            if articles:
                                all_news.extend(articles)
                                successful_sources.add(source)
                                futures.pop(source)
                            else:
                                source_attempts[source] += 1
                                if source_attempts[source] < self.collection_config["retry_attempts"]:
                                    self.logger.warning(f"Tentativa {source_attempts[source]} falhou para {source}, tentando novamente...")
                                    time.sleep(self.config["request_config"]["retry_delay"])
                                    
                                    # Determinar qual método usar baseado no tipo da fonte
                                    collection_method = (self._collect_source_articles 
                                                       if source in self.config["news_sources"]["official_sources"] 
                                                       else self._collect_from_rss)
                                    futures[source] = executor.submit(collection_method, source, days_back)
                                else:
                                    failed_sources[source] = "Sem artigos retornados após todas tentativas"
                                    futures.pop(source)
                        except Exception as e:
                            source_attempts[source] += 1
                            error_msg = str(e)
                            if source_attempts[source] < self.collection_config["retry_attempts"]:
                                self.logger.warning(f"Erro na tentativa {source_attempts[source]} para {source}: {error_msg}")
                                time.sleep(self.config["request_config"]["retry_delay"])
                                
                                collection_method = (self._collect_source_articles 
                                                   if source in self.config["news_sources"]["official_sources"] 
                                                   else self._collect_from_rss)
                                futures[source] = executor.submit(collection_method, source, days_back)
                            else:
                                failed_sources[source] = error_msg
                                futures.pop(source)
                                
                time.sleep(1)  # Evitar sobrecarga
        
        # Log detalhado
        self._log_collection_results(successful_sources, failed_sources, source_attempts)
        
        return all_news

    def _log_collection_results(self, successful_sources: set, failed_sources: dict, source_attempts: dict) -> None:
        """
        Registra resultados detalhados da coleta
        
        Args:
            successful_sources: Conjunto de fontes bem-sucedidas
            failed_sources: Dicionário de fontes que falharam e seus erros
            source_attempts: Dicionário com número de tentativas por fonte
        """
        self.logger.info(f"Fontes bem-sucedidas ({len(successful_sources)}): {', '.join(successful_sources)}")
        
        if failed_sources:
            self.logger.info(f"Fontes com falha ({len(failed_sources)}): ")
            for source, error in failed_sources.items():
                self.logger.error(f"- {source}: {error}")
        
        # Salvar metadados
        self._save_collection_metadata(successful_sources, failed_sources, source_attempts)

    def _save_collection_metadata(self, successful_sources: set, failed_sources: dict, source_attempts: dict) -> None:
        """
        Salva metadados da coleta de notícias
        
        Args:
            successful_sources: Conjunto de fontes bem-sucedidas
            failed_sources: Dicionário de fontes que falharam e seus erros
            source_attempts: Dicionário com número de tentativas por fonte
        """
        try:
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'successful_sources': list(successful_sources),
                'failed_sources': failed_sources,
                'source_attempts': source_attempts,
                'collection_stats': {
                    'total_sources': len(successful_sources) + len(failed_sources),
                    'success_rate': len(successful_sources) / (len(successful_sources) + len(failed_sources)) * 100
                }
            }
            
            metadata_path = Path(self.config['data_paths']['processed_news']) / f"collection_metadata_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            self.logger.info(f"Metadados da coleta salvos em: {metadata_path}")
            
        except Exception as e:
            _, _, tb = sys.exc_info()
            lineno = tb.tb_lineno if tb is not None else "unknown"
            self.logger.error(f"Erro ao salvar metadados da coleta (linha {lineno}): {str(e)}")

    def _process_news_batch(self, all_news: List[Dict]) -> List[Dict]:
        """
        Processa lotes de notícias para análise de impacto
        
        Args:
            all_news: Lista de notícias para processar
            
        Returns:
            List[Dict]: Lista de notícias processadas com análise de impacto
        """
        processed_news = []
        failed_analysis = 0
        skipped_news = 0
        batch_size = 5
        
        for i in range(0, len(all_news), batch_size):
            batch = all_news[i:i + batch_size]
            for news in batch:
                try:
                    # Validar se a notícia foi carregada corretamente
                    if not all(k in news for k in ['title', 'content', 'date', 'source']):
                        self.logger.warning(f"Notícia com campos faltando: {news}")
                        skipped_news += 1
                        continue
                    
                    # Validar conteúdo da notícia
                    if not news['content'] or len(news['content'].strip()) < self.collection_config["min_article_length"]:
                        self.logger.warning(f"Notícia com conteúdo insuficiente: {news['title']}")
                        skipped_news += 1
                        continue
                    
                    # Validar data da notícia
                    try:
                        if isinstance(news['date'], str):
                            news_date = datetime.fromisoformat(news['date'])
                        else:
                            news_date = news['date']
                        
                        if not isinstance(news_date, datetime):
                            self.logger.warning(f"Data inválida para notícia: {news['title']}")
                            skipped_news += 1
                            continue
                    except Exception as e:
                        self.logger.warning(f"Erro ao validar data da notícia: {str(e)}")
                        skipped_news += 1
                        continue
                    
                    # Tentar análise de impacto com retry
                    for attempt in range(self.collection_config["retry_attempts"]):
                        try:
                            impact_analysis = self._analyze_impact(news['title'], news['content'])
                            
                            if not isinstance(impact_analysis, dict):
                                raise ValueError("Análise de impacto retornou formato inválido")
                            
                            required_fields = ['impact_score', 'impact_probability', 'reasoning']
                            if not all(field in impact_analysis for field in required_fields):
                                raise ValueError("Análise de impacto com campos faltando")
                            
                            news_with_impact = {
                                'date': news['date'],
                                'title': news['title'],
                                'description': news['content'],
                                'source': news['source'],
                                'impact_score': float(impact_analysis['impact_score']),
                                'impact_probability': float(impact_analysis['impact_probability']),
                                'impact_reasoning': impact_analysis['reasoning']
                            }
                            
                            processed_news.append(news_with_impact)
                            break
                            
                        except Exception as e:
                            if attempt == self.collection_config["retry_attempts"] - 1:
                                self.logger.error(f"Erro ao analisar impacto após {attempt + 1} tentativas: {str(e)}")
                                failed_analysis += 1
                            time.sleep(1)
                            
                except Exception as e:
                    _, _, tb = sys.exc_info()
                    lineno = tb.tb_lineno if tb is not None else "unknown"
                    self.logger.error(f"Erro ao processar notícia (linha {lineno}): {str(e)}")
                    failed_analysis += 1
            
            time.sleep(0.5)  # Pausa entre lotes

        self.logger.info(f"Total de notícias processadas com sucesso: {len(processed_news)}")
        self.logger.info(f"Total de notícias ignoradas: {skipped_news}")
        self.logger.info(f"Total de análises falhas: {failed_analysis}")
        
        return processed_news
        
    def _display_analysis_summary(self, df: pd.DataFrame) -> None:
        """
        Exibe resumo da análise de notícias
        
        Args:
            df: DataFrame com notícias processadas
        """
        print("\n=== Resumo da Análise de Notícias ===")
        print(f"Total de notícias: {len(df)}")
        
        if not df.empty:
            print(f"Período: {df['date'].min()} até {df['date'].max()}")
            print(f"Fontes únicas: {df['source'].nunique()}")
            print(f"Média de impacto: {df['impact_score'].mean():.2f}")
            
            print("\nDistribuição por fonte:")
            print(df['source'].value_counts().head())
            
            print("\nDistribuição dos scores de impacto:")
            print(df['impact_score'].describe())
        
        print("===================================")

    def _extract_news_with_llm(self, content: str, source: str) -> List[Dict]:
        """
        Usa LLM para extrair notícias do conteúdo HTML/XML
        
        Args:
            content: Conteúdo HTML/XML da página
            source: Nome da fonte
            
        Returns:
            List[Dict]: Lista de notícias extraídas
        """
        try:
            # Normalizar conteúdo removendo caracteres não imprimíveis excessivos
            non_printable = sum(1 for c in content if not c.isprintable() and c not in ['\n', '\r', '\t'])
            if len(content) > 0 and (non_printable / len(content)) > 0.2:
                self.logger.info("Normalizando conteúdo: removendo caracteres não imprimíveis excessivos")
                content = ''.join(c if c.isprintable() or c in ['\n', '\r', '\t'] else ' ' for c in content)

            # Limitar tamanho do conteúdo para respeitar o limite de tokens do modelo GPT-3.5
            # GPT-3.5 tem limite de 16385 tokens. Reservamos:
            # - 2000 tokens para o prompt base e instruções
            # - 500 tokens para a resposta
            # - Resto para o conteúdo (~3.5 chars/token em média)
            MAX_CONTENT_TOKENS = 13885  # 16385 - 2000 - 500
            MAX_CONTENT_CHARS = int(MAX_CONTENT_TOKENS * 3.5)
            
            if len(content) > MAX_CONTENT_CHARS:
                self.logger.warning(f"Conteúdo muito longo ({len(content)} chars), truncando para {MAX_CONTENT_CHARS} chars (~{MAX_CONTENT_TOKENS} tokens)")
                content = content[:MAX_CONTENT_CHARS] + "..."
            
            # Verificar se é JSON válido
            if content.strip().startswith('{') and content.strip().endswith('}'):
                try:
                    json.loads(content)
                except json.JSONDecodeError:
                    self.logger.error("Conteúdo JSON inválido")
                    return []
            
            # Verificar se é XML válido
            elif content.strip().startswith('<?xml') or content.strip().startswith('<'):
                try:
                    ET.fromstring(content)
                except ET.ParseError:
                    # Tentar com BeautifulSoup se falhar como XML puro
                    try:
                        BeautifulSoup(content, 'xml')
                    except Exception:
                        self.logger.error("Conteúdo XML/HTML inválido")
                        return []
            
            # Se não é JSON nem XML, verificar se tem uma quantidade mínima de texto
            elif len(re.sub(r'\s+', '', content)) < 100:
                self.logger.error("Conteúdo de texto muito curto ou inválido")
                return []
            
            # Criar estrutura JSON de exemplo
            json_structure = {
                "news": [
                    {
                        "title": "string (título limpo, sem HTML)",
                        "content": "string (conteúdo completo, sem HTML)",
                        "date": "string (formato ISO 8601)",
                        "url": "string (URL absoluta)"
                    }
                ]
            }
            
            # Criar prompt sem usar f-strings para evitar problemas de formatação
            prompt = '\n'.join([
                'Você é um assistente especializado em extrair e estruturar dados de notícias financeiras.',
                'Sua tarefa é analisar o conteúdo fornecido e retornar APENAS um JSON válido contendo as notícias extraídas.',
                '',
                'REGRAS IMPORTANTES:',
                '1. Retorne APENAS o JSON, sem texto adicional ou explicações',
                '2. O JSON deve seguir EXATAMENTE a estrutura especificada',
                '3. Todos os campos são obrigatórios para cada notícia',
                '4. Datas devem estar em formato ISO 8601 (YYYY-MM-DDTHH:MM:SS)',
                '5. Strings não devem conter caracteres de escape ou formatação HTML',
                '6. URLs devem ser absolutas e começar com http:// ou https://',
                '7. Não inclua comentários ou metadados no JSON',
                '8. Garanta que todas as strings estejam corretamente escapadas',
                '9. Use aspas duplas para todas as chaves e valores string',
                '10. Não use vírgula após o último elemento de arrays/objetos',
                '',
                'ESTRUTURA DO JSON:',
                json.dumps(json_structure, indent=2),
                '',
                'VALIDAÇÕES:',
                '- title: não vazio, sem HTML, máximo 500 caracteres',
                '- content: não vazio, sem HTML, mínimo 100 caracteres',
                '- date: formato ISO 8601 válido',
                '- url: URL absoluta válida começando com http(s)://',
                '',
                'CONTEÚDO PARA EXTRAIR:',
                content,
                '',
                'LEMBRE-SE: Retorne APENAS o JSON válido, nada mais.'
            ])

            completion = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Você é um extrator de dados especializado que retorna apenas JSON válido, sem texto adicional."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1000,  # Aumentado para garantir resposta completa
                top_p=0.1,
                response_format={ "type": "json_object" }
            )
            
            if not completion.choices or not completion.choices[0].message:
                raise ValueError("Resposta vazia do modelo")
            
            result_text = completion.choices[0].message.content
            if not isinstance(result_text, str):
                raise ValueError("Resposta do modelo não é uma string")
            
            # Garantir que temos apenas o JSON
            result_text = result_text.strip()
            if not (result_text.startswith('{') and result_text.endswith('}')):
                raise ValueError("Resposta não é um JSON válido")
            
            # Parsear JSON com tratamento de erro
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                # Tentar limpar caracteres problemáticos
                result_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', result_text)
                result = json.loads(result_text)
            
            if not isinstance(result, dict) or 'news' not in result or not isinstance(result['news'], list):
                raise ValueError("Formato de resposta inválido")
            
            # Validar e normalizar cada notícia
            news_list = []
            for news in result['news']:
                # Validar campos obrigatórios
                if not all(k in news for k in ['title', 'content', 'date']):
                    continue
                    
                # Validar e limpar título
                title = news['title'].strip()
                if not title or len(title) > 500:
                    continue
                    
                # Validar e limpar conteúdo
                content = news['content'].strip()
                if not content or len(content) < 100:
                    continue
                    
                # Validar data
                try:
                    datetime.fromisoformat(news['date'].replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    continue
                    
                # Validar URL se presente
                url = news.get('url', '')
                if url and not url.startswith(('http://', 'https://')):
                    continue
                
                # Adicionar fonte e incluir na lista
                news['source'] = source
                news_list.append(news)
            
            return news_list
            
        except Exception as e:
            _, _, tb = sys.exc_info()
            lineno = tb.tb_lineno if tb is not None else "unknown"
            self.logger.error(f"Erro na extração com LLM (linha {lineno}): {str(e)}")
            return []

    def _extract_news_from_html(self, soup: BeautifulSoup, url: str, cutoff_date: datetime) -> List[Dict]:
        """
        Extrai notícias de uma página HTML usando BeautifulSoup
        
        Args:
            soup: Objeto BeautifulSoup da página
            url: URL da página
            cutoff_date: Data de corte para notícias antigas
            
        Returns:
            List[Dict]: Lista de notícias extraídas
        """
        articles = []
        try:
            # Procurar por artigos/notícias
            for article in soup.find_all(['article', 'div'], class_=lambda x: x and ('news' in x.lower() or 'article' in x.lower())):
                try:
                    # Extrair título
                    title = article.find(['h1', 'h2', 'h3', 'h4'])
                    if not title:
                        continue
                    title = title.get_text(strip=True)

                    # Extrair conteúdo
                    content = article.find(['div', 'p'], class_=lambda x: x and ('content' in x.lower() or 'text' in x.lower()))
                    if not content:
                        content = article.find(['div', 'p'])
                    if not content:
                        continue
                    content = content.get_text(separator=' ', strip=True)

                    # Extrair data
                    date_elem = article.find(['time', 'span', 'div'], class_=lambda x: x and ('date' in x.lower() or 'time' in x.lower()))
                    if not date_elem:
                        continue

                    # Tentar diferentes formatos de data
                    date_text = date_elem.get('datetime') or date_elem.get_text(strip=True)
                    try:
                        news_date = pd.to_datetime(date_text).to_pydatetime()
                    except:
                        continue

                    if news_date >= cutoff_date and len(content) >= self.collection_config["min_article_length"]:
                        # Extrair link
                        link = article.find('a')
                        if link:
                            link = link.get('href', '')
                            if not link.startswith('http'):
                                parsed_url = urlparse(url)
                                link = f"{parsed_url.scheme}://{parsed_url.netloc}{link}"

                        news = {
                            'title': title,
                            'content': content,
                            'date': news_date.isoformat(),
                            'url': link,
                            'source': urlparse(url).netloc
                        }
                        articles.append(news)

                except Exception as e:
                    self.logger.debug(f"Erro ao processar artigo HTML: {str(e)}")
                    continue

        except Exception as e:
            self.logger.error(f"Erro ao extrair notícias do HTML: {str(e)}")

        return articles

def main():
    """Função principal para execução do analisador"""
    try:
        # Inicializar analisador
        analyzer = NewsAnalyzer("config/news_collector_config.json")
        
        # Analisar notícias
        news_df = analyzer.analyze_news(
            days_back=30,
            force_reload=True
        )
        
        # Salvar resultados
        output_path = f"data/processed/news_impact_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        news_df.to_csv(output_path, index=False)
        print(f"\nResultados salvos em: {output_path}")
        
    except Exception as e:
        _, _, tb = sys.exc_info()
        lineno = tb.tb_lineno if tb is not None else "unknown"
        logger.error(f"Erro na execução (linha {lineno}): {str(e)}")
        raise

if __name__ == "__main__":
    main() 