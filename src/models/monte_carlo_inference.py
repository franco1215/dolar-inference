import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonteCarloInference:
    """
    Inferência do dólar usando simulação de Monte Carlo
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Inicializa o modelo de Monte Carlo
        
        Args:
            data: DataFrame com dados históricos
        """
        self.data = data
        self.target_col = 'DOLAR'
        self.date_col = 'Date'
        self.feature_cols: List[str] = []
        self.scaler = RobustScaler()
        
        # Parâmetros da simulação
        self.n_simulations = 10000  # Número de simulações
        self.confidence_level = 0.95  # Nível de confiança
        
        # Modelo base para tendência
        self.trend_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    
    def load_data(self) -> None:
        """
        Carrega dados mais recentes com todas as features disponíveis
        """
        try:
            features_dir = Path("data/processed/features")
            if not features_dir.exists():
                raise FileNotFoundError(f"Diretório não encontrado: {features_dir}")
            
            data_files = [f for f in features_dir.glob("selected_features_data_*.csv")]
            if not data_files:
                raise FileNotFoundError("Nenhum arquivo de features encontrado")
            
            latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Carregando dados de: {latest_file}")
            
            self.data = pd.read_csv(latest_file)
            if self.data.empty:
                raise ValueError("Arquivo de dados vazio")
            
            # Validar colunas necessárias
            self._validate_columns()
            
            # Converter data
            self.data[self.date_col] = pd.to_datetime(self.data[self.date_col])
            
            # Ordenar por data
            self.data = self.data.sort_values(self.date_col)
            
            # Identificar features
            self.feature_cols = [col for col in self.data.columns 
                               if col not in [self.date_col, self.target_col]]
            
            logger.info(f"Dados carregados: {len(self.data)} registros, {len(self.feature_cols)} features")
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {str(e)}")
            raise
    
    def _validate_columns(self) -> None:
        """
        Valida presença das colunas necessárias
        """
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("DataFrame não inicializado")
            
        required_columns = {
            'target': ['DOLAR', 'dolar', 'USD'],
            'date': ['Date', 'DATA', 'date']
        }
        
        # Verificar coluna alvo
        target_found = False
        for possible_name in required_columns['target']:
            if possible_name in self.data.columns:
                if possible_name != self.target_col:
                    logger.info(f"Renomeando coluna alvo de '{possible_name}' para '{self.target_col}'")
                    self.data = self.data.rename(columns={possible_name: self.target_col})
                target_found = True
                break
        
        if not target_found:
            raise ValueError(f"Coluna alvo não encontrada. Esperado uma das: {required_columns['target']}")
        
        # Verificar coluna de data
        date_found = False
        for possible_name in required_columns['date']:
            if possible_name in self.data.columns:
                if possible_name != self.date_col:
                    logger.info(f"Renomeando coluna de data de '{possible_name}' para '{self.date_col}'")
                    self.data = self.data.rename(columns={possible_name: self.date_col})
                date_found = True
                break
        
        if not date_found:
            raise ValueError(f"Coluna de data não encontrada. Esperado uma das: {required_columns['date']}")
            
        logger.info("Validação de colunas concluída com sucesso")
    
    def prepare_data(self) -> None:
        """
        Prepara dados para simulação
        """
        try:
            if not isinstance(self.data, pd.DataFrame) or self.data.empty:
                raise ValueError("Dados não disponíveis")
            
            logger.info("Preparando dados para simulação...")
            
            # Remover valores ausentes
            self.data = self.data.dropna()
            
            # Calcular retornos diários
            self.data['returns'] = self.data[self.target_col].pct_change()
            
            # Calcular volatilidade histórica (21 dias)
            self.data['volatility'] = self.data['returns'].rolling(window=21).std()
            
            # Calcular tendência de curto prazo (5 dias)
            self.data['trend_5d'] = self.data[self.target_col].pct_change(5)
            
            # Calcular tendência de médio prazo (21 dias)
            self.data['trend_21d'] = self.data[self.target_col].pct_change(21)
            
            # Remover linhas com NaN após cálculos
            self.data = self.data.dropna()
            
            logger.info("Dados preparados com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao preparar dados: {str(e)}")
            raise
    
    def simulate_next_day(self) -> Dict:
        """
        Simula preço do dólar para o próximo dia usando Monte Carlo
        """
        try:
            if not isinstance(self.data, pd.DataFrame) or self.data.empty:
                raise ValueError("Dados não disponíveis")
            
            if len(self.data) < 21:
                raise ValueError("Histórico insuficiente para simulação")
            
            logger.info(f"Iniciando {self.n_simulations} simulações...")
            
            # Preparar dados para modelo de tendência
            X = self.data[self.feature_cols]
            y = self.data[self.target_col]
            
            # Treinar modelo de tendência
            X_scaled = self.scaler.fit_transform(X)
            self.trend_model.fit(X_scaled, y)
            
            # Prever tendência base
            last_data = self.data.iloc[-1:]
            X_last = self.scaler.transform(last_data[self.feature_cols])
            base_prediction = float(self.trend_model.predict(X_last)[0])
            
            # Parâmetros para simulação
            current_price = float(self.data[self.target_col].iloc[-1])
            volatility = float(self.data['volatility'].iloc[-1])
            
            # Realizar simulações
            simulated_prices = []
            for _ in range(self.n_simulations):
                # Gerar retorno aleatório baseado na volatilidade
                random_return = np.random.normal(
                    loc=(base_prediction - current_price) / current_price,
                    scale=volatility
                )
                
                # Calcular preço simulado
                simulated_price = current_price * (1 + random_return)
                simulated_prices.append(simulated_price)
            
            # Calcular estatísticas das simulações
            simulated_prices = np.array(simulated_prices)
            mean_price = float(np.mean(simulated_prices))
            std_price = float(np.std(simulated_prices))
            
            # Calcular intervalos de confiança
            confidence_interval = np.percentile(
                simulated_prices,
                [(1 - self.confidence_level) * 100 / 2, (1 + self.confidence_level) * 100 / 2]
            )
            
            # Calcular probabilidades
            prob_up = float(np.mean(simulated_prices > current_price))
            
            next_day = self.data[self.date_col].max() + timedelta(days=1)
            
            result = {
                'date': next_day.strftime("%Y-%m-%d"),
                'current_price': current_price,
                'prediction': mean_price,
                'confidence_interval': {
                    'lower': float(confidence_interval[0]),
                    'upper': float(confidence_interval[1])
                },
                'volatility': volatility,
                'probability_up': prob_up,
                'simulation_stats': {
                    'mean': mean_price,
                    'std': std_price,
                    'n_simulations': self.n_simulations
                }
            }
            
            logger.info(f"Previsão para {result['date']}:")
            logger.info(f"Preço atual: {current_price:.4f}")
            logger.info(f"Previsão: {mean_price:.4f}")
            logger.info(f"Intervalo de confiança: [{result['confidence_interval']['lower']:.4f}, {result['confidence_interval']['upper']:.4f}]")
            logger.info(f"Probabilidade de alta: {prob_up:.1%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na simulação: {str(e)}")
            raise
    
    def save_results(self, result: Dict) -> None:
        """
        Salva resultados da simulação
        
        Args:
            result: Dicionário com resultados da simulação
        """
        try:
            output_dir = Path("data/processed/monte_carlo")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            
            # Salvar previsão
            prediction_df = pd.DataFrame([{
                'date': result['date'],
                'current_price': result['current_price'],
                'predicted_price': result['prediction'],
                'confidence_lower': result['confidence_interval']['lower'],
                'confidence_upper': result['confidence_interval']['upper'],
                'probability_up': result['probability_up']
            }])
            
            pred_path = output_dir / f"monte_carlo_prediction_{timestamp}.csv"
            prediction_df.to_csv(pred_path, index=False)
            
            # Salvar metadados
            metadata = {
                'timestamp': timestamp,
                'features_used': self.feature_cols,
                'n_simulations': self.n_simulations,
                'confidence_level': self.confidence_level,
                'simulation_results': result
            }
            
            metadata_path = output_dir / f"monte_carlo_metadata_{timestamp}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Resultados salvos em: {output_dir}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar resultados: {str(e)}")
            raise

def main():
    try:
        # Inicializar modelo
        inference = MonteCarloInference()
        
        # Carregar e preparar dados
        inference.load_data()
        inference.prepare_data()
        
        # Realizar simulação
        result = inference.simulate_next_day()
        
        # Salvar resultados
        inference.save_results(result)
        
    except Exception as e:
        logger.error(f"Erro na execução: {str(e)}")
        raise

if __name__ == "__main__":
    main() 