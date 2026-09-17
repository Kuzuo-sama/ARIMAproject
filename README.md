# Previsao de Series Temporais com ARIMA

Projeto baseado no dataset **Air Passengers** do Kaggle.

## Preparacao

1. Descarregue o CSV do Kaggle para a pasta do projeto e guarde-o como `AirPassengers.csv`.
2. Confirme que o CSV tem uma coluna de data e uma coluna numerica de passageiros. O formato habitual (`Month,Passengers`) e reconhecido automaticamente.
3. Instale as dependencias:

```bash
python -m pip install -r requirements.txt
```

## Execucao

```bash
python arima_forecast.py --data AirPassengers.csv
```

O programa:


### Justificacao de `p` e `q`

Os graficos sao feitos sobre a serie diferenciada estacionaria. A escolha inicial `p=1` usa o primeiro lag significativo da PACF, comportamento associado ao corte da componente autorregressiva. A escolha `q=1` usa o primeiro lag significativo da ACF, comportamento associado ao corte da componente de medias moveis. Assim, os parametros podem ser comparados visualmente com outras escolhas e o MSE pode ser usado como criterio adicional, sem substituir a leitura dos graficos.

Os p-values ADF e o MSE dependem da versao das bibliotecas e aparecem no terminal quando o script e executado; nao sao valores fixados antecipadamente.

### `d` e `D` no SARIMA

O grafico de diferenciacao mostra a serie original, a primeira diferenca e a segunda diferenca, com o p-value ADF de cada uma. O valor `d` usado pelo programa e o primeiro que apresenta `p-value < 0.05`. No SARIMA, `D` e diferente: representa a diferenciacao sazonal e esta definido como `D=1` no modelo `SARIMA(1,d,1)x(1,1,1,12)`.

## Executar com Docker e noVNC

Com o Docker Desktop iniciado:

```powershell
docker compose up --build
```

No compose agregado, abrir `http://127.0.0.1:8082/arima/` para ver a janela dos
graficos no navegador através do noVNC.