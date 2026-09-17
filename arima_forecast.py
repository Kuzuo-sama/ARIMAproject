"""Previsao de passageiros aereos com ARIMA.

Uso:
    python arima_forecast.py --data AirPassengers.csv

O CSV deve conter uma coluna de data e uma coluna numerica de passageiros.
Por defeito, sao procuradas colunas chamadas Month/Passengers ou Date/Passengers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from matplotlib.widgets import RadioButtons
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX


ALPHA = 0.05
TRAIN_RATIO = 0.80
P = 1
Q = 1
SEASONAL_ORDER = (1, 1, 1, 12)


def load_series(csv_path: Path) -> pd.Series:
    """Carrega o CSV, converte a data e devolve uma serie mensal indexada."""
    data = pd.read_csv(csv_path)
    if data.shape[1] < 2:
        raise ValueError("O CSV precisa de pelo menos uma coluna de data e uma numerica.")

    date_column = next(
        (name for name in ("Month", "Date", "date", "month") if name in data.columns),
        data.columns[0],
    )
    value_column = next(
        (
            name
            for name in ("Passengers", "passengers", "Passengers #")
            if name in data.columns
        ),
        data.columns[1],
    )

    data[date_column] = pd.to_datetime(data[date_column], errors="raise")
    data[value_column] = pd.to_numeric(data[value_column], errors="raise")
    series = data.set_index(date_column)[value_column].sort_index()
    series.index.name = "Date"
    series.name = "Passengers"
    return series.asfreq("MS")


def adf_report(series: pd.Series, label: str) -> float:
    """Imprime o resultado ADF e devolve o p-value."""
    result = adfuller(series.dropna(), autolag="AIC")
    statistic, p_value, used_lag, observations = result[:4]
    print(
        f"ADF ({label}): statistic={statistic:.4f}, p-value={p_value:.6f}, "
        f"lags={used_lag}, observations={observations}"
    )
    return p_value


def find_d(series: pd.Series) -> int:
    """Aplica diferenciacao ate rejeitar a hipotese de raiz unitaria."""
    adf_report(series, "original")
    differenced = series.copy()
    for d in range(1, 4):
        differenced = differenced.diff().dropna()
        p_value = adf_report(differenced, f"d={d}")
        if p_value < ALPHA:
            print(
                f"Conclusao: d={d}, porque p-value={p_value:.6f} < "
                f"{ALPHA} (serie estacionaria)."
            )
            return d
    raise RuntimeError("A serie nao ficou estacionaria ate d=3.")


def show_chart_selector(
    series: pd.Series,
    d: int,
    comparison: pd.DataFrame,
    future_forecast: pd.Series,
    future_sarima_forecast: pd.Series,
) -> None:
    """Mostra todos os graficos numa janela, escolhidos por um menu lateral."""
    decomposition = seasonal_decompose(series, model="additive", period=12)
    stationary_series = series.diff(d).dropna()
    figure = plt.figure(figsize=(10.5, 7), constrained_layout=False)
    selector_axis = figure.add_axes([0.03, 0.22, 0.19, 0.56])
    chart_axes = []
    chart_options = [
        "Serie original",
        "Decomposicao",
        "Diferenciacao",
        "PACF",
        "ACF",
        "Previsao do teste",
        "Previsoes futuras",
    ]
    radio = RadioButtons(selector_axis, chart_options, activecolor="#2563eb")
    selector_axis.set_title("Escolher grafico", pad=15)

    def clear_chart() -> None:
        for axis in chart_axes:
            figure.delaxes(axis)
        chart_axes.clear()

    def draw_chart(label: str) -> None:
        clear_chart()
        figure.suptitle("")
        if label == "Diferenciacao":
            differenced_series = [
                series,
                series.diff().dropna(),
                series.diff().diff().dropna(),
                series.diff().diff().diff().dropna(),
            ]
            chart_axes.extend(
                [
                    figure.add_axes([0.27 + index * 0.17, 0.18, 0.15, 0.64])
                    for index in range(4)
                ]
            )
            for index, (axis, differenced) in enumerate(
                zip(chart_axes, differenced_series)
            ):
                p_value = adfuller(differenced.dropna(), autolag="AIC")[1]
                differenced.plot(ax=axis, color="#1f4e79")
                title = "Original" if index == 0 else f"{index}a diferenca"
                axis.set_title(
                    f"{title}\nADF p-value={p_value:.4f}",
                    fontsize=10,
                    pad=8,
                )
                axis.set_xlabel("Data")
                axis.set_ylabel("Passageiros" if index == 0 else "")
                axis.margins(x=0.03)
            figure.suptitle(
                f"Comparacao da diferenciacao (d escolhido = {d})", y=0.93
            )
        elif label == "Decomposicao":
            chart_axes.extend(
                [figure.add_axes([0.28, 0.75 - index * 0.18, 0.66, 0.13]) for index in range(4)]
            )
            components = [
                (decomposition.observed, "Observada"),
                (decomposition.trend, "Trend"),
                (decomposition.seasonal, "Seasonal"),
                (decomposition.resid, "Residual"),
            ]
            for index, (axis, (component, title)) in enumerate(zip(chart_axes, components)):
                component.plot(ax=axis, color="#1f4e79")
                axis.set_title(title, fontsize=10, pad=2)
                axis.set_ylabel("")
                axis.margins(x=0.02)
                if index < len(chart_axes) - 1:
                    axis.tick_params(axis="x", labelbottom=False)
            chart_axes[-1].set_xlabel("Data")
            figure.suptitle("Decomposicao aditiva", y=0.95)
        else:
            axis = figure.add_axes([0.27, 0.12, 0.68, 0.76])
            chart_axes.append(axis)
            if label == "Serie original":
                series.plot(ax=axis, title="Serie temporal original", color="#1f4e79")
                axis.set_ylabel("Passageiros")
            elif label == "PACF":
                plot_pacf(stationary_series, lags=24, ax=axis, method="ywm")
                axis.set_title(f"PACF da serie diferenciada (d={d})")
            elif label == "ACF":
                plot_acf(stationary_series, lags=24, ax=axis)
                axis.set_title(f"ACF da serie diferenciada (d={d})")
            elif label == "Previsao do teste":
                series.plot(ax=axis, label="Dados observados", color="#1f4e79")
                comparison["Previsao"].plot(
                    ax=axis, label="Previsao no teste", color="#e67e22"
                )
                comparison["Previsao SARIMA"].plot(
                    ax=axis, label="Previsao SARIMA no teste", color="#2ca02c"
                )
                axis.set_title("Previsao no periodo de teste")
                axis.legend()
                axis.set_ylabel("Passageiros")
            else:
                series.plot(ax=axis, label="Dados observados", color="#1f4e79")
                future_forecast.plot(
                    ax=axis, label="Previsao futura ARIMA", color="#e67e22"
                )
                future_sarima_forecast.plot(
                    ax=axis, label="Previsao futura SARIMA", color="#2ca02c"
                )
                axis.set_title("Previsoes futuras: ARIMA vs. SARIMA")
                axis.legend()
                axis.set_ylabel("Passageiros")
            axis.set_xlabel("Data")
            axis.margins(x=0.02)
        figure.canvas.draw_idle()

    radio.on_clicked(draw_chart)
    draw_chart(chart_options[0])
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("AirPassengers.csv"))
    args = parser.parse_args()

    series = load_series(args.data)
    print(f"Observacoes: {len(series)} | periodo: {series.index.min().date()} a {series.index.max().date()}")
    print(f"Nulos apos frequencia mensal: {series.isna().sum()}")
    if series.isna().any():
        raise ValueError("Existem meses sem observacao; complete o CSV antes de continuar.")

    d = find_d(series)
    print(
        f"Escolha dos parametros: p={P} e q={Q}. "
        "Nos graficos, o primeiro lag significativo da PACF sustenta p=1 "
        "e o primeiro lag significativo da ACF sustenta q=1."
    )

    split_index = int(len(series) * TRAIN_RATIO)
    training_data = series.iloc[:split_index]
    test_data = series.iloc[split_index:]
    model = ARIMA(training_data, order=(P, d, Q))
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=len(test_data))
    mse = mean_squared_error(test_data, forecast)

    sarima_model = SARIMAX(
        training_data,
        order=(P, d, Q),
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    sarima_forecast = sarima_model.forecast(steps=len(test_data))
    sarima_mse = mean_squared_error(test_data, sarima_forecast)

    comparison = pd.DataFrame(
        {"Real": test_data, "Previsao": forecast, "Previsao SARIMA": sarima_forecast}
    )
    future_model = ARIMA(series, order=(P, d, Q)).fit()
    future_forecast = future_model.forecast(steps=12)
    future_sarima_model = SARIMAX(
        series,
        order=(P, d, Q),
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    future_sarima_forecast = future_sarima_model.forecast(steps=12)
    show_chart_selector(
        series, d, comparison, future_forecast, future_sarima_forecast
    )

    print(f"Treino: {len(training_data)} observacoes | Teste: {len(test_data)} observacoes")
    print(f"Modelo: ARIMA({P}, {d}, {Q})")
    print(f"MSE no teste: {mse:.4f}")
    print(f"Modelo: SARIMA({P}, {d}, {Q})x{SEASONAL_ORDER}")
    print(f"MSE SARIMA no teste: {sarima_mse:.4f}")
    print(comparison.to_string())


if __name__ == "__main__":
    main()