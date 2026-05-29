# Modelo de Heston — Red Neuronal Surrogate para Valoración de Opciones

> **Trabajo de Fin de Grado**
> Valoración de opciones call europeas mediante una red neuronal entrenada sobre el modelo de Heston (1993).

---

## Descripción

Este repositorio contiene el desarrollo progresivo de un **pricer ** basado en Deep Learning para el modelo de volatilidad estocástica de Heston. El objetivo es aproximar el precio de una opción call europea con la misma precisión que la solución analítica exacta, pero órdenes de magnitud más rápido en inferencia.

El pipeline completo abarca:

1. Implementación de la **función característica de Heston** con corrección de rama logarítmica (Lord-Kahl 2010).
2. Valoración exacta por **fórmula cerrada de Heston (1993)** vía integración numérica.
3. Valoración eficiente por **Carr-Madan FFT** con pesos de Simpson.
4. Generación de un **dataset sintético** de hasta 400 000 puntos de precios sobre superficies (Strike × Madurez).
5. Entrenamiento y evaluación de una **red neuronal MLP** como sustituto  del pricer analítico.
6. **Comparativa de velocidad**: la red neuronal compilada con `tf.function` logra una aceleración de varios órdenes de magnitud frente a la integración numérica exacta.

---

## Estructura del repositorio

| Archivo | Dataset | Arquitectura | Descripción |
|---|---|---|---|
| `TFG_V1.ipynb` | 100 000 filas | 4 × 256, ELU + BN + Dropout 0.1 | Prototipo inicial |
| `TFG_V2.ipynb` | 400 000 filas | 4 × 256, ELU + BN + Dropout 0.05 | Escala datos, añade corrección de sesgo e inferencia con `tf.function` |
| `TFG_V3_Final.ipynb` | 400 000 filas | **4 × 256, ELU + BN + Dropout 0.05** |  **Versión ganadora.** Corrección del algoritmo Carr-Madan. Pipeline definitivo con split por paramset |
| `TFG_V4_Keras.ipynb` | 400 000 filas | 1 × 512, ReLU (Bayesian Opt.) | Exploración de arquitectura optimizada.|

---

## Modelo de Heston

El modelo describe la dinámica del activo subyacente como:

$$dS_t = r S_t \, dt + \sqrt{v_t} \, S_t \, dW_t^S$$
$$dv_t = \kappa(\theta - v_t) \, dt + \sigma \sqrt{v_t} \, dW_t^v$$

con $\langle dW^S, dW^v \rangle = \rho \, dt$.

Solo se generan parámetros que cumplen la **condición de Feller** (2κθ > σ²), garantizando que la varianza no toque cero. La tasa de aceptación en el muestreo aleatorio es aproximadamente el 84 %.

### Parámetros del modelo

| Parámetro | Símbolo | Rango utilizado |
|---|---|---|
| Velocidad de reversión | κ (kappa) | [0.5, 5.0] |
| Varianza a largo plazo | θ (theta) | [0.01, 0.15] |
| Vol. de la vol. | σ (sigma) | [0.01, 0.6] |
| Correlación | ρ (rho) | [−0.9, −0.1] |
| Varianza inicial | v₀ | [0.01, 0.2] |

### Verificación numérica

Para los parámetros de referencia `[κ=2, θ=0.04, σ=0.3, ρ=−0.7, v₀=0.04]`, `T=0.5`, `K=100`:

| Método | Precio (€) |
|---|---|
| Heston 1993 (integración directa) | 5.446870 |
| Carr-Madan FFT | 5.448949 |
| Diferencia | 2.08 × 10⁻³ |

---

## Dataset sintético

La malla de valoración cubre **13 strikes × 4 maturities = 52 puntos por superficie**:

| Dimensión | Valores |
|---|---|
| Strikes (% S₀) | 80 %, 85 %, 90 %, 95 %, 97 %, 99 %, 100 %, 101 %, 102 %, 105 %, 110 %, 115 %, 120 % |
| Maturities (años) | 0.1, 0.25, 0.5, 1.0 |
| Superficies generadas | 7 693 |
| Filas totales | 400 036 |

Cada fila incluye los 5 parámetros de Heston, `T`, `K`, el precio limpio (`price`) y una versión con ruido gaussiano del 1 % (`price_noisy`).

---

## Red neuronal surrogate

### Entradas (7 features)

| Feature | Transformación | Justificación |
|---|---|---|
| Moneyness | log(K / S₀) | Simetría log-normal |
| Madurez | √T | Escala de la volatilidad implícita |
| κ | sin transformar | Ya en escala razonable |
| θ | log(·) | Positivo y con cola larga |
| σ | sin transformar | Ya en escala razonable |
| ρ | sin transformar | Acotado en [−1, 1] |
| v₀ | log(·) | Positivo y con cola larga |

Todas las entradas se estandarizan con `StandardScaler` ajustado únicamente sobre el conjunto de entrenamiento.

### Arquitecturas comparadas

| Versión | Arquitectura | lr | Dataset |
|---|---|---|---|
| **V3 (ganadora)** | **4 capas × 256 neuronas, ELU, BN, Dropout 0.05** | **0.001** | **400k filas** |
| V4 (exploración) | 1 capa × 512 neuronas, ReLU (Bayesian Opt.) | 0.00132 | 400k filas |

### Entrenamiento

| Hiperparámetro | Valor |
|---|---|
| Optimizador | Adam |
| Función de pérdida | MSE sobre price / S₀ |
| Batch size | 512 |
| Epochs máx. | 300 |
| EarlyStopping patience | 30 |
| ReduceLROnPlateau patience | 10 |
| Split train / val / test | 70 % / 15 % / 15 % **por paramset** |

> **Nota sobre el split**: la división se realiza por conjunto de parámetros, no por fila individual, para evitar que superficies del mismo paramset aparezcan a la vez en train y test (data leakage).

### Corrección de sesgo

Tras el entrenamiento se estima el sesgo sistemático del modelo sobre el conjunto de validación y se resta en inferencia:

```python
sesgo = float(np.mean(y_pred_val - y_val))
y_pred_test = model(X_test) - sesgo
```

---

## Resultados principales

### Métricas sobre test (arquitectura ganadora — V3: 4 × 256 ELU)

| Métrica | Valor normalizado | Valor en € |
|---|---|---|
| R² | > 0.99 | — |
| MAE | — | completar con output |
| RMSE | — | completar con output |

> Ejecuta `TFG_V3_Final.ipynb` hasta el **PASO 5** para obtener los valores exactos de tu ejecución.

### Velocidad de inferencia

| Método | Tiempo medio (ms) |
|---|---|
| Heston exacto (integración numérica) | completar con output |
| Red neuronal (`tf.function`) | < 0.5 ms |

---

## Quickstart — Inferencia con el modelo guardado

```python
import joblib
import tensorflow as tf
import numpy as np

# Cargar artefactos
model  = tf.keras.models.load_model('modelo_heston_final.keras')
scaler = joblib.load('escalador_heston_nn.pkl')
sesgo  = joblib.load('sesgo_heston_nn.pkl')

S0 = 100.0

def predecir_precio(kappa, theta, sigma, rho, v0, T, K):
    x_raw = np.array([[
        np.log(K / S0),
        np.sqrt(max(T, 1e-12)),
        kappa,
        np.log(max(theta, 1e-8)),
        sigma,
        rho,
        np.log(max(v0, 1e-8))
    ]], dtype=np.float32)
    x_scaled = scaler.transform(x_raw).astype(np.float32)
    pred_norm = float(model(x_scaled, training=False).numpy()[0, 0])
    return (pred_norm - sesgo) * S0

# Ejemplo: call ATM con parámetros de referencia
precio = predecir_precio(kappa=2.0, theta=0.04, sigma=0.3,
                         rho=-0.7, v0=0.04, T=0.5, K=100.0)
print(f"Precio estimado: {precio:.4f} €")
# → Precio estimado: ~5.45 €
```

---

## Archivos generados en ejecución

| Archivo | Contenido |
|---|---|
| `dataset_heston_400k.csv` | Dataset sintético: kappa, theta, sigma, rho, v0, T, K, price, price_noisy |
| `mejor_modelo_heston.keras` | Pesos del mejor epoch según val_loss |
| `modelo_heston_final.keras` | Modelo final guardado tras entrenamiento |
| `escalador_heston_nn.pkl` | `StandardScaler` ajustado sobre train |
| `sesgo_heston_nn.pkl` | Corrección de sesgo estimada en validación |
| `superficie_heston.png` | Visualización 3D de una superficie de precios de ejemplo |
| `evaluacion_red_heston.png` | Curva de aprendizaje, scatter real vs predicho, histograma de errores |

---

## Requisitos

```bash
pip install numpy scipy pandas matplotlib scikit-learn tensorflow joblib
```

| Librería | Versión mínima recomendada |
|---|---|
| Python | 3.10 |
| TensorFlow | 2.13 |
| scikit-learn | 1.3 |
| NumPy | 1.24 |
| SciPy | 1.11 |

---

## Referencias

- Heston, S. L. (1993). *A closed-form solution for options with stochastic volatility with applications to bond and currency options*. The Review of Financial Studies, 6(2), 327–343.
- Carr, P. & Madan, D. (1999). *Option valuation using the fast Fourier transform*. Journal of Computational Finance, 2(4), 61–73.
- Lord, R. & Kahl, C. (2010). *Complex logarithms in Heston-like models*. Mathematical Finance, 20(4), 671–694.
