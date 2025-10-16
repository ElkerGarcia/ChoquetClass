# 🧠 Librería Fuzzy Choquet — Clasificación Difusa con Integral de Choquet y Combinación Aditiva

## 📘 Descripción general

Esta librería implementa un **sistema de clasificación fuzzy** basado en reglas del tipo *Michigan* y funciones de agregación inspiradas en la **Integral de Choquet** y la **Combinación Aditiva**.  
Permite generar reglas difusas, calcular grados de pertenencia, y optimizar parámetros mediante **algoritmos genéticos**, todo dentro de un marco flexible para experimentación y análisis.

---

## ⚙️ Características principales

- 🧩 **Definición automática de regiones difusas** (bajas, medias, altas).
- 🧩 **Definición manual de regiones difusas** (bajas, medias, altas).
- 🧠 **Generación de reglas fuzzy** a partir de datos de entrenamiento.
- 🧬 **Optimización genética** de:
  - Vector de parámetros \( q \) (para la integral de Choquet o combinación aditiva).
  - Reglas fuzzy mediante operadores de cruce y mutación.
- 📊 **Clasificación** de patrones mediante:
  - Integral de Choquet.
  - Combinación Aditiva.
  - Cópula tipo Choquet.
  -  Integral CF1F2. 
- 🧾 **Evaluación de desempeño** mediante tasa de clasificación (CR).

---

## 🏗️ Arquitectura del sistema
├── ChoquetClass/
│ ├── classifier.py # Clasificador principal
│ ├── rules_generator.py # Generador y mutación de reglas fuzzy
│ ├── algorithm_genetic.py # Optimización genética (Choquet / Aditiva)
│ ├── fuzzy_utils.py # Funciones auxiliares (membership, T-norms, etc.)
│ └── init.py
├── data/
│ └── iris.csv # Dataset de ejemplo
├── examples/
│ ├── demo_choquet.ipynb # Ejemplo de uso con la integral de Choquet
│ └── demo_additive.ipynb # Ejemplo de uso con la combinación aditiva
└── README.md



---

## 🧪 Ejemplo de uso

### 🔹 Clasificación con la Integral de Choquet

```python
from ChoquetClass import algorithm_genetic as cc

# Entrenamiento y prueba
P_best, q_best, historial = cc.algoritmo_genetico_CC_Choquet(
    train_data=train_Data,
    test_data=test_Data,
    col_clase="class",
    variables=variables,
    num_rules=20,
    num_iter=10,
    TQ=30
)

print("Mejor CR:", historial[-1][1])
print("Mejor q:", q_best)


from ChoquetClass import algorithm_genetic as cc

P_best, q_best, historial = cc.algoritmo_genetico_comb_aditive(
    train_data=train_Data,
    test_data=test_Data,
    col_clase="class",
    variables=variables,
    num_rules=20,
    num_iter=10,
    TQ=30
)

print("Mejor CR:", historial[-1][1])
print("Mejor q:", q_best)



Resultados esperados

Tasa de clasificación promedio superior a métodos clásicos de agregación lineal.

Reglas fuzzy interpretables y con pesos ajustados dinámicamente.




