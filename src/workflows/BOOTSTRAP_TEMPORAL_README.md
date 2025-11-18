# Bootstrap Temporal - Documentación

## Resumen de Cambios

Se ha implementado **Bootstrap Temporal** en el notebook `950_WorkFlow_01_senior_bootstrap.ipynb` para mejorar el ensemble de modelos LightGBM mediante muestreo con reemplazo de los datos más recientes.

---

## ¿Qué es el Bootstrap Temporal?

El bootstrap temporal es una técnica de ensemble que crea múltiples conjuntos de entrenamiento mediante **muestreo con reemplazo** de registros en períodos de tiempo específicos. A diferencia del bootstrap tradicional que muestrea todos los datos, esta implementación aplica bootstrap **solo a los meses más recientes**, manteniendo los datos históricos completos.

---

## ¿Cómo Funciona?

### Antes (Solo Bootstrap de Semillas)

```
┌─────────────────────────────────────────┐
│  Dataset Completo (2019-01 a 2021-07)   │
│           [Mismo para todos]            │
└─────────────────────────────────────────┘
           ↓         ↓         ↓
    [Semilla 1] [Semilla 2] ... [Semilla 20]
           ↓         ↓         ↓
    [Modelo 1]  [Modelo 2] ... [Modelo 20]
```

- Los 20 modelos usan exactamente los mismos datos
- Solo difieren en la semilla aleatoria del algoritmo
- Diversidad limitada al azar algorítmico

### Después (Bootstrap de Semillas + Bootstrap Temporal)

```
Dataset se divide en 2 grupos:

┌──────────────────────────────┐  ┌─────────────────────┐
│  Datos Antiguos (2019-2021)  │  │  Datos Recientes    │
│     [COMPLETOS - Sin BS]     │  │ (últimos 4 meses)   │
│                              │  │  [CON BOOTSTRAP]    │
└──────────────────────────────┘  └─────────────────────┘
                                          ↓
                              Muestreo con reemplazo
                                          ↓
    ┌────────────────┬────────────────┬─────────────────┐
    │   Muestra 1    │   Muestra 2    │  ... Muestra 20 │
    └────────────────┴────────────────┴─────────────────┘

Para cada modelo:
Datos Antiguos (completos) + Muestra Bootstrap (recientes) + Semilla única
                    ↓
            [Modelo con doble diversidad]
```

- Cada uno de los 20 modelos recibe:
  - **Datos antiguos completos** (2019-01 a 2021-03)
  - **Muestra bootstrap diferente** de los últimos 4 meses
  - **Semilla aleatoria diferente**
- Mayor diversidad = mejor ensemble

---

## Cambios Implementados

### 1. Nueva Sección: Configuración de Bootstrap Temporal

**Ubicación**: Después de la celda de configuración de semillas (`PARAM$FT$semillas`)

**Celda 1 - Markdown**: Título y descripción

```markdown
### Bootstrap Temporal

Configuramos el bootstrap para hacer muestreo con reemplazo sobre los últimos meses...
```

**Celda 2 - Configuración**:

```r
# Opciones: "ultimos_4_meses" o "ultimo_anio"
PARAM$FT$bootstrap_temporal <- "ultimos_4_meses"

if (PARAM$FT$bootstrap_temporal == "ultimos_4_meses") {
  PARAM$FT$meses_bootstrap <- c(202104, 202105, 202106, 202107)
} else if (PARAM$FT$bootstrap_temporal == "ultimo_anio") {
  PARAM$FT$meses_bootstrap <- c(
    202007, 202008, 202009, 202010, 202011, 202012,
    202101, 202102, 202103, 202104, 202105, 202106, 202107
  )
}

PARAM$FT$bootstrap_sample_ratio <- 1.0
```

**Parámetros**:

- `PARAM$FT$bootstrap_temporal`: Modo de bootstrap (`"ultimos_4_meses"` o `"ultimo_anio"`)
- `PARAM$FT$meses_bootstrap`: Vector con los meses que se muestrearán con reemplazo
- `PARAM$FT$bootstrap_sample_ratio`: Proporción de muestreo (1.0 = mismo tamaño original)

---

### 2. Loop de Entrenamiento Modificado

**Ubicación**: Celda con `for( sem in PARAM$FT$semillas)`

**Cambio Principal**: Dentro del loop, antes de `lgb.train()`:

```r
# ========== BOOTSTRAP TEMPORAL ==========
# Separar datos en dos grupos
indices_no_bootstrap <- which(!(dataset$foto_mes %in% PARAM$FT$meses_bootstrap) &
                               dataset$fold_final_train == TRUE)
indices_bootstrap <- which(dataset$foto_mes %in% PARAM$FT$meses_bootstrap &
                           dataset$fold_final_train == TRUE)

# Muestreo CON reemplazo de los meses de bootstrap
n_bootstrap <- length(indices_bootstrap)
n_sample <- floor(n_bootstrap * PARAM$FT$bootstrap_sample_ratio)
indices_bootstrap_sample <- sample(indices_bootstrap, size = n_sample, replace = TRUE)

# Combinar: todos los registros no-bootstrap + muestra bootstrap
indices_train_bootstrap <- c(indices_no_bootstrap, indices_bootstrap_sample)

# Crear dataset de LightGBM con la muestra bootstrap
dtrain_bootstrap <- lgb.Dataset(
  data = data.matrix(dataset[indices_train_bootstrap, campos_buenos, with = FALSE]),
  label = dataset[indices_train_bootstrap, clase01],
  free_raw_data = FALSE
)

# Entrenar modelo con bootstrap
final_model <- lgb.train(
  data = dtrain_bootstrap,
  param = param_final,
  verbose = -100
)
# ========================================
```

**Qué hace**:

1. Identifica registros que NO están en los meses de bootstrap → se usan completos
2. Identifica registros que SÍ están en los meses de bootstrap → se muestrean
3. Hace muestreo con reemplazo (algunos registros pueden aparecer varias veces, otros ninguna)
4. Combina ambos grupos para crear el dataset de entrenamiento
5. Entrena el modelo con este dataset híbrido

---

### 3. Documentación Adicional

**Celda Markdown - Explicación**:
Describe las ventajas del bootstrap híbrido:

- Mayor diversidad en el ensemble
- Reducción de overfitting
- Mejor estimación de incertidumbre
- Datos antiguos se mantienen completos

**Celda de Visualización (Opcional)**:

```r
# Muestra la distribución de meses y conteo de registros
cat("\n=== Distribución de datos de entrenamiento ===\n")
tabla_meses <- dataset[fold_final_train == TRUE, .N, by = foto_mes][order(foto_mes)]
print(tabla_meses)
```

---

## Cómo Usar

### Opción 1: Bootstrap en Últimos 4 Meses (Default)

```r
PARAM$FT$bootstrap_temporal <- "ultimos_4_meses"
```

**Meses con bootstrap**: 202104, 202105, 202106, 202107
**Meses sin bootstrap**: 201901-202103 (27 meses completos)

**Cuándo usar**:

- Cuando los últimos meses son más relevantes para predicción
- Dataset grande donde bootstrap total sería computacionalmente costoso
- Quieres reducir overfitting en los datos más recientes

### Opción 2: Bootstrap en Último Año

```r
PARAM$FT$bootstrap_temporal <- "ultimo_anio"
```

**Meses con bootstrap**: 202007-202107 (13 meses)
**Meses sin bootstrap**: 201901-202006 (18 meses completos)

**Cuándo usar**:

- Mayor diversidad en el ensemble
- Los datos del último año tienen mayor variabilidad
- Quieres más robustez en las predicciones

### Ajustar Proporción de Muestreo

```r
# Reducir tamaño de muestra bootstrap (más rápido, menos datos)
PARAM$FT$bootstrap_sample_ratio <- 0.8  # 80% del tamaño original

# Aumentar tamaño de muestra bootstrap (más datos, más variabilidad)
PARAM$FT$bootstrap_sample_ratio <- 1.2  # 120% del tamaño original
```

---

## Ventajas del Bootstrap Temporal

### 1. **Doble Diversidad**

- **Diversidad algorítmica**: 20 semillas aleatorias diferentes
- **Diversidad de datos**: 20 muestras bootstrap diferentes

### 2. **Reducción de Overfitting**

- El muestreo con reemplazo previene que el modelo memorice casos específicos
- Algunos registros aparecen múltiples veces, otros no aparecen
- El modelo aprende patrones más generales

### 3. **Mejor Estimación de Incertidumbre**

- Los 20 modelos tienen predicciones ligeramente diferentes
- El promedio es más robusto
- La varianza entre modelos indica incertidumbre

### 4. **Conservación de Datos Históricos**

- Los datos antiguos (2019-2021) se mantienen completos
- Solo se hace bootstrap en meses recientes y relevantes
- No se pierde información histórica importante

### 5. **Flexibilidad**

- Fácil cambiar entre 4 meses o 1 año
- Ajustable el ratio de muestreo
- Se puede extender a otros períodos

---

## Ejemplo de Ejecución

```r
# Configuración
PARAM$FT$bootstrap_temporal <- "ultimos_4_meses"
PARAM$FT$bootstrap_sample_ratio <- 1.0
PARAM$FT$semillerio <- 20

# Durante el entrenamiento se imprime:
# Entrenando modelo con semilla: 123456
#   - Registros fuera de bootstrap: 450000
#   - Registros originales en meses bootstrap: 50000
#   - Registros muestreados con reemplazo: 50000
#   - Total registros para entrenamiento: 500000
#
# Entrenando modelo con semilla: 789012
#   - Registros fuera de bootstrap: 450000
#   - Registros originales en meses bootstrap: 50000
#   - Registros muestreados con reemplazo: 50000  <- Muestra diferente!
#   - Total registros para entrenamiento: 500000
# ...
```

---

## Impacto en el Pipeline

### Etapas NO Modificadas

- Preprocesamiento (9.7.1)
- Feature Engineering
- Hiperparameter Tuning
- Scoring y predicción final

### Etapas Modificadas

- **Solo el entrenamiento final (9.7.3.1)**
- Los modelos se entrenan con muestras bootstrap
- El resto del pipeline permanece igual

### Compatibilidad

- Los modelos entrenados se guardan igual (`modelo_<semilla>.txt`)
- La predicción final funciona igual (promedia las 20 predicciones)
- No requiere cambios en código posterior

---

## Comparación: Antes vs Después

| Aspecto                     | Antes                | Después                  |
| --------------------------- | -------------------- | ------------------------ |
| **Número de modelos**       | 20                   | 20                       |
| **Semillas diferentes**     | ✅ Sí                | ✅ Sí                    |
| **Datos de entrenamiento**  | Idénticos para todos | Diferentes por bootstrap |
| **Datos antiguos**          | Completos            | Completos (sin cambio)   |
| **Datos recientes**         | Completos            | Bootstrap con reemplazo  |
| **Diversidad del ensemble** | Media                | Alta                     |
| **Tiempo de entrenamiento** | X                    | ~X (similar)             |
| **Robustez**                | Buena                | Mejor                    |

---

## Preguntas Frecuentes

### ¿Por qué solo hacer bootstrap en los últimos meses?

Los datos más recientes suelen ser:

- Más relevantes para predecir el futuro
- Más propensos a overfitting (menos historia)
- Más volátiles y con mayor variabilidad

Hacer bootstrap solo aquí maximiza el beneficio mientras conserva la información histórica completa.

### ¿Puedo desactivar el bootstrap temporal?

Sí, simplemente comenta o elimina las celdas nuevas y restaura el código original del loop de entrenamiento. O crea un parámetro condicional:

```r
if (exists("PARAM$FT$bootstrap_temporal") && PARAM$FT$bootstrap_temporal != "ninguno") {
  # código bootstrap
} else {
  # código original
}
```

### ¿Afecta el tiempo de entrenamiento?

Mínimamente. El muestreo de índices es muy rápido. El tiempo principal sigue siendo el entrenamiento de LightGBM.

### ¿Puedo usar otros períodos?

Sí, modifica directamente `PARAM$FT$meses_bootstrap`:

```r
# Ejemplo: Solo bootstrap en los últimos 2 meses
PARAM$FT$meses_bootstrap <- c(202106, 202107)

# Ejemplo: Bootstrap en meses específicos
PARAM$FT$meses_bootstrap <- c(202001, 202002, 202003)  # Meses de pandemia
```

---

## Referencias Técnicas

### Bootstrap Aggregating (Bagging)

El bootstrap temporal es una variante de **bagging** (Bootstrap AGGregatING):

- Propuesto por Leo Breiman (1996)
- Reduce varianza sin aumentar sesgo
- Base de Random Forest y otros métodos ensemble

### Diferencia con Bootstrap Tradicional

- **Bootstrap tradicional**: Muestrea TODO el dataset con reemplazo
- **Bootstrap temporal**: Solo muestrea períodos específicos con reemplazo
- Ventaja: Preserva estructura temporal y reduce computational cost

### Por Qué Funciona

1. **Teorema del Límite Central**: Promediar muchos modelos reduce varianza
2. **Ley de Grandes Números**: Con más modelos, el promedio converge al valor esperado
3. **Diversidad**: Diferentes muestras → diferentes modelos → mejor ensemble

---

## Contacto y Soporte

Para preguntas o mejoras, revisar el notebook:

- `src/workflows/950_WorkFlow_01_senior_bootstrap.ipynb`

Secciones modificadas:

- Celda de configuración bootstrap temporal (después de semillas)
- Loop de entrenamiento final (celda con `for( sem in PARAM$FT$semillas)`)
- Celdas de documentación adicionales

---

**Última actualización**: 2025-11-18
**Versión del notebook**: 950_WorkFlow_01_senior_bootstrap
**Técnica implementada**: Bootstrap Temporal Híbrido (Semillas + Datos)
