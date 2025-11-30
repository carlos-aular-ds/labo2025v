## Experimentos Finales

Despues de la entrega de los experimentos colaborativos no tuve mucho tiempo para hacer experimentos (modalidad sr), Tambien estuve analizando los experimentos de mis conpaneros y la mayoria de las recomendaciones era mantener el workflow
origina, por lo tanto segui mi instinto y quise expander mas una hipotesis que me habia quedado de los colaborativo que es una estrategia de pesos, pero dandole menor pesos a la los meses de pandemia y maximar los ultimos meses,
que fueron 3 de los 4 experimentos que hice, el experimento numero (3) quise cambiar el parametro `boosting` a que usara `dart` por que habia leido en los exp. colaborativos que podria tener una mejor ganancia pero al costo del
poder de computo necesario, por lo tanto este experimentos despues de 30h+ aun no habia terminado la O. bayesiana y decidi de cancelarlo. 

El experimento numero 4 fue el que genero el submit final que seleccione, este experimento tiene una estrategia por pesos implementada -->
```
calcular_pesos_covid <- function(dataset) {
  # Defino los períodos de COVID (marzo 2020 - diciembre 2021)
  inicio_covid <- 202003
  fin_covid <- 202112
  
  # Inicializo los pesos con valor base 1.0
  dataset[, peso := 1.0]
  
  # Reduzco el peso durante los meses de COVID (más inciertos/atípicos)
  # Los meses de COVID tendrán menor peso (0.3 a 0.5)
  dataset[foto_mes >= inicio_covid & foto_mes <= fin_covid, 
          peso := 0.3 + 0.2 * ((foto_mes - inicio_covid) / (fin_covid - inicio_covid))]
  
  # Aumento gradualmente el peso para meses pre-COVID (datos más estables)
  dataset[foto_mes < inicio_covid, 
          peso := 1.0 + 0.5 * ((foto_mes - 201901) / (inicio_covid - 201901))]
  
  # Aumento el peso para meses post-COVID (datos más recientes y relevantes)
  dataset[foto_mes > fin_covid, 
          peso := 1.5 + 1.0 * ((foto_mes - fin_covid) / (202109 - fin_covid))]
  
  # Normalizo los pesos para que el promedio sea 1
  peso_medio <- mean(dataset$peso)
  dataset[, peso_norm := peso / peso_medio]
  
  return(dataset)
}
```

Tambien quise cambiar los cortes que generaba antes del submit de kaggle para poder entender mejor la ganancia que generaba este scritp (exp-4)

El submit elegido fue "KAfinal-pesos-covid-4_s40_11250.csv" que tiene el corte en 11250

### Instrucciones para correr

El script (sr-final-pesos-covid-4.ipynb) no amerita modificacion alguna, tiene un tiempo de corrida en gcp con una maquina de 8 de CPU y 128gb ram de aprox 30h
