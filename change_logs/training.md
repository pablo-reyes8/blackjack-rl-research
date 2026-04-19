# Guía de implementación del training de RL para Blackjack en PyTorch

## Objetivo

Dejar una especificación clara, práctica y coherente de **todo el sistema de entrenamiento**, de modo que se pueda implementar directamente sin ir “parchando” después.

La idea explícita es **no construir una versión bebé y luego rehacerla**, sino implementar desde el inicio una base sólida que soporte:

1. **Baseline Double DQN feedforward**
2. **DRQN / Recurrent Double DQN**
3. **Dueling Recurrent Double DQN**
4. **Dos modos de entrenamiento**
   - `fresh_shoe`
   - `unknown_progress`

La meta de este documento es servir como **tutorial técnico de implementación** del pipeline completo:
- recolección de experiencia,
- replay buffer,
- muestreo,
- forward,
- target TD,
- backward,
- target update,
- evaluación,
- checkpointing,
- métricas,
- buenas prácticas.

---

# 1. Filosofía general del training

En RL el modelo no aprende de un dataset fijo.  
Aprende de su propia interacción con el entorno.

Eso implica que el sistema completo de entrenamiento tiene que manejar bien tres problemas:

1. **datos correlacionados temporalmente**;
2. **distribución de datos cambiante**;
3. **targets no estacionarios**.

Por eso desde el inicio debemos diseñar un pipeline que incluya:

- **replay buffer**
- **target network**
- **exploración controlada**
- **evaluación separada**
- **checkpointing**
- **métricas de entrenamiento y de política**

No basta con definir bien la loss.

---

# 2. Qué vamos a implementar

Queremos un pipeline que soporte desde el primer momento tres tipos de backbone:

## 2.1 Feedforward Double DQN
Entrada:
- `state: [B, D]`

Salida:
- `q_values: [B, 6]`

Uso:
- baseline;
- debugging;
- benchmark.

---

## 2.2 Recurrent Double DQN
Entrada:
- `state: [B, T, D]`
- `padding_mask: [B, T]`

Salida:
- `q_values: [B, T, 6]`

Uso:
- múltiples rondas;
- memoria temporal;
- conteo implícito de cartas.

---

## 2.3 Dueling Recurrent Double DQN
Igual a la recurrente, pero el head produce:
- `V`
- `A`
- y luego combina ambas para formar `Q`

Uso:
- modelo fuerte final;
- mejor estructura de estimación del valor.

---

# 3. Dos modos de entrenamiento del entorno

El training debe soportar explícitamente dos regímenes distintos.

## 3.1 `fresh_shoe`
- la mesa inicia desde shoe nuevo;
- el agente observa desde el principio;
- la memoria puede construir el shoe desde cero.

### Qué permite aprender
- estrategia base;
- estrategia sensible a reglas;
- conteo implícito con historial completo.

---

## 3.2 `unknown_progress`
- el shoe puede venir avanzado;
- el agente conoce reglas y número de mazos;
- no conoce cuánto falta para reshuffle;
- solo puede aprender con lo observado desde que empezó la secuencia.

### Qué vuelve más difícil este modo
- historia truncada;
- estado parcialmente latente;
- la memoria recurrente es más importante.

---

# 4. Arquitectura del sistema de entrenamiento

El sistema completo debe quedar organizado en módulos separados.

## 4.1 Módulos principales

### A. Environment factory
Responsable de instanciar el entorno según:
- reglas,
- seed,
- `StartStateConfig`,
- perfil de observación.

### B. Encoder
Responsable de convertir la respuesta del entorno en tensores.

### C. Agent network
Responsable de producir:
- `q_values`
- y opcionalmente salidas auxiliares (metadata, hidden states, etc.)

### D. Replay buffer
Responsable de almacenar experiencia y devolver batches.

### E. Trainer
Responsable de:
- explorar,
- recolectar experiencia,
- lanzar updates,
- sincronizar target,
- registrar métricas,
- evaluar.

### F. Evaluator
Responsable de correr episodios de validación separados del entrenamiento.

### G. Checkpoint manager
Responsable de guardar:
- pesos,
- optimizer,
- scheduler si existe,
- config,
- métricas,
- mejor modelo.

---

# 5. Replay buffer: qué debe hacer y por qué

El replay buffer no es un accesorio; es central para estabilidad.

## 5.1 Por qué se necesita
Evita entrenar sobre datos consecutivos altamente correlacionados y permite reutilizar experiencia pasada.

---

## 5.2 Dos variantes de buffer

### A. FeedforwardReplayBuffer
Unidad almacenada:
- transición individual

Debe guardar:
- `state`
- `action`
- `reward`
- `next_state`
- `done`
- `action_mask`
- `next_action_mask`

---

### B. RecurrentReplayBuffer
Unidad almacenada:
- secuencia o segmento temporal

Debe guardar:
- `state[t:t+L]`
- `action[t:t+L]`
- `reward[t:t+L]`
- `next_state[t:t+L]`
- `done[t:t+L]`
- `action_mask[t:t+L]`
- `next_action_mask[t:t+L]`
- `padding_mask[t:t+L]`

Esto es obligatorio porque la loss recurrente opera sobre secuencias y usa `padding_mask`.

---

## 5.3 Buenas prácticas del buffer

### Warm-up
Antes de empezar updates, llenar el buffer con experiencia inicial.

Razón:
- evita aprender sobre una muestra microscópica y sesgada.

### Muestreo uniforme
Empezar con replay uniforme.

### Prioritized replay
Se puede considerar más adelante, pero no debería ser lo primero.

### Tamaño suficiente
El buffer debe ser suficientemente grande para mezclar experiencias de políticas recientes y no tan recientes.

---

# 6. Recolección de experiencia

La recolección debe ser compatible con los dos tipos de backbone.

---

## 6.1 Para feedforward

Proceso:
1. observar estado actual
2. codificar con encoder
3. seleccionar acción con epsilon-greedy y action mask
4. ejecutar acción
5. obtener `next_response`
6. codificar `next_state`
7. guardar transición en buffer

---

## 6.2 Para recurrente

Proceso:
1. mantener secuencia actual de interacción
2. acumular pasos consecutivos
3. cerrar secuencia por:
   - longitud máxima,
   - fin de rollout,
   - criterio de segmentación
4. pad si hace falta
5. guardar secuencia o segmento en buffer

### Importante
No se debe entrenar una recurrente como si fueran pasos totalmente independientes.

---

# 7. Exploración

El agente debe explorar, pero siempre respetando acciones legales.

## 7.1 Estrategia recomendada
**epsilon-greedy con action masking**

### Regla
- con probabilidad `epsilon`: elegir acción aleatoria legal
- con probabilidad `1-epsilon`: elegir acción greedy legal

---

## 7.2 Schedule recomendado
Usar un decaimiento de epsilon:

### Inicio
- exploración alta

### Medio
- descenso gradual

### Final
- mantener mínimo positivo pequeño

Esto es importante en blackjack porque acciones como:
- `split`
- `double`
- `insurance`

pueden ser poco frecuentes si el agente explora poco.

---

## 7.3 Buenas prácticas
- no decaer epsilon demasiado rápido;
- no evaluar política con epsilon alto;
- separar claramente training exploration y evaluation policy.

---

# 8. Forward del modelo

El trainer debe soportar las tres arquitecturas con una API coherente.

---

## 8.1 Feedforward
Entrada al modelo:
- `state: [B, D]`

Salida:
- `q_values: [B, 6]`
- opcionalmente `action_mask` si el modelo lo reexpone

---

## 8.2 Recurrente
Entrada:
- `state: [B, T, D]`
- opcionalmente `padding_mask`
- opcionalmente hidden state inicial si se diseña así

Salida:
- `q_values: [B, T, 6]`

---

## 8.3 Dueling recurrente
Igual que la recurrente, pero internamente con:
- stream de valor
- stream de ventaja

---

# 9. Cálculo del target TD

La base debe ser **Double DQN**.

## 9.1 Qué hace Double DQN
En el próximo estado:
- la red online elige la acción greedily;
- la red target evalúa esa acción.

Esto reduce sobreestimación respecto a DQN vanilla.

---

## 9.2 Feedforward
Target:
- `r + gamma * Q_target(s', argmax_a Q_online(s',a))`

con máscara de acciones legales sobre el próximo estado.

---

## 9.3 Recurrente
Misma idea, pero aplicada por paso temporal:
- respetando `padding_mask`
- sin computar pérdida sobre pasos padded

---

# 10. Pérdida recomendada

## 10.1 Default
**Huber loss**

### Por qué
- más robusta a outliers en TD error;
- más estable que MSE al inicio del training.

## 10.2 MSE
Mantenerla como opción experimental, no como default principal.

---

# 11. Backward y actualización de parámetros

Este bloque debe quedar muy claro y estable.

## 11.1 Secuencia recomendada

1. `optimizer.zero_grad()`
2. forward del modelo online
3. cálculo del target con online + target
4. cálculo de TD loss
5. `loss.backward()`
6. gradient clipping
7. `optimizer.step()`

---

## 11.2 Gradient clipping
Especialmente recomendable en recurrente.

### Por qué
- evita explosión de gradientes;
- mejora estabilidad;
- casi obligatorio con GRU/LSTM.

### Recomendación
Aplicarlo siempre en modelos recurrentes.
En feedforward también puede dejarse activado si se desea coherencia.

---

# 12. Target network update

La target network debe mantenerse separada de la online network.

## 12.1 Dos opciones

### Hard update
Cada cierto número de steps:
- copiar pesos online -> target

### Soft update
Cada step:
- actualizar target suavemente con factor `tau`

---

## 12.2 Recomendación práctica

### Feedforward baseline
- hard update

### Recurrente
- empezar con hard update
- si se observa mucha oscilación, considerar soft update

---

# 13. Distinción clara entre feedforward y recurrente

Esto debe quedar explícito en implementación.

---

## 13.1 Qué cambia en feedforward

### Unidad de muestra
- transición individual

### Batch
- `[B, D]`

### Loss
- promedio simple sobre batch

### Hidden state
- no existe

### Buffer
- transiciones

---

## 13.2 Qué cambia en recurrente

### Unidad de muestra
- secuencia o segmento temporal

### Batch
- `[B, T, D]`

### Loss
- promedio sobre pasos válidos usando `padding_mask`

### Hidden state
- existe, pero se recomienda **reiniciarlo por secuencia de entrenamiento**

### Buffer
- secuencias

---

# 14. Hidden state en entrenamiento recurrente

## Recomendación principal
No complicar de entrada el pipeline manteniendo hidden state persistente fuera de cada secuencia.

### Mejor práctica inicial
- cada secuencia de entrenamiento entra con hidden state limpio
- la secuencia contiene suficiente contexto para que la red aprenda

### Ventaja
- más simple;
- más estable;
- más fácil de depurar.

---

# 15. Longitud de secuencia

En recurrente esta es una decisión importante.

## Si la secuencia es muy corta
- la memoria casi no aporta.

## Si es demasiado larga
- entrenamiento más costoso;
- mayor dificultad de optimización;
- más ruido.

## Recomendación conceptual
Usar una longitud suficiente para cubrir:
- varias decisiones,
- más de una ronda,
- potencialmente split y decisiones relacionadas.

La secuencia debe reflejar tu objetivo explícito de “jugar varias rondas seguidas”.

---

# 16. Métricas que hay que registrar

No basta con registrar `loss`.

Debemos separar:

1. **métricas internas de optimización**
2. **métricas de comportamiento/política**
3. **métricas de infraestructura**
4. **métricas de evaluación**

---

## 16.1 Métricas internas de optimización

Registrar por update:

- `loss`
- `mean_q_pred`
- `mean_target`
- `mean_reward`
- `mean_abs_td_error`
- `max_abs_td_error`
- `terminal_fraction`
- `next_legal_fraction`

En recurrente además:
- `num_valid_steps`

Estas métricas ayudan a detectar:
- explosión de targets,
- Q-values descontrolados,
- batches mal construidos.

---

## 16.2 Métricas de comportamiento

Registrar sobre interacción o evaluación:

- reward promedio por ronda
- reward promedio por mano
- EV por 1000 manos
- win rate
- push rate
- loss rate
- frecuencia de cada acción:
  - `hit`
  - `stand`
  - `double`
  - `split`
  - `surrender`
  - `insurance`

Esto es importante porque una loss “linda” no garantiza una política buena.

---

## 16.3 Métricas por tipo de situación

Muy recomendable registrar desempeño segmentado por:

- hard hands
- soft hands
- pairs
- insurance offers
- post-split states
- tipo de mesa / configuración de reglas

Esto permite ver si el agente está aprendiendo solo heurísticas gruesas o realmente blackjack fino.

---

## 16.4 Métricas de infraestructura

Registrar también:

- steps totales de entorno
- updates de gradiente
- tamaño actual del buffer
- epsilon actual
- learning rate actual
- gradient norm
- tiempo por update
- tiempo por rollout
- número de secuencias por batch si es recurrente

---

# 17. Evaluación separada del entrenamiento

La evaluación no debe contaminar el entrenamiento.

## 17.1 Qué hacer
Cada cierto número de updates o episodios:
- congelar la política;
- correr episodios de evaluación;
- usar epsilon muy bajo o cero;
- no almacenar esas trayectorias en el buffer.

---

## 17.2 Qué medir en evaluación
- reward promedio
- EV por 1000 manos
- métricas por acción
- métricas por tipo de mesa
- métricas por tipo de mano

---

## 17.3 Por qué es crucial
El reward de training mezcla:
- exploración,
- ruido,
- distribución cambiante.

La evaluación te dice si la política realmente mejora.

---

# 18. Checkpointing

El checkpointing debe diseñarse desde el inicio, no al final.

## 18.1 Qué guardar en cada checkpoint

### Mínimo obligatorio
- pesos del modelo online
- pesos del modelo target
- estado del optimizer
- estado del scheduler si existe
- contador de steps
- contador de updates
- epsilon actual
- config del experimento
- métricas recientes

### Si es recurrente
No hace falta guardar hidden states de entrenamiento como parte central del checkpoint salvo que tu loop lo requiera de forma especial.

---

## 18.2 Tipos de checkpoint

### A. `latest`
Siempre sobrescribir el checkpoint más reciente.

### B. `best_eval`
Guardar el mejor modelo según métrica de evaluación.

### C. `periodic`
Guardar checkpoints periódicos cada cierta cantidad de updates para poder volver atrás si algo diverge.

---

## 18.3 Métrica para decidir “best”
Recomendación:
- usar una métrica de evaluación estable, no la loss de entrenamiento

Por ejemplo:
- EV por 1000 manos
- reward promedio por ronda en evaluación

---

## 18.4 Buenas prácticas
- no guardar solo el mejor;
- guardar también algunos snapshots recientes;
- registrar claramente con qué seed y config se entrenó cada checkpoint.

---

# 19. Logging y trazabilidad

Todo experimento debe poder reconstruirse.

## 19.1 Qué registrar
- seed
- modo de entrenamiento (`fresh_shoe` / `unknown_progress`)
- perfil de observación
- perfil del encoder
- tipo de backbone
- tipo de recurrente (`GRU` / `LSTM`)
- gamma
- loss type
- epsilon schedule
- target update config
- replay buffer config
- batch size
- sequence length
- learning rate
- gradient clipping
- métricas por evaluación

---

## 19.2 Por qué importa
En RL hay mucha varianza entre runs.  
Si no registras bien, luego no sabrás si una mejora fue real o casualidad.

---

# 20. Curriculum recomendado

Aunque quieres implementar todo desde el inicio, el **orden de entrenamiento** sí importa.

Esto no significa rehacer el código; significa elegir bien en qué régimen correr primero.

---

## 20.1 Orden sugerido

### Fase 1
Entrenar el baseline feedforward en `fresh_shoe`

Objetivo:
- validar pipeline completo;
- validar buffer;
- validar target network;
- validar métricas;
- validar checkpointing.

---

### Fase 2
Entrenar recurrente en `fresh_shoe`

Objetivo:
- explotar memoria;
- verificar que secuencias y padding funcionan bien;
- aprender múltiples rondas.

---

### Fase 3
Entrenar recurrente en `unknown_progress`

Objetivo:
- robustez a historia truncada;
- escenario más realista.

---

### Fase 4
Entrenar dueling recurrente

Objetivo:
- mejorar estructura de valor;
- producir el modelo final fuerte.

---

# 21. Buenas prácticas de optimización

---

## 21.1 Optimizador
Una opción estándar y razonable:
- Adam o AdamW

No hace falta complicar esto al inicio.

---

## 21.2 Learning rate
No usar learning rate agresivo al principio.

La prioridad en RL es:
- estabilidad
- no velocidad bruta

---

## 21.3 Scheduler
No es obligatorio en la primera versión.

Se puede empezar sin scheduler y luego añadirlo si hace falta.

---

## 21.4 Batch size
Debe ser suficientemente grande para estabilidad, pero no tan grande que reduzca demasiado la frecuencia efectiva de updates.

En recurrente hay que pensar en:
- batch size de secuencias
- longitud de secuencia

no solo en número bruto de tensores.

---

# 22. Cosas que suelen romper el training

Checklist de fallas típicas:

- empezar updates sin warm-up del buffer
- epsilon decay demasiado rápido
- targets mal enmascarados
- acciones ilegales durante exploración
- no usar target network correctamente
- usar padding mal y contaminar la loss recurrente
- secuencias demasiado cortas para DRQN
- gradient explosion en LSTM/GRU
- learning rate demasiado alto
- evaluar con muy pocas rondas
- confiar solo en loss y no en métricas de política
- mezclar demasiadas mesas distintas desde el inicio sin control

---

# 23. Estructura recomendada del trainer

El trainer debe tener responsabilidades explícitas.

## 23.1 Componentes del trainer
- environment(s)
- encoder
- online_network
- target_network
- replay_buffer
- optimizer
- scheduler opcional
- epsilon scheduler
- checkpoint manager
- evaluator
- logger

---

## 23.2 Funciones o métodos clave

### `collect_experience(...)`
Interactúa con el entorno y agrega experiencia al buffer.

### `sample_batch(...)`
Obtiene batch del buffer.

### `train_step(...)`
Hace forward, target, loss, backward, optimizer step.

### `update_target(...)`
Sincroniza target network.

### `evaluate(...)`
Corre episodios de evaluación.

### `save_checkpoint(...)`
Guarda estado completo.

### `load_checkpoint(...)`
Permite reanudar entrenamiento.

---

# 24. Flujo recomendado del training loop

## 24.1 Inicialización
1. crear entorno(s)
2. crear encoder
3. crear modelo online
4. clonar target
5. crear optimizer
6. crear replay buffer
7. inicializar schedulers
8. inicializar logger/checkpointing

---

## 24.2 Warm-up
1. interactuar con el entorno
2. llenar buffer
3. no actualizar todavía

---

## 24.3 Training loop principal
Repetir:

1. recolectar experiencia nueva
2. almacenar en buffer
3. samplear batch
4. computar loss TD
5. backward
6. gradient clipping
7. optimizer step
8. actualizar target si corresponde
9. registrar métricas
10. evaluar periódicamente
11. guardar checkpoints

---

# 25. Qué debe diferenciar el trainer feedforward del recurrente

El trainer puede compartir gran parte del código, pero debe separar claramente:

## Feedforward
- buffer de transiciones
- sample batch simple
- no hidden state
- no padding

## Recurrente
- buffer de secuencias
- sample batch secuencial
- padding mask
- loss sobre pasos válidos
- posibilidad de elegir GRU o LSTM desde config

La lógica de alto nivel puede ser la misma, pero las unidades de datos son distintas.

---

# 26. Recomendación final de implementación

Si hay que implementarlo todo desde ya, mi recomendación es:

## Implementar desde el inicio:
- replay buffer feedforward y recurrente
- trainer general con dos rutas
- evaluation loop
- checkpoint manager
- logging de métricas
- target update configurable
- epsilon scheduler
- gradient clipping
- soporte para `fresh_shoe` y `unknown_progress`

## No implementar todavía si no hace falta:
- prioritized replay
- schedulers sofisticados
- trucos avanzados adicionales
- demasiada ingeniería extra

La prioridad es una base sólida, estable y trazable.

---

# 27. Resumen ejecutivo final

## Qué queremos construir
Un pipeline de entrenamiento completo para blackjack RL en PyTorch que soporte:

- Double DQN feedforward
- DRQN / Recurrent Double DQN
- Dueling Recurrent Double DQN

y además:

- `fresh_shoe`
- `unknown_progress`

---

## Piezas obligatorias
- replay buffer
- warm-up
- epsilon-greedy con action masking
- target network
- Huber loss
- gradient clipping
- evaluación separada
- checkpointing
- logging completo

---

## Métricas obligatorias
- loss TD
- Q pred / target
- mean abs TD error
- reward promedio
- EV por 1000 manos
- action frequencies
- win/push/loss rate
- métricas por tipo de mano y tipo de mesa

---

## Distinción clave
- **Feedforward** aprende de transiciones individuales
- **Recurrente** aprende de secuencias y necesita `padding_mask`

---

## Regla de oro
No juzgar el modelo solo por la loss.  
La política se juzga con evaluación separada y métricas de comportamiento.

---

## Orden lógico
1. pipeline completo
2. baseline feedforward
3. recurrente
4. dueling recurrente
5. `unknown_progress`
6. refinamiento fino

Este documento debe servir como blueprint de implementación del sistema completo de training.
