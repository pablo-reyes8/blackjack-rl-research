# Especificación del loss Bellman para las tres redes de Blackjack RL

## Objetivo

Definir exclusivamente la lógica del **loss Bellman / TD loss** para las tres arquitecturas ya construidas:

1. **Baseline Double DQN feedforward**
2. **Recurrent Double DQN (DRQN)**
3. **Dueling Recurrent Double DQN**

La meta en este documento es dejar completamente claro:

- qué entradas necesita el loss;
- cómo se calcula el target Bellman;
- cómo se selecciona la acción siguiente en Double DQN;
- cómo se incorporan las máscaras de acciones legales;
- cómo se incorpora `done`;
- cómo se incorpora `padding_mask` en modelos recurrentes;
- qué partes comparten las tres arquitecturas.

Este documento **no cubre todavía**:
- replay buffer;
- training loop;
- optimizer;
- epsilon schedule;
- target update schedule.

---

# 1. Principio fundamental

En estas arquitecturas, el loss **no viene dado por PyTorch** como una pérdida ya hecha para RL.

PyTorch solo aporta:
- tensores;
- autograd;
- pérdidas base como `SmoothL1Loss` o `MSELoss`.

La lógica Bellman se construye manualmente.

---

# 2. Qué queremos estimar

En un problema de control discreto como blackjack, la red aproxima:

$$
Q_	heta(s, a)
$$

donde:

- $ s $ = estado o representación codificada del estado;
- $ a $ = acción discreta legal o potencial;
- $ Q_	heta(s,a) $ = retorno esperado descontado si se toma la acción $a$ en el estado $s$ y luego se sigue la política implícita del agente.

---

# 3. Ecuación objetivo: Double DQN

La forma base del target Bellman para Double DQN es:

$$
y = r + \gamma (1-d)\, Q_{	ext{target}}igl(s', rg\max_{a'} Q_{	ext{online}}(s', a')igr)
$$

donde:

- $ r $ = recompensa inmediata;
- $ \gamma $ = factor de descuento;
- $ d \in \{0,1\} $ = indicador terminal (`done`);
- `online network` = red entrenable actual;
- `target network` = copia estabilizada de la red online.

---

# 4. Adaptación necesaria a este proyecto

En este proyecto, el target Bellman no puede usar un `argmax` libre sobre todas las acciones, porque el entorno tiene:

- acciones ilegales por estado;
- `action_mask` por estado;
- `next_action_mask` por estado siguiente.

Por tanto, tanto en la selección de acción siguiente como en el valor siguiente, hay que restringirse a acciones legales.

---

# 5. Regla general para acciones ilegales

Cuando se calcule el máximo sobre acciones siguientes, se debe aplicar máscara:

- acciones legales: conservan su Q-value;
- acciones ilegales: se rellenan con `-inf` efectivo o el mínimo representable del dtype.

Esto ya es consistente con la función:

```python
apply_action_mask(q_values, action_mask)
```

que enmascara con el valor mínimo representable del tensor. fileciteturn2file1L16-L18

---

# 6. Pérdida base recomendada

## Recomendación
Usar **Huber loss**, es decir:

- `torch.nn.SmoothL1Loss(reduction="none")`

## Razón
En DQN es preferible a MSE como default porque:
- es más robusta a outliers;
- amortigua targets ruidosos;
- suele estabilizar mejor el aprendizaje bootstrap.

---

# 7. Caso 1: Baseline Double DQN feedforward

---

## 7.1 Entradas mínimas requeridas

El loss feedforward debe recibir un batch con al menos:

```python
{
    "state": ...,
    "action": Tensor[B],
    "reward": Tensor[B],
    "next_state": ...,
    "done": Tensor[B],
    "action_mask": Tensor[B, 6],
    "next_action_mask": Tensor[B, 6],
}
```

donde:

- `state` y `next_state` pueden venir crudos o ya codificados;
- `action` es el índice entero de la acción tomada;
- `reward` es la recompensa escalar;
- `done` indica si la transición termina el episodio o la ronda;
- `next_action_mask` define qué acciones son legales en `next_state`.

---

## 7.2 Salidas necesarias de la red

La red feedforward ya devuelve:

- `q_values`
- `masked_q_values`

para el batch. fileciteturn2file0L145-L163

### Interpretación
- `q_values`: Q-values sin máscara, útiles para gather sobre acción ejecutada
- `masked_q_values`: Q-values con acciones ilegales anuladas, útiles para argmax legal

---

## 7.3 Cálculo de la predicción actual

La predicción TD para cada transición es:

$$
Q_{	ext{pred}} = Q_{	ext{online}}(s, a)
$$

Implementacionalmente:
- pasar `state` por la red online;
- tomar `q_values`;
- hacer `gather` usando `action`.

Resultado esperado:
- `q_pred: [B]`

---

## 7.4 Cálculo del target Double DQN

### Paso 1: red online sobre `next_state`
Pasar `next_state` por la red online y obtener:

- `next_online_masked_q: [B, 6]`

### Paso 2: acción siguiente greedy legal
Seleccionar:

$$
a^* = arg\max_{a'} Q_{	ext{online}}^{	ext{masked}}(s', a')
$$

Resultado:
- `next_action: [B]`

### Paso 3: red target sobre `next_state`
Pasar `next_state` por la red target y obtener:

- `next_target_q_values: [B, 6]`

### Paso 4: gather del valor target
Tomar:

$$
Q_{	ext{next}} = Q_{	ext{target}}(s', a^*)
$$

Resultado:
- `next_q: [B]`

### Paso 5: construir Bellman target
$$
y = r + \gamma (1-d)\, Q_{	ext{next}}
$$

Resultado:
- `target: [B]`

---

## 7.5 Cálculo del TD error y loss

$$
\delta = Q_{	ext{pred}} - y
$$

y luego:

$$
L = 	ext{Huber}(Q_{	ext{pred}}, y)
$$

como `reduction="none"` inicialmente.

Resultado intermedio:
- `loss_per_sample: [B]`

### Loss final
$$
L_{	ext{final}} = \frac{1}{B}\sum_i L_i
$$

---

# 8. Caso 2: Recurrent Double DQN (DRQN)

---

## 8.1 Entradas mínimas requeridas

Para la versión recurrente, el batch debe incluir:

```python
{
    "state": ...,
    "action": Tensor[B, T],
    "reward": Tensor[B, T],
    "next_state": ...,
    "done": Tensor[B, T],
    "action_mask": Tensor[B, T, 6],
    "next_action_mask": Tensor[B, T, 6],
    "padding_mask": Tensor[B, T],
}
```

donde:

- `padding_mask[b, t] = True` significa paso real;
- `padding_mask[b, t] = False` significa padding.

---

## 8.2 Salidas necesarias de la red recurrente

La red recurrente devuelve:

- `q_values: [B, T, 6]`
- `masked_q_values: [B, T, 6]`
- `padding_mask` propagado
- `hidden_state` si se quiere inspección adicional. fileciteturn2file0L224-L258

---

## 8.3 Cálculo de la predicción actual

Para cada paso temporal real:

$$
Q_{	ext{pred}, t} = Q_{	ext{online}}(s_t, a_t)
$$

Implementacionalmente:
- usar `q_values [B, T, 6]`;
- hacer `gather` con `action [B, T]`.

Resultado:
- `q_pred: [B, T]`

---

## 8.4 Cálculo del target Double DQN por paso

### Paso 1: red online sobre `next_state`
Obtener:
- `next_online_masked_q: [B, T, 6]`

### Paso 2: acción siguiente greedy legal
$$
a_t^* = rg\max_{a'} Q_{	ext{online}}^{	ext{masked}}(s_{t+1}, a')
$$

Resultado:
- `next_action: [B, T]`

### Paso 3: red target sobre `next_state`
Obtener:
- `next_target_q_values: [B, T, 6]`

### Paso 4: gather del valor target
$$
Q_{	ext{next}, t} = Q_{	ext{target}}(s_{t+1}, a_t^*)
$$

Resultado:
- `next_q: [B, T]`

### Paso 5: target Bellman por paso
$$
y_t = r_t + \gamma (1-d_t)Q_{	ext{next}, t}
$$

Resultado:
- `target: [B, T]`

---

## 8.5 TD error y loss por paso

$$
\delta_t = Q_{	ext{pred}, t} - y_t
$$

Luego:
- Huber con `reduction="none"`

Resultado:
- `loss_per_timestep: [B, T]`

---

## 8.6 Aplicación obligatoria de `padding_mask`

Aquí está la diferencia clave respecto al feedforward.

Los pasos padded **no deben contribuir al loss**.

Entonces:

$$
L_{b,t}^{	ext{masked}} = L_{b,t}\cdot 	ext{padding\_mask}_{b,t}
$$

donde el `padding_mask` se convierte a float.

### Loss final
$$
L_{	ext{final}} =
\frac{\sum_{b,t} L_{b,t}\cdot M_{b,t}}
{\sum_{b,t} M_{b,t}}
$$

donde $ M_{b,t} $ es el `padding_mask`.

---

# 9. Caso 3: Dueling Recurrent Double DQN

---

## 9.1 Punto clave

El loss Bellman **no cambia** por ser dueling.

La diferencia de dueling está solo en cómo la red parametriza internamente los Q-values:

- value stream
- advantage stream
- recombinación a `q_values`

Tu implementación ya devuelve `q_values` finales desde el `DuelingQHead`, por lo tanto el loss puede trabajar exactamente igual que en el DRQN simple. 

---

## 9.2 Entradas requeridas

Las mismas que en DRQN:

```python
{
    "state": ...,
    "action": Tensor[B, T],
    "reward": Tensor[B, T],
    "next_state": ...,
    "done": Tensor[B, T],
    "action_mask": Tensor[B, T, 6],
    "next_action_mask": Tensor[B, T, 6],
    "padding_mask": Tensor[B, T],
}
```

---

## 9.3 Salidas usadas para el loss

Aunque el modelo también devuelve:
- `state_value`
- `advantages`

el loss Bellman solo debe usar:

- `q_values`
- `masked_q_values`

No se define una pérdida separada para `value` ni para `advantage`.

---

## 9.4 Cálculo del loss

Exactamente igual que en DRQN:

1. `q_pred = gather(q_values, action)`
2. `next_action = argmax(next_online_masked_q)`
3. `next_q = gather(next_target_q_values, next_action)`
4. `target = reward + gamma * (1-done) * next_q`
5. `loss_raw = huber(q_pred, target)`
6. aplicar `padding_mask`
7. promediar solo sobre pasos válidos

---

# 10. Resumen de qué comparten las tres redes

## Igual en las tres
- lógica Double DQN;
- gather sobre la acción ejecutada;
- argmax legal con máscara en `next_state`;
- target obtenido con red target;
- Huber loss;
- manejo de `done`.

## Solo cambia entre arquitecturas
- forma del input (`[B,D]` vs `[B,T,D]`);
- presencia de `padding_mask`;
- arquitectura de la red;
- en dueling, la forma de producir `q_values`, no la forma de calcular el loss.

---

# 11. Convenciones recomendadas de implementación

---

## 11.1 Usar `with torch.no_grad()` en el target
El cálculo del target Bellman debe ir sin gradientes:

- forward de red online sobre `next_state` para seleccionar acción
- forward de red target sobre `next_state` para evaluar esa acción

Esto evita:
- gradientes accidentales por el branch target;
- gasto innecesario de memoria.

---

## 11.2 Usar `q_values` sin máscara para `gather` actual
Para la acción ejecutada en el estado actual:
- usar `q_values`, no `masked_q_values`.

Razón:
- la acción ejecutada ya debería ser válida por construcción;
- no necesitas llenar con `-inf` ahí.

---

## 11.3 Usar `masked_q_values` para el `argmax` siguiente
Para elegir la acción greedy en `next_state`:
- usar `masked_q_values`
- nunca usar `q_values` sin máscara

Razón:
- el target no puede elegir acciones ilegales.

---

## 11.4 Convertir `done` a float
Para la fórmula Bellman:
- `done_float = done.float()`

y usar:
$$
1 - done
$$

---

## 11.5 Validar que toda fila tenga al menos una acción legal en `next_action_mask`
Esto es un chequeo de sanidad importante.

Si un `next_state` no tiene ninguna acción legal, entonces:
- el `argmax` sobre valores todos enmascarados puede ser inválido;
- eso debe prevenirse en datos o manejarse explícitamente.

---

## 11.6 En recurrente, no dejar que padding entre al denominator si no hay pasos válidos
El denominador del promedio debe ser:

```python
num_valid = padding_mask.sum()
```

y conviene proteger contra el caso degenerado `num_valid == 0`.

---

# 12. Interfaces conceptuales que conviene implementar

Aunque aquí no se escribe código, conviene que la lógica quede separada en funciones limpias.

---

## 12.1 Feedforward

```python
compute_double_dqn_targets_feedforward(
    online_network,
    target_network,
    batch,
    gamma,
) -> target
```

Devuelve:
- `target: [B]`
- opcionalmente estadísticas auxiliares

```python
compute_td_loss_feedforward(
    online_network,
    target_network,
    batch,
    gamma,
    loss_type="huber",
) -> dict
```

Devuelve:
- `loss`
- `loss_per_sample`
- `q_pred`
- `target`
- `td_error`

---

## 12.2 Recurrente

```python
compute_double_dqn_targets_recurrent(
    online_network,
    target_network,
    batch,
    gamma,
) -> target
```

Devuelve:
- `target: [B, T]`

```python
compute_td_loss_recurrent(
    online_network,
    target_network,
    batch,
    gamma,
    loss_type="huber",
) -> dict
```

Devuelve:
- `loss`
- `loss_per_timestep`
- `q_pred`
- `target`
- `td_error`
- `num_valid_steps`

---

# 13. Shapes finales esperados

---

## 13.1 Feedforward

### Inputs
- `action: [B]`
- `reward: [B]`
- `done: [B]`

### Outputs intermedios
- `q_values: [B, 6]`
- `masked_q_values: [B, 6]`
- `q_pred: [B]`
- `target: [B]`
- `loss_per_sample: [B]`

### Output final
- `loss: scalar`

---

## 13.2 Recurrente / Dueling recurrente

### Inputs
- `action: [B, T]`
- `reward: [B, T]`
- `done: [B, T]`
- `padding_mask: [B, T]`

### Outputs intermedios
- `q_values: [B, T, 6]`
- `masked_q_values: [B, T, 6]`
- `q_pred: [B, T]`
- `target: [B, T]`
- `loss_per_timestep: [B, T]`

### Output final
- `loss: scalar`

---

# 14. Chequeos mínimos de consistencia antes de seguir al trainer

Antes de pasar al training loop, conviene verificar que el módulo de loss cumpla esto:

## Feedforward
1. `target` no requiere gradiente
2. `loss` sí requiere gradiente
3. `q_pred.shape == reward.shape`
4. `target.shape == reward.shape`
5. acciones ilegales no son elegidas en `next_state`

## Recurrente
1. `target` no requiere gradiente
2. `loss` sí requiere gradiente
3. `q_pred.shape == reward.shape == done.shape`
4. `loss` no depende de pasos padded
5. `num_valid_steps > 0`
6. acciones ilegales no son elegidas en `next_state`

---

# 15. Resumen ejecutivo

## Para las tres redes
El loss base es **Double DQN + Huber loss**.

## Feedforward
- sin `padding_mask`
- target y loss por muestra

## Recurrente
- target y loss por paso temporal
- requiere `padding_mask`

## Dueling recurrente
- exactamente la misma lógica Bellman que el recurrente simple
- solo cambia cómo la red produce `q_values`

## Regla crítica
- `q_values` para gather actual
- `masked_q_values` para argmax legal en `next_state`

## Regla crítica 2
- target siempre bajo `torch.no_grad()`

## Regla crítica 3
- en recurrente, promediar solo sobre pasos válidos según `padding_mask`

---

# 16. Siguiente paso natural después de este documento

Una vez implementado este loss, lo siguiente será:

1. validarlo con batches sintéticos;
2. comparar feedforward vs recurrente en shapes y estabilidad;
3. recién entonces montar:
   - optimizer,
   - target updates,
   - replay buffer,
   - training loop.
