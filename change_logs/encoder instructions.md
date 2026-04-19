# Especificación del Encoder para el Agente de RL en Blackjack

## Objetivo

Definir un **encoder modular** que transforme la observación del entorno en tensores listos para un agente en PyTorch, permitiendo:

1. entrenar un **baseline simple** (`minimal_basic_strategy`);
2. entrenar un agente **realista con memoria** (`table_realistic_default`);
3. soportar **múltiples tipos de mesa** con reglas distintas;
4. dejar la interfaz preparada para un agente **recurrente (LSTM/GRU)** que procese varias rondas seguidas.

La idea es separar claramente:

- **lo que produce el entorno**;
- **lo que consume la red**;
- **qué módulos del encoder se activan o no** según el perfil de observación.

El entorno ya devuelve una observación bien separada de `public_state` y `debug_state`, lo cual permite diseñar el encoder sin filtrar información oculta por accidente. fileciteturn0file3L392-L419 fileciteturn0file4L46-L57

---

# 1. Filosofía de diseño

## 1.1 Principios

El encoder debe ser:

- **modular**: cada bloque codifica un subconjunto semántico;
- **estable**: la dimensionalidad de salida no cambia durante entrenamiento;
- **sin leakage**: solo usa `observation`, `table_rules` y opcionalmente `action_mask`;
- **compatible con distintos perfiles** de observación (`minimal_basic_strategy`, `table_realistic_default`, `fully_observable_sim`). fileciteturn0file0L33-L62 fileciteturn0file4L84-L118
- **compatible con memoria temporal**: la salida por paso debe poder alimentar una LSTM.

## 1.2 Separación conceptual

El encoder no debe mezclar todo “a mano” en un solo vector sin estructura.  
Debe construir varios sub-bloques:

1. `hand_features`
2. `rule_features`
3. `context_features`
4. `history_features`
5. `temporal_features`
6. `mask_features` (opcional, no siempre como input)

Luego se concatenan en un tensor final de estado.

---

# 2. Qué entra al encoder

La entrada del encoder será el resultado de `env.reset()` o `env.step()`, específicamente:

```python
{
    "observation": {...},
    "table_rules": {...},
    "action_mask": [...],
    "action_mask_by_name": {...},
    "reward": float,
    "done": bool,
    "info": {...}
}
```

Las partes relevantes para el encoder son:

- `response["observation"]`
- `response["table_rules"]`
- opcionalmente `response["action_mask"]`

No usar:

- `response["info"]["public_state"]` como entrada principal del agente;
- `get_debug_state()`;
- ninguna variable no visible para el agente. fileciteturn0file3L392-L419

---

# 3. Tipos de encoder que queremos soportar

Tendremos tres configuraciones principales.

## 3.1 Encoder tipo A: `minimal_basic_strategy`

Pensado para baseline rápido.

Usa información mínima:

- `dealer_upcard`
- `dealer_upcard_value`
- `current_hand_total`
- `current_hand_is_soft`
- `hand_context`
- `insurance_context`
- `table_rules` visibles mínimas

Este perfil existe explícitamente en el config. fileciteturn0file0L58-L93

### Caso de uso
- validar pipeline;
- comparar contra basic strategy;
- entrenar un DQN/MLP no recurrente.

---

## 3.2 Encoder tipo B: `table_realistic_default`

Pensado como perfil principal del proyecto.

Incluye típicamente:

- cartas visibles actuales de la mano;
- upcard del dealer;
- otras manos visibles tras split;
- contexto de apuesta;
- contexto de insurance;
- reglas visibles;
- historial observable de cartas;
- descarte resumido;
- contexto temporal;
- progreso estimado del shoe. fileciteturn0file4L84-L118

### Caso de uso
- agente serio;
- observación parecida a un scraper visual;
- entrenamiento con memoria entre rondas.

---

## 3.3 Encoder tipo C: `fully_observable_sim`

No es el foco principal, pero es útil como upper bound.

Puede incluir:

- composición exacta del shoe;
- reglas ocultas;
- `n_decks`;
- `shoe_penetration`. fileciteturn0file0L38-L57

### Caso de uso
- benchmark;
- verificar techo de performance;
- detectar si el límite es de observación o de arquitectura.

---

# 4. Salidas esperadas del encoder

## 4.1 Salida base por paso

Todo encoder debe devolver un diccionario tensorial con al menos:

```python
{
    "state_vector": Tensor[float32]    # [D]
    "action_mask": Tensor[bool or float32]   # [6]
    "metadata": dict
}
```

donde:

- `state_vector` = representación final concatenada del estado;
- `action_mask` = máscara alineada con `ACTION_ORDER`;
- `metadata` = información auxiliar para debug, logging o reconstrucción.

La máscara tiene dimensión 6 porque el espacio de acciones es:
`stand, hit, double, split, surrender, insurance`. fileciteturn0file1L14-L23

---

## 4.2 Salida para entrenamiento feedforward

Para un agente no recurrente:

- entrada de la red: `state_vector` con forma `[B, D]`
- máscara: `[B, 6]`

---

## 4.3 Salida para entrenamiento recurrente

Para LSTM/GRU:

- entrada: secuencia de estados con forma `[B, T, D]`
- máscara de acciones por paso: `[B, T, 6]`
- máscara de padding temporal: `[B, T]`
- opcionalmente `episode_id` o `sequence_breaks`

La LSTM no necesita que el encoder sepa entrenar secuencias; solo necesita que el encoder produzca un vector fijo por paso.

---

# 5. Módulos del encoder

---

## 5.1 Módulo `HandFeatureEncoder`

### Objetivo
Codificar la mano actual del jugador y la carta visible del dealer.

### Entrada posible

Para `basic_strategy`:
- `current_hand_total`
- `current_hand_is_soft`
- `dealer_upcard`
- `dealer_upcard_value`

Para `table_raw`:
- `current_hand_cards`
- `dealer_upcard`
- `dealer_upcard_value`

### Salida sugerida

```python
hand_features: Tensor[float32]  # [D_hand]
```

### Implementación sugerida

#### Variante minimal
- `current_hand_total`: escalar normalizado a `[0,1]` o dividido por 21
- `current_hand_is_soft`: binaria
- `dealer_upcard_value`: escalar normalizado o one-hot de 10 clases

#### Variante raw
- codificar hasta `MAX_CURRENT_HAND_CARDS`
- cada carta como vector one-hot de 13 ranks
- máscara de longitud real
- además:
  - total actual
  - flag soft
  - número de cartas

### Recomendación práctica
Aunque tengas cartas raw, añade también:
- total;
- is_soft;
- hand_length.

Es redundancia útil y estable.

---

## 5.2 Módulo `OtherHandsEncoder`

### Objetivo
Codificar otras manos visibles del jugador cuando hubo split.

### Entrada
- `other_player_hands_visible`

Esto existe en modo `table_raw` cuando el config lo permite. fileciteturn0file3L242-L246

### Salida

```python
other_hands_features: Tensor[float32]  # [D_other]
```

### Implementación sugerida

Fijar:
- `MAX_OTHER_HANDS = max_hands_after_split - 1`
- `MAX_CARDS_PER_HAND = 12` (o el número que definas)

Por cada mano:
- cartas padded;
- bet;
- from_split;
- split_aces;
- closed.

Luego:
- o concatenas todo plano;
- o haces pooling por mano;
- o pasas por un mini encoder por mano y luego agregas.

### Recomendación
Para primera versión:
- encoding plano + padding.
Más adelante:
- mini encoder por mano + mean/max pooling.

---

## 5.3 Módulo `HandContextEncoder`

### Objetivo
Codificar contexto estructural de la mano actual.

### Entrada
`observation["hand_context"]`, que puede incluir:
- `current_hand_index`
- `n_player_hands`
- `from_split`
- `split_aces`
- `first_decision_on_hand` fileciteturn0file3L251-L258

### Salida

```python
hand_context_features: Tensor[float32]  # [D_hand_ctx]
```

### Comentario
Este bloque es muy importante para aprender:
- cuándo tiene sentido doblar;
- cuándo una mano viene de split;
- cuándo ya no estás en la primera decisión.

---

## 5.4 Módulo `InsuranceContextEncoder`

### Objetivo
Codificar si hay oferta de seguro y el estado asociado.

### Entrada
`observation["insurance_context"]`:
- `insurance_offer_active`
- `insurance_bet` fileciteturn0file3L260-L265

### Salida

```python
insurance_features: Tensor[float32]  # [D_ins]
```

### Comentario
Este módulo debe existir incluso si el seguro aparece raramente.  
La red debe distinguir estados donde `insurance` es una decisión económicamente relevante.

---

## 5.5 Módulo `BetEncoder`

### Objetivo
Codificar la apuesta actual.

### Entrada
- `current_bet`

### Salida
```python
bet_features: Tensor[float32]  # [D_bet]
```

### Comentario
Con `base_bet` fijo este módulo es pequeño, pero conviene dejarlo desde ya porque luego puedes:
- variar tamaño de apuesta;
- introducir bankroll;
- convertir esto en política conjunta de play + bet sizing.

---

## 5.6 Módulo `RuleEncoder`

### Objetivo
Permitir que el agente aprenda a jugar distinto en distintas mesas.

### Entrada
`table_rules`, que puede incluir reglas visibles como:
- `dealer_hits_soft_17`
- `blackjack_payout`
- `double_allowed_on`
- `double_after_split_allowed`
- `split_rule`
- `max_hands_after_split`
- `resplit_aces_allowed`
- `hit_split_aces_allowed`
- `surrender_allowed`
- `insurance_allowed`
- `base_bet` fileciteturn0file3L320-L361 fileciteturn0file4L137-L152

### Salida
```python
rule_features: Tensor[float32]  # [D_rules]
```

### Implementación sugerida

#### Booleanos
- codificación 0/1

#### Categóricos
- `double_allowed_on`: one-hot de 3 categorías
- `split_rule`: one-hot de 2 categorías

#### Continuos
- `blackjack_payout`
- `base_bet`
- opcionalmente `shoe_penetration`
- opcionalmente `n_decks`

### Comentario clave
Este bloque es central para el objetivo de **generalización entre mesas**.  
Sin esto, el agente no puede adaptar política a reglas distintas.

---

## 5.7 Módulo `ObservedCardsHistoryEncoder`

### Objetivo
Representar la información que el jugador ha visto del shoe.

### Entrada
`observation["observed_cards_history"]`

Según config puede venir como:
- `rank_counts`
- `low_neutral_high`
- `recent_cards_sequence` fileciteturn0file0L18-L24 fileciteturn0file3L292-L308

### Salida
```python
history_features: Tensor[float32]  # [D_hist]
```

### Estrategias posibles

#### Opción A. `rank_counts`
Vector de 13 dimensiones:
- conteos por rank `A,2,...,K`

Muy buena opción para empezar.

#### Opción B. `low_neutral_high`
Vector de 3 dimensiones:
- low / neutral / high

Muy compacto, pero pierde detalle.

#### Opción C. `recent_cards_sequence`
Secuencia de ranks observados recientemente:
- usar embedding de rank + pooling o mini-RNN

Más expresivo, más complejo.

### Recomendación
Para primera versión seria:
- usar `rank_counts`
- normalizar por número total observado o por total teórico del shoe

Ejemplo:
```python
rank_fraction = observed_rank_count / max(1, observed_cards_count)
```

### Comentario
Este módulo es el más cercano a “aprender que las cartas se acaban”.

---

## 5.8 Módulo `DiscardSummaryEncoder`

### Objetivo
Codificar resumen agregado del descarte observable.

### Entrada
`observation["discard_summary"]`:
- `observed_cards_count`
- `by_group`
- `recent_cards` fileciteturn0file3L310-L317

### Salida
```python
discard_features: Tensor[float32]  # [D_discard]
```

### Comentario
Hay redundancia con `observed_cards_history`, pero puede ser útil como shortcut para la red.

---

## 5.9 Módulo `TemporalFeatureEncoder`

### Objetivo
Codificar dinámica entre rondas y progreso del shoe.

### Entrada
`observation["temporal_context"]`, que puede incluir:
- `shuffle_count`
- `rounds_played_total`
- `rounds_since_shuffle`
- `player_hands_seen_since_shuffle`
- `dealer_hands_seen_since_shuffle`
- `player_hands_seen_total`
- `dealer_hands_seen_total`
- `estimated_shoe_progress`
- `last_round_outcome`
- `recent_actions` fileciteturn0file3L268-L290 fileciteturn0file4L119-L135

### Salida
```python
temporal_features: Tensor[float32]  # [D_temp]
```

### Submódulos internos

#### a. Progress encoder
Para:
- `fraction_used`
- `bucket` (`early`, `mid`, `late`) fileciteturn0file3L653-L663

#### b. Last outcome encoder
Para:
- reward de la ronda pasada
- dealer total pasado
- dealer blackjack pasado
- resultado por mano

#### c. Recent actions encoder
Para:
- ventana corta de acciones visibles públicas

### Comentario
En un agente con LSTM este bloque puede volverse opcional o más pequeño, pero inicialmente conviene incluirlo porque resume historia útil.

---

## 5.10 Módulo `ExactShoeEncoder` (solo simulación upper bound)

### Objetivo
Codificar composición exacta del shoe restante.

### Entrada
- `exact_shoe_composition`

### Salida
```python
exact_shoe_features: Tensor[float32]  # [13]
```

### Comentario
No usar en el agente realista principal.  
Solo para benchmark fully observable.

---

# 6. Construcción del vector final

## 6.1 Forma general

El encoder final concatena:

```python
state_vector = concat([
    hand_features,
    other_hands_features,
    hand_context_features,
    insurance_features,
    bet_features,
    rule_features,
    history_features,
    discard_features,
    temporal_features,
    exact_shoe_features,   # opcional
])
```

Salida final:

```python
state_vector: Tensor[float32]  # [D_total]
```

---

## 6.2 Dimensionalidad fija

Aunque algunos módulos no existan en ciertos perfiles, la salida total debe mantenerse fija por configuración de encoder.

Dos enfoques válidos:

### Enfoque 1
Tener un encoder distinto por perfil:
- `MinimalObservationEncoder`
- `RealisticObservationEncoder`
- `FullObservationEncoder`

### Enfoque 2
Tener un solo encoder configurable que rellena con ceros módulos ausentes.

### Recomendación
Para trazabilidad y limpieza:
- implementar **un encoder base configurable**;
- pero guardar configs explícitas por perfil.

---

# 7. Interfaces sugeridas

## 7.1 Clase base

```python
class BaseBlackjackEncoder(nn.Module):
    def forward(self, response: dict) -> dict[str, Tensor]:
        ...
```

Salida:

```python
{
    "state_vector": Tensor[D],
    "action_mask": Tensor[6],
    "module_tensors": {
        "hand": Tensor[D_hand],
        "rules": Tensor[D_rules],
        ...
    }
}
```

Esto permite inspeccionar qué parte del estado está entrando.

---

## 7.2 Encoders específicos

```python
class MinimalObservationEncoder(BaseBlackjackEncoder):
    ...

class RealisticObservationEncoder(BaseBlackjackEncoder):
    ...

class FullObservationEncoder(BaseBlackjackEncoder):
    ...
```

O alternativamente:

```python
class BlackjackObservationEncoder(nn.Module):
    def __init__(self, encoder_config: EncoderConfig):
        ...
```

---

# 8. Configuración recomendada del encoder

Definir un `EncoderConfig` separado del `ObservationConfig`.

Ejemplo conceptual:

```python
@dataclass
class EncoderConfig:
    profile: str
    encode_rules: bool = True
    encode_other_hands: bool = True
    encode_temporal: bool = True
    encode_observed_history: bool = True
    encode_discard_summary: bool = True
    encode_recent_actions: bool = False
    encode_exact_shoe: bool = False
    card_encoding: str = "one_hot_rank"
    history_encoding: str = "rank_counts"
    normalize_counts: bool = True
```

### Razón
`ObservationConfig` controla lo que entrega el entorno.  
`EncoderConfig` controla cómo lo conviertes en tensores.

---

# 9. Recomendación por régimen de entrenamiento

---

## 9.1 Régimen 1: baseline no recurrente

### Perfil de observación
- `minimal_basic_strategy`

### Encoder activo
- `HandFeatureEncoder`
- `HandContextEncoder`
- `InsuranceContextEncoder`
- `RuleEncoder`

### Salida
```python
state_vector: [D_min]
```

### Uso
- MLP / DQN clásico
- sanity check

---

## 9.2 Régimen 2: agente principal realista

### Perfil de observación
- `table_realistic_default`

### Encoder activo
- `HandFeatureEncoder`
- `OtherHandsEncoder`
- `HandContextEncoder`
- `InsuranceContextEncoder`
- `BetEncoder`
- `RuleEncoder`
- `ObservedCardsHistoryEncoder`
- `DiscardSummaryEncoder`
- `TemporalFeatureEncoder`

### Salida
```python
state_vector: [D_realistic]
```

### Uso
- DQN con memoria
- LSTM/GRU
- generalización a varias mesas

---

## 9.3 Régimen 3: upper bound simulación

### Perfil de observación
- `fully_observable_sim`

### Encoder activo
- todo lo anterior
- `ExactShoeEncoder`

### Salida
```python
state_vector: [D_full]
```

### Uso
- benchmark máximo
- ablation de información

---

# 10. Compatibilidad con LSTM y “dos rondas seguidas”

Tu idea de que el agente aprenda sobre más de una ronda es totalmente coherente con este diseño.

## Cómo se soporta eso

Cada decisión genera un vector:

```python
x_t = encoder(response_t)["state_vector"]   # [D]
```

Luego la secuencia alimenta la LSTM:

```python
X = [x_1, x_2, ..., x_T]
```

con forma:

```python
[B, T, D]
```

donde `T` puede cubrir:

- varias decisiones dentro de una mano;
- varias manos por split;
- varias rondas seguidas;
- incluso varios resets si tú decides que la secuencia no se corta al final de una ronda.

## Recomendación práctica

Para tu primer diseño recurrente:

- secuencia = **todas las decisiones observadas durante 2 rondas seguidas**
- reset del hidden state:
  - no en cada mano;
  - sí cuando tú cortes la secuencia de entrenamiento.

Eso permite que la red use:
- el descarte observado de la ronda anterior;
- el progreso del shoe;
- la mesa actual y sus reglas.

---

# 11. Decisiones de implementación concretas

## 11.1 Lo que sí conviene hacer ya

1. Crear un `EncoderConfig`
2. Crear un `BaseBlackjackEncoder`
3. Implementar primero:
   - `HandFeatureEncoder`
   - `HandContextEncoder`
   - `InsuranceContextEncoder`
   - `RuleEncoder`
4. Después agregar:
   - `ObservedCardsHistoryEncoder`
   - `TemporalFeatureEncoder`
   - `OtherHandsEncoder`
5. Dejar `ExactShoeEncoder` solo para benchmark

---

## 11.2 Lo que no conviene hacer todavía

1. No usar Transformers para el encoder desde el inicio
2. No codificar `recent_actions` con demasiado lujo al principio
3. No meter `public_state` entero como shortcut
4. No usar `debug_state`
5. No cambiar la dimensionalidad del estado según el paso actual

---

# 12. Entregable esperado del encoder

Al final, el encoder debe dejar una interfaz clara como esta:

```python
encoded = encoder(response)

encoded["state_vector"]      # Tensor[D]
encoded["action_mask"]       # Tensor[6]
encoded["module_tensors"]    # dict para debug
```

Y para batch:

```python
batch_state      # [B, D]
batch_mask       # [B, 6]
```

Y para LSTM:

```python
batch_seq_state  # [B, T, D]
batch_seq_mask   # [B, T, 6]
batch_pad_mask   # [B, T]
```

---

# 13. Resumen ejecutivo

## Baseline simple
Usar `minimal_basic_strategy` con un encoder pequeño.

## Agente principal
Usar `table_realistic_default` con:
- mano actual,
- reglas,
- historia observable,
- temporalidad,
- contexto de split/insurance,
- progreso del shoe.

## Agente recurrente
El encoder produce un vector fijo por paso; la LSTM se encarga de integrar información entre decisiones y entre rondas.

## Generalización a distintas mesas
Esto depende críticamente de incluir `RuleEncoder`.

## Aprender agotamiento de cartas
Esto depende críticamente de incluir:
- `ObservedCardsHistoryEncoder`
- `DiscardSummaryEncoder`
- `TemporalFeatureEncoder`
o una LSTM que procese secuencias suficientemente largas.

---

# 14. Recomendación final de orden

1. Implementar `EncoderConfig`
2. Implementar `MinimalObservationEncoder`
3. Validar shapes
4. Implementar `RealisticObservationEncoder`
5. Validar consistencia en una mesa fija
6. Entrenar baseline feedforward
7. Pasar a secuencias para LSTM
8. Finalmente entrenar con randomización de reglas entre mesas
