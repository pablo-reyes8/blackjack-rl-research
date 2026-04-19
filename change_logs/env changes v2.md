# Diseño de observación del agente: opciones configurables

## Objetivo
El ambiente debe permitir controlar explícitamente **qué información ve el agente** mediante flags al momento de crear la mesa.  
La idea es soportar distintos niveles de observabilidad, desde un simulador totalmente informado hasta una configuración más realista pensada para un futuro **scraper visual**.

---

# 1. Principio de diseño

## Regla general
- El motor puede conocer toda la verdad del juego.
- El agente solo debe recibir el subconjunto de información definido por los flags de observación.
- La `action_mask` se mantiene separada y siempre la controla el ambiente.

## Meta
- Poder entrenar distintos agentes bajo distintos supuestos de observación.
- Mantener como configuración por defecto una versión **realista y compatible con un scraper visual**.

---

# 2. Capas de información

## A. Estado interno del motor
Información completa, no necesariamente visible al agente:
- hole card del dealer
- shoe completo
- composición exacta restante
- cartas no visibles
- flags internos de resolución

## B. Información pública de mesa
Información visible o razonablemente inferible:
- cartas del jugador
- carta visible del dealer
- manos activas
- progreso observable de la ronda
- descarte observado o resumen de cartas vistas
- reglas visibles o anunciadas por la mesa

## C. Observación efectiva del agente
Subconjunto final que realmente entra a la red.

---

# 3. Flags recomendados de observación

## Flags de reglas de mesa

### `obs_include_table_rules`
- Incluye un bloque `table_rules` en la observación del agente.
- Default recomendado: `False`

### `obs_include_visible_rules_only`
- Si `True`, solo pasa reglas razonablemente visibles o conocidas por un jugador real.
- Default recomendado: `True`

### `obs_include_hidden_rules`
- Si `True`, pasa reglas internas no necesariamente visibles de forma directa.
- Ejemplo: `n_decks` exacto, `shoe_penetration`.
- Default recomendado: `False`

---

## Flags de estado actual de la mano

### `obs_current_hand_mode`
Opciones sugeridas:
- `"basic_strategy"`:
  - total actual
  - soft/hard
  - upcard dealer
- `"table_raw"`:
  - cartas exactas de la mano actual
  - otras manos visibles
  - upcard dealer
- Default recomendado: `"table_raw"`

### `obs_include_other_player_hands`
- Incluye las otras manos activas del jugador después de split.
- Default recomendado: `True`

### `obs_include_current_bet`
- Incluye apuesta actual de la mano.
- Default recomendado: `True`

### `obs_include_hand_context`
- Incluye:
  - `current_hand_index`
  - `n_player_hands`
  - `from_split`
  - `split_aces`
  - `first_decision_on_hand`
- Default recomendado: `True`

### `obs_include_insurance_context`
- Incluye si hay oferta activa de insurance.
- Default recomendado: `True`

---

## Flags de contexto temporal

### `obs_include_temporal_context`
- Activa features temporales agregadas.
- Default recomendado: `True`

### `obs_include_hands_since_shuffle`
- Incluye número de manos jugadas desde el último reshuffle.
- Fácil de calcular y razonablemente útil.
- Default recomendado: `True`

### `obs_include_estimated_shoe_progress`
- Incluye una medida resumida del avance del shoe.
- Ejemplo:
  - proporción consumida
  - bucket discreto de progreso (`early`, `mid`, `late`)
- Default recomendado: `True`

### `obs_include_last_hand_outcome`
- Incluye resultado de la mano anterior.
- Puede ayudar a modelar secuencias, aunque no es estrictamente necesario.
- Default recomendado: `False`

### `obs_include_recent_actions`
- Incluye una ventana corta de acciones recientes.
- Ejemplo: últimas 3 o 5 acciones.
- Default recomendado: `False`

---

## Flags de historial de cartas observadas

### `obs_include_observed_cards_history`
- Activa historial de cartas observadas públicamente.
- Default recomendado: `True`

### `obs_observed_cards_mode`
Opciones sugeridas:
- `"rank_counts"`:
  - conteo acumulado por rango observado
  - Ejemplo: cuántos A, 2, ..., K han sido vistos
- `"low_neutral_high"`:
  - resumen agregado:
    - bajas `2-6`
    - neutrales `7-9`
    - altas `10-A`
- `"recent_cards_sequence"`:
  - secuencia corta de cartas observadas recientemente
- Default recomendado: `"rank_counts"`

### `obs_recent_cards_window`
- Número de cartas recientes si se usa modo secuencial.
- Default recomendado: `20`

### `obs_reset_history_on_shuffle`
- Reinicia historial observado al reshuffle.
- Default recomendado: `True`

---

## Flags de shoe y descarte

### `obs_include_exact_shoe_composition`
- Incluye composición exacta restante del shoe.
- Muy útil para simulador fully observable.
- Poco realista para scraper visual.
- Default recomendado: `False`

### `obs_include_discard_summary`
- Incluye resumen del descarte observado en lugar del shoe exacto.
- Más realista visualmente.
- Default recomendado: `True`

### `obs_include_n_decks`
- Incluye número exacto de mazos.
- Realista solo si quieres asumir que la mesa lo anuncia o el sistema lo conoce.
- Default recomendado: `False`

### `obs_include_shoe_penetration_rule`
- Incluye parámetro exacto de penetración configurada.
- Poco realista como input directo.
- Default recomendado: `False`

---

# 4. Qué reglas sí debería poder ver el agente

Estas son razonables de pasar si quieres una mesa realista pero no ciega:

- `dealer_hits_soft_17`
- `blackjack_payout`
- `double_allowed_on`
- `double_after_split_allowed`
- `split_rule`
- `surrender_allowed`
- `insurance_allowed`

Estas reglas suelen ser:
- conocidas por la mesa
- anunciadas
- o inferibles tras pocas manos

---

# 5. Qué cosas no debería ver por default

Estas no deberían ir por defecto al agente en una configuración realista:

- `n_decks` exacto
- `shoe_penetration` exacta
- composición exacta restante del shoe
- hole card del dealer
- cualquier variable tipo:
  - `can_hit`
  - `can_double`
  - `can_split`
  - `can_surrender`

La legalidad puntual de acciones debe seguir viniendo por `action_mask`, no como feature del estado.

---

# 6. Qué cosas son plausibles desde un scraper visual

Estas features son razonables y no “locas” si piensas en visión o scraping temporal:

## Muy plausibles
- cartas actuales del jugador
- upcard del dealer
- otras manos activas del jugador
- apuesta actual
- insurance activa
- número de manos activas
- historial de cartas observadas
- resultado de manos previas
- progreso estimado del shoe
- tamaño relativo del descarte visible

## Plausibles con algo más de trabajo
- secuencia reciente de cartas observadas
- conteo agregado por grupos de cartas
- estimación de si el shoe está en fase temprana/media/tardía

## Poco realistas como input directo
- composición exacta restante del shoe
- número exacto de mazos si la mesa no lo muestra
- penetración exacta configurada internamente

---

# 7. Configuración por defecto recomendada

## Objetivo del default
La configuración default debe ser la **más realista pensada para un scraper visual**, sin volver el problema innecesariamente difícil.

## Default recomendado

- `obs_include_table_rules = True`
- `obs_include_visible_rules_only = True`
- `obs_include_hidden_rules = False`

- `obs_current_hand_mode = "table_raw"`
- `obs_include_other_player_hands = True`
- `obs_include_current_bet = True`
- `obs_include_hand_context = True`
- `obs_include_insurance_context = True`

- `obs_include_temporal_context = True`
- `obs_include_hands_since_shuffle = True`
- `obs_include_estimated_shoe_progress = True`
- `obs_include_last_hand_outcome = False`
- `obs_include_recent_actions = False`

- `obs_include_observed_cards_history = True`
- `obs_observed_cards_mode = "rank_counts"`
- `obs_reset_history_on_shuffle = True`

- `obs_include_exact_shoe_composition = False`
- `obs_include_discard_summary = True`
- `obs_include_n_decks = False`
- `obs_include_shoe_penetration_rule = False`

---

# 8. Perfiles de observación sugeridos

## Perfil 1: `fully_observable_sim`
Pensado para experimentos controlados.

- incluye `table_rules` completos
- incluye `n_decks`
- incluye `shoe_penetration`
- incluye composición exacta del shoe
- no realista, pero útil para upper bound

## Perfil 2: `table_realistic_default`
Pensado como default del proyecto.

- reglas visibles de mesa
- cartas visibles actuales
- contexto de mano
- historial de cartas observadas
- progreso estimado del shoe
- sin composición exacta
- sin `n_decks` exacto
- sin penetración exacta

## Perfil 3: `minimal_basic_strategy`
Pensado para baseline rápido.

- total de mano
- soft/hard
- upcard dealer
- flags mínimos de contexto
- sin historial temporal
- sin reglas ocultas

---

# 9. Recomendación de implementación

## Crear configuración explícita de observación
Ejemplo conceptual:

- `ObservationConfig`
- integrada dentro de `BlackjackConfig`
- o pasada por separado al ambiente

## Métodos sugeridos
- `get_agent_observation()`
- `get_visible_table_rules()`
- `get_temporal_features()`
- `get_observed_cards_summary()`

## Regla de implementación
- el ambiente construye todo internamente
- luego filtra según flags
- la observación final del agente se arma a partir de esos flags

---

# 10. Decisión de diseño recomendada

## Mantener siempre separados
- `observation`
- `table_rules`
- `action_mask`
- `public_state`
- `debug_state`

## Default final recomendado
La observación por defecto debe representar:

- **lo que una mesa real deja ver**
- **lo que un scraper visual temporal podría reconstruir con relativa facilidad**
- **y un resumen temporal suficiente para que memoria/LSTM tenga señal útil**

Ese default debe priorizar:
- cartas visibles actuales
- reglas visibles de mesa
- historial observado de cartas
- progreso aproximado del shoe

y no debe priorizar:
- reglas internas exactas
- composición perfecta del shoe
- features derivadas de legalidad