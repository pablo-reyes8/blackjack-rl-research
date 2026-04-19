# Blackjack Environment Rules

## Regla base de mesa

La mesa base sigue un blackjack de casino parametrizable con estas reglas por defecto:

- `n_decks = 6`
- `shoe_penetration = 0.8`
- `dealer_hits_soft_17 = False` (`S17`)
- `blackjack_payout = 1.5` (`3:2`)
- `dealer_peeks_for_blackjack = True`
- `double_allowed_on = "any_two_cards"`
- `double_after_split_allowed = True` (`DAS`)
- `split_rule = "same_value"`
- `max_hands_after_split = 4`
- `resplit_aces_allowed = True`
- `hit_split_aces_allowed = False`
- `surrender_allowed = True`
- `insurance_allowed = True`
- `reward_mode = "round_end"`

## Reglas fijas del motor

- El shoe reparte sin reemplazo.
- El shoe persiste entre manos y solo se reshuffla al final de una ronda cuando se cruza la penetración.
- El reparto inicial es:
  - jugador
  - dealer visible
  - jugador
  - dealer oculta
- Blackjack natural solo cuenta con 2 cartas y sin venir de split.
- El dealer:
  - pide con menos de 17
  - se planta con mas de 17
  - en 17 suave depende de `dealer_hits_soft_17`
- `insurance` es una acción explícita.
- El reward es monetario limpio y se entrega al final de la ronda.

## Reglas de mesa parametrizables

Toda variación importante vive en `BlackjackConfig` y puede cambiar entre episodios:

- `n_decks`
- `shoe_penetration`
- `dealer_hits_soft_17`
- `blackjack_payout`
- `dealer_peeks_for_blackjack`
- `double_allowed_on`
- `double_after_split_allowed`
- `split_rule`
- `max_hands_after_split`
- `resplit_aces_allowed`
- `hit_split_aces_allowed`
- `surrender_allowed`
- `insurance_allowed`
- `base_bet`

## Modos de arranque de mesa

El arranque del episodio no vive en `BlackjackConfig`, sino en `StartStateConfig`.

### `fresh_shoe`

- el primer round visible empieza con shoe nuevo
- el agente observa la mesa desde el inicio del shoe
- útil para baseline y depuración

### `unknown_progress`

- la mesa puede arrancar ya avanzada dentro del shoe
- antes del primer round visible se queman rondas ocultas
- el shoe real sí avanza
- el historial visible del agente se limpia después del burn
- el agente solo puede reconstruir el pasado desde que empieza a observar

Config mínima:

```python
StartStateConfig(
    mode="unknown_progress",
    min_burned_rounds=5,
    max_burned_rounds=30,
    clear_visible_histories_after_burn=True,
    hide_reshuffle_progress_from_observation=True,
)
```

## Capas separadas del ambiente

Cada `reset()` y `step(action)` devuelve:

```python
{
    "observation": {...},         # input efectivo del agente
    "table_rules": {...},        # reglas visibles para el agente
    "action_mask": [...],        # mascara discreta alineada a ACTION_ORDER
    "action_mask_by_name": {...},
    "reward": float,
    "done": bool,
    "info": {
        "public_state": {...},   # estado visible para UI/debug
        "last_transition": {...} # trazabilidad del ultimo paso
    }
}
```

La verdad completa del juego queda en `get_debug_state()`.

## Configuración explícita de observación

El agente no ve toda la verdad del motor. Lo que entra a la red se controla con `ObservationConfig`.

### Flags principales

- `obs_include_table_rules`
- `obs_include_visible_rules_only`
- `obs_include_hidden_rules`
- `obs_current_hand_mode`
- `obs_include_other_player_hands`
- `obs_include_current_bet`
- `obs_include_hand_context`
- `obs_include_insurance_context`
- `obs_include_temporal_context`
- `obs_include_hands_since_shuffle`
- `obs_include_estimated_shoe_progress`
- `obs_include_last_hand_outcome`
- `obs_include_recent_actions`
- `obs_recent_actions_window`
- `obs_include_observed_cards_history`
- `obs_observed_cards_mode`
- `obs_recent_cards_window`
- `obs_reset_history_on_shuffle`
- `obs_include_exact_shoe_composition`
- `obs_include_discard_summary`
- `obs_include_n_decks`
- `obs_include_shoe_penetration_rule`

## Perfil default del proyecto

El default es `table_realistic_default`, pensado para una mesa observable por scraper visual y útil para modelos con memoria.

Incluye por defecto:

- cartas visibles actuales del jugador
- upcard del dealer
- otras manos propias visibles tras split
- apuesta actual
- contexto de la mano actual
- contexto de insurance
- historial observado de cartas por rango
- resumen visible del descarte
- contexto temporal entre manos
- progreso estimado del shoe
- reglas visibles de la mesa

No incluye por defecto:

- hole card del dealer
- composición exacta del shoe
- `n_decks` exacto
- penetración exacta
- features derivadas de legalidad como `can_hit` o `can_double`

## Perfil adicional para mesa ya iniciada

Existe `table_realistic_unknown_progress` para usar junto con `StartStateConfig(mode="unknown_progress")`.

Es equivalente al perfil realista principal, pero por defecto:

- no expone `estimated_shoe_progress`
- no expone contadores `*_since_shuffle`
- mantiene historial observado de cartas
- mantiene discard summary

El encoder soporta este perfil sin cambios estructurales.

## Perfiles incluidos

### `table_realistic_default`

Pensado como default realista.

- `table_rules` visibles
- modo `table_raw`
- historial de cartas observadas
- progreso estimado del shoe
- sin composición exacta del shoe

### `fully_observable_sim`

Pensado para upper bound y simulación totalmente informada.

- incluye reglas ocultas
- incluye `n_decks`
- incluye `shoe_penetration`
- incluye composición exacta restante del shoe

### `minimal_basic_strategy`

Pensado para baseline rápido.

- total actual
- soft/hard
- upcard dealer
- contexto mínimo de mano
- sin historial temporal
- sin historial de cartas observadas

## Contexto temporal persistente

La observación puede incluir memoria agregada entre manos. Esto permite entrenar agentes secuenciales que no traten cada mano como i.i.d.

Las features temporales disponibles incluyen:

- `rounds_since_shuffle`
- `player_hands_seen_since_shuffle`
- `dealer_hands_seen_since_shuffle`
- `player_hands_seen_total`
- `dealer_hands_seen_total`
- `estimated_shoe_progress`
- `last_round_outcome`
- `recent_actions`

Esto soporta modelos con memoria tipo LSTM que exploten el hecho de que el blackjack depende del historial del shoe y no solo de la mano actual.

## Historial observable de cartas

El motor mantiene `observed_cards_history` con cartas que un jugador real puede haber visto:

- cartas del jugador
- upcard del dealer
- cartas de split
- cartas robadas visibles
- hole card del dealer cuando se revela

Modos disponibles:

- `rank_counts`
- `low_neutral_high`
- `recent_cards_sequence`

## Historial público de acciones

El motor también mantiene `public_action_history` con acciones visibles o razonablemente inferibles:

- `table:deal_round`
- `dealer:offer_insurance`
- `player:hit`
- `player:stand`
- `player:double`
- `player:split`
- `player:surrender`
- `player:insurance`
- `dealer:reveal_hole`
- `dealer:hit`
- `dealer:stand`
- `table:settle_round`
- `table:reshuffle`

Si `obs_include_recent_actions=True`, el agente recibe una ventana corta de este historial.

## Reglas visibles vs ocultas

Por defecto el agente solo recibe reglas razonablemente visibles:

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
- `base_bet`

Reglas internas o menos visibles quedan fuera salvo que el perfil lo permita:

- `dealer_peeks_for_blackjack`
- `n_decks`
- `shoe_penetration`

## Validación de `load_shoe()`

Siempre valida:

- rangos válidos
- `total_cards >= len(cards)`
- que no haya más cartas cargadas que las posibles para `n_decks`

Con `strict_shoe_validation=True` o `load_shoe(..., strict=True)` también valida:

- que `total_cards` no exceda la capacidad real de los mazos configurados
- que no existan conteos imposibles por rango
