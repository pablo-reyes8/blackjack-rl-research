# Mejoras pendientes para blindar el ambiente de Blackjack

## 1. Tests deterministas
- Crear tests con `load_shoe()` para escenarios críticos:
  - blackjack jugador
  - blackjack dealer
  - push de blackjack
  - insurance ganado/perdido
  - split de 8s
  - split de ases
  - resplit de ases
  - double after split
  - surrender
  - dealer H17 / S17
  - reshuffle por penetración
- Verificar en cada test:
  - cartas finales
  - acciones legales
  - reward final
  - settlement por mano
  - estado del dealer
  - estado del shoe

## 2. Separar claramente tres capas
### A. Estado interno del motor
- Mantener un `engine_state` completo con toda la verdad del juego:
  - shoe completo
  - hole card del dealer
  - todas las manos
  - apuestas
  - flags de split/double/surrender/insurance
  - estado de la ronda

### B. Observación del agente
- Crear una `agent_observation` separada.
- La observación solo debe contener información que el agente realmente puede usar como input.

### C. Restricciones de acción
- Mantener una `action_mask` separada.
- La máscara no debe mezclarse con la observación como feature principal.

## 3. Quitar features derivadas del input del agente
- No pasar como input de la red:
  - `can_hit`
  - `can_double`
  - `can_split`
  - `can_surrender`
  - `pair_for_split`
  - `legal_actions`
- Estas variables sí pueden existir internamente para construir la máscara de acciones.

## 4. Mantener máscara de acciones
- Conservar `action_mask` para filtrar acciones ilegales.
- La red puede producir Q-values para todas las acciones.
- Antes de elegir acción:
  - anular acciones ilegales con la máscara
  - o asignarles `-inf`
- La máscara es parte del control del ambiente, no del conocimiento experto del agente.

## 5. Agregar reglas de mesa como input explícito
- Si el objetivo es jugar en diferentes mesas, el agente sí debe conocer la configuración de la mesa.
- Crear un bloque `table_rules` separado del estado de la mano.
- Incluir como mínimo:
  - `n_decks`
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
- Esto permite que el agente aprenda políticas distintas según la mesa.

## 6. Definir observación mínima del agente
- Dejar como input base:
  - cartas de la mano actual
  - carta visible del dealer
  - índice de mano actual
  - número de manos activas
  - apuesta actual
  - `from_split`
  - `split_aces`
  - `first_decision_on_hand`
  - `insurance_offer_active`
- Opcional:
  - resumen del shoe
  - cartas visibles de las otras manos del jugador
- No incluir la hole card del dealer.

## 7. Crear dos modos de observación
### `basic_strategy`
- Estado compacto para que aprenda rápido.
- Ejemplo:
  - total del jugador
  - soft/hard
  - upcard dealer
  - primera decisión
  - si viene de split
  - insurance activa
  - reglas de la mesa

### `table_raw`
- Estado más crudo y cercano a una mesa real.
- Ejemplo:
  - cartas exactas de la mano actual
  - cartas visibles de otras manos propias
  - upcard dealer
  - índice de mano
  - apuesta actual
  - insurance activa
  - reglas de la mesa

## 8. Formalizar qué aprende el agente
- El agente no debe aprender si una acción es legal.
- La legalidad la controla el ambiente con `action_mask`.
- El agente debe aprender:
  - cuándo conviene hit
  - cuándo conviene stand
  - cuándo conviene double
  - cuándo conviene split
  - cuándo conviene surrender
  - cuándo conviene tomar insurance
- Condicionado por:
  - estado visible de la mesa
  - reglas de la mesa

## 9. Validar más fuerte el shoe
- Verificar que todas las cartas en `load_shoe()` sean válidas.
- Verificar que `total_cards >= len(cards)`.
- Agregar modo estricto opcional para composición consistente con `n_decks`.
- Verificar que no se puedan cargar shoes imposibles por error.

## 10. Congelar reglas por escrito
- Documentar explícitamente:
  - peek rule
  - insurance
  - surrender
  - split rule
  - split aces
  - resplit aces
  - payout de blackjack
  - H17 / S17
  - DAS
  - reshuffle
- Toda variante debe venir de `table_rules`, no de lógica escondida.

## 11. Logging por transición
- Guardar por paso:
  - observación antes
  - acción elegida
  - máscara de acciones
  - carta robada
  - observación después
  - cierre de mano
  - reward de la mano
  - reward acumulado
  - settlement parcial/final
- Esto servirá para depurar Bellman, targets y episodios raros.

## 12. Mantener reward monetario limpio
- No meter reward shaping raro.
- Conservar reward final basado en:
  - win/loss/push
  - blackjack
  - double
  - split
  - surrender
  - insurance
- Dejar explícito si el reward se entrega:
  - solo al final de la ronda
  - o parcialmente por mano

## 13. Diseñar API estable del ambiente
- Definir claramente métodos como:
  - `reset()`
  - `step(action)`
  - `legal_actions()` o `action_mask()`
  - `get_agent_observation()`
  - `get_table_rules()`
  - `get_debug_state()`
- La salida debe separar explícitamente:
  - observación del agente
  - reglas de mesa
  - máscara de acciones
  - reward
  - done
  - info/debug

## 14. Estructura final recomendada por paso
- Cada paso del ambiente debería devolver algo conceptualmente así:

```python
{
    "observation": {...},      # estado visible de la mesa
    "table_rules": {...},      # configuración de la mesa
    "action_mask": [...],      # acciones legales
    "reward": float,
    "done": bool,
    "info": {...}              # debug / trazabilidad
}
```

## 15. Prioridades inmediatas
1. Escribir tests deterministas.
2. Separar `engine_state`, `agent_observation`, `table_rules` y `action_mask`.
3. Quitar `can_*` del input del agente.
4. Agregar `table_rules` como input explícito.
5. Congelar la API del ambiente y documentar reglas exactas.


## 16. Generalización a diferentes mesas
- El objetivo no será entrenar una política para una sola mesa fija.
- El ambiente debe permitir variar reglas entre episodios.
- El agente debe recibir un bloque `table_rules` como parte del input.
- La legalidad puntual de cada acción se sigue controlando con `action_mask`.

### Idea central
- Si el agente entrena solo con una configuración fija, aprenderá una política específica para esa mesa.
- Si después cambian reglas como `split_rule`, `dealer_hits_soft_17` o `double_after_split_allowed`, la política puede quedar subóptima.
- Para evitar reentrenar desde cero, el entrenamiento debe incluir múltiples configuraciones de mesa.

### Qué debe variar entre episodios
- `split_rule`
- `double_allowed_on`
- `double_after_split_allowed`
- `dealer_hits_soft_17`
- `surrender_allowed`
- `insurance_allowed`
- `blackjack_payout`
- `n_decks`

### Resultado esperado
- El agente no aprende “una sola estrategia de blackjack”.
- El agente aprende una política condicionada por:
  - el estado visible de la mano
  - las reglas de la mesa actual