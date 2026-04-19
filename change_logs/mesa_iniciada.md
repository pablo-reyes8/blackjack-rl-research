# Cambios mínimos al setup para soportar dos modos de mesa en Blackjack RL

## Objetivo

Definir **qué modificar mínimamente** en el environment actual para soportar dos regímenes de entrenamiento sin romper la modularidad:

1. **Modo toy / limpio**
   - la mesa empieza desde shoe nuevo;
   - el agente observa el historial desde el inicio;
   - útil para validar aprendizaje base.

2. **Modo realista parcial**
   - conocemos reglas de la mesa y número de mazos;
   - **no conocemos cuánto falta para el reshuffle**;
   - el agente puede seguir aprendiendo por conteo de cartas a partir de lo observado;
   - útil para el caso más realista antes de CV.

La meta es que **lo que cambie viva en la configuración e inicialización del ambiente**, no en la lógica principal del motor ni en el encoder ni en el loop de entrenamiento.

---

# 1. Principio general

El environment ya tiene buena separación entre:

- reglas de mesa (`BlackjackConfig`);
- observación (`ObservationConfig`);
- estado real del juego;
- historia observable. fileciteturn0file0L65-L101 fileciteturn0file3L42-L55

Entonces, la estrategia correcta es **no rehacer el motor**, sino agregar una **capa explícita de “starting state policy”** o política de arranque del shoe/mesa.

---

# 2. Qué queremos poder instanciar

Idealmente algo como esto:

```python
env = BlackjackEnvironment(
    config=BlackjackConfig(...),
    seed=42,
    start_state=StartStateConfig(
        mode="fresh_shoe",   # o "unknown_progress"
        ...
    ),
)
```

o equivalentemente:

```python
env = BlackjackEnvironment(
    config=BlackjackConfig(...),
    seed=42,
    episode_mode="fresh_shoe",
)
```

Pero si queremos mantenerlo limpio y escalable, es mejor crear un `StartStateConfig`.

---

# 3. Nueva pieza mínima: `StartStateConfig`

## Propuesta

Agregar un nuevo dataclass en config, separado de reglas y de observación:

```python
@dataclass(slots=True)
class StartStateConfig:
    mode: str = "fresh_shoe"   # "fresh_shoe" | "unknown_progress"
    randomize_before_first_round: bool = False
    min_burned_rounds: int = 0
    max_burned_rounds: int = 0
    hide_reshuffle_progress_from_observation: bool = False
    expose_initial_observed_history: bool = True
```

---

## Razón conceptual

Esto separa tres cosas que hoy no conviene mezclar:

### A. Reglas de mesa
Ejemplo:
- S17/H17
- DAS
- n_decks
- surrender

### B. Qué ve el agente
Ejemplo:
- historial observable
- reglas visibles
- discard summary
- temporal context

### C. Cómo arranca el episodio
Ejemplo:
- shoe nuevo
- mesa ya corrida
- historial parcial o total
- progreso real oculto o visible

La pieza nueva debe vivir en **C**, no en A ni en B.

---

# 4. Dos modos concretos a soportar

## 4.1 Modo 1: `fresh_shoe`

### Comportamiento
- al `reset()`, si se inicia episodio nuevo:
  - shoe recién mezclado;
  - historia observable limpia;
  - rounds_since_shuffle = 0;
  - el agente ve todo desde el inicio.

### Interpretación
Es el modo toy / limpio.

### Ventaja
- más fácil de aprender;
- útil para baseline;
- ideal para depuración.

### No requiere casi cambios
Este modo es muy cercano a lo que ya hace el env hoy.

---

## 4.2 Modo 2: `unknown_progress`

### Comportamiento deseado
- sabemos reglas de la mesa;
- sabemos `n_decks`;
- la mesa puede ya venir avanzada;
- **no se expone al agente cuánto falta para reshuffle**;
- el agente debe inferir información del shoe solo a partir de cartas observadas desde que empieza a mirar.

### Importante
Aquí hay dos variantes posibles:

#### Variante A: mesa avanzada pero historial observable arranca en cero
El agente entra “a mitad de shoe” pero solo sabe lo que ve desde ese momento.

Esto es realista y duro.

#### Variante B: mesa avanzada y además recibe historial visible acumulado desde antes
Eso sería menos realista si el sistema no estuvo observando antes.

### Recomendación
Para tu caso:
usar la **Variante A** como principal.

---

# 5. Qué cambiar mínimamente en el environment

---

## 5.1 Agregar `start_state` al constructor del environment

### Hoy
```python
class BlackjackEnvironment:
    def __init__(self, config: BlackjackConfig | None = None, seed: int | None = None) -> None:
        ...
```

### Cambio mínimo
```python
class BlackjackEnvironment:
    def __init__(
        self,
        config: BlackjackConfig | None = None,
        seed: int | None = None,
        start_state: StartStateConfig | None = None,
    ) -> None:
        ...
```

Y guardar:

```python
self.start_state = start_state or StartStateConfig()
```

### Razón
No afecta el resto del motor.  
Solo introduce una política explícita de arranque.

---

## 5.2 Crear un hook privado de inicialización del episodio

Agregar un método como:

```python
def _prepare_episode_start(self) -> None:
    ...
```

Este método se llama al inicio de `reset()` antes del primer reparto real.

### Razón
Todo lo relacionado con “cómo arranca el episodio” queda concentrado en un solo lugar, en vez de regarse por `reset()`.

---

## 5.3 Implementar la lógica de `fresh_shoe` dentro del hook

```python
if self.start_state.mode == "fresh_shoe":
    self.shoe.shuffle()
    self._reshuffle_pending = False
    self._start_new_shoe_tracking(reason="episode_fresh_shoe", record_action=False)
    return
```

### Comentario
Esto deja explícito que el episodio empieza con shoe limpio.

---

## 5.4 Implementar la lógica de `unknown_progress` dentro del hook

La idea mínima no es reescribir el entorno, sino **quemar** una cantidad aleatoria de juego antes de la primera ronda visible.

### Estructura conceptual
```python
if self.start_state.mode == "unknown_progress":
    self.shoe.shuffle()
    self._reshuffle_pending = False
    self._start_new_shoe_tracking(reason="episode_unknown_progress", record_action=False)

    k = rng.randint(min_burned_rounds, max_burned_rounds)
    self._burn_hidden_rounds(k)
    self._clear_public_histories_after_hidden_burn()
    return
```

---

# 6. Nueva función mínima: `_burn_hidden_rounds(k)`

## Objetivo
Avanzar el shoe antes de que el agente empiece a jugar.

## Qué debe hacer
- simular `k` rondas ocultas;
- consumir cartas del shoe;
- opcionalmente usar una política dummy simple;
- actualizar el shoe real;
- actualizar contadores internos consistentes con el avance real.

## Qué NO debe hacer
- no debe dejar trazas visibles de esas rondas en:
  - `observed_cards_history`
  - `public_action_history`
  - `last_round_outcome`

## Razón
El agente no observó esas rondas; por tanto, no deberían contaminar su historia visible.

---

# 7. Nueva función mínima: `_clear_public_histories_after_hidden_burn()`

Después de quemar rondas ocultas, hay que limpiar lo que el agente no debería saber.

## Debe resetear al menos

```python
self.observed_cards_history = []
self.public_action_history = []
self.last_round_outcome = None
```

Y dependiendo del criterio, posiblemente también:

```python
self.rounds_since_shuffle = 0
self.player_hands_seen_since_shuffle = 0
self.dealer_hands_seen_since_shuffle = 0
```

### Recomendación importante
Para el modo `unknown_progress`, sí conviene resetear esos contadores visibles si van en la observación, porque si no le estarías filtrando progreso oculto del shoe.

---

# 8. Qué hacer con `estimated_shoe_progress`

Este punto es crítico.

Hoy el entorno puede construir:

- `fraction_used`
- `bucket` (`early`, `mid`, `late`) dentro de `temporal_context`. fileciteturn0file3L653-L663

Pero tú quieres un modo donde:

- conocemos reglas y mazos;
- **no sabemos cuánto falta para reshuffle**.

Entonces el cambio mínimo es:

## Opción recomendada
No cambiar la lógica real del shoe.  
Cambiar solo la observación.

### Es decir:
en `get_temporal_features()` o en `_build_estimated_shoe_progress()`:

- si `start_state.mode == "unknown_progress"` y `hide_reshuffle_progress_from_observation=True`,
  entonces **no incluir** `estimated_shoe_progress`.

o devolver:

```python
{"fraction_used": None, "bucket": None}
```

### Mejor aún
Controlarlo desde observación con un flag nuevo:

```python
obs_include_estimated_shoe_progress = False
```

para ese régimen.

### Conclusión
No hay que modificar el motor real del shoe; solo asegurarse de que esa señal no llegue al agente en ese modo.

---

# 9. Qué cambiar en `ObservationConfig`

## Cambio mínimo recomendado
No tocar mucho. Solo permitir un perfil adicional, o un override limpio.

### Opción A: nuevo perfil
Agregar algo como:

```python
"table_realistic_unknown_progress"
```

que sea igual a `table_realistic_default` excepto:

- `obs_include_estimated_shoe_progress = False`
- `obs_include_hands_since_shuffle = False` si esos conteos filtran progreso oculto
- mantener `obs_include_observed_cards_history = True`
- mantener `obs_include_discard_summary = True`

### Opción B: usar el mismo perfil con overrides
Esto es más minimalista.

Ejemplo:
```python
obs = ObservationConfig.for_profile("table_realistic_default")
obs.obs_include_estimated_shoe_progress = False
obs.obs_include_hands_since_shuffle = False
```

### Recomendación
Para mantener todo modular, prefiero **overrides** antes que multiplicar perfiles.

---

# 10. Qué cambiar en el encoder

## Respuesta corta
Casi nada.

Ese es justamente el objetivo de hacerlo modular.

### Para `fresh_shoe`
Puedes usar el encoder actual sin cambios.

### Para `unknown_progress`
Solo asegúrate de que el encoder acepte naturalmente que algunos campos estén ausentes o en cero:

- `estimated_shoe_progress`
- algunos contadores temporales

Tu encoder ya está bastante preparado para esto porque usa defaults seguros y funciones como `safe_float`, `safe_bool`, etc. fileciteturn1file8L11-L28

### Conclusión
No debes mover la arquitectura del encoder; solo la config de observación.

---

# 11. Qué cambiar en el loop de entrenamiento

## Respuesta corta
Tampoco demasiado.

La idea es que el loop solo cambie en la instanciación del env/config.

Ejemplo:

```python
env = BlackjackEnvironment(
    config=blackjack_config,
    seed=seed,
    start_state=StartStateConfig(mode="fresh_shoe"),
)
```

o

```python
env = BlackjackEnvironment(
    config=blackjack_config_unknown_progress,
    seed=seed,
    start_state=StartStateConfig(
        mode="unknown_progress",
        min_burned_rounds=5,
        max_burned_rounds=30,
        hide_reshuffle_progress_from_observation=True,
    ),
)
```

El bucle RL no debería enterarse de nada más.

---

# 12. Diseño mínimo recomendado de API

## 12.1 Config nueva

```python
@dataclass(slots=True)
class StartStateConfig:
    mode: str = "fresh_shoe"  # fresh_shoe | unknown_progress
    min_burned_rounds: int = 0
    max_burned_rounds: int = 0
    clear_visible_histories_after_burn: bool = True
    hide_reshuffle_progress_from_observation: bool = False
```

---

## 12.2 Constructor del env

```python
env = BlackjackEnvironment(
    config=BlackjackConfig(...),
    seed=123,
    start_state=StartStateConfig(...),
)
```

---

## 12.3 Hook interno

```python
def _prepare_episode_start(self) -> None:
    ...
```

---

## 12.4 Helper interno

```python
def _burn_hidden_rounds(self, n_rounds: int) -> None:
    ...
```

---

## 12.5 Helper interno

```python
def _clear_public_histories_after_hidden_burn(self) -> None:
    ...
```

---

# 13. Qué NO cambiar

Para mantener el sistema limpio, no conviene tocar:

1. La mecánica principal de `step()`
2. La lógica de settlement
3. La legalidad de acciones
4. La estructura base del encoder
5. El replay buffer o batching
6. La red DQN/LSTM por culpa de este cambio

El nuevo comportamiento debe vivir en:
- config de arranque,
- observación,
- reset del ambiente.

---

# 14. Resumen exacto de cambios mínimos

## Cambios obligatorios

### A. Nueva config
Agregar `StartStateConfig`.

### B. Nuevo parámetro del env
Agregar `start_state` al constructor.

### C. Nuevo hook
Agregar `_prepare_episode_start()` llamado dentro de `reset()`.

### D. Nuevo helper
Agregar `_burn_hidden_rounds(k)`.

### E. Limpieza de historia visible
Agregar `_clear_public_histories_after_hidden_burn()`.

### F. Observación
En el modo `unknown_progress`, ocultar:
- `estimated_shoe_progress`
- y posiblemente contadores desde shuffle si filtran progreso oculto.

---

# 15. Resumen conceptual final

## Modo toy
- shoe limpio;
- historia completa desde el inicio;
- más fácil.

## Modo realista parcial
- reglas conocidas;
- mazos conocidos;
- progreso real del shoe no observable;
- el agente aprende contando cartas desde que empieza a observar;
- más realista.

## Ventaja del diseño propuesto
Con estos cambios:

- el **motor del blackjack sigue limpio**;
- el **encoder no se rompe**;
- el **loop RL casi no cambia**;
- toda la diferencia entre problemas vive en:
  - `StartStateConfig`
  - `ObservationConfig`
  - `reset()`

---

# 16. Recomendación final de implementación

Orden sugerido:

1. Crear `StartStateConfig`
2. Agregar `start_state` a `BlackjackEnvironment`
3. Implementar `_prepare_episode_start()`
4. Implementar `fresh_shoe`
5. Implementar `unknown_progress` con `burn_hidden_rounds`
6. Limpiar historia visible tras el burn
7. Ocultar `estimated_shoe_progress` en observación para ese modo
8. Reusar el encoder actual sin cambios estructurales

Si este diseño queda bien, lo siguiente natural es escribir la especificación exacta de `_burn_hidden_rounds()`:
- qué política dummy usar,
- qué contadores sí actualizar,
- cuáles reiniciar,
- y cómo evitar inconsistencias con el shoe real.