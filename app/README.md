# Blackjack RL Casino App

Interfaz local tipo casino para jugar contra el ambiente real de `enviroment_bj` con recomendaciones del agente entrenado.

## Ejecutar

Desde la raiz del repo:

```bash
python3 app/server.py --model 05A --port 8765
```

Abrir en el navegador:

```text
http://127.0.0.1:8765
```

El primer request puede tardar unos segundos porque carga PyTorch, reconstruye el modelo con `run_blackjack_stage(run_training=False)` y carga el checkpoint.

## Modelos disponibles

- `05A`: `outputs/models/KEEP_05A_count_aux_representation_acc0815.pt`
- `04D`: `outputs/models/KEEP_04D_betting_weighted_ce_best.pt`

Para iniciar directamente con 04D:

```bash
python3 app/server.py --model 04D --port 8765
```

## Interfaz

- Mesa fija tipo casino: dealer arriba, jugador abajo, shoe/discard en bandeja lateral.
- Acciones separadas entre `Apuesta` y `Jugada`.
- `Jugar sugerida`: ejecuta la accion greedy del agente en el estado actual.
- `Auto mano`: deja que el agente termine la mano actual.
- `Auto continuo`: juega manos sucesivas hasta que se presione de nuevo.
- `Siguiente mano`: conserva el mismo shoe, historial observado y memoria de mesa.
- `Nueva mesa`: reinicia la sesion y puede alternar entre 05A y 04D.

La recomendacion del agente aparece en el panel lateral y la accion recomendada se marca con dorado/estrella en los botones. Las pintas de las cartas son solo visuales: el ambiente y el modelo siguen usando los ranks que entrega `enviroment_bj`.

## Paneles

- Recomendacion: Q-values por accion y accion greedy del agente.
- Conteo auxiliar: bucket estimado por la cabeza auxiliar cuando el checkpoint la tiene.
- Score: ganadas, perdidas, push y EV/100 de la sesion.
- Bankroll: reward acumulado, rondas, promedio y seed.

## API local

- `GET /api/state`
- `POST /api/action` con `{"action": "hit"}` o cualquier accion legal.
- `POST /api/play-suggestion`
- `POST /api/autoplay`
- `POST /api/new-round`
- `POST /api/new-table`
