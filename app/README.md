# Blackjack RL Casino App

App local para jugar contra el ambiente real de `enviroment_bj` con sugerencias del checkpoint entrenado.

```bash
python3 app/server.py --model 05A --port 8765
```

Abre:

```text
http://127.0.0.1:8765
```

Modelos incluidos:

- `05A`: `outputs/models/KEEP_05A_count_aux_representation_acc0815.pt`
- `04D`: `outputs/models/KEEP_04D_betting_weighted_ce_best.pt`

El primer request carga PyTorch y reconstruye el modelo desde el checkpoint, por eso puede tardar unos segundos.
