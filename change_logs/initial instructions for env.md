# Especificación funcional de un ambiente de Blackjack canónico para RL

## Objetivo

Implementar un entorno de **Blackjack estilo casino** lo más canónico posible, con reglas realistas y parametrizables, de forma que un agente de Reinforcement Learning pueda aprender estrategia bajo condiciones cercanas a una mesa real.

---

# 1. Componentes generales del juego

## 1.1 Número de barajas
El entorno debe permitir configurar:

- `n_decks`: número de barajas en el shoe.
- Valor típico de casino:
  - 6 barajas
  - 8 barajas

## 1.2 Shoe y agotamiento de cartas
El entorno debe modelar que:

- Las cartas **no se reponen después de cada mano**.
- Se extraen de un **shoe** común.
- El shoe se va agotando progresivamente.
- Debe existir un umbral de **penetración** tras el cual se reshufflea.

Parámetros sugeridos:

- `shoe_penetration`: proporción usada antes de barajar de nuevo.
  - típico: `0.75`, `0.80`, `0.85`

Regla:

- Si el número de cartas restantes cae por debajo del umbral, al finalizar la mano actual:
  - se reconstruye y baraja el shoe completo.

Esto es importante para:
- realismo de casino
- conteo implícito de cartas
- no i.i.d. entre manos

---

# 2. Representación de cartas

## 2.1 Rangos
Cartas posibles:

- A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K

## 2.2 Valores
- 2–10 valen su número
- J, Q, K valen 10
- As vale:
  - 11 si no hace bust
  - 1 en caso contrario

## 2.3 Palos
Los palos normalmente **no importan** en blackjack.
Puedes ignorarlos completamente salvo que quieras simular una baraja explícita.

---

# 3. Inicio de la mano

## 3.1 Reparto inicial
Al inicio de cada ronda:

- El jugador recibe 2 cartas
- El dealer recibe 2 cartas:
  - 1 carta visible (`upcard`)
  - 1 carta oculta (`hole card`)

Orden típico:
1. jugador
2. dealer visible
3. jugador
4. dealer oculta

## 3.2 Información observable
El estado debe permitir observar, como mínimo:

- mano del jugador
- valor total actual
- si la mano es soft o hard
- carta visible del dealer
- si la mano del jugador puede dividirse
- si puede doblar
- composición restante del shoe (opcional, si quieres un entorno totalmente observable)

---

# 4. Evaluación de manos

## 4.1 Hard hand
Una mano es **hard** cuando no hay un As contando como 11 sin pasarse.

Ejemplos:
- 10 + 6 = 16 hard
- A + 9 + 7 = 17 hard

## 4.2 Soft hand
Una mano es **soft** cuando un As está contando como 11 sin bust.

Ejemplos:
- A + 6 = 17 soft
- A + 3 + 5 = 19 soft

## 4.3 Blackjack natural
Definición:

- Una mano inicial de 2 cartas que suma 21:
  - A + 10
  - A + J
  - A + Q
  - A + K

Debe distinguirse de un 21 obtenido con más de 2 cartas.

---

# 5. Acciones del jugador

El entorno debe soportar estas acciones canónicas.

## 5.1 Stand
El jugador se planta y termina su turno sobre esa mano.

## 5.2 Hit
El jugador pide una carta adicional.

Regla:
- Puede repetir hit hasta:
  - plantarse
  - bust
  - llegar a una restricción específica del casino

## 5.3 Double Down
El jugador:

- duplica su apuesta en esa mano
- recibe exactamente **una sola carta adicional**
- queda automáticamente plantado después

Configuración:
- `double_allowed_on`: define cuándo puede doblar
  - típico canónico:
    - solo en las 2 primeras cartas
  - variantes:
    - solo en 9, 10, 11
    - en cualquier total inicial

Recomendación para versión casino típica:
- permitir doblar **solo con 2 cartas iniciales**
- y permitirlo sobre cualquier total inicial

## 5.4 Split
Si las dos cartas iniciales del jugador tienen el mismo valor de split, puede dividir la mano en dos.

Ejemplos típicos de split permitido:
- 8 + 8
- A + A
- K + Q si usas regla “mismo valor 10” (depende del casino)
- 10 + 10 normalmente sí por valor, aunque estratégicamente mala idea

Debes decidir entre estas dos políticas:

### Opción A: split por mismo rango exacto
Solo:
- 8+8, A+A, K+K, etc.

### Opción B: split por mismo valor
Permite:
- 10+J
- Q+K
- etc.

La opción más común en motores de simulación de blackjack es:
- **split por mismo rango**
o bien
- **split por mismo valor de carta**

Debes parametrizar esto:
- `split_rule = "same_rank"` o `"same_value"`

## 5.5 Reglas del split
Tras dividir:

- se crean 2 manos independientes
- cada nueva mano recibe una carta adicional
- el jugador juega una mano completa antes de pasar a la otra

Parámetros importantes:

- `max_splits`: número máximo de resplits
  - típico: 3 o 4 manos totales
- `resplit_aces_allowed`: si se pueden volver a dividir ases
- `hit_split_aces_allowed`: si se puede pedir hit tras dividir ases

Regla de casino muy típica:
- los ases divididos reciben **una sola carta cada uno**
- no se puede seguir pidiendo
- a veces no se permite re-split de ases o se permite solo una vez

---

# 6. Seguro (Insurance)

## 6.1 Cuándo se ofrece
Si la carta visible del dealer es un As, se puede ofrecer seguro.

## 6.2 Cómo funciona
- El jugador apuesta una apuesta lateral de hasta la mitad de la apuesta principal.
- Si el dealer tiene blackjack:
  - el seguro paga 2:1
- si no:
  - el seguro se pierde


---

# 7. Even Money

Si el jugador tiene blackjack natural y el dealer muestra As, algunos casinos permiten “even money”.

Funcionalmente equivale a cobrar 1:1 inmediatamente en lugar de esperar a ver si el dealer también tiene blackjack.

Esto puede modelarse como:
- una regla opcional
- o derivarse del sistema de seguro

---

# 8. Reglas del dealer (crupier)

## 8.1 Dealer revela la hole card
Una vez termina el turno del jugador (o cuando corresponda por regla de blackjack natural), el dealer revela su carta oculta.

## 8.2 Dealer debe pedir hasta cierto umbral
Regla canónica:

- el dealer debe pedir con total menor a 17

Es decir:
- hit en 16 o menos
- stand en 17 o más

## 8.3 Soft 17
Este punto es crucial y debe ser parámetro.

Dos variantes estándar:

- `dealer_hits_soft_17 = True`  → H17
- `dealer_hits_soft_17 = False` → S17

Interpretación:
- H17: el dealer pide con A+6
- S17: el dealer se planta con A+6

Ambas existen en casinos reales.

## 8.4 El dealer no toma decisiones estratégicas
El dealer sigue reglas fijas, no una política aprendida.

---

# 9. Resolución de la mano

## 9.1 Bust del jugador
Si el jugador supera 21:

- pierde inmediatamente esa mano
- no importa si luego el dealer también se pasa

## 9.2 Bust del dealer
Si el dealer supera 21 y el jugador no:

- el jugador gana

## 9.3 Comparación final
Si ni jugador ni dealer se pasan:

- mayor total gana
- igualdad = push
- menor total pierde

## 9.4 Blackjack natural
Debe pagarse distinto a un 21 normal.

Regla típica:
- blackjack natural paga `3:2`

Variantes modernas menos favorables:
- `6:5`
- `1:1`

Parámetro:
- `blackjack_payout = 1.5`  para 3:2

## 9.5 Push de blackjack
Si:
- jugador tiene blackjack natural
- dealer también tiene blackjack natural

Entonces:
- push

---

# 10. Apuestas

## 10.1 Apuesta base
Cada ronda inicia con una apuesta principal:
- `base_bet`

Para un entorno RL básico:
- puedes fijarla en 1 unidad

## 10.2 Doblar
Si el jugador hace double:
- la exposición de esa mano pasa a 2 unidades

## 10.3 Split
Cada mano nueva producto del split requiere:
- una nueva apuesta igual a la apuesta original

---

# 11. Orden de juego con múltiples manos

Cuando el jugador splittea:

1. se crean las manos hijas
2. se juega la primera mano completa
3. luego la segunda
4. y así sucesivamente

Cada mano debe almacenar:
- cartas
- apuesta asociada
- si fue doblada
- si proviene de split
- si es split de ases
- si ya está cerrada

---

# 12. Estados terminales

Una mano termina cuando ocurre cualquiera de estos eventos:

- stand
- bust
- double completado
- mano bloqueada tras split de ases
- dealer resuelve y compara resultados

Una ronda termina cuando:
- todas las manos del jugador fueron resueltas
- y se liquidaron pagos

---

# 13. Reglas opcionales pero muy importantes para realismo

## 13.1 Dealer hole-card / peek rule
Cuando el dealer muestra:
- As
- o 10

en muchos casinos revisa inmediatamente si tiene blackjack.

Parámetro:
- `dealer_peeks_for_blackjack = True`

Esto afecta:
- si el jugador puede perder apuestas extra de split/double antes de saber que el dealer ya tenía blackjack

## 13.2 Double after split (DAS)
Se debe parametrizar si se puede doblar después de dividir.

- `double_after_split_allowed = True/False`

Típico:
- muchos casinos sí lo permiten


---

# 14. Funcionalidades mínimas a implementar en el entorno RL

## 14.1 Gestión del shoe
- construir shoe con `n_decks`
- barajar
- repartir cartas sin reemplazo
- controlar penetración
- reshuffle automático

## 14.2 Lógica de valores
- cálculo correcto de hand value
- manejo de As flexible
- detección de hard/soft
- detección de blackjack natural
- detección de bust

## 14.3 Gestión de acciones legales
El entorno debe exponer qué acciones están permitidas en cada estado:

- hit
- stand
- double
- split
- insurance
- surrender

No todas deben estar siempre disponibles.

## 14.4 Transición de estado
Implementar la evolución exacta tras cada acción:

- hit → roba carta, recalcula total
- stand → pasa a siguiente mano o dealer
- double → duplica apuesta, roba una carta, cierra mano
- split → crea nuevas manos y redistribuye flujo
- insurance → registra side bet
- surrender → cierra mano con media pérdida

## 14.5 Política fija del dealer
- revelar hole card
- aplicar regla H17/S17
- detenerse correctamente

## 14.6 Liquidación de pagos
Debe calcular el reward final considerando:

- win/loss/push
- blackjack payout
- doubles
- splits
- insurance
- surrender

---

# 15. Espacio de acciones sugerido

## 15.1 Acción discreta base
Puedes usar algo como:

- `0 = stand`
- `1 = hit`
- `2 = double`
- `3 = split`
- `4 = surrender`
- `5 = insurance`

Luego el entorno invalida las no legales.

## 15.2 Máscara de acciones
Muy recomendable:
- devolver `action_mask`
para indicar cuáles acciones son legales en el estado actual.

Esto es casi obligatorio si vas a entrenar RL seriamente.

---

# 16. Estado sugerido para el agente

## 16.1 Estado 
- total del jugador
- indicador soft/hard
- carta visible del dealer
- indicador de pair para split

Además incluir:
- composición restante del shoe o conteo resumido (esto dejalo parametizado porque en un casino real no puedes llegar a preguntar cuantas faltas quedan)
- número de manos activas
- índice de mano actual
- si la mano viene de split
- si puede doblar
- si puede dividir
- si el dealer mostró As o 10 y ya hizo peek
- apuesta actual de la mano

---

# 17. Recompensas

## 17.1 Reward natural
Usar como reward la ganancia monetaria de la ronda o de la mano:

- perder apuesta base: `-1`
- ganar apuesta base: `+1`
- push: `0`
- blackjack natural 3:2: `+1.5`
- double ganado: `+2`
- double perdido: `-2`
- surrender: `-0.5`

## 17.2 Recompensa por mano o por ronda
Debes decidir si el paso del entorno representa:

### Opción A
Una acción por mano activa, y reward solo al final de la ronda.

### Opción B
Reward parcial por cada mano cuando se cierra.

La opción más limpia suele ser:
- reward acumulado al final de la ronda

---

# 18. Configuración canónica recomendada

Si quieres una versión muy estándar de casino, una configuración razonable sería:

- `n_decks = 6`
- `shoe_penetration = 0.8`
- `dealer_hits_soft_17 = False`  (S17)
- `blackjack_payout = 1.5`
- `dealer_peeks_for_blackjack = True`
- `double_allowed_on = "any_two_cards"`
- `double_after_split_allowed = True`
- `split_rule = "same_value"` o `"same_rank"` según diseño
- `max_hands_after_split = 4`
- `resplit_aces_allowed = True`
- `hit_split_aces_allowed = False`
- `surrender_allowed = True` o `False` según qué tan completa quieras la mesa
- `insurance_allowed = True`

---

# 19. Checklist final de implementación

## Motor base
- [ ] Crear shoe con múltiples barajas
- [ ] Barajar y extraer sin reemplazo
- [ ] Rebarajar según penetración

## Mano y scoring
- [ ] Calcular total con As flexible
- [ ] Detectar soft/hard
- [ ] Detectar blackjack natural
- [ ] Detectar bust

## Flujo de ronda
- [ ] Reparto inicial correcto
- [ ] Dealer con una carta visible y una oculta
- [ ] Peek rule opcional
- [ ] Turno del jugador
- [ ] Turno del dealer
- [ ] Resolución final

## Acciones
- [ ] Stand
- [ ] Hit
- [ ] Double
- [ ] Split
- [ ] Insurance
- [ ] Surrender

## Split
- [ ] Crear múltiples manos
- [ ] Controlar resplits
- [ ] Regla especial para ases divididos
- [ ] Double after split

## Dealer
- [ ] Regla H17/S17
- [ ] Política determinística

## Recompensas
- [ ] Win/loss/push
- [ ] Blackjack 3:2
- [ ] Dobles
- [ ] Splits
- [ ] Seguro
- [ ] Surrender

## Interfaz RL
- [ ] Observación del estado
- [ ] Máscara de acciones legales
- [ ] `reset()`
- [ ] `step(action)`
- [ ] estado terminal
- [ ] reward acumulado

---

# 20. Nota importante de diseño

Aunque esto busca ser “canónico”, **no existe una única versión universal de blackjack**. Lo correcto para un entorno serio es:

1. definir una configuración base “casino típica”
2. parametrizar todas las variantes importantes
3. entrenar o evaluar al agente bajo reglas bien especificadas

Las reglas que más cambian la estrategia son:

- H17 vs S17
- 3:2 vs 6:5
- DAS o no
- surrender o no
- hit en split aces o no
- número de barajas
- penetración del shoe
- peek rule