# Arquitecturas recomendadas para Blackjack RL en PyTorch

## Objetivo

Definir una especificación clara de las **tres arquitecturas** que se van a considerar para el agente de blackjack, manteniendo consistencia con:

- el environment ya modular;
- el encoder ya modular;
- los dos modos de entrenamiento que queremos soportar;
- un pipeline en PyTorch limpio y extensible.

Las tres arquitecturas seleccionadas son:

1. **Baseline Double DQN feedforward**
2. **DRQN / Recurrent Double DQN**
3. **Dueling Recurrent Double DQN**

La idea no es crear familias totalmente distintas de modelos, sino una línea coherente de complejidad creciente sobre el mismo encoder.

---

# 1. Dos modos de entrenamiento que el modelo debe soportar

Antes de hablar de arquitecturas, dejamos fijado que habrá **dos regímenes de entrenamiento**.

## 1.1 Modo A: `fresh_shoe`

### Descripción
- el shoe inicia desde cero;
- el agente observa la evolución desde el inicio;
- puede construir memoria completa del descarte observado.

### Qué aprende mejor aquí
- estrategia básica y avanzada de blackjack;
- adaptación a reglas distintas;
- conteo implícito de cartas desde el inicio del shoe;
- dependencia entre rondas.

### Uso principal
- baseline limpio;
- entorno toy/controlado;
- validación de que la red y el loop funcionan.

---

## 1.2 Modo B: `unknown_progress`

### Descripción
- el shoe real puede venir avanzado;
- el agente conoce reglas y número de mazos;
- **no conoce cuánto falta para el reshuffle**;
- el agente solo puede contar cartas a partir de lo observado desde que empieza la secuencia.

### Qué vuelve más difícil este modo
- el agente entra con información incompleta;
- el estado real del shoe es parcialmente latente;
- la memoria recurrente tiene más valor.

### Uso principal
- setting más realista;
- puente previo al caso CV;
- robustez a historia truncada.

---

# 2. Principio de diseño general para las tres arquitecturas

Las tres arquitecturas deben compartir estas ideas base:

## 2.1 Mismo encoder de entrada
El encoder ya produce:

- `state_vector`
- `action_mask`
- opcionalmente `module_tensors`

El modelo no debe rehacer el trabajo del encoder.  
Debe consumir la representación ya estructurada.

## 2.2 Action masking obligatorio
La salida siempre es sobre 6 acciones discretas:

- `stand`
- `hit`
- `double`
- `split`
- `surrender`
- `insurance`

Pero la selección de acción y el target del DQN deben respetar `action_mask`.

## 2.3 Double DQN como base
En los tres casos la lógica base será **Double DQN**, no DQN vanilla.

Razón:
- reduce sobreestimación;
- es estándar más robusto;
- especialmente importante cuando hay pocas acciones pero decisiones delicadas.

## 2.4 Modularidad
La arquitectura debe poder reutilizar:

- el mismo encoder;
- el mismo replay buffer base;
- el mismo loop;
- el mismo esquema de target network;
- el mismo esquema de evaluación.

La diferencia principal entre modelos debe vivir en el **backbone de la red**, no en todo el pipeline.

---

# 3. Arquitectura 1: Baseline Double DQN feedforward

## 3.1 Cuándo se usa
Este modelo sirve como:

- baseline obligatorio;
- sanity check;
- comparación contra los modelos recurrentes.

Se debe entrenar al menos en:

- `fresh_shoe`
- opcionalmente también en `unknown_progress` como benchmark débil

---

## 3.2 Estructura conceptual

### Entrada
- `state_vector: [B, D]`

### Backbone
- MLP de 2 o 3 capas totalmente conectadas

Ejemplo conceptual:
- Linear(D -> H1)
- activación
- Linear(H1 -> H2)
- activación
- Linear(H2 -> H3) opcional
- activación

### Head
- Linear(H_last -> 6)

Salida:
- `Q(s, a)` para las 6 acciones

---

## 3.3 Recomendaciones de diseño

### Activaciones
Usar preferiblemente:
- **ReLU** si quieres algo estándar y estable
- **GELU** si quieres una alternativa más suave

### Recomendación
Para empezar:
- **ReLU**
porque en RL sigue siendo la opción más simple y robusta.

---

### BatchNorm
**No es recomendable como default** en DQN feedforward clásico.

Razones:
- las distribuciones del replay buffer cambian con el tiempo;
- mezcla estados de política antigua y nueva;
- puede introducir inestabilidad.

### Mejor alternativa
- **LayerNorm** si realmente necesitas estabilización
- o ninguna normalización al principio

---

### Dropout
**No usar como default** en este baseline.

Razón:
- en value-based RL suele meter ruido innecesario;
- puede dificultar convergencia del estimador Q.

### Recomendación
- dropout = 0 al inicio
- solo considerar dropout pequeño si luego detectas sobreajuste serio

---

### Residuales
No son necesarios en la primera versión.

---

## 3.4 Ventajas
- simple;
- fácil de depurar;
- útil para comprobar que el encoder tiene señal.

## 3.5 Limitaciones
- no tiene memoria interna;
- depende totalmente de lo que el encoder ya resumió;
- probablemente limitado para capturar dinámica temporal rica entre rondas.

---

# 4. Arquitectura 2: DRQN / Recurrent Double DQN

## 4.1 Cuándo se usa
Este debe ser el **primer modelo serio** para aprender multironda.

Se debe usar principalmente en:
- `fresh_shoe`
- `unknown_progress`

---

## 4.2 Idea principal

En vez de tratar cada decisión como i.i.d., la red procesa una **secuencia de estados**.

### Entrada
- `state_vector: [B, T, D]`
- `action_mask: [B, T, 6]`
- `padding_mask: [B, T]`

### Backbone
1. proyección inicial por paso
2. bloque recurrente (GRU o LSTM)
3. head lineal a Q-values por paso

Salida:
- `Q: [B, T, 6]`

---

## 4.3 Estructura conceptual

### Etapa A: proyección inicial
Antes de la recurrente conviene tener una proyección densa:

- Linear(D -> H_proj)
- activación
- opcional LayerNorm

Razón:
- compacta el estado;
- reduce ruido;
- facilita a la recurrente trabajar sobre una representación más estable.

---

### Etapa B: bloque recurrente
Elegir una de dos opciones:

#### Opción 1: GRU
Ventajas:
- menos parámetros;
- más simple;
- más fácil de entrenar;
- muy buena opción inicial.

#### Opción 2: LSTM
Ventajas:
- más capacidad de memoria;
- útil si luego ves que GRU se queda corta;
- especialmente interesante en `unknown_progress`.

### Recomendación práctica
- empezar con **GRU**
- dejar **LSTM** como switch configurable

---

### Etapa C: head de Q-values
Sobre la salida recurrente por paso:
- Linear(H_rec -> H_head)
- activación
- Linear(H_head -> 6)

---

## 4.4 Recomendaciones de diseño

### Activaciones
- **ReLU** o **GELU**
- ReLU sigue siendo muy razonable
- GELU puede funcionar bien en proyecciones densas

### Recomendación
- ReLU en el baseline
- puedes probar GELU en el trunk recurrente si quieres suavidad extra

---

### Normalización
En modelos recurrentes, si quieres normalizar, preferir:
- **LayerNorm** en la proyección previa o en el head

Evitar como default:
- **BatchNorm dentro de la parte recurrente**

Razón:
- en secuencias RL el batch puede ser heterogéneo;
- padding y longitudes distintas complican uso limpio;
- BatchNorm rara vez es la primera opción aquí.

---

### Dropout
En recurrentes:
- no usar dropout alto;
- si lo usas, que sea pequeño.

Recomendación:
- `dropout = 0.0` si una sola capa recurrente
- o `0.05 - 0.10` solo en proyección/head, no como primera decisión

---

### Número de capas recurrentes
Empezar con:
- **1 capa recurrente**

No recomiendo empezar con 2 o más capas:
- más complejidad;
- más inestabilidad;
- no necesitas eso para blackjack inicialmente.

---

### Hidden size
Debe ser suficientemente grande para:
- resumir historia;
- integrar reglas y mano actual;
- mantener señal útil entre rondas.

Pero no tan grande que vuelva el entrenamiento innecesariamente pesado.

---

## 4.5 Ventajas
- permite jugar varias rondas seguidas;
- integra información entre decisiones;
- puede aprender conteo implícito mejor que un MLP plano.

## 4.6 Limitaciones
- más costoso;
- entrenamiento más delicado;
- debugging algo más difícil.

---

# 5. Arquitectura 3: Dueling Recurrent Double DQN

## 5.1 Cuándo se usa
Este es el modelo más fuerte de los tres y probablemente el candidato final.

Debe probarse en:
- `fresh_shoe`
- `unknown_progress`

Especialmente útil cuando:
- el valor del estado importa mucho;
- no todas las acciones cambian radicalmente el valor en todos los estados.

---

## 5.2 Idea principal

La arquitectura es igual a la recurrente, salvo que el head final no predice Q directamente.

Predice dos cosas:

1. **Value stream**: \( V(s_t) \)
2. **Advantage stream**: \( A(s_t, a) \)

Y luego combina ambas para obtener:
\( Q(s_t, a) \)

---

## 5.3 Estructura conceptual

### Entrada
- `state_vector: [B, T, D]`

### Etapas compartidas
1. proyección inicial
2. GRU o LSTM
3. embedding temporal por paso

### Head dueling

#### Stream de valor
- Linear(H_rec -> H_v)
- activación
- Linear(H_v -> 1)

#### Stream de ventaja
- Linear(H_rec -> H_a)
- activación
- Linear(H_a -> 6)

#### Combinación
- Q = V + (A - mean(A))

---

## 5.4 Por qué tiene sentido en blackjack

En blackjack hay muchos estados donde:

- el valor global del estado ya está bastante determinado;
- pero la diferencia entre acciones es relativamente estructurada;
- algunas acciones son simplemente ilegales o irrelevantes en muchos estados.

Entonces separar:
- valor del estado
- ventaja relativa de cada acción

puede ayudar a aprender mejor.

---

## 5.5 Recomendaciones de diseño

### Trunk recurrente
Mismo criterio que en DRQN:
- 1 capa recurrente
- GRU primero
- LSTM como opción

### Streams del dueling head
Ambos streams pueden ser:
- pequeños MLPs de 1 capa oculta

No hace falta que sean profundos.

---

### Activaciones
- ReLU está perfectamente bien
- GELU también es válida, pero no es indispensable

### Normalización
- LayerNorm opcional en proyección
- no BatchNorm en recurrente

### Dropout
- muy bajo o cero al inicio

---

## 5.6 Ventajas
- suele aprender mejor la estructura de valor;
- puede estabilizar la estimación de Q;
- muy buena opción final para tu caso.

## 5.7 Limitaciones
- algo más complejo;
- más piezas para depurar;
- conviene implementarlo después del DRQN simple.

---

# 6. Cómo se relacionan las tres arquitecturas

La progresión natural es:

## Modelo 1
**Double DQN feedforward**
- sirve para verificar pipeline
- no tiene memoria

## Modelo 2
**DRQN / Recurrent Double DQN**
- agrega memoria
- mantiene head simple

## Modelo 3
**Dueling Recurrent Double DQN**
- misma memoria
- head más expresivo y estructurado

---

# 7. Recomendaciones por modo de entrenamiento

## 7.1 Para `fresh_shoe`

### Baseline
- Double DQN feedforward

### Modelo serio recomendado
- DRQN con GRU

### Modelo fuerte final
- Dueling DRQN con GRU

### Comentario
Como el agente ve el shoe desde el principio, la memoria recurrente puede explotar muy bien la secuencia completa.

---

## 7.2 Para `unknown_progress`

### Baseline
- Double DQN feedforward, solo como referencia

### Modelo serio recomendado
- DRQN con GRU o LSTM

### Modelo fuerte final
- Dueling DRQN con GRU o LSTM

### Comentario
Aquí la memoria importa todavía más, porque la historia inicial está truncada y el agente necesita inferir mejor el estado latente del shoe.

---

# 8. Buenas prácticas de diseño para estas redes en RL

---

## 8.1 Activaciones

### Recomendación principal
- **ReLU** como default

### Alternativa razonable
- **GELU** en trunks más suaves

### No priorizar
- activaciones exóticas sin necesidad

---

## 8.2 Normalización

### Recomendado
- **LayerNorm** si necesitas estabilización
- especialmente en la proyección inicial o en el trunk denso

### No recomendado como default
- **BatchNorm** en DQN / DRQN
- especialmente no dentro del recurrente

Razón:
- replay buffer no i.i.d.
- secuencias de distinta longitud
- padding
- distribución cambiante durante entrenamiento

---

## 8.3 Dropout

### Recomendación general
- usar **muy poco o nada** al inicio

### Por qué
En value-based RL, dropout puede:
- aumentar ruido en targets;
- dificultar estimación estable de Q.

### Si se usa
- solo pequeño
- preferiblemente fuera del recurrente
- por ejemplo en el head

---

## 8.4 Inicialización
Mantener inicialización estándar de PyTorch suele ser suficiente al inicio.

No hace falta inventar una inicialización especial en la primera versión.

---

## 8.5 Tamaño del modelo
Evitar redes enormes de entrada.

El objetivo es:
- estabilidad,
- interpretabilidad,
- iteración rápida.

Primero una red mediana y estable vale más que una red muy grande y difícil de entrenar.

---

## 8.6 Profundidad
- MLP: 2 o 3 capas basta
- recurrente: 1 capa basta al inicio
- dueling head: streams pequeños

---

## 8.7 Máscara de padding
En las arquitecturas recurrentes, el entrenamiento debe respetar:
- `padding_mask`
- no calcular pérdida sobre pasos padded

Esto es una buena práctica obligatoria, no opcional.

---

## 8.8 Target network
Las tres arquitecturas deben mantener:
- red online
- red target

Y la target debe actualizarse de forma:
- periódica
o
- suave (soft update)

---

## 8.9 Gradient clipping
En las arquitecturas recurrentes es altamente recomendable.

Razón:
- evita explosión de gradientes;
- mejora estabilidad en secuencias largas.

---

# 9. Modularidad esperada en implementación

Aunque aquí no estamos escribiendo código, el diseño debe soportar una implementación como esta:

## Bloques compartidos
- encoder
- replay buffer
- trainer base
- utilities de masked action selection
- target computation

## Backbone intercambiable
- `FeedForwardQBackbone`
- `RecurrentQBackbone`
- `DuelingRecurrentQBackbone`

## Recurrent cell intercambiable
- `GRU`
- `LSTM`

La elección entre GRU y LSTM debe ser una **configuración**, no otra familia de modelo totalmente distinta.

---

# 10. Orden recomendado de desarrollo

## Paso 1
Implementar **Baseline Double DQN feedforward**

Objetivo:
- validar loop,
- validar losses,
- validar action masking,
- validar integración con encoder.

---

## Paso 2
Implementar **DRQN / Recurrent Double DQN**

Objetivo:
- soportar secuencias,
- jugar múltiples rondas,
- aprender memoria temporal.

Primero con:
- GRU

Luego opcionalmente:
- LSTM

---

## Paso 3
Implementar **Dueling Recurrent Double DQN**

Objetivo:
- mejorar estimación de valor;
- construir el candidato final.

---

# 11. Recomendación final

Si hay que priorizar:

## Primer modelo a correr
**Baseline Double DQN feedforward**

## Primer modelo serio
**DRQN con GRU**

## Modelo final candidato
**Dueling DRQN con GRU**

## Variante a comparar después
**Dueling DRQN con LSTM**

---

# 12. Resumen ejecutivo

## Arquitectura 1
**Double DQN feedforward**
- simple
- baseline
- sin memoria

## Arquitectura 2
**DRQN / Recurrent Double DQN**
- secuencial
- memoria explícita
- ideal para múltiples rondas

## Arquitectura 3
**Dueling Recurrent Double DQN**
- secuencial
- memoria explícita
- mejor estructura de estimación de Q
- candidato más fuerte

## Buenas prácticas
- ReLU como default
- LayerNorm si hace falta
- evitar BatchNorm como default
- poco o nada de dropout
- 1 capa recurrente al inicio
- gradient clipping
- action masking
- target network
- pérdida solo sobre pasos no padded

## Dos modos de entrenamiento
- `fresh_shoe`: historia completa desde el inicio
- `unknown_progress`: historia truncada, mayor parcial observabilidad

La arquitectura base puede ser la misma; lo que cambia principalmente es el régimen de entrenamiento y el tipo de secuencias que se le presentan a la red.
