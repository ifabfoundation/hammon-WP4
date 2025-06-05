# Appunti

## 1. HEADING_TMP: Orientamento della Facciata

- **Calcolo iniziale**: `heading = np.arctan2(hvp[2], hvp[0])` - Calcolato da un punto di fuga orizzontale (hvp), rappresenta la direzione principale (azimut) della facciata rilevata.

- **Uso nella rettificazione**: `calculate_no_adaptive_coor(h_fov=154, ...)` utilizza `headings_tmp` (insieme a pitch e roll) per definire l'orientamento della "camera virtuale" che guarda la facciata. Il centro orizzontale dell'immagine rettificata corrisponderà esattamente a questa direzione.

- **Campo visivo**: L'immagine rettificata ha un campo visivo orizzontale (h_fov, ad esempio 154 gradi). Se il centro dell'immagine corrisponde all'heading H = headings_tmp, allora:
  - Pixel all'estrema sinistra ≈ H - h_fov/2
  - Pixel all'estrema destra ≈ H + h_fov/2

- **Raffinamento dei punti di fuga**:
  ```python
  if refine_radians > np.pi / 2:
      refine_radians = refine_radians - np.pi
  elif refine_radians < -np.pi / 2:
      refine_radians = refine_radians + np.pi
  ```
  - Non è una normalizzazione ma un cambio intenzionale della direzione del punto di fuga
  - In una facciata, le linee parallele possono generare punti di fuga opposti (170° è equivalente a -10°)
  - Il codice verifica se l'angolo è "estremo" (> 90° o < -90°) e inverte la direzione se necessario
  - Vantaggio: più conveniente lavorare con angoli vicini a 0° durante il raffinamento (10° è più stabile di 170°)

## 2. MAPPA DI HEADING: Direzioni Assolute

- **Calcolo**: `angle_x = np.arctan2(point1[:, 0], point1[:, 2])` - Per ogni punto 3D calcola l'angolo di heading guardando la proiezione sul piano x-z.

- **Significato**: Ogni valore rappresenta l'heading assoluto che si avrebbe se la camera fosse orientata verso quel punto specifico nella scena. Fornisce l'heading corretto per ogni direzione nella panoramica.

## 3. NORMALIZZAZIONE DEGLI ANGOLI

- **Normalizzazione completa** (2π):
  ```python
  if angle > np.pi:
      angle -= 2*np.pi
  elif angle < -np.pi:
      angle += 2*np.pi
  ```
  - Mantiene l'angolo nell'intervallo [-π, π] preservando la direzione originale
  - Necessario quando un angolo supera π o è inferiore a -π

- **Inversione di direzione** (π):
  ```python
  if angle > np.pi/2:
      angle -= np.pi
  elif angle < -np.pi/2:
      angle += np.pi
  ```
  - Non normalizza, ma inverte la direzione (180° di differenza)
  - Converte angoli "estremi" nelle loro direzioni opposte più gestibili

## 4. DIFFERENZE TRA SISTEMI DI COORDINATE

### Sistema geografico 2D (piano x-y)

- **Funzione**: `yaw_radians(cam_x, cam_y, midpoint.x, midpoint.y)`
- **Formula**: `math.atan2(tgt_y - cam_y, tgt_x - cam_x)`
- **Caratteristiche**:
  - Mappa 2D (vista dall'alto)
  - Asse x: direzione est-ovest
  - Asse y: direzione nord-sud
  - Angolo 0° punta verso est (asse x positivo)
  - Calcola l'angolo tra asse x positivo e vettore camera→target
  - Risponde alla domanda: "In quale direzione devo guardare per vedere il target?"

### Sistema panoramico 3D (piano x-z)

- **Funzione**: `np.arctan2(point1[:, 0], point1[:, 2])`
- **Caratteristiche**:
  - Spazio 3D (mondo visto dalla camera)
  - Asse z: direzione "avanti"
  - Asse x: direzione "destra"
  - Asse y: direzione "su"
  - Angolo 0° punta "avanti" (asse z positivo)
  - Calcola l'angolo nel piano orizzontale x-z
  - Risponde alla domanda: "Quale direzione sto guardando in questo punto dell'immagine?"

### Differenze chiave

1. **Orientamento degli assi**:
   - Sistema geografico: 0° = est
   - Sistema panoramico: 0° = avanti

2. **Ordine degli argomenti**:
   - `atan2(tgt_y - cam_y, tgt_x - cam_x)`: ordine y, x
   - `arctan2(point1[:, 0], point1[:, 2])`: ordine x, z

3. **Esempio pratico**:
   - Edificio direttamente a est della camera:
     - Sistema geografico: `atan2(0, 1) = 0°`
     - Sistema panoramico: `arctan2(1, 0) = 90°`

## 5. CONVERSIONE TRA SISTEMI

Per allineare i valori tra i due sistemi:

```python
# Trova l'offset tra la mappa di heading e i valori di yaw_rad
central_j = heading_map.shape[1] // 2
central_heading = heading_map[-1, central_j]
heading_offset = central_heading - heading_midpoint
heading_map_normalized = heading_map - heading_offset
```

Questo compensa sia le differenze di orientamento sia eventuali inversioni di segno.

## 6. FUNZIONI DI CONVERSIONE

### Conversione diretta degli angoli

```python
def convert_geographic_yaw_to_heading(yaw_rad):
    """Converte un angolo di yaw geografico in un heading nel sistema panoramico."""
    # La relazione esatta dipende dall'orientamento degli assi nei due sistemi
    # Questa è una approssimazione basata su quanto discusso
    heading_rad = np.pi/2 - yaw_rad  # Offset di 90° (π/2)
    
    # Normalizza nell'intervallo [-π, π]
    if heading_rad > np.pi:
        heading_rad -= 2*np.pi
    elif heading_rad < -np.pi:
        heading_rad += 2*np.pi
        
    return heading_rad
```

### Calibrazione empirica (metodo robusto)

```python
def calibrate_and_convert_headings(geographic_yaws, reference_heading_map):
    """
    Calibra e converte angoli di yaw geografici in heading della mappa.
    
    Args:
        geographic_yaws: Lista di angoli yaw nel sistema geografico
        reference_heading_map: Mappa di heading di riferimento
    
    Returns:
        Lista di angoli heading calibrati
    """
    # Prendi un punto di riferimento (ad es. il centro)
    central_j = reference_heading_map.shape[1] // 2
    central_heading = reference_heading_map[-1, central_j]
    
    # Assumi che il primo yaw geografico corrisponda al punto centrale
    reference_yaw = geographic_yaws[0]
    
    # Calcola l'offset e l'eventuale inversione
    offset = central_heading - reference_yaw
    
    # Applica la trasformazione a tutti i yaw
    converted_headings = []
    for yaw in geographic_yaws:
        heading = yaw + offset
        
        # Normalizza nell'intervallo [-π, π]
        if heading > np.pi:
            heading -= 2*np.pi
        elif heading < -np.pi:
            heading += 2*np.pi
            
        converted_headings.append(heading)
    
    return converted_headings
```

### Allineamento della mappa di heading

```python
def align_heading_map(heading_map, heading_midpoint):
    """
    Allinea la mappa degli heading con il valore di heading_midpoint.
    
    Args:
        heading_map: La mappa degli heading da normalizzare
        heading_midpoint: Il valore di heading centrale di riferimento
    
    Returns:
        La mappa degli heading normalizzata
    """
    # Trova l'offset tra la mappa di heading e i valori di yaw_rad
    central_j = heading_map.shape[1] // 2  # Colonna centrale
    central_heading = heading_map[-1, central_j]  # Usando l'ultima riga come riferimento
    
    heading_offset = central_heading - heading_midpoint
    heading_map_normalized = heading_map - heading_offset

    # Gestisci il wrap-around degli angoli
    heading_map_normalized = np.where(heading_map_normalized > np.pi, 
                                  heading_map_normalized - 2*np.pi, 
                                  heading_map_normalized)
    heading_map_normalized = np.where(heading_map_normalized < -np.pi, 
                                  heading_map_normalized + 2*np.pi, 
                                  heading_map_normalized)
    
    return heading_map_normalized
```
