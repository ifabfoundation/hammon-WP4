
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os.path
import skimage.io
import numpy as np
from default_params import default_params
from V import V

class todo:
    save_results_image = 1
    benchmark = 0
    calibrate = 1
    ortho_rectify = 1
    save_ortho_images = 1

# plot options (0 if not plotted, figure number otherwise)
class plots:
    hvps = 0 #1
    z = 0 #1
    hl = 0 #1
    gthl = 0 #1
    benchmark = 0 #2 display presision curve and AUC for each dataset (the figure number will be plots.benchmark*100+#dataset)
    manhattan = 0 #1
    orthorectify = 0


def simon_rectification(img, num, folder, root, tmp_count):
    """
    Funzione che implementa l'algoritmo di rettificazione di Simon per immagini panoramiche
    
    Args:
        img: Può essere un percorso (string) all'immagine o un array NumPy dell'immagine
        num: Numero identificativo dell'immagine nella sequenza
        folder: Directory dove salvare i risultati visualizzati
        root: Directory principale del progetto
        tmp_count: Contatore temporaneo utilizzato per il batch processing
    
    Returns:
        Lista contenente: [hl, hvps, hvp_groups, z, z_group, ls, z_homo, hvp_homo, ls_homo, params]
        dove:
        - hl: coordinate della linea dell'orizzonte
        - hvps: punti di fuga orizzontali
        - hvp_groups: gruppi di punti di fuga orizzontali
        - z: punto di zenith (punto di fuga verticale)
        - z_group: gruppo di linee che definiscono il punto di zenith
        - ls: segmenti di linea rilevati
        - z_homo, hvp_homo, ls_homo: versioni in coordinate omogenee
        - params: parametri utilizzati nell'algoritmo
    """
    
    # Gestione dell'input: può essere un percorso file o un array NumPy
    if type(img) == str:
        im_path = img                      # Salva il percorso dell'immagine
        im = Image.open(im_path)           # Carica l'immagine come oggetto PIL
        im_useless = im.copy()             # Copia dell'immagine (non utilizzata)
        im_array = skimage.io.imread(im_path)  # Carica l'immagine come array NumPy

    elif len(img.shape) == 3:              # Se l'input è già un array NumPy a 3 canali
        im = Image.fromarray(img)          # Converte l'array in oggetto PIL
        im_useless = im.copy()             # Copia dell'immagine (non utilizzata)
        im_array = img                     # Usa l'array direttamente

    else:
        raise ValueError('input type is wrong')  # Errore per input non validi

    # Ottiene le dimensioni dell'immagine
    width = im.width
    height = im.height

    # Configura i parametri per l'algoritmo di Simon
    params = default_params()               # Carica i parametri di default
    params.include_infinite_hvps = 1        # Include i punti di fuga all'infinito
    params.return_z_homo = 1                # Restituisce le coordinate omogenee dello zenith

    # Stima della lunghezza focale (approssimata)
    focal = max(width, height) / 2         # Usa la metà della dimensione maggiore

    # Chiamata alla funzione V che implementa l'algoritmo principale di rilevamento
    # di linee orizzontali, punti di fuga e punto zenith
    [hl, hvps, hvp_groups, z, z_group, ls, z_homo, hvp_homo, ls_homo] = V(im, width, height, focal, params, tmp_count)

    # Crea una mappa di colori per la visualizzazione dei diversi gruppi di linee
    # Utilizza la colormap HSV di matplotlib per generare 4 colori distinti
    cmap = plt.cm.hsv(np.linspace(0, 1, 4))[:, :3]

    # Disegna i punti di fuga orizzontali (HVPs) se richiesto dalla configurazione
    if plots.hvps:
        im_hvps = im  # Usa l'immagine originale
        draw = ImageDraw.Draw(im_hvps)  # Crea un oggetto per disegnare sull'immagine
        
        # Per ogni gruppo di punti di fuga orizzontali
        for j in range(len(hvp_groups)):
            hg = hvp_groups[j]  # Prendi il gruppo j-esimo
            # Disegna tutte le linee associate a questo gruppo
            for k in range(len(hg)):
                # Estrai le coordinate di inizio e fine della linea
                pt1 = (ls[hg[k], 0], ls[hg[k], 1])  # Punto iniziale
                pt2 = (ls[hg[k], 2], ls[hg[k], 3])  # Punto finale
                # Disegna la linea con un colore specifico per questo gruppo
                draw.line((pt1, pt2), fill=tuple((cmap[j] * 255).astype(int)), width=5)
        
        # Prepara il percorso dove salvare l'immagine (attualmente commentato)
        hvps_name = os.path.join(folder, str(num) + '_im_hvps.jpg')
        #im_hvps.save(hvps_name)  # Salvataggio disabilitato

    # Visualizzazione delle linee che definiscono il punto di zenith (punto di fuga verticale)
    if plots.z:
        im_z = im  # Usa l'immagine originale
        # Codice commentato che offriva alternative per l'immagine di base
        # if plots.hvps:
        #     im_z = im_hvps.copy()
        # else:
        #     im_z = im.copy()
        
        draw = ImageDraw.Draw(im_z)  # Crea un oggetto per disegnare sull'immagine
        zg = z_group  # Gruppo di linee che definiscono il punto di zenith
        
        # Per ogni linea nel gruppo del punto zenith
        for k in range(len(zg)):
            # Estrai le coordinate di inizio e fine della linea
            pt1 = (ls[zg[k], 0], ls[zg[k], 1])  # Punto iniziale
            pt2 = (ls[zg[k], 2], ls[zg[k], 3])  # Punto finale
            # Disegna la linea usando il terzo colore della mappa colori
            draw.line((pt1, pt2), fill=tuple((cmap[2] * 255).astype(int)), width=5)
        
        # Prepara il percorso dove salvare l'immagine (attualmente commentato)
        z_name = os.path.join(folder, str(num) + '_im_z.jpg')
        #im_z.save(z_name)  # Salvataggio disabilitato

    # Visualizzazione della linea dell'orizzonte
    if plots.hl:
        im_hl = im  # Usa l'immagine originale
        # Codice commentato che offriva alternative per l'immagine di base
        # if plots.hvps and plots.z:
        #     im_hl = im_z.copy()
        # else:
        #     im_hl = im.copy()
        
        draw = ImageDraw.Draw(im_hl)  # Crea un oggetto per disegnare sull'immagine
        # Estrai i punti di inizio e fine della linea dell'orizzonte
        pt1 = (hl[0, 0], hl[0, 1])  # Punto iniziale della linea dell'orizzonte
        pt2 = (hl[1, 0], hl[1, 1])  # Punto finale della linea dell'orizzonte
        # Disegna la linea dell'orizzonte in colore ciano (0, 255, 255)
        draw.line((pt1, pt2), fill=tuple([0, 255, 255]), width=7)
        
        # Prepara il percorso e salva l'immagine con la linea dell'orizzonte
        hl_name = os.path.join(folder, str(num) + '_im_hl.jpg')
        im_hl.save(hl_name)  # Questa è l'unica immagine che viene effettivamente salvata

        # CODICE COMMENTATO: SALVATAGGIO DELLA MASCHERA DELL'ORIZZONTE
        # Crea un'immagine nera delle stesse dimensioni dell'originale
        # horizon_mask = np.zeros([height, width, 3], dtype=np.uint8)
        # horizon_mask = Image.fromarray(horizon_mask)
        # draw = ImageDraw.Draw(horizon_mask)
        # pt1 = (hl[0, 0], hl[0, 1])
        # pt2 = (hl[1, 0], hl[1, 1])
        # Disegna solo la linea dell'orizzonte in bianco (255, 255, 255)
        # draw.line((pt1, pt2), fill=tuple([255, 255, 255]), width=8)
        # hl_mask = os.path.join(root, 'horizon_mask', str(num) + 'hl_msk.jpg')
        # horizon_mask.save(hl_mask)

    # Restituisce tutti i risultati dell'algoritmo di rettificazione:
    # - hl: linea dell'orizzonte
    # - hvps: punti di fuga orizzontali
    # - hvp_groups: gruppi di punti di fuga orizzontali
    # - z: punto zenith (fuga verticale)
    # - z_group: linee che definiscono il punto zenith
    # - ls: tutti i segmenti di linea rilevati
    # - z_homo, hvp_homo, ls_homo: versioni in coordinate omogenee
    # - params: parametri utilizzati nell'algoritmo
    return [hl, hvps, hvp_groups, z, z_group, ls, z_homo, hvp_homo, ls_homo, params]