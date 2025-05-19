import os
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
import skimage.io
from math import acos, cos, degrees, radians, sin
import glob
from Panos.Pano_visualization import R_roll, R_pitch, R_heading
from Panos.Pano_new_pano import calculate_new_pano, save_heading_pitch_json, save_heading_only
from Panos.Pano_refine_project import simon_refine
import json

# with open('/home/zhup/Desktop/GSV_Pano_val/Val/projections_views/useful_projections.json') as f:
#     useful_projections = json.load(f)

# with open('/home/zhup/Desktop/GSV_Pano_val/Val/projections_views/random_projections.json') as f:
#     useful_projections = json.load(f)

# with open('/home/zhup/Desktop/GSV_Pano_val/Val/projections_views/prior_projections.json') as f:
#     useful_projections = json.load(f)

# with open('/home/zhup/Desktop/GSV_Pano_val/Val/projections_views/12viewpoints_projections.json') as f:
#     useful_projections = json.load(f)

# with open('/home/zhup/Desktop/GSV_Pano_val/Val/projections_views/4viewpoints_projections.json') as f:
#     useful_projections = json.load(f)

# with open('/home/zhup/Desktop/GSV_Pano_val/Val/projections_views/16-inlier12viewpoints_projections.json') as f:
#     useful_projections = json.load(f)




def stitch_tiles(num, tiles, directory):
    test_img= skimage.io.imread(tiles[0])
    height = test_img.shape[0]
    width = test_img.shape[1]

    stiched_img = np.zeros([height, width * num, 3])
    for i, tile in enumerate(tiles):
        tmp = skimage.io.imread(tile)
        stiched_img[:, i * width: (i + 1) * width] = tmp

    skimage.io.imsave(os.path.join(directory, "hl_stiched.jpg"), stiched_img/255)


def rotation_matrix_Z(angle):
    # https://developers.google.com/streetview/spherical-metadata#euler_overview


    def R_Z(angle):
        return np.array([[cos(angle), -sin(angle), 0],
                         [sin(angle), cos(angle), 0], [0., 0., 1.]])

    def R_X(angle):
        return np.array([[1, 0, 0], [0, cos(angle), -sin(angle)],
                         [0, sin(angle), cos(angle)]])

    def R_Y(angle):
        return np.array([[cos(angle), 0, sin(angle)], [0, 1, 0],
                         [-sin(angle), 0, cos(angle)]])

    R = R_Z(radians(angle))

    return np.linalg.inv(R)


def rotation_matrix_X(angle):
    # https://developers.google.com/streetview/spherical-metadata#euler_overview


    def R_Z(angle):
        return np.array([[cos(angle), -sin(angle), 0],
                         [sin(angle), cos(angle), 0], [0., 0., 1.]])

    def R_X(angle):
        return np.array([[1, 0, 0], [0, cos(angle), -sin(angle)],
                         [0, sin(angle), cos(angle)]])

    def R_Y(angle):
        return np.array([[cos(angle), 0, sin(angle)], [0, 1, 0],
                         [-sin(angle), 0, cos(angle)]])

    R = R_X(radians(angle))
    return np.linalg.inv(R)

def get_the_top():
    height = 20
    width = 20
    mpp = 0.0125
    p0 = np.array([-10., 10., 0.])
    p1 = np.array([10., 10, 0.])
    middle = (p0 + p1) / 2

    up = np.array([0., 0., 1.])
    vec = p1 - p0
    dist = np.linalg.norm(vec)
    vec /= dist
    rot = rotation_matrix_X(-90)

    # if width is not None:
    #     center = (p0 + p1) / 2.
    #     dist = width
    #     p0 = center - 0.5 * vec * dist

    # m - Number of rows in the output image
    # n - Number of columns in the output image
    m = int(np.ceil(height / mpp))
    n = int(np.ceil((width) / mpp))

    # Generate barycentric coordinates for every point on the face-grid
    u, v = np.mgrid[-height / 2:height / 2:m * 1j, -width / 2:width / 2:n * 1j]

    # Generate enu-coordinates for each point on the face-grid
    xy = np.outer(u, up) + np.outer(v, vec) + middle
    xy = rot.dot(xy.T).T

    # Generate a m x n x 2 array of headings and pitches
    heading = (np.degrees(np.arctan2(xy[:, 0], xy[:, 1])))
    pitch = np.degrees(np.arctan2(xy[:, 2], np.hypot(xy[:, 0], xy[:, 1])))
    projected = np.column_stack((heading, pitch)).reshape(m, n, 2)

    root = '/home/zhup/Desktop/Pano'
    img_folder = os.path.join(root, 'Pano_img/')
    output_folder = os.path.join(root, 'Pano_render/')
    imageList = glob.glob(img_folder + '*.jpg')
    imageList.sort()

    panorama_img = imageList[0]
    render_num = 8
    start = int(-render_num / 2) + 1
    img = skimage.io.imread(panorama_img)

    coordinates = projected.transpose(2, 0, 1, )

    # I am getting 'heading, pitch', I want 'pitch, heading' since columns correspond to different headings
    coordinates = np.roll(coordinates, 1, axis=0)

    # Map heading from  -180 ..180 to  0...360
    coordinates[1] += 180.
    coordinates[0] = 90 - coordinates[0]  # 0 ->90 (horizontal), 90->0 (top/up)

    coordinates[0] *= img.shape[0] / 180.
    coordinates[1] *= img.shape[1] / 360.

    sub = np.dstack([
        map_coordinates(img[:, :, 0], coordinates, order=0),
        map_coordinates(img[:, :, 1], coordinates, order=0),
        map_coordinates(img[:, :, 2], coordinates, order=0)
    ])
    sub = sub[::-1, :, :]

    save_path = os.path.join(output_folder, 'top/Render_top.jpg')
    save_img = skimage.io.imsave(save_path, sub)





def project_face(num, degree):
    # enu = self.enu_matrix
    # Codice precedente commentato: rotazione e conversione coordinate
    # rot = self.rotation_matrix_Z

    # Codice precedente commentato: calcolo punti di bordo dalle coordinate geografiche
    # p0 = (enu @ lla2xyz(lat=edge[0][1], lon=edge[0][0], alt=ground))[:3]
    # p1 = (enu @ lla2xyz(lat=edge[1][1], lon=edge[1][0], alt=ground))[:3]

    # Definizione delle dimensioni fisiche della facciata in metri
    height = 20  # Altezza della facciata in metri (abbassare per avere un'immagine più piccola)
    width = 20   # Larghezza della facciata in metri (abbassare per avere un'immagine più piccola)
    mpp = 0.0125  # Metri per pixel - definisce la risoluzione dell'immagine risultante
    
    # Definizione dei punti che rappresentano il bordo superiore della facciata
    p0 = np.array([-10., 10., 0.])  # Punto iniziale del bordo superiore: coordinate (x=-10, y=10, z=0)
    p1 = np.array([10., 10., 0.])    # Punto finale del bordo superiore: coordinate (x=10, y=10, z=0)
    middle = (p0 + p1) / 2          # Punto centrale del bordo superiore: media tra p0 e p1

    # Definizione dei vettori di orientamento della facciata
    up = np.array([0., 0., 1.])     # Vettore verticale (direzione +z, verso l'alto)
    vec = p1 - p0                   # Vettore orizzontale lungo il bordo superiore della facciata
    dist = np.linalg.norm(vec)      # Calcolo della lunghezza del vettore orizzontale
    vec /= dist                     # Normalizzazione del vettore orizzontale (lunghezza unitaria)
    
    # Creazione della matrice di rotazione attorno all'asse Z
    # Ruota la facciata dell'angolo specificato (degree * num)
    rot = rotation_matrix_Z(degree * (num))

    # Codice precedente commentato: calcolo alternativo della larghezza
    # if width is not None:
    #     center = (p0 + p1) / 2.
    #     dist = width
    #     p0 = center - 0.5 * vec * dist

    # Calcolo delle dimensioni dell'immagine di output in pixel
    m = int(np.ceil(height / mpp))  # Numero di righe nell'immagine di output
    n = int(np.ceil((width) / mpp)) # Numero di colonne nell'immagine di output

    # Generazione di una griglia di coordinate baricentriche per ogni punto della facciata
    # Crea due matrici u e v di dimensione m×n che coprono l'intera facciata
    # Il parametro 1j indica che vogliamo m/n punti equispaziati nel range specificato
    u, v = np.mgrid[-height/2:height/2:m * 1j, -width/2:width/2:n * 1j]

    # Conversione delle coordinate baricentriche in coordinate 3D (ENU - East, North, Up)
    # Per ogni punto (u,v) della griglia, calcola la corrispondente posizione 3D 
    xy = np.outer(u, up) + np.outer(v, vec) + middle  # Combinazione dei vettori di base più il punto centrale
    xy = rot.dot(xy.T).T  # Applicazione della rotazione a tutti i punti

    # Calcolo delle coordinate sferiche (heading e pitch) per ogni punto 3D
    # Queste coordinate saranno utilizzate per campionare l'immagine panoramica
    heading = (np.degrees(np.arctan2(xy[:, 0], xy[:, 1])))  # Azimut (angolo orizzontale) in gradi
    pitch = np.degrees(np.arctan2(xy[:, 2], np.hypot(xy[:, 0], xy[:, 1])))  # Elevazione (angolo verticale) in gradi
    
    # Organizzazione dell'output in un array di forma (m, n, 2)
    # Per ogni pixel (i,j) dell'immagine di output, projected[i,j] contiene [heading, pitch]
    projected = np.column_stack((heading, pitch)).reshape(m, n, 2)
    return projected




# def project_facade_output(final_hvps_rectified, im, pitch, roll, im_path, root):
#
#     y = 100
#     h_fov = 120  # -60~60
#     v_fov1 = -20
#     v_fov2 = 70
#
#     x1 = np.tan(np.radians(-h_fov / 2)) * y
#     x2 = np.tan(np.radians(h_fov / 2)) * y
#     width = x2 - x1
#
#     z1 = np.tan(np.radians(v_fov1)) * y
#     z2 = np.tan(np.radians(v_fov2)) * y
#     height = z2 - z1
#
#
#     m = int(np.ceil(6656 / 180 * (v_fov2 - v_fov1)))
#     n = int(np.ceil(6656 / 180 * h_fov))
#
#
#     p0 = np.array([x1, y, 0.])
#     p1 = np.array([x2, y, 0.])
#     middle = (p0 + p1) / 2
#
#
#     up = np.array([0., 0., 1.])
#     vec = p1 - p0
#     dist = np.linalg.norm(vec)
#     vec /= dist
#
#
#     # Generate barycentric coordinates for every point on the face-grid
#     u, v = np.mgrid[-z2:-z1:m * 1j, x1:x2:n * 1j]
#
#     # Generate enu-coordinates for each point on the face-grid
#     xy = np.outer(u, up) + np.outer(v, vec) + middle
#
#     new_xy = np.vstack([xy[:, 0], xy[:, 2], xy[:, 1]])
#
#     for i in range(len(final_hvps_rectified)):
#         hvp = final_hvps_rectified[i]
#
#         heading = np.arctan2(hvp[2], hvp[0])
#         headings = [heading, heading + np.pi]
#
#         for j in range(2):
#
#             coordinates = (R_pitch(pitch).dot(R_roll(roll).dot(R_heading(-headings[j])))).dot(new_xy).T
#             coordinates = calculate_new_pano(coordinates, im)
#
#             coordinates = coordinates.reshape(2, m, n)
#
#             img = skimage.io.imread(im_path)
#             sub = np.dstack([
#                 map_coordinates(img[:, :, 0], coordinates, order=0),
#                 map_coordinates(img[:, :, 1], coordinates, order=0),
#                 map_coordinates(img[:, :, 2], coordinates, order=0)
#             ])
#
#             save_path = os.path.join(root, 'Pano_facades', '{}_{}.jpg'.format(heading, j))
#             skimage.io.imsave(save_path, sub)


def calculate_adaptive_coor():
    y = 10
    h_fov = 160  # -60~60
    v_fov1 = -70
    v_fov2 = 70

    h_fov_main = 140
    mpp = 0.0125

    x1 = np.tan(np.radians(-h_fov / 2)) * y
    x2 = np.tan(np.radians(h_fov / 2)) * y
    width = x2 - x1

    z1 = np.tan(np.radians(v_fov1)) * y
    z2 = np.tan(np.radians(v_fov2)) * y
    height = z2 - z1

    x1_main = np.tan(np.radians(-h_fov_main / 2)) * y
    x2_main = np.tan(np.radians(h_fov_main / 2)) * y
    width_main = x2_main - x1_main


    m = int(np.ceil(height / mpp))
    n = int(np.ceil((width) / mpp))
    n_main = int(np.ceil((width_main) / mpp))

    if m % 2 == 1:
        m = m + 1
    if n % 2 == 1:
        n = n + 1
    if n_main == 1:
        n_main = n_main + 1


    # m = int(np.ceil(6656 / 180 * (v_fov2 - v_fov1)))
    # n = int(np.ceil(6656 / 180 * h_fov))

    p0 = np.array([x1, y, 0.])
    p1 = np.array([x2, y, 0.])
    middle = (p0 + p1) / 2

    up = np.array([0., 0., 1.])
    vec = p1 - p0
    dist = np.linalg.norm(vec)
    vec /= dist

    # Generate barycentric coordinates for every point on the face-grid
    u, v = np.mgrid[-z2:-z1:m * 1j, x1:x2:n * 1j]

    # Generate enu-coordinates for each point on the face-grid
    xy = np.outer(u, up) + np.outer(v, vec) + middle

    new_xy = np.vstack([xy[:, 0], xy[:, 2], xy[:, 1]])

    return new_xy, m, n, n_main



def calculate_no_adaptive_coor(h_fov, v_fov1, v_fov2, mpp=0.0125):
    y = 10

    x1 = np.tan(np.radians(-h_fov / 2)) * y
    x2 = np.tan(np.radians(h_fov / 2)) * y
    width = x2 - x1

    z1 = np.tan(np.radians(v_fov1)) * y
    z2 = np.tan(np.radians(v_fov2)) * y
    height = z2 - z1


    m = int(np.ceil(height / mpp))
    n = int(np.ceil((width) / mpp))

    if m % 2 == 1:
        m = m + 1
    if n % 2 == 1:
        n = n + 1


    # m = int(np.ceil(6656 / 180 * (v_fov2 - v_fov1)))
    # n = int(np.ceil(6656 / 180 * (h_fov2 - h_fov1)))

    p0 = np.array([x1, y, 0.])
    p1 = np.array([x2, y, 0.])
    middle = (p0 + p1) / 2

    up = np.array([0., 0., 1.])
    vec = p1 - p0
    dist = np.linalg.norm(vec)
    vec /= dist

    # Generate barycentric coordinates for every point on the face-grid
    u, v = np.mgrid[-z2:-z1:m * 1j, x1:x2:n * 1j]

    # Generate enu-coordinates for each point on the face-grid
    xy = np.outer(u, up) + np.outer(v, vec) + middle

    new_xy = np.vstack([xy[:, 0], xy[:, 2], xy[:, 1]])
    focal = y / mpp

    return new_xy, m, n, focal






def project_facade_for_refine(final_hvps_rectified, im, pitch, roll, im_path, root, tmp_folder, rendering_img_base, tmp_count):
    """
    Funzione che proietta e rettifica le facciate architettoniche da un'immagine panoramica
    utilizzando i punti di fuga orizzontali rilevati.
    
    Args:
        final_hvps_rectified: Lista dei punti di fuga orizzontali rettificati
        im: Immagine panoramica (oggetto PIL)
        pitch: Angolo di beccheggio (inclinazione verticale) della fotocamera in radianti
        roll: Angolo di rollio (rotazione laterale) della fotocamera in radianti
        im_path: Percorso dell'immagine panoramica originale
        root: Directory principale del progetto
        tmp_folder: Directory temporanea per i file intermedi
        rendering_img_base: Nome base per i file di output
        tmp_count: Contatore per il batch processing
    """

    # Determina la modalità di proiezione: adattiva o non adattiva
    no_adaptive = True  # Attualmente usa sempre la modalità non adattiva
    
    if no_adaptive:
        # Calcola le coordinate di proiezione con un campo visivo specificato
        # h_fov: Campo visivo orizzontale (140°)
        # v_fov1, v_fov2: Campo visivo verticale (da -70° a +70°)
        [new_xy, m, n, focal] = calculate_no_adaptive_coor(h_fov=140, v_fov1=-70, v_fov2=70)
    else:
        # Alternativa: usa una proiezione adattiva (attualmente non utilizzata)
        [new_xy, m, n, n_main] = calculate_adaptive_coor()
        # Calcola i limiti per la regione principale dell'immagine
        adat_1 = int((n - n_main) / 2)  # Inizio della regione principale
        adat_2 = int(n - (n - n_main) / 2)  # Fine della regione principale


    headings_list = []

    # Ciclo principale: elabora ogni punto di fuga orizzontale rilevato
    for i in range(len(final_hvps_rectified)):
        hvp = final_hvps_rectified[i]  # Punto di fuga corrente

        # Calcola l'angolo orizzontale (heading/azimut) del punto di fuga
        # arctan2(z, x) restituisce l'angolo nel piano x-z
        heading = np.arctan2(hvp[2], hvp[0])

        # Codice commentato: regolazione manuale fine dell'angolo
        #heading = heading + 0.0139
        
        # Crea due direzioni di vista: originale e opposta (+180°)
        # Questo permette di rettificare le facciate in entrambe le direzioni
        # lungo lo stesso asse del punto di fuga
        headings = [heading, heading + np.pi]  # [direzione originale, direzione opposta]

        # Elabora entrambe le direzioni (0° e 180°)
        for j in range(2):
            # Ottiene l'angolo di heading corrente
            headings_tmp = headings[j]
            
            # Calcola la matrice di rotazione combinata (in ordine di applicazione):
            # 1. R_heading(-headings_tmp): Rotazione attorno all'asse y per allineare con la direzione corrente
            # 2. R_roll(roll): Applicazione della correzione di roll (rotazione attorno all'asse z)
            # 3. R_pitch(pitch): Applicazione della correzione di pitch (rotazione attorno all'asse x)
            # Questa matrice viene applicata alle coordinate di base (new_xy)
            coordinates = (R_pitch(pitch).dot(R_roll(roll).dot(R_heading(-headings_tmp)))).dot(new_xy).T
            
            # Converte le coordinate 3D in coordinate 2D nell'immagine panoramica
            coordinates = calculate_new_pano(coordinates, im)

            # Riorganizza le coordinate nella forma richiesta per il campionamento dell'immagine
            coordinates = coordinates.reshape(2, m, n)

            # Carica l'immagine panoramica originale
            img = skimage.io.imread(im_path)
            
            # Campiona l'immagine panoramica alle coordinate calcolate per creare la vista rettificata
            # map_coordinates esegue un'interpolazione per ciascun canale colore (R, G, B)
            # order=0 indica un'interpolazione nearest-neighbor (pixel più vicino)
            sub = np.dstack([
                map_coordinates(img[:, :, 0], coordinates, order=0),  # Canale R
                map_coordinates(img[:, :, 1], coordinates, order=0),  # Canale G
                map_coordinates(img[:, :, 2], coordinates, order=0)   # Canale B
            ])

            # Gestione del salvataggio delle immagini rettificate

            # Modalità di rettificazione non adattiva (attualmente l'unica utilizzata)
            if no_adaptive:
                # Codice commentato: salvare direttamente l'immagine parzialmente rettificata
                # save_path_main = os.path.join(root, 'Pano_refine', 'VP_{}_{}.jpg'.format(i, j))
                # skimage.io.imsave(save_path_main, sub)

                # Codice commentato: salvare temporaneamente l'immagine per verifiche
                # sub_im_path = os.path.join(tmp_folder, 'tmp.jpg')
                # skimage.io.imsave(sub_im_path, sub)

                # Applica l'algoritmo di raffinamento all'immagine parzialmente rettificata
                # Questo è un passaggio cruciale: analizza l'immagine per trovare aggiustamenti più precisi
                # is_main_vp=i indica quale punto di fuga si sta elaborando (principale o secondario)
                refine_radians = simon_refine(sub.copy(), focal=focal, is_main_vp=i, tmp_count=tmp_count)
                # Codice commentato: pulizia del file temporaneo
                # os.remove(sub_im_path)

                # Se il raffinamento ha avuto successo (ha trovato una correzione angolare)
                if refine_radians != None:
                    # Definisce il percorso di output per l'immagine rettificata finale
                    # Il formato del nome è: [base]_VP_[i]_[j].jpg dove:
                    # - i: indice del punto di fuga (0, 1, ...)
                    # - j: direzione (0 = originale, 1 = opposta/180°)
                    save_path_main = rendering_img_base + '_VP_{}_{}.jpg'.format(i, j)

                    # Applica la correzione angolare ottenuta dal raffinamento
                    # Questo migliora l'allineamento della facciata
                    headings_tmp += refine_radians

                    # Codice commentato: opzioni alternative per i parametri di proiezione
                    #[tmp_xy, m_tmp, n_tmp, _] = calculate_no_adaptive_coor(h_fov=160, v_fov1=-45, v_fov2=80, mpp=0.0125*2)
                    # Requisiti rilassati (alternativa commentata)
                    # [tmp_xy, m_tmp, n_tmp, _] = calculate_no_adaptive_coor(h_fov=146, v_fov1=-40, v_fov2=75,
                    #                                                        mpp=0.0125 * 2)

                    # Calcola nuove coordinate di proiezione con parametri ottimizzati per l'immagine finale
                    # h_fov=154: campo visivo orizzontale di 154°
                    # v_fov1=-44, v_fov2=83: campo visivo verticale da -44° a +83° (asimmetrico)
                    # mpp=0.0125*2: risoluzione dimezzata rispetto alla proiezione originale
                    [tmp_xy, m_tmp, n_tmp, _] = calculate_no_adaptive_coor(h_fov=154, v_fov1=-44, v_fov2=83,
                                                                           mpp=0.0125 * 2)
                    
                    # Calcola la matrice di rotazione finale combinando pitch, roll e heading corretto
                    super_R = R_pitch(pitch).dot(R_roll(roll).dot(R_heading(-headings_tmp)))
                    # Applica la rotazione alle coordinate di base
                    tmp_coordinates = super_R.dot(tmp_xy).T


                    ########### Codice commentato: salvataggio delle informazioni di heading e pitch
                    # Questa parte salverebbe le coordinate di heading e pitch per ogni pixel dell'immagine finale
                    #ttttt_heading_pitch = save_heading_pitch_json(tmp_coordinates, im, m_tmp, n_tmp)
                    #ttttt_heading_pitch_json_path = rendering_img_base + '_VP_{}_{}_heading_pitch.npy'.format(i, j)
                    #with open(ttttt_heading_pitch_json_path, 'w') as f:
                    #    json.dump(ttttt_heading_pitch.tolist(), f)
                    #np.save(ttttt_heading_pitch_json_path, ttttt_heading_pitch)
                    #########################################

                    # Calcola le coordinate di heading per ogni pixel
                    ttttt_heading = save_heading_only(tmp_coordinates, im, m_tmp, n_tmp)
                    ttttt_heading_json_path = rendering_img_base + '_VP_{}_{}_heading_map.npy'.format(i, j)
                    with open(ttttt_heading_json_path, 'w') as f:
                        json.dump(ttttt_heading.tolist(), f)
                    # Salva solo l'ultima riga della matrice (la riga più bassa)
                    np.save(ttttt_heading_json_path, ttttt_heading[-1, :])

                    # Converte le coordinate 3D in coordinate 2D nell'immagine panoramica
                    tmp_coordinates = calculate_new_pano(tmp_coordinates, im)

                    # Riorganizza le coordinate per il campionamento dell'immagine
                    tmp_coordinates = tmp_coordinates.reshape(2, m_tmp, n_tmp)

                    # Campiona l'immagine panoramica originale con le nuove coordinate raffinate
                    # per generare l'immagine finale rettificata
                    tmp_sub = np.dstack([
                        map_coordinates(img[:, :, 0], tmp_coordinates, order=0),  # Canale R
                        map_coordinates(img[:, :, 1], tmp_coordinates, order=0),  # Canale G
                        map_coordinates(img[:, :, 2], tmp_coordinates, order=0)   # Canale B
                    ])

                    # Salva l'immagine rettificata finale
                    skimage.io.imsave(save_path_main, tmp_sub)
                    
                    # Crea il percorso per il file JSON che contiene la matrice di rotazione
                    # Questo file può essere utile per successive elaborazioni o analisi
                    json_path = rendering_img_base + '_VP_{}_{}.json'.format(i, j)

                    heading_data = {
                        "heading": float(headings_tmp),
                        "i": i,
                        "j": j
                    }

                    headings_list.append(heading_data)

                    # Salva la matrice di rotazione finale in formato JSON
                    with open(json_path, 'w') as f:
                        json.dump(super_R.tolist(), f)

                # Se il raffinamento non ha avuto successo (non è stato trovato un punto di fuga valido)
                else:
                    pass  # Non fa nulla, nessuna immagine viene salvata
                    # print('no vp')  # Codice commentato: stampa di debug


            else:
                assert no_adaptive == True
                # not implement
                sub_main =  sub[:, adat_1:adat_2, :]
                sub_left = sub[:, 0:int(n/2), :]
                sub_right = sub[:, int(n / 2):n, :]

                # save_path_main = os.path.join(root, 'Pano_refine', 'VP_{}_{}_main.jpg'.format(i, j))
                # save_path_left = os.path.join(root, 'Pano_refine', 'VP_{}_{}_left.jpg'.format(i, j))
                # save_path_right = os.path.join(root, 'Pano_refine', 'VP_{}_{}_right.jpg'.format(i, j))
                # skimage.io.imsave(save_path_main, sub_main)
                # skimage.io.imsave(save_path_left, sub_left)
                # skimage.io.imsave(save_path_right, sub_right)

    heading_json = rendering_img_base + 'heading_facade.json'
    with open(heading_json, 'w') as f:
        json.dump(headings_list, f)




def render_imgs(panorama_img, tmp_dir, tmp_dir_ifab, save_directly):

    #render_num = 16
    render_num = 4

    start = int(-render_num/2) + 1
    end = render_num + start
    degree = 360 / render_num
    img = panorama_img.copy()
    output_tiles = []

    for i in range(start, end):
        # interleaved -> planar representation of the coordinates
        coordinates = project_face(i, degree)
        coordinates = coordinates.transpose(2, 0, 1,)

        # I am getting 'heading, pitch', I want 'pitch, heading' since columns correspond to different headings
        coordinates = np.roll(coordinates, 1, axis=0) # [pitch, heading, 2]

        # Map heading from  -180 ..180 to  0...360
        coordinates[1] += 180. # Heading
        coordinates[0] = 90 - coordinates[0]  # 0 ->90 (horizontal), 90->0 (top/up) # Pitch

        coordinates[0] *= img.shape[0] / 180.
        coordinates[1] *= img.shape[1] / 360.

        # Map Coordinates require [pitch, heading, channel]
        sub = np.dstack([
            map_coordinates(img[:, :, 0], coordinates, order=0),
            map_coordinates(img[:, :, 1], coordinates, order=0),
            map_coordinates(img[:, :, 2], coordinates, order=0)
        ])
        sub = sub[::-1, :, :]
        output_tiles.append(sub)
        if not save_directly:
            save_path = os.path.join(tmp_dir, 'Render_' + str(i - start) + '.jpg')
            save_path_2 = os.path.join(tmp_dir_ifab, 'Render_' + str(i - start) + '.jpg')
            skimage.io.imsave(save_path, sub)
            skimage.io.imsave(save_path_2, sub)

    # Get the top images
    #get_the_top()
    return output_tiles

if __name__ == "__main__":
    # render_imgs()
    # project_facade_output()
    print(100)
    calculate_adaptive_coor()
