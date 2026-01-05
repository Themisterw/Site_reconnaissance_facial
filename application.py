import hashlib

import os
import shutil
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from cryptography.fernet import Fernet  # Ajout pour le chiffrement

# Désactiver les avertissements TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Répertoire personnalisé pour l'utilisation du deep-face
custom_weights_path = r'H:\user2\deepface_models'
os.environ['DEEPFACE_HOME'] = custom_weights_path
import json
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
import face_recognition
import numpy as np
from functools import wraps
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import cv2
from deepface import DeepFace


# Créer une session de l'application flask
app = Flask(__name__)
app.secret_key = 'session'
app.jinja_env.globals.update(zip=zip)


# Chemin vers le fichier de clé
KEY_FILE = 'secret.key'

# Fonction pour générer et sauvegarder la clé
def generate_key():
    """
    Génère une clé et la sauvegarde dans un fichier.
    """
    key = Fernet.generate_key()
    with open(KEY_FILE, 'wb') as key_file:
        key_file.write(key)

# Fonction pour charger la clé depuis le fichier
def load_key():
    """
    Charge la clé depuis le fichier.
    """
    return open(KEY_FILE, 'rb').read()

# Si le fichier de clé n'existe pas, générer une nouvelle clé
if not os.path.exists(KEY_FILE):
    generate_key()

# Charger la clé depuis le fichier
key = load_key()
cipher_suite = Fernet(key)


# Chemin JSON pour stocker les utilisateurs (répertoire: JSON)
FICHIER_UTILISATEURS = 'json/utilisateurs.json'
os.makedirs(os.path.dirname(FICHIER_UTILISATEURS), exist_ok=True)  # Vérifier si le JSON existe


# Fonction pour chiffrer les valeurs
def chiffrer_valeurs(data):
    def chiffrer_valeur(v):
        if v is None:
            return None
        elif isinstance(v, (str, int, float)):
            return cipher_suite.encrypt(str(v).encode()).decode()
        elif isinstance(v, dict):
            return chiffrer_valeurs(v)
        elif isinstance(v, list):
            return [chiffrer_valeur(item) for item in v]
        else:
            raise TypeError(f"Type non pris en charge pour le chiffrement: {type(v)}")

    return {k: chiffrer_valeur(v) for k, v in data.items()}

# Fonction pour déchiffrer les valeurs
def dechiffrer_valeurs(data):
    def dechiffrer_valeur(v):
        if v is None:
            return None
        elif isinstance(v, str):
            try:
                return cipher_suite.decrypt(v.encode()).decode()
            except:
                return v
        elif isinstance(v, dict):
            return dechiffrer_valeurs(v)
        elif isinstance(v, list):
            return [dechiffrer_valeur(item) for item in v]
        else:
            raise TypeError(f"Type non pris en charge pour le déchiffrement: {type(v)}")

    return {k: dechiffrer_valeur(v) for k, v in data.items()}

# Fonction pour déchiffrer les utilisateurs
def dechiffrer_utilisateurs(utilisateurs):
    utilisateurs_dechiffres = {}
    for login, details in utilisateurs.items():
        utilisateurs_dechiffres[login] = {
            'prenom': dechiffrer_valeurs({'prenom': details['prenom']})['prenom'],
            'nom': dechiffrer_valeurs({'nom': details['nom']})['nom'],
            'login': details['login'],
            'email': dechiffrer_valeurs({'email': details['email']})['email'],
            'mot_de_passe': details['mot_de_passe'],  # On peut aussi déchiffrer le mot de passe si nécessaire
            'date_naissance': dechiffrer_valeurs({'date_naissance': details['date_naissance']})['date_naissance'],
            'adresse': dechiffrer_valeurs({'adresse': details['adresse']})['adresse'],
            'role': details['role'],
            'abonnement': details['abonnement']
        }
    return utilisateurs_dechiffres

# Fonction pour lire les utilisateurs du fichier JSON
def lire_utilisateurs():
    utilisateurs = {}

    if os.path.exists(FICHIER_UTILISATEURS):
        with open(FICHIER_UTILISATEURS, 'r', encoding='utf-8') as fichier:
            try:
                data = json.load(fichier)
                utilisateurs = dechiffrer_valeurs(data)
            except json.JSONDecodeError:
                pass

    # Vérifier s'il existe un super_admin
    super_admin_existe = any(utilisateur['role'] == 'super_admin' for utilisateur in utilisateurs.values())

    if not super_admin_existe:
        # Ajouter un utilisateur par défaut
        utilisateur_par_defaut = {
            'admin': {
                'prenom': 'Default',
                'nom': 'Admin',
                'login': 'admin',
                'email': 'admin@example.com',
                'mot_de_passe': 'admin',  # À modifier pour plus de sécurité
                'date_naissance': '1970-01-01',
                'adresse': 'Default Address',
                'role': 'super_admin',
                'abonnement': None
            }
        }
        utilisateurs.update(utilisateur_par_defaut)
        ecrire_utilisateurs(utilisateurs)

    return utilisateurs


# Fonction pour écrire les utilisateurs dans le fichier JSON
def ecrire_utilisateurs(utilisateurs):
    with open(FICHIER_UTILISATEURS, 'w', encoding='utf-8') as fichier:
        json.dump(chiffrer_valeurs(utilisateurs), fichier, ensure_ascii=False, indent=4)

# Charger les utilisateurs au démarrage de l'application
utilisateurs = lire_utilisateurs()


# Pour la détection, créer une liste vide contenant l'encodage des visages et le nom.
encodages_visages_connus = []
noms_visages_connus = []

# Cherche le dossier dans visages_connus et vérifier si le dossier existe
visages_connus_dossier = 'visages_connus'

if not os.path.exists(visages_connus_dossier):
    print(f"Le dossier {visages_connus_dossier} n'existe pas.")
else:
    # Il prend toute la liste des noms dans le répertoire dans visages_connus_dossier
    for nom_personnage in os.listdir(visages_connus_dossier):
        chemin_dossier = os.path.join(visages_connus_dossier, nom_personnage)
        if os.path.isdir(chemin_dossier):
            for fichier in os.listdir(chemin_dossier):
                if fichier.endswith('.jpg') or fichier.endswith('.png'):
                    # créer le chemin
                    chemin_image = os.path.join(chemin_dossier, fichier)
                    # ouvre l'image
                    image = face_recognition.load_image_file(chemin_image)
                    # et prend l'encodage
                    encodage_visage = face_recognition.face_encodings(image)
                    if encodage_visage:
                        # le met dans le dossier pour l'utiliser plus tard
                        encodages_visages_connus.append(encodage_visage[0])
                        noms_visages_connus.append(nom_personnage)

                    else:
                        # Boucle si il ne la trouve pas
                        print(f"Aucun encodage trouvé pour {nom_personnage} dans {fichier}")
        else:
            print(f"{chemin_dossier} n'est pas un dossier")
# A l'ouverture de l'application, on a donc la liste des encodages de visages et leur nom prêt à être utilisé


# Permet de sécuriser le code et autorise la fonction que aux administrateurs
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'utilisateur' not in session or session.get('role') not in ['admin', 'super_admin']:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/")
def acceuil():
    flash_message = session.pop('flash_message', None)
    return render_template("index.html", flash_message=flash_message)

# La personne peut se connecter au travers de login.html avec des informations récupérées à partir d'un JSON (lire utilisateur)
@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        pseudonyme = request.form['pseudonyme']
        mot_de_passe = request.form['mot_de_passe']
        if pseudonyme in utilisateurs and utilisateurs[pseudonyme]['mot_de_passe'] == mot_de_passe:
            session['utilisateur'] = pseudonyme
            session['role'] = utilisateurs[pseudonyme]['role']
            return redirect(url_for('acceuil'))
        else:
            return render_template('login.html', erreur="Pseudonyme ou mot de passe incorrect.")
    return render_template('login.html')

# Peut se déconnecter en supprimant l'utilisateur de la session
@app.route("/logout")
def logout():
    session.pop('utilisateur', None)
    session.pop('role', None)
    return redirect(url_for('acceuil'))

# L'inscription consiste à prendre les données postées par l'utilisateur lors de l'inscription et de les réécrire
@app.route("/inscription", methods=['GET', 'POST'])
def inscription():
    if request.method == 'POST':
        prenom = request.form['prenom']
        nom = request.form['nom']
        login = request.form['login']
        email = request.form['email']
        mot_de_passe = request.form['mot_de_passe']
        confirm_mot_de_passe = request.form['confirm_mot_de_passe']
        date_naissance = request.form['date_naissance']
        adresse = request.form['adresse']

        if mot_de_passe == confirm_mot_de_passe:
            if login in utilisateurs:
                flash("Ce nom d'utilisateur est déjà pris.", "error")
                return redirect(url_for('inscription'))

            utilisateur = {
                'prenom': prenom,
                'nom': nom,
                'login': login,
                'email': email,
                'mot_de_passe': mot_de_passe,
                'date_naissance': date_naissance,
                'adresse': adresse,
                'role': 'user',
                'abonnement': None
            }
            utilisateurs[login] = utilisateur
            ecrire_utilisateurs(utilisateurs)
            flash("Inscription réussie!", "success")
            return redirect(url_for('inscription', success=True))
        else:
            flash("Les mots de passe ne correspondent pas.", "error")
            return redirect(url_for('inscription'))

    return render_template('inscription.html')

# Menureconnaissancefacial
@app.route("/ReconnaissanceImageMenu")
def ReconnaissanceImageMenu():
    return render_template("menureconnaissancefacial.html")

# Création d'un dossier temporaire et si il existe le mettre dans la session
UPLOAD_FOLDER = 'uploads_temp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Fonction reconnaîtra le visage et à partir de la base de données encodée précédemment déterminera le  encodage le plus proche
@app.route("/reconnaitre", methods=['POST'])
def reconnaitre():
    fichier = request.files.get('image')
    if not fichier:
        return redirect('/ReconnaissanceImageMenu')

    # Charger l'image téléchargée sans redimension (pour ne pas perdre de l'information)
    image = Image.open(fichier)
    print(f"Original image size: {image.size}")

    image_array = np.array(image)

    encodages_visages_telecharges = face_recognition.face_encodings(image_array)
    print(f"Number of faces detected: {len(encodages_visages_telecharges)}")

    noms = []
    if len(encodages_visages_telecharges) > 0:
        for encodage_visage_telecharge in encodages_visages_telecharges:
            # Comparer le visage téléchargé aux visages connus avec une tolérance ajustable
            tolerance = 0.7  # Valeur ajustable
            correspondances = face_recognition.compare_faces(encodages_visages_connus, encodage_visage_telecharge, tolerance=tolerance)
            distances_visages = face_recognition.face_distance(encodages_visages_connus, encodage_visage_telecharge)
            # mesure la distance de chaque visage
            print(f"Face distances: {distances_visages}")

            if len(distances_visages) > 0:
                indice_meilleure_correspondance = np.argmin(distances_visages)
                print(f"Meilleure correspondance index: {indice_meilleure_correspondance}, correspondance: {correspondances[indice_meilleure_correspondance]}")
                if correspondances[indice_meilleure_correspondance]:
                    nom = noms_visages_connus[indice_meilleure_correspondance]
                else:
                    nom = "Inconnu"
            else:
                nom = "Aucun visage détecté"

            # Ajouter des tests pour vérifier les correspondances
            print(f"Encodage téléchargé: {encodage_visage_telecharge}")
            print(f"Correspondances: {correspondances}")
            print(f"Distances: {distances_visages}")

            noms.append(nom)
    else:
        noms.append("Aucun visage détecté")

    # Stocker l'image temporairement
    filename = secure_filename(fichier.filename)
    temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(temp_image_path)

    session['noms'] = noms
    session['temp_image_path'] = temp_image_path  # Stocker le chemin de l'image temporaire

    print(f"Résultats: {noms}")
    return redirect(url_for('resultat', noms=",".join(noms)))


# Afficher le résultat et prépare aussi les données pour les statistiques
@app.route("/resultat")
def resultat():
    noms = request.args.get('noms', '').split(',')
    tous_inconnus = all(nom == "Inconnu" for nom in noms)
    session['tous_inconnus'] = tous_inconnus
    session['noms'] = noms  # Stocker les noms dans la session
    return render_template('resultat.html', noms=noms)

"""def redimensionner_image(image, largeur, hauteur):
    # Utiliser l'interpolation bicubique pour une meilleure qualité
    return image.resize((largeur, hauteur), Image.BICUBIC)"""

# Route pour détecter une personne à partir d'un mot et délimiter une personne spécifique dans une photo
@app.route("/detecter_personne", methods=['POST'])
def detecter_personne():
    fichier = request.files.get('image')
    nom_personne = request.form.get('nom')

    if not fichier or not nom_personne:
        flash("Veuillez fournir une image et un nom.", "error")
        return redirect(url_for('DetectionVisageMenu'))
    # Transforme le nom du dossier en nom et prénom
    nom_personne = nom_personne.replace('-', ' ')

    try:
        image = Image.open(fichier)
        max_size = (800, 600)  # ne pas dépasser cette taille car sinon c'est trop gros
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        image_array = np.array(image)  # le transformer en matrice

    except Exception as e:
        flash("Erreur lors du chargement de l'image.", "error")
        print(f"Erreur lors du chargement de l'image : {e}")
        return redirect(url_for('DetectionVisageMenu'))

    print(f"Image dimensions: {image_array.shape}")

    encodages_visages_telecharges = face_recognition.face_encodings(image_array)  # prendre son encodage
    face_locations = face_recognition.face_locations(image_array)
    print(f"Number of faces detected: {len(encodages_visages_telecharges)}")
    detections = []

    if len(encodages_visages_telecharges) > 0:
        for i, encodage_visage_telecharge in enumerate(encodages_visages_telecharges):
            correspondances = face_recognition.compare_faces(encodages_visages_connus, encodage_visage_telecharge)
            distances_visages = face_recognition.face_distance(encodages_visages_connus, encodage_visage_telecharge)
            print(f"Face distances: {distances_visages}")

            if any(correspondances):
                for j, match in enumerate(correspondances):
                    print(f"Matching with known face {noms_visages_connus[j]}: {match}")

                    if match:
                        nom_connu = noms_visages_connus[j].replace('-', ' ')
                        print(f"Nom connu: {nom_connu}, Nom fourni: {nom_personne}")
                        if nom_connu.lower() == nom_personne.lower():
                            nom = noms_visages_connus[j]
                            location = face_locations[i]
                            print(f"Found {nom} at {location}")
                            top, right, bottom, left = location
                            # détecte la tête et dessine un rectangle où il y a la tête
                            draw = ImageDraw.Draw(image)
                            draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=3)
                            buffered = BytesIO()
                            image.save(buffered, format="JPEG")
                            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                            detections.append({
                                'nom': nom,
                                'location': location,
                                'image_data': img_str
                            })  # mets la détection au travers d'une liste appelée détection
                            # prendre le nom dans
                            session['detections'] = detections[0]['nom']
                            print(f"Détection réussie enregistrée : {detections}")
                            return redirect(url_for('resultatDetection', nom=nom, image_data=img_str))

        # Si aucun visage reconnu correspond
        detections = "Inconnu"
    else:
        detections = "Aucun visage détecté"

    session['detections'] = detections
    print(f"Session detections mises à jour : {detections}")
    return redirect(url_for('resultatDetection', nom="Inconnu", image_data=None))

# Route pour afficher le résultat, les valeurs sont également gardées pour les statistiques
@app.route("/resultatDetection")
def resultatDetection():
    nom = request.args.get('nom')
    image_data = request.args.get('image_data')
    return render_template('resultatdetectionvisage.html', nom=nom, image_data=image_data)

# Menu pour la détection des visages au moment où on appuie sur le menu gauche
@app.route("/DetectionVisageMenu")
def DetectionVisageMenu():
    return render_template("menudetectionvisage.html")

"""@app.route('/admin_dashboard')
@admin_required
def admin_dashboard():
    return render_template('admin_dashboard.html')"""

# Route qui permet aux administrateurs de gérer les rôles
@app.route('/user_management', methods=['GET', 'POST'])
@admin_required
def user_management():
    if request.method == 'POST':
        # Récupérer les données du formulaire pour modifier les rôles des utilisateurs
        pseudonyme = request.form['pseudonyme']
        nouveau_role = request.form['role']
        if pseudonyme in utilisateurs:
            # Empêcher un administrateur de dégrader un autre administrateur ou le super admin
            if utilisateurs[pseudonyme]['role'] == 'super_admin' and session['utilisateur'] != pseudonyme:
                flash("Vous ne pouvez pas dégrader le super administrateur.", "error")
            elif pseudonyme == session['utilisateur'] and nouveau_role != utilisateurs[pseudonyme]['role']:
                flash("Vous ne pouvez pas modifier votre propre rôle.", "error")
            elif utilisateurs[pseudonyme]['role'] == 'admin' and session['role'] != 'super_admin':
                flash("Seul le super administrateur peut dégrader un administrateur.", "error")
            else:
                utilisateurs[pseudonyme]['role'] = nouveau_role
                ecrire_utilisateurs(utilisateurs)
                flash(f"Le rôle de {pseudonyme} a été mis à jour en {nouveau_role}.", "success")
        else:
            flash(f"L'utilisateur {pseudonyme} n'existe pas.", "error")
    utilisateurs_dechiffres = dechiffrer_utilisateurs(utilisateurs)
    return render_template('gestionUtilisateur.html', utilisateurs=utilisateurs_dechiffres)

# Menu qui permet à l'utilisateur la photo pour la reconnaissance d'émotion
@app.route('/menureconnaissanceemotion')
def menu_reconnaissance_emotion():
    return render_template('menureconnaissanceemotion.html')

# Utilisation de DeepFace pour la reconnaissance des émotions en anglais, donc on les traduit à l'aide d'un dictionnaire

# Dictionnaire de traduction des émotions
traduction_emotions = {
    "angry": "en colère",
    "disgust": "dégoût",
    "fear": "peur",
    "happy": "heureux",
    "sad": "triste",
    "surprise": "surpris",
    "neutral": "neutre"
}

def redimensionner_image(image, largeur):
    ratio_aspect = image.shape[1] / image.shape[0]
    nouvelle_hauteur = int(largeur / ratio_aspect)
    return cv2.resize(image, (largeur, nouvelle_hauteur))

# Algorithme de reconnaissance d'émotion reprenant les deux principes mais ajoutant une détection d'émotion et d'âge
@app.route('/reconnaissanceemotion', methods=['POST'])
def reconnaissanceemotion():
    conclusions = []
    fichier = request.files['image']
    chemin_fichier = 'image_telechargee.jpg'
    fichier.save(chemin_fichier)

    # Charger l'image téléchargée
    image = face_recognition.load_image_file(chemin_fichier)
    image = redimensionner_image(image, 800)  # Redimensionner l'image
    print("Image chargée et redimensionnée")

    # Trouver tous les emplacements des visages dans l'image
    emplacement_visages = face_recognition.face_locations(image)
    print(f"Visages détectés : {emplacement_visages}")

    emotions = []
    for emplacement_visage in emplacement_visages:
        # Extraire le visage
        haut, droite, bas, gauche = emplacement_visage
        image_visage = image[haut:bas, gauche:droite]

        # Convertir l'image du visage en BGR pour OpenCV
        image_visage = cv2.cvtColor(image_visage, cv2.COLOR_RGB2BGR)

        # Analyser l'émotion du visage extrait
        try:
            resultat = DeepFace.analyze(image_visage, actions=['emotion'])
            print(f"Résultat de l'analyse : {resultat}")

            # Traduire les émotions
            emotion_traduite = {traduction_emotions[emo]: score for emo, score in resultat[0]['emotion'].items()}
            emotions.append(emotion_traduite)  # Ajouter les émotions traduites
            # Déterminer l'émotion dominante
            emotion_dominante = max(emotion_traduite, key=emotion_traduite.get)
            conclusions.append(f"Le personnage est principalement {emotion_dominante}.")
        except Exception as e:
            print(f"Erreur lors de l'analyse de l'émotion: {e}")
            emotions.append({"message": "Erreur lors de l'analyse de l'émotion."})

    # Si aucun visage n'est détecté
    if not emplacement_visages:
        emotions = [{"message": "Aucun visage détecté dans l'image."}]
        print("Aucun visage détecté dans l'image")
        conclusions.append("Aucun visage détecté dans l'image.")

    # Dessiner des cadres autour des visages détectés et ajouter le texte de l'émotion dominante
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    # Définir la police et la taille du texte pour qu'il soit plus gros pour qu'on observe bien sur l'image la détection en rouge avec l'émotion qui est écrite
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for (haut, droite, bas, gauche), emotion, conclusion in zip(emplacement_visages, emotions, conclusions):
        draw.rectangle([gauche, haut, droite, bas], outline="red", width=2)
        # Ajouter le texte de l'émotion dominante sous le rectangle
        if 'message' not in emotion:
            emotion_dominante = conclusion.split(' ')[-1].rstrip('.')
            text = f"{emotion_dominante}"
            draw.text((gauche, bas + 5), text, fill="red", font=font)

    # Convertir l'image avec les cadres en base64 (cela permet de l'envoyer)
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    print("Image convertie en base64")
    # Sauvegarder les émotions et le chemin de l'image dans la session
    session['emotions'] = emotions

    return render_template('resultatreconnaissanceemotion.html', image_base64=image_base64, face_locations=emplacement_visages, emotions=emotions, conclusions=conclusions)


# Age et sexe et reconnaissance

# Utilisation de différents modèles pour la reconnaissance âge sexe deep
models = [
    "age_model_weights.h5",
    "gender_model_weights.h5"
]

# Bien vérifier si ces modèles existent
os.makedirs(custom_weights_path, exist_ok=True)
@app.route('/menureconnaissanceagesexe')
def menu_reconnaissance_agesexe():
    return render_template('menureconnaissanceagesexe.html')


@app.route('/reconnaissanceagesexe', methods=['POST'])
def reconnaissanceagesexe():
    fichier = request.files['image']
    image = face_recognition.load_image_file(fichier)
    image = redimensionner_image(image, 800)  # Redimensionner l'image
    print("Image chargée et redimensionnée")

    # Trouver tous les emplacements des visages dans l'image
    emplacement_visages = face_recognition.face_locations(image)
    print(f"Visages détectés : {emplacement_visages}")

    ages_et_sexes = []
    for emplacement_visage in emplacement_visages:
        # Extraire le visage
        haut, droite, bas, gauche = emplacement_visage
        image_visage = image[haut:bas, gauche:droite]

        # Convertir l'image du visage en BGR pour OpenCV (problème)
        image_visage_bgr = cv2.cvtColor(image_visage, cv2.COLOR_RGB2BGR)

        # Analyser l'âge et le sexe du visage extrait
        try:
            resultat = DeepFace.analyze(image_visage_bgr, actions=['age', 'gender'])
            print(f"Résultat de l'analyse : {resultat}")
            ages_et_sexes.append(resultat[0])  # Accéder correctement aux résultats
        except Exception as e:
            print(f"Erreur lors de l'analyse de DeepFace: {e}")
            ages_et_sexes.append({"message": "Erreur lors de l'analyse du visage."})

    # Dessiner des cadres autour des visages détectés
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    # Définir la police et la taille du texte
    try:
        font = ImageFont.truetype("arial.ttf", 20)  # Assurez-vous que le fichier de police arial.ttf est disponible
    except IOError:
        font = ImageFont.load_default()

    for (haut, droite, bas, gauche), age_sexe in zip(emplacement_visages, ages_et_sexes):
        draw.rectangle([gauche, haut, droite, bas], outline="red", width=2)
        if 'age' in age_sexe and 'dominant_gender' in age_sexe:
            text = f"{age_sexe['age']}, {age_sexe['dominant_gender']}"
            # Calculer la position du texte sous le rectangle
            text_position = (gauche, bas + 5)
            draw.text(text_position, text, fill="red", font=font)

    # Convertir l'image avec les cadres en base64 pour pouvoir être envoyée directement mais il y a un moyen de changer l'image
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    session['ages_et_sexes'] = ages_et_sexes

    return render_template('resultatagegenre.html', image_base64=image_base64, ages_et_sexes=ages_et_sexes)

CHEMIN_FICHIER_JSON = 'json/avis.json'

def charger_avis(CHEMIN_FICHIER_JSON):
    if os.path.exists(CHEMIN_FICHIER_JSON):
        with open(CHEMIN_FICHIER_JSON, 'r') as fichier:
            return json.load(fichier)
    return {'vrai_positif': 0, 'faux_positif': 0, 'vrai_negatif': 0, 'faux_negatif': 0}


def sauvegarder_avis(donnees_avis, CHEMIN_FICHIER_JSON):
    with open(CHEMIN_FICHIER_JSON, 'w') as fichier:
        json.dump(donnees_avis, fichier)
# Utilisation d'une fonction de hachage car envoi très lourd et peut être lent avec l'image
def calculer_hachage(fichier_path):
    hachage = hashlib.md5()
    with open(fichier_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hachage.update(chunk)
    return hachage.hexdigest()
# On peut donner notre avis cela aidera l'administrateur de plus tous les vrai positifs iront dans la base de données
@app.route('/soumettre_avis', methods=['POST'])
def soumettre_avis():
    avis = request.form['feedback']
    noms = session.get('noms', [])
    tous_inconnus = session.get('tous_inconnus', False)
    temp_image_path = session.get('temp_image_path', '')  # Récupérer le chemin de l'image temporaire
    donnees_avis = charger_avis(CHEMIN_FICHIER_JSON)

    if not tous_inconnus:
        if avis == 'bon':
            donnees_avis['vrai_positif'] += 1
            if temp_image_path and os.path.exists(temp_image_path):
                for nom in noms:
                    if nom != "Inconnu":
                        person_folder = os.path.join('visages_connus', nom)
                        os.makedirs(person_folder, exist_ok=True)

                        # Vérifier si l'image existe déjà
                        temp_image_hachage = calculer_hachage(temp_image_path)
                        image_deja_existante = False
                        for existing_file in os.listdir(person_folder):
                            existing_file_path = os.path.join(person_folder, existing_file)
                            if calculer_hachage(existing_file_path) == temp_image_hachage:
                                image_deja_existante = True
                                break
                        # Lorsque l'on soumet un avis vrai positif pour cette détection, cette image alimente notre base de données
                        if not image_deja_existante:
                            # Compter le nombre de fichiers dans le dossier pour générer un nom de fichier unique
                            compteur = len(os.listdir(person_folder)) + 1
                            extension = os.path.splitext(temp_image_path)[1]
                            filename = f"{nom}{compteur}{extension}"
                            file_path = os.path.join(person_folder, filename)
                            shutil.move(temp_image_path, file_path)  # Déplacer l'image vers le dossier spécifique
                            break  # arrêter après avoir enregistré dans le premier dossier correspondant
        else:
            donnees_avis['vrai_negatif'] += 1
            if temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)
    else:
        if avis == 'bon':
            donnees_avis['faux_positif'] += 1
            if temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)
        else:
            donnees_avis['faux_negatif'] += 1
            if temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)

    sauvegarder_avis(donnees_avis, CHEMIN_FICHIER_JSON)
    return redirect(url_for('merci'))

# Création des statistiques et de la matrice de confusion qu'utilisera l'admin
@app.route('/statistiques')
@admin_required
def statistiques():
    donnees_avis = charger_avis(CHEMIN_FICHIER_JSON)
    vrai_positif = donnees_avis['vrai_positif']
    faux_positif = donnees_avis['faux_positif']
    vrai_negatif = donnees_avis['vrai_negatif']
    faux_negatif = donnees_avis['faux_negatif']

    total = vrai_positif + faux_positif + vrai_negatif + faux_negatif

    if total == 0:
        accuracy = precision = recall = f1 = mcc = 0
    else:
        accuracy = (vrai_positif + vrai_negatif) / total
        precision = vrai_positif / (vrai_positif + faux_positif) if (vrai_positif + faux_positif) > 0 else 0
        recall = vrai_positif / (vrai_positif + faux_negatif) if (vrai_positif + faux_negatif) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        mcc = ((vrai_positif * vrai_negatif) - (faux_positif * faux_negatif)) / (
                ((vrai_positif + faux_positif) * (vrai_positif + faux_negatif) * (vrai_negatif + faux_positif) * (vrai_negatif + faux_negatif)) ** 0.5) \
            if ((vrai_positif + faux_positif) * (vrai_positif + faux_negatif) * (vrai_negatif + faux_positif) * (vrai_negatif + faux_negatif)) > 0 else 0

    cm = [[vrai_positif, faux_positif], [faux_negatif, vrai_negatif]]

    return render_template('statistiques.html', cm=cm, accuracy=accuracy, precision=precision, recall=recall, f1=f1, mcc=mcc)


CHEMIN_FICHIER_JSON2 = 'json/avisagegenre.json'

# Soumission avis age et genre
@app.route('/soumettre_avis_age_genre', methods=['POST'])
def soumettre_avis_age_genre():
    avis = request.form['feedback']
    print(f"Feedback reçu: {avis}")
    ages_et_sexes = session.get('ages_et_sexes', [])
    temp_image_path = session.get('temp_image_path', '')

    print(f"Session ages_et_sexes: {ages_et_sexes}")
    print(f"Session temp_image_path: {temp_image_path}")

    donnees_avis = charger_avis(CHEMIN_FICHIER_JSON2)
    print(f"Chargement des données d'avis: {donnees_avis}")

    for age_sexe in ages_et_sexes:
        print(f"Traitement de age_sexe: {age_sexe}")
        if 'message' not in age_sexe:
            if avis == 'bon':
                donnees_avis['vrai_positif'] += 1
                print(f"Vrai positif incrémenté")
            else:
                donnees_avis['vrai_negatif'] += 1
                print(f"Faux positif incrémenté")
        else:
            if avis == 'bon':
                donnees_avis['faux_positif'] += 1
                print(f"Vrai négatif incrémenté")
            else:
                donnees_avis['faux_negatif'] += 1
                print(f"Faux négatif incrémenté")

    if temp_image_path and os.path.exists(temp_image_path):
        os.remove(temp_image_path)
        print(f"Image temporaire supprimée: {temp_image_path}")

    # Sauvegarder les données sans initialiser 'age' et 'genre'
    sauvegarder_avis(donnees_avis, CHEMIN_FICHIER_JSON2)
    print(f"Sauvegarde des données d'avis: {donnees_avis}")

    return redirect(url_for('merci'))

# Remercie quand la personne a donné son avis
@app.route('/merci')
def merci():
    return render_template('merci.html')

# Statistiques age genre
@app.route('/statistiquesagesexe')
@admin_required
def statistiquesagesexe():
    donnees_avis = charger_avis(CHEMIN_FICHIER_JSON2)
    vrai_positif = donnees_avis['vrai_positif']
    faux_positif = donnees_avis['faux_positif']
    vrai_negatif = donnees_avis['vrai_negatif']
    faux_negatif = donnees_avis['faux_negatif']

    total = vrai_positif + faux_positif + vrai_negatif + faux_negatif

    if total == 0:
        accuracy = precision = recall = f1 = mcc = 0
    else:
        accuracy = (vrai_positif + vrai_negatif) / total
        precision = vrai_positif / (vrai_positif + faux_positif) if (vrai_positif + faux_positif) > 0 else 0
        recall = vrai_positif / (vrai_positif + faux_negatif) if (vrai_positif + faux_negatif) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        mcc = ((vrai_positif * vrai_negatif) - (faux_positif * faux_negatif)) / (
                ((vrai_positif + faux_positif) * (vrai_positif + faux_negatif) * (vrai_negatif + faux_positif) * (vrai_negatif + faux_negatif)) ** 0.5) \
            if ((vrai_positif + faux_positif) * (vrai_positif + faux_negatif) * (vrai_negatif + faux_positif) * (vrai_negatif + faux_negatif)) > 0 else 0

    cm = [[vrai_positif, faux_positif], [faux_negatif, vrai_negatif]]

    return render_template('statistiquesagesexe.html', cm=cm, accuracy=accuracy, precision=precision, recall=recall, f1=f1, mcc=mcc)


CHEMIN_FICHIER_JSON3 = 'json/avisemotion.json'
# Soumission avis détection émotion
@app.route('/soumettre_emotion', methods=['POST'])
def soumettre_emotion():
    avis = request.form['feedback']
    print(f"Feedback reçu: {avis}")
    emotions = session.get('emotions', [])
    temp_image_path = session.get('temp_image_path', '')

    print(f"Session emotions: {emotions}")
    print(f"Session temp_image_path: {temp_image_path}")

    donnees_avis = charger_avis(CHEMIN_FICHIER_JSON3)
    print(f"Chargement des données d'avis: {donnees_avis}")

    for emotion in emotions:
        print(f"Traitement de emotion: {emotion}")
        if 'message' not in emotion:
            if avis == 'bon':
                donnees_avis['vrai_positif'] += 1
                print(f"Vrai positif incrémenté")
            else:
                donnees_avis['vrai_negatif'] += 1
                print(f"Faux positif incrémenté")
        else:
            if avis == 'bon':
                donnees_avis['faux_positif'] += 1
                print(f"Vrai négatif incrémenté")
            else:
                donnees_avis['faux_negatif'] += 1
                print(f"Faux négatif incrémenté")

    if temp_image_path and os.path.exists(temp_image_path):
        os.remove(temp_image_path)
        print(f"Image temporaire supprimée: {temp_image_path}")

    # Sauvegarder les données sans initialiser 'age' et 'genre'
    sauvegarder_avis(donnees_avis, CHEMIN_FICHIER_JSON3)
    print(f"Sauvegarde des données d'avis: {donnees_avis}")

    return redirect(url_for('merci'))

# Statistiques émotions
@app.route('/statistiquesemotion')
@admin_required
def statistiquesemotion():
    donnees_avis = charger_avis(CHEMIN_FICHIER_JSON3)
    vrai_positif = donnees_avis['vrai_positif']
    faux_positif = donnees_avis['faux_positif']
    vrai_negatif = donnees_avis['vrai_negatif']
    faux_negatif = donnees_avis['faux_negatif']

    total = vrai_positif + faux_positif + vrai_negatif + faux_negatif

    if total == 0:
        accuracy = precision = recall = f1 = mcc = 0
    else:
        accuracy = (vrai_positif + vrai_negatif) / total
        precision = vrai_positif / (vrai_positif + faux_positif) if (vrai_positif + faux_positif) > 0 else 0
        recall = vrai_positif / (vrai_positif + faux_negatif) if (vrai_positif + faux_negatif) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        mcc = ((vrai_positif * vrai_negatif) - (faux_positif * faux_negatif)) / (
                ((vrai_positif + faux_positif) * (vrai_positif + faux_negatif) * (vrai_negatif + faux_positif) * (vrai_negatif + faux_negatif)) ** 0.5) \
            if ((vrai_positif + faux_positif) * (vrai_positif + faux_negatif) * (vrai_negatif + faux_positif) * (vrai_negatif + faux_negatif)) > 0 else 0

    cm = [[vrai_positif, faux_positif], [faux_negatif, vrai_negatif]]

    return render_template('statistiquesemotion.html', cm=cm, accuracy=accuracy, precision=precision, recall=recall, f1=f1, mcc=mcc)

CHEMIN_FICHIER_JSON4 = 'json/avisdetection.json'
# Soumission avis détections personnages
@app.route('/soumettre_avis_detection', methods=['POST'])
def soumettre_avis_detection():
    avis = request.form['feedback']
    print(f"Feedback reçu: {avis}")
    detections = session.get('detections', '')
    temp_image_path = session.get('temp_image_path', '')

    print(f"Session detections avant traitement: {detections}")
    print(f"Session temp_image_path: {temp_image_path}")

    donnees_avis = charger_avis(CHEMIN_FICHIER_JSON4)
    print(f"Chargement des données d'avis: {donnees_avis}")
    print(f"Détection avant analyse: {detections}")

    faux_detecte = detections in ["Aucun visage détecté", "Inconnu"]
    print(f"Faux détecté ? {faux_detecte}")

    if faux_detecte:
        if avis == 'bon':
            donnees_avis['faux_positif'] += 1
            print(f"Faux positif incrémenté")
        else:
            donnees_avis['faux_negatif'] += 1
            print(f"Vrai négatif incrémenté")
    else:
        if avis == 'bon':
            donnees_avis['vrai_positif'] += 1
            print(f"Vrai positif incrémenté")
        else:
            donnees_avis['vrai_negatif'] += 1
            print(f"Faux négatif incrémenté")

    if temp_image_path and os.path.exists(temp_image_path):
        os.remove(temp_image_path)
        print(f"Image temporaire supprimée: {temp_image_path}")

    # Sauvegarder les données
    sauvegarder_avis(donnees_avis, CHEMIN_FICHIER_JSON4)
    print(f"Sauvegarde des données d'avis: {donnees_avis}")

    return redirect(url_for('merci'))

# Statistiques détections visages
@app.route('/statistiquesdetection')
@admin_required
def statistiquesdetection():
    donnees_avis = charger_avis(CHEMIN_FICHIER_JSON4)
    vrai_positif = donnees_avis['vrai_positif']
    faux_positif = donnees_avis['faux_positif']
    vrai_negatif = donnees_avis['vrai_negatif']
    faux_negatif = donnees_avis['faux_negatif']

    total = vrai_positif + faux_positif + vrai_negatif + faux_negatif

    if total == 0:
        accuracy = precision = recall = f1 = mcc = 0
    else:
        accuracy = (vrai_positif + vrai_negatif) / total
        precision = vrai_positif / (vrai_positif + faux_positif) if (vrai_positif + faux_positif) > 0 else 0
        recall = vrai_positif / (vrai_positif + faux_negatif) if (vrai_positif + faux_negatif) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        mcc = ((vrai_positif * vrai_negatif) - (faux_positif * faux_negatif)) / (
                ((vrai_positif + faux_positif) * (vrai_positif + faux_negatif) * (vrai_negatif + faux_positif) * (vrai_negatif + faux_negatif)) ** 0.5) \
            if ((vrai_positif + faux_positif) * (vrai_positif + faux_negatif) * (vrai_negatif + faux_positif) * (vrai_negatif + faux_negatif)) > 0 else 0

    cm = [[vrai_positif, faux_positif], [faux_negatif, vrai_negatif]]

    return render_template('statistiquesdetection.html', cm=cm, accuracy=accuracy, precision=precision, recall=recall, f1=f1, mcc=mcc)


# Gestion de la base de données ajout suppressions données

# Ajouter dossier/image
app.config['DOSSIER_VISAGES'] = 'visages_connus'

# Créer le dossier de visages connus s'il n'existe pas
if not os.path.exists(app.config['DOSSIER_VISAGES']):
    os.makedirs(app.config['DOSSIER_VISAGES'])

def lister_dossiers(repertoire):
    dossiers = {}
    for entree in os.scandir(repertoire):
        if entree.is_dir():
            dossiers[entree.name] = [f.name for f in os.scandir(entree.path) if f.is_file()]
    return dossiers

@app.route('/telecharger', methods=['POST'])
def telecharger_fichier():
    if 'fichier' not in request.files:
        flash('Aucun fichier sélectionné')
        return redirect(request.url)

    fichier = request.files['fichier']

    if fichier.filename == '':
        flash('Aucun fichier sélectionné')
        return redirect(request.url)

    if fichier:
        nom_fichier = fichier.filename
        fichier.save(os.path.join(app.config['DOSSIER_VISAGES'], nom_fichier))
        flash('Fichier téléchargé avec succès')
        return redirect(url_for('gestionphoto'))

@app.route('/fichiers')
def fichiers():
    fichiers = lister_dossiers(app.config['DOSSIER_VISAGES'])
    return render_template('fichiers.html', fichiers=fichiers)

@app.route('/telechargements/<path:nom_fichier>')
def fichier_telecharge(nom_fichier):
    return send_from_directory(app.config['DOSSIER_VISAGES'], nom_fichier)

@app.route('/gestionPhoto')
def gestionphoto():
    dossiers = lister_dossiers(app.config['DOSSIER_VISAGES'])
    return render_template('gestionphoto.html', dossiers=dossiers)

@app.route('/creer_dossier', methods=['POST'])
def creer_dossier():
    nom_dossier = request.form.get('nom_dossier')
    if nom_dossier:
        chemin_dossier = os.path.join(app.config['DOSSIER_VISAGES'], nom_dossier)
        os.makedirs(chemin_dossier, exist_ok=True)
        flash('Dossier créé avec succès')
    else:
        flash('Le nom du dossier ne peut pas être vide')
    return redirect(url_for('gestionphoto'))

@app.route('/ajouter_image', methods=['POST'])
def ajouter_image():
    dossier = request.form.get('dossier')
    fichier = request.files['fichier']

    if fichier and dossier:
        chemin_dossier = os.path.join(app.config['DOSSIER_VISAGES'], dossier)
        if os.path.exists(chemin_dossier):
            fichier.save(os.path.join(chemin_dossier, fichier.filename))
            flash('Image ajoutée avec succès')
        else:
            flash('Le dossier spécifié n\'existe pas')
    else:
        flash('Veuillez sélectionner un dossier et un fichier')
    return redirect(url_for('gestionphoto'))

@app.route('/supprimer_dossier/<dossier>', methods=['POST'])
def supprimer_dossier(dossier):
    chemin_dossier = os.path.join(app.config['DOSSIER_VISAGES'], dossier)
    if os.path.exists(chemin_dossier):
        shutil.rmtree(chemin_dossier)
        flash('Dossier supprimé avec succès')
    else:
        flash('Le dossier spécifié n\'existe pas')
    return redirect(url_for('gestionphoto'))

@app.route('/supprimer_image/<dossier>/<image>', methods=['POST'])
def supprimer_image(dossier, image):
    chemin_image = os.path.join(app.config['DOSSIER_VISAGES'], dossier, image)
    if os.path.exists(chemin_image):
        os.remove(chemin_image)
        flash('Image supprimée avec succès')
    else:
        flash('L\'image spécifiée n\'existe pas')
    return redirect(url_for('gestionphoto'))


@app.route('/mettre_a_jour_abonnement', methods=['POST'])
def mettre_a_jour_abonnement():
    utilisateurs = lire_utilisateurs()
    utilisateur_login = request.form.get('login')

    if utilisateur_login in utilisateurs:
        utilisateurs[utilisateur_login]['abonnement'] = datetime.now().strftime('%Y-%m-%d')
        ecrire_utilisateurs(utilisateurs)
        flash("Abonnement mis à jour avec succès!", "success")

        # Mettre à jour la session utilisateur
        session['utilisateurs'] = utilisateurs
        session.modified = True
    else:
        flash("Utilisateur non trouvé!", "error")

    return redirect(url_for('acceuil'))


def is_subscribed(utilisateur):
    if not utilisateur.get("abonnement"):
        return False
    abonnement_date = datetime.strptime(utilisateur["abonnement"], "%Y-%m-%d")
    expiration_date = abonnement_date + timedelta(days=6*30)  # Approximation de 6 mois
    return datetime.now() <= expiration_date


def mettre_a_jour_statut_abonnement(utilisateurs):
    for login, utilisateur in utilisateurs.items():
        if utilisateur.get("abonnement") and not is_subscribed(utilisateur):
            utilisateurs[login]['abonnement'] = None
    ecrire_utilisateurs(utilisateurs)


@app.context_processor
def inject_abonnement_status():
    utilisateurs = lire_utilisateurs()
    mettre_a_jour_statut_abonnement(utilisateurs)
    utilisateur = utilisateurs.get(session.get('utilisateur')) if 'utilisateur' in session else None
    abonnement_status = "Abonné" if utilisateur and is_subscribed(utilisateur) else "Non abonné"
    return dict(abonnement_status=abonnement_status)


@app.route("/")
def retoursession():
    utilisateurs = lire_utilisateurs()
    mettre_a_jour_statut_abonnement(utilisateurs)
    utilisateur = utilisateurs.get(session['utilisateur']) if 'utilisateur' in session else None
    if utilisateur:
        print(f"Utilisateur trouvé: {utilisateur['login']}")
    else:
        print("Aucun utilisateur trouvé dans la session")
    abonnement_status = "Abonné" if utilisateur and is_subscribed(utilisateur) else "Non abonné"
    print(f"Statut d'abonnement: {abonnement_status}")
    return render_template("barreDeNavigation.html", abonnement_status=abonnement_status)

if __name__ == '__main__':
    app.run(debug=True)
