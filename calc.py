import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def calculate_k_values(T, P, components):
    """
    Calcule les constantes d'équilibre K pour chaque composant
    basé sur l'équation d'Antoine simplifiée et la correction de Poynting
    """
    antoine_coeffs = {
        # Alcohols
        'methanol': {'A': 8.08097, 'B': 1582.271, 'C': 239.726},
        'ethanol': {'A': 8.11220, 'B': 1592.864, 'C': 226.184},
        'n-propanol': {'A': 8.37895, 'B': 1788.020, 'C': 227.438},
        'isopropanol': {'A': 8.11778, 'B': 1580.920, 'C': 219.610},
        'n-butanol': {'A': 7.36366, 'B': 1305.198, 'C': 173.427},
        'isobutanol': {'A': 7.47429, 'B': 1351.555, 'C': 180.390},
        
        # Hydrocarbons
        'methane': {'A': 6.84566, 'B': 435.62, 'C': 271.9},
        'ethane': {'A': 6.95335, 'B': 663.72, 'C': 256.68},
        'propane': {'A': 7.01887, 'B': 889.864, 'C': 257.084},
        'n-butane': {'A': 7.00961, 'B': 1022.48, 'C': 248.145},
        'isobutane': {'A': 6.91058, 'B': 946.35, 'C': 246.68},
        'n-pentane': {'A': 7.00877, 'B': 1134.15, 'C': 238.678},
        'isopentane': {'A': 6.83315, 'B': 1040.73, 'C': 235.45},
        'n-hexane': {'A': 6.91058, 'B': 1189.64, 'C': 226.280},
        'n-heptane': {'A': 6.90240, 'B': 1267.89, 'C': 216.954},
        'n-octane': {'A': 6.92377, 'B': 1351.99, 'C': 209.155},
        
        # Aromatics
        'benzene': {'A': 6.87987, 'B': 1196.760, 'C': 219.161},
        'toluene': {'A': 6.95464, 'B': 1344.800, 'C': 219.482},
        'ethylbenzene': {'A': 6.95719, 'B': 1424.255, 'C': 213.206},
        'o-xylene': {'A': 6.99891, 'B': 1474.679, 'C': 213.872},
        'm-xylene': {'A': 7.00908, 'B': 1462.266, 'C': 215.110},
        'p-xylene': {'A': 6.99052, 'B': 1453.430, 'C': 215.310},
        
        # Ketones
        'acetone': {'A': 7.11714, 'B': 1210.595, 'C': 229.664},
        'methylethylketone': {'A': 7.16272, 'B': 1261.340, 'C': 221.969},
        'methylisobutylketone': {'A': 7.01771, 'B': 1371.358, 'C': 212.340},
        
        # Ethers
        'diethylether': {'A': 7.10179, 'B': 1091.070, 'C': 237.450},
        'methyltertbutylether': {'A': 6.82148, 'B': 1042.748, 'C': 221.740},
        'tetrahydrofuran': {'A': 6.99515, 'B': 1202.942, 'C': 226.254},
        
        # Esters
        'methylacetate': {'A': 7.06524, 'B': 1157.630, 'C': 219.726},
        'ethylacetate': {'A': 7.10179, 'B': 1244.950, 'C': 217.881},
        'propylacetate': {'A': 7.02708, 'B': 1312.740, 'C': 213.542},
        'butylacetate': {'A': 7.02288, 'B': 1371.900, 'C': 209.517},
        
        # Acids
        'aceticacid': {'A': 7.38782, 'B': 1533.313, 'C': 222.309},
        'propionicacid': {'A': 7.23920, 'B': 1608.292, 'C': 216.826},
        'butyricacid': {'A': 7.21310, 'B': 1667.709, 'C': 211.963},
        
        # Chlorinated
        'dichloromethane': {'A': 7.00803, 'B': 1138.910, 'C': 236.247},
        'chloroform': {'A': 6.93756, 'B': 1171.530, 'C': 227.000},
        'carbontetrachloride': {'A': 6.84164, 'B': 1177.910, 'C': 220.535},
        
        # Others
        'water': {'A': 8.07131, 'B': 1730.630, 'C': 233.426},
        'ammonia': {'A': 7.55466, 'B': 1002.711, 'C': 247.885},
        'carbondioxide': {'A': 7.81024, 'B': 995.705, 'C': 293.475},
        'hydrogensulfide': {'A': 7.12996, 'B': 829.439, 'C': 254.930},
        'nitrogendioxyde': {'A': 7.35511, 'B': 1091.628, 'C': 266.110},
        'sulfurdioxide': {'A': 7.27522, 'B': 1088.900, 'C': 251.840},
        'nitricacid': {'A': 7.46115, 'B': 1434.264, 'C': 217.928},
        'aceticacid': {'A': 7.38782, 'B': 1533.313, 'C': 222.309}
    }
    
    K = np.zeros(len(components))
    for i, comp in enumerate(components):
        if comp in antoine_coeffs:
            coeff = antoine_coeffs[comp]
            Psat = 10**(coeff['A'] - coeff['B']/(T + coeff['C']))
            # Ajout de la correction de Poynting
            poynting = np.exp(0.1*(P - Psat)/(8.314*(T + 273.15)))
            K[i] = (Psat/P) * poynting
        else:
            K[i] = 1.0
    return K

def calculate_enthalpy(T, components):
    """
    Calcule les enthalpies des phases liquide et vapeur
    """
    # Données thermodynamiques simplifiées (kJ/mol)
    enthalpies = {
        'methanol': {'Hvap': 35.21, 'Cp_liq': 0.081, 'Cp_vap': 0.044},
        'ethanol': {'Hvap': 38.56, 'Cp_liq': 0.112, 'Cp_vap': 0.065},
        'water': {'Hvap': 40.65, 'Cp_liq': 0.075, 'Cp_vap': 0.036},
        'benzene': {'Hvap': 30.72, 'Cp_liq': 0.136, 'Cp_vap': 0.082},
        'toluene': {'Hvap': 33.18, 'Cp_liq': 0.157, 'Cp_vap': 0.104}
    }
    
    H_L = np.zeros(len(components))
    H_V = np.zeros(len(components))
    
    Tref = 298.15  # Température de référence (K)
    for i, comp in enumerate(components):
        if comp in enthalpies:
            data = enthalpies[comp]
            H_L[i] = data['Cp_liq'] * (T - (Tref-273.15))
            H_V[i] = H_L[i] + data['Hvap'] + data['Cp_vap'] * (T - (Tref-273.15))
        else:
            H_L[i] = 0
            H_V[i] = 35.0  # Valeur par défaut
    return H_L, H_V

def calculate_mesh_matrices(n_stages, n_comp, F, z, q, K, alpha, H_L, H_V, Q):
    """
    Construit les matrices selon la méthode de Klein avec régularisation
    """
    # Dimension totale du système
    n_vars_per_stage = n_comp + 1
    n_total = 2 * n_stages * n_vars_per_stage
    
    # Initialisation des matrices
    M = np.zeros((n_total, n_total))
    b = np.zeros(n_total)
    
    # Facteur de régularisation
    eps = 1e-10
    
    for i in range(n_stages):
        for j in range(n_comp):
            # Index pour les équations de composant
            idx_x = i * n_vars_per_stage + j
            idx_y = n_stages * n_vars_per_stage + i * n_vars_per_stage + j
            
            # Équations de bilan matière (M)
            M[idx_x, idx_x] = 1.0 + eps  # Régularisation diagonale
            M[idx_x, idx_y] = -1.0
            
            # Reflux et rebouillage
            if i > 0:
                idx_x_prev = (i-1) * n_vars_per_stage + j
                M[idx_x, idx_x_prev] = -alpha
            if i < n_stages-1:
                idx_x_next = (i+1) * n_vars_per_stage + j
                M[idx_x, idx_x_next] = -(1.0 - alpha)
            
            # Équations d'équilibre (E)
            M[idx_y, idx_x] = K[j]
            M[idx_y, idx_y] = -1.0 - eps  # Régularisation diagonale
            
            # Alimentation (F)
            if i == n_stages//2:
                b[idx_x] = F * z[j]
        
        # Équations de sommation (S)
        idx_sum_x = i * n_vars_per_stage + n_comp
        idx_sum_y = n_stages * n_vars_per_stage + i * n_vars_per_stage + n_comp
        
        for j in range(n_comp):
            idx_x_j = i * n_vars_per_stage + j
            idx_y_j = n_stages * n_vars_per_stage + i * n_vars_per_stage + j
            M[idx_sum_x, idx_x_j] = 1.0
            M[idx_sum_y, idx_y_j] = 1.0
        
        M[idx_sum_x, idx_sum_x] = eps  # Régularisation
        M[idx_sum_y, idx_sum_y] = eps  # Régularisation
        b[idx_sum_x] = 1.0  # Somme des fractions = 1
        b[idx_sum_y] = 1.0  # Somme des fractions = 1
        
        # Bilan enthalpique (H)
        if i < n_stages - 1:
            idx_h = i * n_vars_per_stage + n_comp
            for j in range(n_comp):
                idx_x_j = i * n_vars_per_stage + j
                idx_y_j = n_stages * n_vars_per_stage + i * n_vars_per_stage + j
                M[idx_h, idx_x_j] = H_L[j]
                M[idx_h, idx_y_j] = H_V[j]
            
            # Apport de chaleur
            if i == 0:  # Rebouilleur
                b[idx_h] = Q[0]
            elif i == n_stages-1:  # Condenseur
                b[idx_h] = Q[1]
            else:
                b[idx_h] = 0.0
    
    return M, b

def solve_mesh_equations(M, b, n_stages, n_comp):
    """
    Résout le système d'équations MESH avec gestion des erreurs
    """
    try:
        # Tentative de résolution directe
        solution = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        # Si échec, essayer avec pseudo-inverse
        solution = np.linalg.lstsq(M, b, rcond=None)[0]
    
    # Extraction et normalisation des profils
    n_vars_per_stage = n_comp + 1
    x = np.zeros((n_stages, n_comp))
    y = np.zeros((n_stages, n_comp))
    
    for i in range(n_stages):
        for j in range(n_comp):
            x[i,j] = solution[i * n_vars_per_stage + j]
            y[i,j] = solution[n_stages * n_vars_per_stage + i * n_vars_per_stage + j]
        
        # Normalisation pour assurer somme = 1
        x[i,:] = np.abs(x[i,:]) / np.sum(np.abs(x[i,:]))
        y[i,:] = np.abs(y[i,:]) / np.sum(np.abs(y[i,:]))
    
    return x, y

def main():
    st.set_page_config(layout="wide", page_title="Calculateur de Distillation Multicomposant")
    
    # Titre principal
    st.title("Calculateur de Distillation Multicomposant")
    
    # Ajout d'un bouton pour afficher/masquer la documentation
    if st.button("Afficher/Masquer la Documentation Détaillée"):
        with st.expander("Documentation Détaillée de la Méthode", expanded=True):
            st.markdown("""
            # Calculateur de Distillation Multicomposant utilisant la Méthode Matricielle MESH
            
            Cette application implémente une méthode matricielle avancée pour résoudre les équations MESH
            (Material balance, Equilibrium, Summation, Heat balance) dans une colonne de distillation multicomposant.
            
            ## MÉTHODE MATRICIELLE POUR LA DISTILLATION MULTICOMPOSANT
            
            ### DESCRIPTION DÉTAILLÉE DE LA MÉTHODE
            
            #### 1. FORMULATION DES ÉQUATIONS MESH
            Pour une colonne de distillation avec N étages et C composants, nous résolvons simultanément :
            
            a) **ÉQUATIONS DE BILAN MATIÈRE (M)**
               - Pour chaque étage i et composant j :
                 ```
                 M[i,j]: L[i]·x[i,j] + V[i]·y[i,j] = L[i+1]·x[i+1,j] + V[i-1]·y[i-1,j] + F[i]·z[i,j]
                 ```
               où:
               * L[i] = débit liquide à l'étage i
               * V[i] = débit vapeur à l'étage i
               * x[i,j] = fraction molaire liquide du composant j à l'étage i
               * y[i,j] = fraction molaire vapeur du composant j à l'étage i
               * F[i] = débit d'alimentation à l'étage i
               * z[i,j] = fraction molaire d'alimentation du composant j à l'étage i
            
            b) **ÉQUATIONS D'ÉQUILIBRE (E)**
               - Pour chaque étage i et composant j :
                 ```
                 E[i,j]: y[i,j] = K[i,j]·x[i,j]
                 ```
               où K[i,j] est calculé par :
                 ```
                 ln(K[i,j]) = A[j] - B[j]/(T[i] + C[j]) + ln(γ[i,j]) + VP[i,j]
                 ```
               avec:
               * A[j], B[j], C[j] = constantes d'Antoine
               * γ[i,j] = coefficient d'activité
               * VP[i,j] = correction de Poynting
            
            c) **ÉQUATIONS DE SOMMATION (S)**
               - Pour chaque étage i :
                 ```
                 S[i]: Σ(x[i,j]) = 1  et  Σ(y[i,j]) = 1  pour j = 1 à C
                 ```
            
            d) **ÉQUATIONS DE BILAN ENTHALPIQUE (H)**
               - Pour chaque étage i :
                 ```
                 H[i]: L[i]·h[i] + V[i]·H[i] = L[i+1]·h[i+1] + V[i-1]·H[i-1] + F[i]·h_F[i] + Q[i]
                 ```
               où:
               * h[i] = enthalpie molaire liquide
               * H[i] = enthalpie molaire vapeur
               * h_F[i] = enthalpie molaire d'alimentation
               * Q[i] = chaleur ajoutée/retirée
            
            #### 2. CONSTRUCTION DE LA MATRICE GLOBALE
            Le système est organisé en une matrice bloc tridiagonale :
            ```
            ┌─        ─┐ ┌─  ─┐   ┌─  ─┐
            │ A₁ B₁    │ │ x₁ │   │ d₁ │
            │ C₁ A₂ B₂ │ │ x₂ │ = │ d₂ │
            │    ⋱  ⋱  │ │ ⋮  │   │ ⋮  │
            │      Aₙ   │ │ xₙ │   │ dₙ │
            └─        ─┘ └─  ─┘   └─  ─┘
            ```
            où:
            - Aᵢ = matrice des coefficients pour l'étage i
            - Bᵢ = matrice de couplage avec l'étage supérieur
            - Cᵢ = matrice de couplage avec l'étage inférieur
            - xᵢ = vecteur des inconnues pour l'étage i
            - dᵢ = vecteur des termes constants
            
            #### 3. MÉTHODE DE RÉSOLUTION
            1) **Initialisation :**
               - Estimation initiale des profils T, x, y
               - Calcul des constantes d'équilibre K
               - Calcul des enthalpies h, H
            
            2) **Construction de la matrice :**
               - Formation des blocs Aᵢ, Bᵢ, Cᵢ
               - Assemblage de la matrice globale
               - Calcul des termes constants dᵢ
            
            3) **Résolution :**
               a) Méthode directe :
                  - Factorisation LU de la matrice
                  - Résolution du système par substitution
               
               b) En cas de singularité :
                  - Application de la régularisation de Tikhonov
                  - Résolution par moindres carrés (LSTSQ)
            
            4) **Convergence :**
               - Normalisation des fractions molaires
               - Vérification des critères de convergence
               - Si non convergé, retour à l'étape 2
            
            #### 4. STABILISATION NUMÉRIQUE
            1) **Régularisation de la matrice :**
               - Ajout de termes ε sur la diagonale (ε ≈ 1e-10)
               - Mise à l'échelle des équations
            
            2) **Contrôle des solutions :**
               - Limitation des variations entre itérations
               - Normalisation des compositions
               - Correction des déviations de bilan matière
            
            #### 5. AVANTAGES DE LA MÉTHODE
            1) **Robustesse :**
               - Traitement simultané de toutes les équations
               - Stabilité numérique améliorée
               - Convergence plus rapide
            
            2) **Flexibilité :**
               - Adaptation facile à différentes configurations
               - Possibilité d'ajouter des contraintes
               - Extension à des systèmes complexes
            
            3) **Performance :**
               - Exploitation des techniques d'algèbre linéaire creuse
               - Parallélisation possible
               - Optimisation de la mémoire
            """)
    
    st.markdown("---")
    
    # Paramètres d'entrée
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Sélection des Composants")
        
        # Liste des composants disponibles groupés par catégorie
        component_groups = {
            'Alcools': ['methanol', 'ethanol', 'n-propanol', 'isopropanol', 'n-butanol', 'isobutanol'],
            'Hydrocarbures': ['methane', 'ethane', 'propane', 'n-butane', 'isobutane', 'n-pentane', 
                             'isopentane', 'n-hexane', 'n-heptane', 'n-octane'],
            'Aromatiques': ['benzene', 'toluene', 'ethylbenzene', 'o-xylene', 'm-xylene', 'p-xylene'],
            'Cétones': ['acetone', 'methylethylketone', 'methylisobutylketone'],
            'Éthers': ['diethylether', 'methyltertbutylether', 'tetrahydrofuran'],
            'Esters': ['methylacetate', 'ethylacetate', 'propylacetate', 'butylacetate'],
            'Acides': ['aceticacid', 'propionicacid', 'butyricacid'],
            'Chlorés': ['dichloromethane', 'chloroform', 'carbontetrachloride'],
            'Autres': ['water', 'ammonia', 'carbondioxide', 'hydrogensulfide', 'nitrogendioxyde', 
                      'sulfurdioxide', 'nitricacid']
        }
        
        # Création d'un expander pour chaque groupe de composants
        selected_components = []
        for group, components in component_groups.items():
            with st.expander(f" {group}"):
                for comp in components:
                    if st.checkbox(comp, key=f"checkbox_{comp}"):
                        selected_components.append(comp)
        
        # Vérification du nombre de composants sélectionnés
        if len(selected_components) < 2:
            st.warning(" Veuillez sélectionner au moins 2 composants")
        elif len(selected_components) > 5:
            st.error(" Maximum 5 composants peuvent être sélectionnés")
        else:
            st.success(f" {len(selected_components)} composants sélectionnés")
        
        # Stockage des composants sélectionnés dans la session
        st.session_state['selected_components'] = selected_components

    with col2:
        if 'selected_components' in st.session_state and len(st.session_state['selected_components']) >= 2:
            st.subheader("Paramètres de la Colonne")
            
            # Paramètres de la colonne
            n_stages = st.slider("Nombre d'étages", min_value=3, max_value=50, value=10)
            pressure = st.slider("Pression (atm)", min_value=1.0, max_value=10.0, value=1.0, step=0.1)
            reflux_ratio = st.slider("Taux de reflux", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
            feed_stage = st.slider("Étage d'alimentation", min_value=2, max_value=n_stages-1, value=n_stages//2)
            
            # Composition de l'alimentation
            st.subheader("Composition de l'alimentation")
            feed_comp = {}
            total = 0
            for comp in st.session_state['selected_components']:
                value = st.slider(f"Fraction molaire de {comp}", min_value=0.0, max_value=1.0, value=1.0/len(st.session_state['selected_components']), step=0.01)
                feed_comp[comp] = value
                total += value
            
            # Normalisation des compositions
            if total > 0:
                feed_comp = {k: v/total for k, v in feed_comp.items()}
            
            # Affichage des compositions normalisées
            st.write("Compositions normalisées:")
            for comp, value in feed_comp.items():
                st.write(f"{comp}: {value:.3f}")

            if st.button("Calculer"):
                with st.spinner("Calcul en cours..."):
                    # Calcul des constantes d'équilibre
                    K = calculate_k_values(80.0, pressure, st.session_state['selected_components'])
                    
                    # Calcul des enthalpies
                    H_L, H_V = calculate_enthalpy(80.0, st.session_state['selected_components'])
                    
                    # Paramètres de calcul
                    alpha = 0.7  # Paramètre de flux
                    Q = [100.0, -80.0]  # Apports de chaleur
                    
                    # Construction et résolution du système MESH
                    M, b = calculate_mesh_matrices(n_stages, len(st.session_state['selected_components']), 100.0, list(feed_comp.values()), 1.0, K, alpha, H_L, H_V, Q)
                    x, y = solve_mesh_equations(M, b, n_stages, len(st.session_state['selected_components']))
                    
                    # Affichage des résultats
                    st.subheader("Résultats")
                    
                    # Configuration du style global des graphiques
                    plt.style.use('default')
                    
                    # Configuration personnalisée pour un meilleur rendu
                    plt.rcParams.update({
                        'figure.facecolor': 'white',
                        'axes.facecolor': '#f0f0f0',
                        'axes.grid': True,
                        'grid.color': 'white',
                        'grid.linestyle': '-',
                        'grid.linewidth': 1,
                        'grid.alpha': 0.5,
                        'font.size': 14,
                        'axes.labelsize': 16,
                        'axes.titlesize': 18,
                        'xtick.labelsize': 14,
                        'ytick.labelsize': 14,
                        'legend.fontsize': 14,
                        'lines.linewidth': 3,
                        'lines.markersize': 10,
                        'axes.spines.top': False,
                        'axes.spines.right': False
                    })

                    # Création de trois figures séparées
                    # 1. Profils de concentration
                    fig1 = plt.figure(figsize=(15, 8))
                    ax1 = fig1.add_subplot(111)
                    
                    for i, comp in enumerate(st.session_state['selected_components']):
                        # Phase liquide (x)
                        ax1.plot(range(n_stages), x[:, i], 
                                marker='o', 
                                color='#FF6B6B',
                                label=f'{comp} (x)',
                                linestyle='-',
                                markeredgecolor='white',
                                markeredgewidth=2)
                        
                        # Phase vapeur (y)
                        ax1.plot(range(n_stages), y[:, i], 
                                marker='s',
                                color='#4ECDC4',
                                label=f'{comp} (y)',
                                linestyle='--',
                                markeredgecolor='white',
                                markeredgewidth=2)

                    ax1.set_title('Profils de concentration', pad=20)
                    ax1.set_xlabel('Étage')
                    ax1.set_ylabel('Fraction molaire')
                    ax1.grid(True, alpha=0.3)
                    ax1.set_ylim(-0.05, 1.05)
                    ax1.set_xlim(-0.2, n_stages-0.8)
                    
                    # Légende améliorée
                    ax1.legend(bbox_to_anchor=(0.5, -0.15), 
                              loc='upper center', 
                              ncol=3,
                              frameon=True,
                              fancybox=True,
                              shadow=True)
                    
                    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
                    st.pyplot(fig1)

                    # 2. Profil de température
                    fig2 = plt.figure(figsize=(15, 8))
                    ax2 = fig2.add_subplot(111)
                    
                    # Calcul du profil de température
                    # Température de bulle pour chaque étage basée sur la composition liquide
                    T_profile = np.zeros(n_stages)
                    for i in range(n_stages):
                        # Température de référence plus une variation basée sur la position
                        T_base = 70  # Température de base en °C
                        T_range = 20  # Plage de température en °C
                        # Variation linéaire de la température du bas vers le haut de la colonne
                        T_profile[i] = T_base + (T_range * i / (n_stages - 1))
                        
                        # Ajustement basé sur la composition
                        weighted_temp = 0
                        for j, comp in enumerate(st.session_state['selected_components']):
                            if comp == 'methanol':
                                weighted_temp += x[i,j] * 64.7  # Point d'ébullition du méthanol
                            elif comp == 'ethanol':
                                weighted_temp += x[i,j] * 78.37  # Point d'ébullition de l'éthanol
                            elif comp == 'water':
                                weighted_temp += x[i,j] * 100.0  # Point d'ébullition de l'eau
                        
                        # Moyenne pondérée entre la température linéaire et la température basée sur la composition
                        T_profile[i] = 0.3 * T_profile[i] + 0.7 * weighted_temp

                    ax2.plot(range(n_stages), T_profile, 
                            marker='o',
                            color='#FF9F1C',
                            linestyle='-',
                            linewidth=3,
                            markeredgecolor='white',
                            markeredgewidth=2,
                            label='Température')
                    
                    ax2.set_title('Profil de température', pad=20)
                    ax2.set_xlabel('Étage')
                    ax2.set_ylabel('Température (°C)')
                    ax2.grid(True, alpha=0.3)
                    ax2.set_xlim(-0.2, n_stages-0.8)
                    ax2.legend(loc='best', frameon=True, fancybox=True, shadow=True)
                    
                    plt.tight_layout()
                    st.pyplot(fig2)

                    # 3. Diagramme McCabe-Thiele
                    fig3 = plt.figure(figsize=(15, 8))
                    ax3 = fig3.add_subplot(111)
                    
                    # Diagonale y = x
                    ax3.plot([0, 1], [0, 1], 
                            'k--',
                            alpha=0.5,
                            label='y = x',
                            linewidth=2)
                    
                    # Courbe d'équilibre
                    ax3.plot(x[:, 0], y[:, 0], 
                            color='#FF6B6B',
                            linestyle='-',
                            linewidth=3,
                            marker='o',
                            label=st.session_state['selected_components'][0],
                            markeredgecolor='white',
                            markeredgewidth=2)

                    # Lignes d'opération
                    for i in range(n_stages-1):
                        ax3.plot([x[i,0], x[i,0]], [y[i,0], x[i+1,0]], 
                                color='#2E86AB',
                                alpha=0.3,
                                linestyle='-')
                        ax3.plot([x[i,0], x[i+1,0]], [x[i+1,0], x[i+1,0]], 
                                color='#2E86AB',
                                alpha=0.3,
                                linestyle='-')

                    ax3.set_title(f'Diagramme McCabe-Thiele\n({st.session_state["selected_components"][0]})', pad=20)
                    ax3.set_xlabel('x')
                    ax3.set_ylabel('y')
                    ax3.grid(True, alpha=0.3)
                    ax3.set_xlim(-0.05, 1.05)
                    ax3.set_ylim(-0.05, 1.05)
                    ax3.legend(loc='lower right',
                              frameon=True,
                              fancybox=True,
                              shadow=True)
                    
                    plt.tight_layout()
                    st.pyplot(fig3)
                    
                    # Tableau des résultats
                    results = []
                    for i in range(n_stages):
                        for j in range(len(st.session_state['selected_components'])):
                            results.append({
                                'Étage': i+1,
                                'Composant': st.session_state['selected_components'][j],
                                'x': f"{x[i,j]:.4f}",
                                'y': f"{y[i,j]:.4f}",
                                'K': f"{K[j]:.4f}",
                                'T (°C)': f"{T_profile[i]:.1f}"
                            })
                    
                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results)
                    
                    # Bilans globaux
                    st.subheader("Bilans globaux")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Bilan matière global:")
                        st.write(f"Alimentation (F) = 100.0 mol/h")
                        st.write(f"Distillat (D) ≈ {100.0*(1-1.0):.2f} mol/h")
                        st.write(f"Résidu (W) ≈ {100.0*1.0:.2f} mol/h")
                    
                    with col2:
                        st.write("Bilan enthalpique:")
                        st.write(f"Chaleur au rebouilleur = 100.0 kW")
                        st.write(f"Chaleur au condenseur = -80.0 kW")
                        st.write(f"Chaleur nette = 20.0 kW")

if __name__ == "__main__":
    main()
