# ML-Implementation
Projet de fin d'année d'implémentation de viola jones en version GPU + parrallélisé
Réalisation par :  BELKARFA LINA ET CHARLES CANAVAGGIO

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <ul>
        <li><a href="#introduction">Introduction</a></li>
      </ul>
      <ul>
        <li><a href="#le-projet"> Le projet </a></li>
      </ul>
       <ul>
        <li><a href="#results">Results</a></li>
      </ul>
    </li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
# Implementation du ML

### Introduction

## Viola Jones
Dans ce projet vous allez améliorer une implémentation existante de Viola-Jones qui s'exécute sur CPU. Vous devez remplacer les parties du code qui font du calcul intensif par leurs version sur parallèle sur GPU. Le but est d'avoir une version plus rapide que la version séquentielle.

* L'idée générale pour trouver un visage est de chercher des zones de dont le contraste est particulier. 
* Par exemple, les yeux sont en général séparés par une partie plus sombre (le nez). 
* Cette analyze est couteuse car elle doit être faite sur toutes les zones de l'image et à différentes échelles. 
* Pour améliorer la performance, on a recours à du machine learning.
* Les deux tutoriaux de la section précédente utilisent du code partagé sur github. Nous allons l'utiliser comme base de notre projet. Attention, il nécessite d'installer le framework Pickle avec PIP.
* L'avantage de cette version de Viola-Jones est qu'elle vient avec un classifieur déjà entrainé.
* Trouvez comment charger le classifieur déjà entrainé et testez quelques images.

### Un peu de code

Les deux tutoriaux de la section précédente utilisent du code partagé sur github. Nous allons l'utiliser comme base de notre projet. Attention, il nécessite d'installer le framework Pickle avec PIP.

    1. Clonez le dépot
    2. À quoi correspondent chacun des fichiers pickle contenus dans le projet ?
    3. Quel jeu de données a été utilisé pour l'entrainement et les tests ?

L'avantage de cette version de Viola-Jones est qu'elle vient avec un classifieur déjà entrainé.

    4. Trouvez comment charger le classifieur déjà entrainé et testez quelques images.


### Le projet
Votre mission, si vous l'acceptez, consistera à ajouter des versions parallèles/GPU des algorithmes utilisés dans ce code et montrer leur intérêt. Plus précisément, vous devez

* Écrire une version Numba/Cuda pour calculer l'image intégrale
* Écrire une méthode qui parallelise le calcul des features sur GPU
* Créer un fichier projet.py qui contient deux méthodes :
    * **bench_train() qui exécute un entrainement sur un nombre variable d'images et affiche le temps à chaque fois.**
    * **bench_accuracy() qui charge les modèles précédents et mesure la précision sur un jeu d'image test**


### Results
* Nous avons réussi à mettre en place l'image integrale mais nous avons mal réalisé l'allocation de mémoire.. En effet, notre algorithme est plus lent qu'avec celui de base. 
* Pour la partie transpose utilisé dans l'image intégrale, nous avons essayé de faire un code par nous même sans succès, nous avons donc utilisé les ressources du net pour adapter un code existant
* Détail des fichiers : 
  * Vous trouverez dans projet.py, le projet que nous avons réalisé et fonctionne malgré sa lenteur comparé à l'algo CPU
  * Dans un autre fichier projet.ipynb, vous trouverez un jupyter notebook (pas besoin de le charger nous avons run toutes les cellules pour que vous puissiez le consulter en ligne directement sur git). Dans celui ci se trouve les réponses aux questions en début de projet, sur l'analyse des pkl, l'essaie de l'algo etc... 

