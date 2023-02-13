Voici mon rendu de projet pour le test technique proposé par Servier.

Vous trouverez dans ce repo la liste de choses suivantes :

    - report.ipynb : Un notebook d'analyse contenant toutes mes recherches concernant la data augmentation,
      data viz et la construction des modèles.
    - Une dockerfile conprenant les instructions pour dockeriser l'application.


Afin de faire tourner l'application, il suffit de lancer le main.py. J'ai recontré quelques soucis avec le docker,
car il m'est impossible de télécharger une librairie essentielle au bon fonctionnement du programme.

Cependant vous pouvez lancer :

'docker build -t docker-ml-model -f Dockerfile .'

pour voir le problème. 
