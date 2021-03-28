# Work on High Performance Computing with ECM

This CUDA script generate a histogram of the letters in a CSV file with a given txt file that contains a list of words seprated in lines.

## Answers to the Practice

Sujet du TP : http://info.iut-bm.univ-fcomte.fr/staff/perrot/ECM-GPU/sujets/histoText/histoText.html

### Avez-vous eu des difficultés à réaliser correctement l’optimisation ?
Les différentes diffiultés furent :
1. Charger le fichier en entier peu importe sa taille.
2. Passer les appels de la fonction kernel dans une fonction séparée du programme pour une lecture plus simple du code.

### Quelles sont les optimisations les plus bénéfiques ?
550 ms avec tout les atomicAdd() réalisés sur la **mémoire globale**.\
540 ms avec l'utilisation de la **mémoire partagée**.

### Combien de lectures de la mémoire globale sont-elles effectuées par votre kernel calculant l’histogramme ? expliquez.
Par ligne le programe va lire toute les lettres une par une dans la mémoire globale plus une pour vérifier la condition d'arrêt.\
Une fois toutes les valeurs stockées dans la mémoire partagée, le 1er thread de chaque bloc va ajouter la valeur contenu dans la mémoire partagée dans l'array de mémoire globale. Cela correspond à une lecture en mémoire globale pour chaque index du tableau.

Au total, on a :
Sans utilisation de la mémoire partagée : glbMemRead = nbLine * (2 * averageLineSize + 1)
Avec utilisation de la mmémoire partagée: glbMemRead = nbLine * (averageLineSize + 1) + nbBloc * NB_ASCII

### Combien d’opérations atomiques sont effectuées par votre kernel calculant l’histogramme ? expliquez.

### La plupart des fichiers texte se composent uniquement de lettres, de chiffres et de caractères d’espacement. Que pouvons-nous dire sur les conflits d’accès concernant le nombre de threads qui essaient simultanément d’incrémenter atomiquement un histogramme privé ?
