# Work on High Performance Computing with ECM

This CUDA script generate a histogram of the letters in a CSV file with a given txt file that contains a list of words seprated in lines.

## Answers to the Practice

Sujet du TP : http://info.iut-bm.univ-fcomte.fr/staff/perrot/ECM-GPU/sujets/histoText/histoText.html

* Avez-vous eu des difficultés à réaliser correctement l’optimisation ?
* Quelles sont les optimisations les plus bénéfiques ?
* Combien de lectures de la mémoire globale sont-elles effectuées par votre kernel calculant l’histogramme ? expliquez.
* Combien d’opérations atomiques sont effectuées par votre kernel calculant l’histogramme ? expliquez.
* La plupart des fichiers texte se composent uniquement de lettres, de chiffres et de caractères d’espacement. Que pouvons-nous dire sur les conflits d’accès concernant le nombre de threads qui essaient simultanément d’incrémenter atomiquement un histogramme privé ?
