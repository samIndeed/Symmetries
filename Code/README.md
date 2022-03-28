# The scripts
*Written by:* Johannes Borgqvist<br>
*Date:* 2022-02-17<br>
The Code-folder contains the following six scripts: 

1. "*fit\_to\_data.py*",
2. "*read\_data.py*",
3. "*write\_output.py*",
4. "*symmetry\_toolbox.py*",
5. "*symmetry\_analysis\_PLM\_vs\_IM\_III.py*",
6. "*visualise\_orthogonal\_distance.py*".<br>
The first script contains all the functions that are used for the model fitting. The second script reads the time series from the csv-files that are stored in the Data-folder (for more information read the README-file with relative path "../Data/README.md"). The third script writes the output from the model-fitting to csv-files in the Output-folder (for more information read the README-file with relative path "../Output/README.md"). The fourth script contains all the functions that are relevant for the symmetry based analysis. The fifth script is the script that conducts the whole analysis and generates all the relevant results presented in the article. Lastly, the sixth script visualises how the orthogonal distance between a point and a curve is calculated in Python, and this script is entirely based on the script by John Kitchin that is licensed under the creative commons license: [*CC BY-SA 4.0*](https://creativecommons.org/licenses/by-sa/4.0/deed.en_US#) that can be accessed at the [Kitchin Research Group](https://kitchingroup.cheme.cmu.edu/blog/2013/02/14/Find-the-minimum-distance-from-a-point-to-a-curve/)'s web-page. This code is the basis for calculating the orthogonal distance between the transformed data (that is transformed by the symmetries of the candidate models) and the transformed solution curves which, in turn, is the basis for the implementation of the symmetry based methodology for model selection that is found in the script "*symmetry\_toolbox.py*". 

