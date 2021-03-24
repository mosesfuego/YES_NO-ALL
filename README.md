# YES_NO-ALL
 
README YES/NO CLASSIFICATION
This folder contains 3 scripts and one folder. 
MUST BE RUN IN GIVEN ORDER ( May need to delete DS Store on Mac, run in cmd : 
find . -name ".DS_Store" -delete )

	0.	Clean.py 
	⁃	Folder that cleans and slices up audio data files in the folders in wav. 
	⁃	See source https://github.com/seth814/Audio-Classification/blob/master/clean.py
	⁃	Ensure you put all the files you wish to clean and chop up in a folder called waffles. This folder gets called at the bottom of clean.py 
	⁃	Place the different classes in their own folders. In this case, wavefiles contains 2 folders (Binary classification). In the case of say classifying among 10 instruments there would be 10 different folders each with audio file samples of that folders class.
	⁃	All the cleaned up samples are stored in a folder called clean
2.
MFCC_clean
	This parses through the clean folder and takes the different files, extracts MFCCs from the files and saves them into a panda dataframe. 
These CSV files contain the important information we will use to carry out classification of different speech related sounds. There are many other features that can be extracted other than MFCCs. The code for those can be incorporated as needed.

	0.	SoundClassifier.py
 Sound Classifier is where the actual classification happens. This uses common python ML libraries.  3 different classifiers are used in pipelines, SVM, Logistic Regression and KNN.
	0.	Wavefiles  - As mentioned earlier this contains the audio data you are classifying. The files must be wav files as the name suggests and must be arranged in folders buy class. Does not matter how you name the folders.

