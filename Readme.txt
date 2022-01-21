#QUESTION ANSWERING SYSTEM
 
Required Installations:
1.Download Anaconda from following Links:
  -> https://repo.anaconda.com/archive/Anaconda3-2021.05-Windows-x86_64.exe (windows 64-bit)
  -> https://repo.anaconda.com/archive/Anaconda3-2021.05-Windows-x86.exe (Windows 32-bit)
  -> https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh (Linux)
2.Install Anaconda Navigator and Launch jupyter notebook and spyder.
3.Install the modules like numpy,sklearn,tensorflow,seaborn,keras,pandas and matplotlib using the Command:
            -> pip install MODULENAME in commandprompt/terminal 

FILES/FOLDERS INFO:
1."babi_tasks_1-20_v1-2.tar" is the dataset for this project.
2."QAS.py" is the implimentation of MACHINE LEARNING Model step by step.
3."static" and "template" folders for styling the web application.
4."app.py" is implimentation for deploying web application.
5."qa1_single-supporting-fact-model.h5" is the trained model for first task of babi tasks
6."qa2_two-supporting-facts-model.h5" is the trained model for second task of babi tasks (like this there are four trained models in the zip file)

EXECUTION STEPS:
1.Download the "CSE-E8" zip file and extract all.
2.open spyder and go to location of FILE and follow steps:
       CSE-E8->QAS.py.
3.Execute the each cell step by step.
4.open command prompt and go to location of app.py using the following command:
               cd FILEPATH.
5.Run python app.py command in commandprompt/terminal.
6.Open Local host adress generated in any web browser to view web application.
7.Enter all the fields and press "Predict" button to view output.

               