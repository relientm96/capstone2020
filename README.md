# Australian Sign Language Recognition using Human Pose Estimation

Repository for the Australian Sign Language Recognition using Human Pose Estimation project. For project documentation please visit the following link https://relientm96.github.io/capstone2020/

Team Members:

* Matthew Yong
* Tsz Kiu Pang
* Yong Yick Wong

## Repository Structure
```
|--- src - Folder containing source code of project
|       |
|       +--- training  - Data processing pipeline (both pre & post processing)  
|       +--- webserver - New web server code using WebRTC
|       +--- desktop   - Desktop version of our application using OpenCV Python.
|       +--- openpose-python - OpenPose compiled binaries for Python
|       +--- scrapper  - Python web scrapper for video data collection
|
+--- documents - Folder containing project documents
|       
+--- meeting-logs - Folder containing all meeting logs
|
+--- Archive - Old Code not in use 
```

## Project Information

#### Project Poster Summary
![poster](docs/images/Electrical_Matthew_Yong_Poster.png)

#### Project Presentation
[![oral_presentation](http://img.youtube.com/vi/AMlnNzqC3Bs/0.jpg)](http://www.youtube.com/watch?v=AMlnNzqC3Bs "Endeavour Presentation - Australian Sign Language Recognition")

#### Our Project Journey
[![road_to_endeavour](http://img.youtube.com/vi/zT6ssMtPTGA/0.jpg)](http://www.youtube.com/watch?v=zT6ssMtPTGA "Road to Endeavour")

## Info On Workflow
Guides to follow when making changes to repository.

### How to get repository on pc ( do this once )
1. Install [git bash](https://gitforwindows.org/) if you do not have git command prompt.  
2. Clone this repository by running 
`git clone "https://github.com/relientm96/capstone2020.git"`

### How to change/add/remove files on repository
Can only do after installing git bash and cloning. Changes should be made on a local branch first before merging to main "master".     

1. Pull latest changes from repository   
`git pull`  
2. Create a new branch    
`git checkout -b MyNewBranchName`  
3. Make the necessary changes on your local PC.   
4. Stage files for commit by running:  
`git add --all` for all files or `git add <filename1> <filename2> etc...` for individual changes.  
5. Commit changes and write a commit message:  
`git commit -m "description of what you changed"`
6. Upload new changes to main repository using:
`git push origin MyNewBranchName`  
7. Go to the main repo on GitHub where you should now see your new branch under branch dropdown tab.     
8. (WARNING, ONLY DO THIS WHEN READY TO MERGE TO MASTER) Click on to your branch and click Pull Request and Send Pull Request to be reviewed by others before merging.  
9. Once pull request is sent, delete the branch by running
```
git branch -d MyNewBranchName

git push origin :MyNewBranchName
```

## Links
Project links :    
* [Projet HackMD Page](https://hackmd.io/team/capstone2020?nav=overview)
* [Project EMI Link](https://apps2.eng.unimelb.edu.au/emi-capstone-projects/index.php?r=project%2Fview&id=194&ajaxView=yes)

Tutorial/Reading resources links :  
* [Useful cheat sheet to format README.md files](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)
* [Great Git command line tutorial](http://gitimmersion.com/)


