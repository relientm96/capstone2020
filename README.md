# Music Gesture Control with Real Time Pose Estimation

Repository for the 2020 Capstone Real Time Pose Estimation project.    
Link to project description [here](./documents/projectOutline.md).   

Members:

* Matthew Yong
* Tsz Kiu Pang
* Yong Yick Wong

## Repository Structure
```
|--- README.md - Main markdown for this repository
|
+--- documents - Folder containing project documents
|       |
|       +--- Resources - Folder containing helpful project document resources
|       
+--- media - Folder containing all images, videos or other forms of media
|       
+--- meeting-logs - Folder containing all meeting logs
|       |
|       +--- JonathanLogs - Folder containing all meeting logs with Jonathan
|       |  
|       +--- GroupLogs - Folder containing all group logs
|
\--- src - Folder containing source code of project
```

## Links
Project links :    
* [Slack Link](https://nebula-m78.slack.com/)
* [Project Timeline](https://docs.google.com/document/d/1RuPlROiwp9qh14LQtcvLYuOO0s0pgH-rktbnG3PjgCE/edit)
* [Project EMI Link](https://apps2.eng.unimelb.edu.au/emi-capstone-projects/index.php?r=project%2Fview&id=194&ajaxView=yes)
* [Project Proposals + Discussion Google Sheets](https://docs.google.com/spreadsheets/d/15XxvRazeiOha9PAxoPqFTfDsIj7EWxJ6Ipewfeq0vCw/edit?fbclid=IwAR26_xDexQU5j8ucYRdEgOGRn9WnBZEb_gNmb5hwk7R50zRKGe-IrorVelA#gid=1278852641)
* [Project Assessments](https://drive.google.com/open?id=1vqbPQjNfL9CqeaYIf6xfPV044M-8mn2Q)
    * Assessment 01
        * Team Charter
        * Ganntt Chart
        * RACIX Matrix
    * General Risk Assessment 


Tutorial/Reading resources links :  

* [Useful cheat sheet to format README.md files](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)
* [Great Git command line tutorial](http://gitimmersion.com/)

## Info On Workflow

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




