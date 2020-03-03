# Real Time Pose Estimation Capstone 2020

Repository to store files for the 2020 Capstone Real Time Pose Estimation project. 
* [Project EMI Link](https://apps2.eng.unimelb.edu.au/emi-capstone-projects/index.php?r=project%2Fview&id=194&ajaxView=yes)

Members:

* Matthew Yong
* Tsz Kiu Pang
* Yong Yick Wong

## Links
Project links :  

* [Project Proposals + Discussion Google Sheets](https://docs.google.com/spreadsheets/d/15XxvRazeiOha9PAxoPqFTfDsIj7EWxJ6Ipewfeq0vCw/edit?fbclid=IwAR26_xDexQU5j8ucYRdEgOGRn9WnBZEb_gNmb5hwk7R50zRKGe-IrorVelA#gid=1278852641)

Tutorial/Reading resources links :  

* [Useful cheat sheet to format README.md files](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)

* [Great Git command line tutorial](http://gitimmersion.com/)

## To-do List

- [ ] Meet up to discuss for first meetup with Jonathan (on [doodle](https://doodle.com/poll/zz4nwysca4szcryd6u96y9ns/private?utm_campaign=poll_invitecontact_participant_invitation_with_message&utm_medium=email&utm_source=poll_transactional&utm_content=participatenow-cta)).  
- [ ] Set on a communication platform (FB/Slack etc..)  
- [ ] Set a weekly meetup day and time with Jonathan.  
- [ ] Set individual tasks. 

## How to get repository on pc ( do this once )
1. Install [git bash](https://gitforwindows.org/) if you do not have git command prompt.  
2. Clone this repository by running 
`git clone "https://github.com/relientm96/capstone2020.git"`

## How to change/add/remove files on repository
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
8. (Only do this when we want to merge to master) Click on to your branch and click Pull Request and Send Pull Request.  
9. After merging, delete the branch by running
```
git branch -d MyNewBranchName

git push origin :MyNewBranchName
```




