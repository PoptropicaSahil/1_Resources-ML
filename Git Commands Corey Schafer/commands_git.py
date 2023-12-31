# check version
!git --version

# set config values
!git config --global user.name "Sahil G"
!git config --global user.email "abcd@example.com"

# show all config values
!git config --list

# get help of command
# !git <config/version/..> --help

# list all files
!ls -la

# initialise repo from code
!git init

# not trac it as repo
# just delete .git
!rm -rf .git

# make gitignore
!touch .gitignore

# status
!git status

# add to staging area
!git add -A # all files
!git add calc.py # specific file

# remove from staging area
!git reset # all files
!git reset calc.py # specific file


# commit
!git commit -m "first commit" 
!git status
!git log # shows commit history with hashes

# clone remote repo
!git clone <url.git> <where to clone>
!git clone remote_repo.git .

# info about remote repo
!git remote -v
!git branch -a  # show branches
!git branch # show current branch highlighted

# show changes
!git diff
!git status
!git commit -m "message"

# push
# origin is name of remote repo
# master is the branch where pushing
!git pull origin master
!git push origin master


# common workflow
!git branch calc-divide
!git checkout calc-divide
!git branch # calc-divide is now active
# now we can do some changes
!git status
!git add -A
!git commit -m "message"
# push to remote
!git push -u origin calc-divide
!git branch -a  # calc-divide branch is also pushed to remote repo
!git checkout master
!git pull origin master
!git branch --merged # list merged branches, calc-divide not shows up 
!git merge calc-divide
!git push origin master
# delete branch
!git branch --merged # calc-divide shows up now
!git branch -d calc-divide # delete locally
!git branch -a # calc-divide present in remote repo
!git push origin --delete calc-divide 
!git branch -a # calc-divide deleted from remote










