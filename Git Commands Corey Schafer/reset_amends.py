# have made changes to file
# have to reset
!git status # check if file is actually modified
!git diff # see changes
!git branch # see branches
!git checkout calc.py # reset done
!git diff # no changes seen


# changes between two commits
!git diff 1v1v1v1 2cecd45d

# Made bad commit but others have pulled
# Make new commit to revert prev changes
# so no rewriting history
!git revert 0b0f8cbc
!git log

# Change a commit - change only message
!git commit -m "incorrect message here"
!git log # see commit
!git commit --amend -m "correct message" # hash also changed
# fine if we are the only ones to pull this so far

# Change a commit - add a file in prev commit only
!git add .gitignore # added a file
!git commit --amend
!git log # see prev commit only
!git log --stat # exact change in commit
# again changed hash

# Change a commit - made commit to wrong branch - 
# want to move the commit to another branch
!git log # grab hash of commit we want to cherry pick
!git checkout subtract-feature # switch to branch
!git log # will not have the new commit ofc
!git cherry-pick 0b0f8cbc # hash number
!git log # see new commit
# now delete commit from master branch
!git checkout master
!git log
!git reset --soft 2cecd45d # hash number of commit till where we want to go
!git status # new changes will be in staging area - didn't lose any work

!git reset 2cecd45d # default to mixed
!git status # new changes will be in working dir, not in staging

!git reset --hard 2cecd45d
!git status # new changes all gone, just untracked files are left
!git clean -df # get rid of all changes. d: dir, f: force
# use case: unzipped a file - created a lot of new file
# but now want to delete them


# recover files even if they were hard reset
# upto 30days maybe?
!git reflog
!git checkout 1bsd34k # till where you want to recover
# gives us info on a detached head state
!git branch backup # create new branch from this
!git branch # list out branches
