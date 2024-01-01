# STASHES CARRY OVER FROM BRANCH TO BRANCH
!git stash save "message"
!git status # nothing to commit
!git checkout add
!git stash pop
!git diff
!git add .
!git commit -m "Add func"


!git branch add
!git checkout add

# make changes to file

!git stash save "message" # save changes to stash, but not to file
!git diff # no change
!git status # nothing to commit

!git stash list # list all stashes

!git stash apply stash@{0} # apply first stash
!git stash list # stash still there

!git stash pop
!git stash list # no stash anymore

# drop a stash
!git stash drop stash@{0}

# drop all stashes
!git stash clear