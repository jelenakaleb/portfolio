# Tower of Hanoi
# There are three rods and 64 golden disks arranged in ascending order on one of the rods.
# The goal is to move the disks to the third rod and arrange them there in ascending order as well.
# Only one disk can be moved at a time and no disk may be placed on a smaller disk.

nDisks <- 4 # set lower for shorter runtime
nPoles <- 3
startLoc <- 3 # first rod
goalLoc <- 1 # third rod
locMat <- matrix(c(rep(startLoc, nDisks), NA), nrow = 1)
colnames(locMat) <- c(paste0("disk", 1:nDisks), "movedDisk")

moveDisk <- function(index, target, locMat) {
  diskLoc <- locMat[nrow(locMat), -ncol(locMat)]
  if (index == 1 || diskLoc[index-1] != diskLoc[index]) {
    diskLoc[index] <- target
    locMat <- rbind(locMat, c(diskLoc, index))
  } else {
    targetNew  <- setdiff(1:nPoles, c(target, diskLoc[index]))
    locMat <-  moveDisk(index - 1, targetNew, locMat)
    diskLoc[index] <- target
    locMat <- rbind(locMat, c(diskLoc, index))
    locMat <- moveDisk(index - 1, target, locMat)
  }
  locMat
}

recordMovement <- moveDisk(nDisks, goalLoc, locMat)
