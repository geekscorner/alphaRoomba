# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: auto_pickle=True
# cython: profile=True

import numpy as np

cdef class Board():
    """
    Roomba Board.
    """

    cdef int height
    cdef int width
    cdef int length
    cdef int win_length
    cdef public int[:,:] pieces

    def __init__(self):
        """Set up initial board configuration."""
        self.pieces = np.zeros((4, 4), dtype=np.intc)
        startingVariation = "A"
        forcedLearningPosition = False

        if (forcedLearningPosition == True):
            #Can we put the AI in the spot where I have beaten it to try and learn from its mistake?
            self.pieces[0, 0] = 3  # black piece 1
            self.pieces[1, 2] = 3  # black piece 2
            self.pieces[1, 3] = 3  # black piece 3

            self.pieces[2, 2] = -1  # white piece 1
            self.pieces[2, 3] = -1  # white piece 2
            self.pieces[3, 1] = -2  # white piece 3
        elif (startingVariation == "A"):
            #Position A
            self.pieces[0, 0] = 3  # black piece 1
            self.pieces[1, 2] = 3  # black piece 2
            self.pieces[0, 3] = 3  # black piece 3

            self.pieces[2, 1] = -1  # white piece 1
            self.pieces[3, 0] = -1  # white piece 2
            self.pieces[3, 3] = -1  # white piece 3
        elif (startingVariation == "B"):
            #Position B
            self.pieces[0, 0] = 3  # black piece 1
            self.pieces[0, 2] = 3  # black piece 2
            self.pieces[1, 3] = 3  # black piece 3

            self.pieces[2, 0] = -1  # white piece 1
            self.pieces[3, 1] = -1  # white piece 2
            self.pieces[3, 3] = -1  # white piece 3
        elif (startingVariation == "C"):
            #Position C
            self.pieces[0, 1] = 3  # black piece 1
            self.pieces[0, 2] = 3  # black piece 2
            self.pieces[1, 3] = 4  # black piece 3

            self.pieces[2, 0] = -2  # white piece 1
            self.pieces[3, 1] = -1  # white piece 2
            self.pieces[3, 2] = -1  # white piece 3

    def __getstate__(self):
        return self.height, self.width, self.win_length, np.asarray(self.pieces)

    def __setstate__(self, state):
        self.height, self.width, self.win_length, pieces = state
        self.pieces = np.asarray(pieces)

    def move(self, int row, int col):

        cdef int movingPiece = self.pieces[row, col]
        cdef int movingDirection = abs(movingPiece)
        cdef int new_row = row, new_col = col
        cdef bint piecePushed = False
        cdef int pushedPiece
        cdef int pushedDirection

        # check the direction of the moving piece
        # 1 is up, 2 is right, 3 is down, 4 is left

        if movingDirection == 1:
            new_row = new_row - 1
        elif movingDirection == 2:
            new_col = new_col + 1
        elif movingPiece == 3:
            new_row = new_row + 1
        elif movingPiece == 4:
            new_col = new_col - 1

        # check to see if the piece is moving off the board (this should never happen)
        if new_row < 0 or new_row > 3 or new_col < 0 or new_col > 3:
            return

        pushedPiece = self.pieces[new_row, new_col]
        pushedDirection = abs(pushedPiece)

        # check to see if the piece is moving into an empty space or pushing another piece        
        if pushedPiece == 0:
            self.pieces[new_row, new_col] = movingPiece #set the new location to the moving piece since its empty
            self.pieces[row,col] = 0    # clear out the old space to zero, nothing is there anymore
            return
        else:
            # check to see if the piece is facing the pushing piece, because then it cannot push
            if movingDirection == 1 and pushedDirection == 3:
                return
            elif movingDirection == 2 and pushedDirection == 4:
                return
            elif movingPiece == 3 and pushedDirection == 1:
                return
            elif movingPiece == 4 and pushedDirection == 2:
                return
            else:
                #push it
                piecePushed = self.push(new_row, new_col, movingDirection)
                if piecePushed == True:
                    self.pieces[new_row, new_col] = movingPiece #set the new location to the moving piece since its empty
                    self.pieces[row,col] = 0    # clear out the old space to zero, nothing is there anymore
                    return
                

    def push(self, int row, int col, int direction):
        cdef int piece = self.pieces[row, col], pushedPiece, pushedDirection
        cdef int new_row = row, new_col = col
        cdef bint piecePushed = False
        
         # check the direction of the moving piece
        # 1 is up, 2 is right, 3 is down, 4 is left

        if direction == 1:
            new_row = new_row - 1
        elif direction == 2:
            new_col = new_col + 1
        elif direction == 3:
            new_row = new_row + 1
        elif direction == 4:
            new_col = new_col - 1

         # check to see if the piece is getting pushed off the board
        if new_row < 0 or new_row > 3 or new_col < 0 or new_col > 3:
            self.pieces[row,col] = 0   # clear out the old space to zero, nothing is there anymore because it got pushed "off"
            return True                # return true because the push was successful

        pushedPiece = self.pieces[new_row, new_col]
        pushedDirection = abs(pushedPiece)

        # check to see if the piece is moving into an empty space or pushing another piece        
        if pushedPiece == 0:
            self.pieces[new_row, new_col] = piece #set the new location to the moving piece since its empty
            self.pieces[row,col] = 0    # clear out the old space to zero, nothing is there anymore
            return True
        else:
            # check to see if the piece is facing the pushing piece, because then it cannot push
            if direction == 1 and pushedDirection == 3:
                return False    # return false because the piece cannot be pushed
            elif direction == 2 and pushedDirection == 4:
                return False    # return false because the piece cannot be pushed
            elif direction == 3 and pushedDirection == 1:
                return False    # return false because the piece cannot be pushed
            elif direction == 4 and pushedDirection == 2:
                return False    # return false because the piece cannot be pushed
            else:
                #push it, calling recursively
                 piecePushed = self.push(new_row, new_col, direction)
                 if piecePushed == True:
                    self.pieces[new_row, new_col] = piece #set the new location to the moving piece since its empty
                    self.pieces[row,col] = 0    # clear out the old space to zero, nothing is there anymore
                    return True
            return False

    def canRotate(self, int start, int end):
        return abs(start - end) % 4 == 1 or abs(start - end) % 4 == 4 - 1

    def rotate(self, int row, int col, int rotation):
        cdef int piece = self.pieces[row, col]
        
        if piece == 0:
            return
        elif self.canRotate(piece, rotation) == False:
            return
        else:
            if piece < 0:
                self.pieces[row, col] = rotation * -1
            else:
                self.pieces[row, col] = rotation

    def makeMove(self, int action, int player):

        #print("DEBUG: makeMove being called ", action, player)
        cdef Py_ssize_t row, col, r, c
        cdef int rotation

        if action <= 127 and action >= 0:  # If action is between 0 and 127, try to make a move

            if action > 63:  # If action is greater than 63, it is a rotation
                row = (action - 64) // 16  # Integer division to get the row index
                col = (action - 64) % 16 // 4  # Modulus to get the column index
                if self.pieces[row, col] == 0:  # there is no piece there, this is invalid somehow
                    raise ValueError("Can't play action %s on board %s" % (action, self))
                else:
                    rotation = (action % 4) + 1 
                    self.rotate(row, col, rotation)
            else:
                row = action // 16  # Integer division to get the row index
                col = action % 16 // 4  # Modulus to get the column index
                if self.pieces[row, col] == 0:  # there is no piece there, this is invalid somehow
                    raise ValueError("Can't play action %s on board %s" % (action, self))
                else:
                    self.move(row, col)
        else:   # tried to make a move that is bigger than 127 somehow
            raise ValueError("Can't play action %s on board %s" % (action, self))

    def checkValidMove(self, int row, int col, int pushingDirection):
        cdef int movingDirection = abs(self.pieces[row, col]) if pushingDirection == 0 else pushingDirection
        cdef int destination
        cdef int new_row = row, new_col = col

        if movingDirection == 1:
            new_row = new_row - 1
        elif movingDirection == 2:
            new_col = new_col + 1
        elif movingDirection == 3:
            new_row = new_row + 1
        elif movingDirection == 4:
            new_col = new_col - 1

        if new_row < 0 or new_row > 3 or new_col < 0 or new_col > 3:
            if pushingDirection == 0:
                return False
            else:
                return True
        else:
            destination = abs(self.pieces[new_row, new_col])
            if destination == 0:
                return True
            else:
                # check to see if the piece is facing the pushing piece, because then it cannot push
                if movingDirection == 1 and destination == 3:
                    return False    # return false because the piece cannot be pushed
                elif movingDirection == 2 and destination == 4:
                    return False    # return false because the piece cannot be pushed
                elif movingDirection == 3 and destination == 1:
                    return False    # return false because the piece cannot be pushed
                elif movingDirection == 4 and destination == 2:
                    return False    # return false because the piece cannot be pushed
                else:
                    #push it
                    return self.checkValidMove(new_row, new_col, movingDirection)
        


    def get_valid_moves(self, int player):
        cdef Py_ssize_t r, c
        cdef int[:] valid = np.zeros(128, dtype=np.intc)

        #print("DEBUG: get_valid_moves being called ", self.pieces)
        
        for r in range(4): 
            for c in range(4):
                if (player == -1 and self.pieces[r,c] < 0) or (player == 1 and self.pieces[r,c] > 0):
                    if abs(self.pieces[r,c]) == 1:
                        # check if the piece can move up
                        if r > 0 and self.checkValidMove(r,c,0) == True:
                            valid[r*16 + c*4] = 1
                        # piece can rotate to the east
                        valid[r*16 + c*4 + 1 + 64] = 1
                        # piece can rotate to the west
                        valid[r*16 + c*4 + 3 + 64] = 1
                    elif abs(self.pieces[r,c]) == 2:
                        # check if the piece can move right
                        if c < 3 and self.checkValidMove(r,c,0) == True:
                            valid[r*16 + c*4 + 1] = 1
                        # piece can rotate to the north
                        valid[r*16 + c*4 + 0 + 64] = 1
                        # piece can rotate to the south
                        valid[r*16 + c*4 + 2 + 64] = 1
                    elif abs(self.pieces[r,c]) == 3:
                        # check if the piece can move down
                        if r < 3 and self.checkValidMove(r,c,0) == True:
                            valid[r*16 + c*4 + 2] = 1
                        # piece can rotate to the east
                        valid[r*16 + c*4 + 1 + 64] = 1
                        # piece can rotate to the west
                        valid[r*16 + c*4 + 3 + 64] = 1
                    elif abs(self.pieces[r,c]) == 4:
                        # check if the piece can move left
                        if c > 0 and self.checkValidMove(r,c,0) == True:
                            valid[r*16 + c*4  + 3] = 1
                        # piece can rotate to the north
                        valid[r*16 + c*4 + 0 + 64] = 1
                        # piece can rotate to the south
                        valid[r*16 + c*4 + 2 + 64] = 1
        return valid

    def get_win_state(self):
        cdef int player
        cdef int total
        cdef int good
        cdef Py_ssize_t r, c, x

        for player in [1, -1]:
            # check the number of pieces this player has
            total = 0
            for r in range(4): 
                for c in range(4):
                    if self.pieces[r, c] != 0: 
                        if player == 1 and self.pieces[r, c] > 0:
                            total += 1
                        elif player == -1 and self.pieces[r, c] < 0:
                            total += 1
            if total == 2:
                return (True, -player)  # Game is ended and the _other_ player wins

        # Game is not ended yet.
        return (False, 0)

    def __str__(self):
        return str(np.asarray(self.pieces))
