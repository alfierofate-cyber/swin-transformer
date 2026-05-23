import random

class Minesweeper:
    def __init__(self, rows=9, cols=9, mines=10):
        self.rows, self.cols, self.mines = rows, cols, mines
        self.board = [[0]*cols for _ in range(rows)]
        self.revealed = [[False]*cols for _ in range(rows)]
        self.flagged = [[False]*cols for _ in range(rows)]
        self.game_over = False
        self.place_mines()
        self.calculate_numbers()

    def place_mines(self):
        mine_positions = random.sample([(r, c) for r in range(self.rows) for c in range(self.cols)], self.mines)
        for r, c in mine_positions:
            self.board[r][c] = -1

    def calculate_numbers(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] != -1:
                    count = 0
                    for dr in (-1,0,1):
                        for dc in (-1,0,1):
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.board[nr][nc] == -1:
                                count += 1
                    self.board[r][c] = count

    def reveal(self, r, c):
        if not (0 <= r < self.rows and 0 <= c < self.cols) or self.revealed[r][c] or self.flagged[r][c]:
            return
        self.revealed[r][c] = True
        if self.board[r][c] == -1:
            self.game_over = True
            return
        if self.board[r][c] == 0:
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    self.reveal(r+dr, c+dc)

    def flag(self, r, c):
        if not self.revealed[r][c]:
            self.flagged[r][c] = not self.flagged[r][c]

    def print_board(self):
        for r in range(self.rows):
            line = []
            for c in range(self.cols):
                if self.flagged[r][c]:
                    line.append('F')
                elif not self.revealed[r][c]:
                    line.append('.')
                elif self.board[r][c] == -1:
                    line.append('*')
                else:
                    line.append(str(self.board[r][c]))
            print(' '.join(line))

# 示例用法
game = Minesweeper()
game.reveal(0, 0)
game.print_board()