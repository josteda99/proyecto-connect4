const board = document.querySelector(".board");
const cells = document.querySelectorAll(".cell");
const ROWS = 6;
const COLS = 7;
let currentPlayer = "red";

cells.forEach((cell) => {
  cell.addEventListener("click", handleClick);
});

function handleClick(event) {
  const cell = event.target;
  const colIndex = Array.from(cell.parentNode.children).indexOf(cell);

  if (isColumnFull(colIndex)) return;

  const rowIndex = getRowIndex(colIndex);
  const cellToFill = cells[rowIndex * COLS + colIndex];

  cellToFill.classList.add(currentPlayer);

  if (checkWin(rowIndex, colIndex)) {
    announceWinner();
    return;
  }

  currentPlayer = currentPlayer === "red" ? "yellow" : "red";
}

function isColumnFull(colIndex) {
  const cellsInColumn = Array.from(cells).filter((cell, index) => index % COLS === colIndex);
  return cellsInColumn.every((cell) => cell.classList.contains("red") || cell.classList.contains("yellow"));
}

function getRowIndex(colIndex) {
  for (let i = ROWS - 1; i >= 0; i--) {
    const cell = cells[i * COLS + colIndex];
    if (!cell.classList.contains("red") && !cell.classList.contains("yellow")) return i;
  }
  return -1;
}

function checkWin(rowIndex, colIndex) {
  const directions = [
    { x: 0, y: 1 },
    { x: 1, y: 0 },
    { x: 1, y: 1 },
    { x: 1, y: -1 },
  ];

  for (const direction of directions) {
    let count = 1;
    const dx = direction.x;
    const dy = direction.y;

    let x = colIndex - dx;
    let y = rowIndex - dy;
    while (x >= 0 && y >= 0 && x < COLS && y < ROWS && cells[y * COLS + x].classList.contains(currentPlayer)) {
      count++;
      x -= dx;
      y -= dy;
    }

    x = colIndex + dx;
    y = rowIndex + dy;
    while (x >= 0 && y >= 0 && x < COLS && y < ROWS && cells[y * COLS + x].classList.contains(currentPlayer)) {
      count++;
      x += dx;
      y += dy;
    }

    if (count >= 4) return true;
  }

  return false;
}

function announceWinner() {
  alert("¡Ganó el jugador " + currentPlayer + "!");
  resetGame();
}

function resetGame() {
  cells.forEach((cell) => {
    cell.classList.remove("red", "yellow");
  });
  currentPlayer = "red";
}
