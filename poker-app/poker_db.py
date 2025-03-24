import sqlite3


def initialize_db() -> None:
    conn = sqlite3.connect("poker_games.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS game_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        stages TEXT,
        score REAL,
        justifications TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def get_past_game_examples(limit: int = 5, min_score: float = 0.8) -> str:
    conn = sqlite3.connect("poker_games.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT stages, score, justifications FROM game_history WHERE score > ? LIMIT ?",
        (min_score, limit),
    )
    past_games = cursor.fetchall()
    conn.close()

    xml = ""
    for game in past_games:
        xml += f"""
        <game>
            <stages>{game[0]}</stages>
            <performance_score>{game[1]}</performance_score>
            <justifications>{game[2]}</justifications>
        </game>
    """
    return f"<games>{xml}</games>"


def save_game_data(stages: str, score: float, justifications: str) -> None:
    conn = sqlite3.connect("poker_games.db")
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO game_history (stages, score, justifications) VALUES (?, ?, ?)",
        (stages, score, justifications),
    )

    conn.commit()
    conn.close()
