import sqlite3
import json
import pandas as pd

_conn = None


def get_conn():
    global _conn
    if _conn is None:
        _conn = sqlite3.connect('omr_data.db', check_same_thread=False)
    return _conn


def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS answer_keys (
            subject_code TEXT,
            exam_set TEXT,
            exam_title TEXT,
            exam_format TEXT,
            key_data TEXT,
            PRIMARY KEY (subject_code, exam_set)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            seat_number TEXT PRIMARY KEY,
            first_name TEXT,
            last_name TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS exam_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            seat_number TEXT,
            student_name TEXT,
            subject_code TEXT,
            exam_title TEXT,
            exam_set TEXT,
            score REAL,
            wrong_questions TEXT,
            empty_questions TEXT,
            double_questions TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Migrate existing DB that doesn't have the new columns yet
    for col in ['empty_questions', 'double_questions']:
        try:
            c.execute(f"ALTER TABLE exam_history ADD COLUMN {col} TEXT DEFAULT '-'")
        except Exception:
            pass  # Column already exists
    conn.commit()


# --- Answer Keys ---

def get_answer_key(subject_code, exam_set):
    """Returns (exam_format, key_data, exam_title) or None."""
    c = get_conn().cursor()
    c.execute(
        "SELECT exam_format, key_data, exam_title FROM answer_keys WHERE subject_code=? AND exam_set=?",
        (subject_code, exam_set)
    )
    return c.fetchone()


def get_all_answer_keys():
    """Returns DataFrame with all answer keys including key_data column."""
    return pd.read_sql_query(
        "SELECT subject_code, exam_set, exam_title, exam_format, key_data FROM answer_keys",
        get_conn()
    )


def save_answer_key(subject_code, exam_set, exam_title, exam_format, key_data):
    conn = get_conn()
    conn.cursor().execute(
        'REPLACE INTO answer_keys VALUES (?, ?, ?, ?, ?)',
        (subject_code, exam_set, exam_title, exam_format, json.dumps(key_data))
    )
    conn.commit()


def delete_answer_key(subject_code, exam_set):
    conn = get_conn()
    conn.cursor().execute(
        "DELETE FROM answer_keys WHERE subject_code=? AND exam_set=?",
        (subject_code, exam_set)
    )
    conn.commit()


# --- Students ---

def get_students():
    return pd.read_sql_query(
        "SELECT seat_number, first_name, last_name FROM students",
        get_conn()
    )


def get_student(seat_number):
    c = get_conn().cursor()
    c.execute("SELECT first_name, last_name FROM students WHERE seat_number=?", (seat_number,))
    return c.fetchone()


def add_student(seat_number, first_name, last_name):
    conn = get_conn()
    conn.cursor().execute("INSERT INTO students VALUES (?,?,?)", (seat_number, first_name, last_name))
    conn.commit()


def update_student(seat_number, first_name, last_name):
    conn = get_conn()
    conn.cursor().execute(
        "UPDATE students SET first_name=?, last_name=? WHERE seat_number=?",
        (first_name, last_name, seat_number)
    )
    conn.commit()


def delete_students(seat_numbers):
    conn = get_conn()
    placeholders = ','.join(['?'] * len(seat_numbers))
    conn.cursor().execute(
        f"DELETE FROM students WHERE seat_number IN ({placeholders})",
        seat_numbers
    )
    conn.commit()


def import_students(df):
    df.to_sql('students', get_conn(), if_exists='append', index=False)


# --- Exam History ---

def get_exam_history():
    return pd.read_sql_query('''
        SELECT timestamp as เวลาตรวจ, filename as ชื่อไฟล์, seat_number as เลขที่นั่งสอบ,
               student_name as ชื่อนักเรียน, exam_title as วิชาที่สอบ, exam_set as ชุด,
               score as คะแนนรวม, wrong_questions as ข้อที่ผิด,
               empty_questions as ข้อที่ไม่ตอบ, double_questions as ข้อที่ตอบซ้ำ
        FROM exam_history ORDER BY id DESC
    ''', get_conn())


def save_exam_result(filename, seat_number, student_name, subject_code, exam_title, exam_set,
                     score, wrong_str, empty_str='-', double_str='-'):
    conn = get_conn()
    conn.cursor().execute('''
        INSERT INTO exam_history (filename, seat_number, student_name, subject_code, exam_title, exam_set,
                                  score, wrong_questions, empty_questions, double_questions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (filename, seat_number, student_name, subject_code, exam_title, exam_set,
          score, wrong_str, empty_str, double_str))
    conn.commit()


def clear_exam_history():
    conn = get_conn()
    conn.cursor().execute("DELETE FROM exam_history")
    conn.commit()


def delete_history_by_filename(filename):
    conn = get_conn()
    conn.cursor().execute("DELETE FROM exam_history WHERE filename=?", (filename,))
    conn.commit()
