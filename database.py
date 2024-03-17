import sqlite3

def create_connection(db_file):
    """Create a database connection to a SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)
    return conn

def create_table(conn):
    """Create the videos table if it doesn't already exist."""
    sql_create_videos_table = """
    CREATE TABLE IF NOT EXISTS videos (
        video_base_path TEXT PRIMARY KEY,
        txt_file_path TEXT NOT NULL,
        audio_path TEXT NOT NULL,
        images_folder_path TEXT NOT NULL
    );
    """
    try:
        c = conn.cursor()
        c.execute(sql_create_videos_table)
    except sqlite3.Error as e:
        print(e)

        
def insert_video_data(conn, video_data):
    """
    Insert a new video record into the videos table.
    :param conn: Database connection object.
    :param video_data: Tuple containing video base path, txt file path, audio path, and images folder path.
    """
    sql = ''' INSERT INTO videos(video_base_path, txt_file_path, audio_path, images_folder_path)
              VALUES(?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, video_data)
    conn.commit()
    return cur.lastrowid
