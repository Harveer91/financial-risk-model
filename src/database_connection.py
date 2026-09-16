import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

with engine.connect() as connection:
    result = connection.execute(
        text("SELECT COUNT(*) FROM clean_loans_model")
    )
    count = result.scalar()

print("Connected successfully.")
print(f"Rows in clean_loans_model: {count:,}")