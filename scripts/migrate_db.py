#!/usr/bin/env python3
"""
Database Migration Script for Sherman QC System

Supports:
- SQLite to PostgreSQL migration
- Schema updates and versioning
- Data export/import

Usage:
    python scripts/migrate_db.py --help
    python scripts/migrate_db.py status
    python scripts/migrate_db.py migrate-to-postgres --sqlite-path data/qc_jobs.db --postgres-url postgresql://user:pass@host/db
    python scripts/migrate_db.py export --format json --output backup.json
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


def get_sqlite_connection(db_path: str):
    """Get SQLite connection"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_postgres_connection(database_url: str):
    """Get PostgreSQL connection"""
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        return psycopg2.connect(database_url)
    except ImportError:
        print("ERROR: psycopg2 not installed. Install with: pip install psycopg2-binary")
        sys.exit(1)


def check_status(args):
    """Check database status"""
    from database import DATABASE_PATH, DATABASE_URL

    print("=== Sherman QC Database Status ===\n")

    # Check SQLite
    sqlite_path = Path(DATABASE_PATH)
    if sqlite_path.exists():
        conn = get_sqlite_connection(str(sqlite_path))
        cursor = conn.execute("SELECT COUNT(*) FROM jobs")
        count = cursor.fetchone()[0]
        conn.close()
        print(f"SQLite Database: {sqlite_path}")
        print(f"  - Jobs: {count}")
        print(f"  - Size: {sqlite_path.stat().st_size / 1024:.1f} KB")
    else:
        print(f"SQLite Database: Not found at {sqlite_path}")

    print()

    # Check PostgreSQL
    if DATABASE_URL and DATABASE_URL.startswith("postgresql"):
        try:
            conn = get_postgres_connection(DATABASE_URL)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM jobs")
            count = cursor.fetchone()[0]
            conn.close()
            print(f"PostgreSQL Database: Connected")
            print(f"  - Jobs: {count}")
            print(f"  - URL: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'localhost'}")
        except Exception as e:
            print(f"PostgreSQL Database: Connection failed")
            print(f"  - Error: {e}")
    else:
        print("PostgreSQL Database: Not configured")
        print("  - Set DATABASE_URL environment variable to enable")


def migrate_to_postgres(args):
    """Migrate data from SQLite to PostgreSQL"""
    sqlite_path = args.sqlite_path or str(Path(__file__).parent.parent / "data" / "qc_jobs.db")
    postgres_url = args.postgres_url

    if not postgres_url:
        print("ERROR: PostgreSQL URL required. Use --postgres-url")
        sys.exit(1)

    if not Path(sqlite_path).exists():
        print(f"ERROR: SQLite database not found at {sqlite_path}")
        sys.exit(1)

    print(f"=== Migrating SQLite to PostgreSQL ===\n")
    print(f"Source: {sqlite_path}")
    print(f"Target: {postgres_url.split('@')[1] if '@' in postgres_url else postgres_url}")
    print()

    # Read from SQLite
    sqlite_conn = get_sqlite_connection(sqlite_path)
    cursor = sqlite_conn.execute("SELECT * FROM jobs")
    rows = cursor.fetchall()
    columns = [description[0] for description in cursor.description]
    sqlite_conn.close()

    print(f"Found {len(rows)} jobs to migrate")

    if len(rows) == 0:
        print("No data to migrate.")
        return

    # Connect to PostgreSQL
    pg_conn = get_postgres_connection(postgres_url)
    pg_cursor = pg_conn.cursor()

    # Initialize PostgreSQL schema
    from database import PostgreSQLDatabaseManager
    pg_db = PostgreSQLDatabaseManager(postgres_url)

    # Migrate data
    migrated = 0
    skipped = 0
    errors = 0

    for row in rows:
        row_dict = dict(zip(columns, row))
        job_id = row_dict['job_id']

        try:
            # Check if job already exists
            pg_cursor.execute("SELECT job_id FROM jobs WHERE job_id = %s", (job_id,))
            if pg_cursor.fetchone():
                if args.skip_existing:
                    skipped += 1
                    continue
                else:
                    # Update existing
                    pg_cursor.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))

            # Parse result_json if it's a string
            result_json = row_dict.get('result_json')
            if result_json and isinstance(result_json, str):
                try:
                    result_json = json.loads(result_json)
                except json.JSONDecodeError:
                    result_json = None

            # Insert into PostgreSQL
            pg_cursor.execute("""
                INSERT INTO jobs (
                    job_id, status, progress, stage, message, error,
                    part_id, part_name, material, tolerance,
                    reference_path, scan_path, drawing_path, report_path, pdf_path,
                    result_json, created_at, updated_at, completed_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
            """, (
                row_dict['job_id'],
                row_dict['status'],
                row_dict['progress'],
                row_dict['stage'],
                row_dict['message'],
                row_dict['error'],
                row_dict['part_id'],
                row_dict['part_name'],
                row_dict['material'],
                row_dict['tolerance'],
                row_dict['reference_path'],
                row_dict['scan_path'],
                row_dict['drawing_path'],
                row_dict['report_path'],
                row_dict['pdf_path'],
                json.dumps(result_json) if result_json else None,
                row_dict['created_at'],
                row_dict['updated_at'],
                row_dict['completed_at']
            ))

            migrated += 1

        except Exception as e:
            print(f"  ERROR migrating job {job_id}: {e}")
            errors += 1

    pg_conn.commit()
    pg_conn.close()

    print()
    print(f"Migration complete:")
    print(f"  - Migrated: {migrated}")
    print(f"  - Skipped:  {skipped}")
    print(f"  - Errors:   {errors}")


def export_data(args):
    """Export database to JSON or CSV"""
    from database import get_db

    db = get_db()
    jobs = db.list_jobs(limit=10000)

    output_format = args.format or 'json'
    output_path = args.output or f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"

    print(f"Exporting {len(jobs)} jobs to {output_path}")

    if output_format == 'json':
        data = [job.to_dict() for job in jobs]
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    elif output_format == 'csv':
        import csv
        if jobs:
            fieldnames = list(jobs[0].to_dict().keys())
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for job in jobs:
                    row = job.to_dict()
                    # Flatten result dict to JSON string
                    if 'result' in row and row['result']:
                        row['result'] = json.dumps(row['result'])
                    writer.writerow(row)
    else:
        print(f"ERROR: Unknown format '{output_format}'. Use 'json' or 'csv'.")
        sys.exit(1)

    print(f"Export complete: {output_path}")


def import_data(args):
    """Import data from JSON backup"""
    from database import get_db

    input_path = args.input
    if not Path(input_path).exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Importing from {input_path}")

    with open(input_path, 'r') as f:
        data = json.load(f)

    db = get_db()
    imported = 0
    skipped = 0
    errors = 0

    for job_data in data:
        job_id = job_data.get('job_id')
        try:
            # Check if exists
            existing = db.get_job(job_id)
            if existing:
                if args.skip_existing:
                    skipped += 1
                    continue
                else:
                    db.delete_job(job_id)

            # Create job
            db.create_job(
                job_id=job_id,
                part_id=job_data.get('part_id', ''),
                part_name=job_data.get('part_name', ''),
                material=job_data.get('material', ''),
                tolerance=job_data.get('tolerance', 0.1),
                reference_path=job_data.get('reference_path', ''),
                scan_path=job_data.get('scan_path', ''),
                drawing_path=job_data.get('drawing_path')
            )

            # Update status if not pending
            if job_data.get('status') == 'completed':
                db.update_job_result(
                    job_id=job_id,
                    result=job_data.get('result', {}),
                    report_path=job_data.get('report_path', ''),
                    pdf_path=job_data.get('pdf_path', '')
                )
            elif job_data.get('status') == 'failed':
                db.update_job_error(job_id, job_data.get('error', 'Unknown error'))

            imported += 1

        except Exception as e:
            print(f"  ERROR importing job {job_id}: {e}")
            errors += 1

    print()
    print(f"Import complete:")
    print(f"  - Imported: {imported}")
    print(f"  - Skipped:  {skipped}")
    print(f"  - Errors:   {errors}")


def main():
    parser = argparse.ArgumentParser(
        description="Sherman QC Database Migration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s status
  %(prog)s migrate-to-postgres --postgres-url postgresql://user:pass@localhost/sherman_qc
  %(prog)s export --format json --output backup.json
  %(prog)s import --input backup.json
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Status command
    status_parser = subparsers.add_parser('status', help='Check database status')

    # Migrate command
    migrate_parser = subparsers.add_parser('migrate-to-postgres', help='Migrate SQLite to PostgreSQL')
    migrate_parser.add_argument('--sqlite-path', help='Path to SQLite database')
    migrate_parser.add_argument('--postgres-url', required=True, help='PostgreSQL connection URL')
    migrate_parser.add_argument('--skip-existing', action='store_true', help='Skip existing jobs instead of overwriting')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export database to file')
    export_parser.add_argument('--format', choices=['json', 'csv'], default='json', help='Output format')
    export_parser.add_argument('--output', '-o', help='Output file path')

    # Import command
    import_parser = subparsers.add_parser('import', help='Import data from backup')
    import_parser.add_argument('--input', '-i', required=True, help='Input file path')
    import_parser.add_argument('--skip-existing', action='store_true', help='Skip existing jobs')

    args = parser.parse_args()

    if args.command == 'status':
        check_status(args)
    elif args.command == 'migrate-to-postgres':
        migrate_to_postgres(args)
    elif args.command == 'export':
        export_data(args)
    elif args.command == 'import':
        import_data(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
