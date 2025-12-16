import os
import mysql.connector
import datetime


class MySQLConnector:
    def __init__(self):
        self.host = os.getenv("DB_HOST")
        self.port = 3306 # Default to 3306 if not set
        self.username = os.getenv("DB_USERNAME")
        self.password = os.getenv("DB_PASSWORD")
        self.database = os.getenv("DB_DATABASE")
        self.connection = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.username,
                password=self.password,
                database=self.database
            )
            print("Connected to the database successfully!")
        except mysql.connector.Error as err:
            print(f"Error connecting to the database: {err}")

    def disconnect(self):
        if self.connection:
            self.connection.close()
            print("Disconnected from the database.")


    def execute_query(self, sql_query, params=None):
        """
        Executes a SQL query and returns the results.
        Use params for parameterized queries to prevent SQL injection.
        """

        if not self.connection or not self.connection.is_connected():
            print("Not connected to the database. Please connect first.")
            return None

        cursor = None # Initialize cursor to None
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(sql_query, params)
            if sql_query.strip().upper().startswith(("SELECT", "SHOW", "DESCRIBE")):
                result = cursor.fetchall()
            else:
                self.connection.commit() # Commit changes for INSERT, UPDATE, DELETE
                result = cursor.rowcount # Return number of affected rows
            return result

        except mysql.connector.Error as err:
            print(f"Error executing query: {err}")
            return None

        finally:
            if cursor:
                cursor.close() 

    def create_book(self, book_data):
        try:
            cursor = self.connection.cursor()
            insert_query = """
                INSERT INTO cliperest_book
                (user_id, name, slug, rendered, version, category_id, modified, addEnd, coverImage, sharing,
                coverColor, dollarsGiven, privacy, type, created, coverHexColor, numLikers, description,
                tags, thumbnailImage, numClips, numViews, userLanguage, embed_code, thumbnailImageSmall,
                humanModified, coverV3, typeFilters)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, tuple(book_data.values()))
            self.connection.commit()
            book_id = cursor.lastrowid  # Obtiene el ID del registro insertado
            print(f"Book record inserted successfully with ID: {book_id}")
            return book_id  # Devuelve el ID del libro insertado
        except mysql.connector.Error as err:
            print(f"Error creating book: {err}")
            return None  # Devuelve None si hay un error
    
    # Interactive SQL Querier
    def SQL_queries(self):
        if self.connection and self.connection.is_connected():
            print("\n--- Interactive Query Executor ---")

            while True:
                query = input("Enter SQL query (or press ENTER to exit): ").strip()
                if not query:
                    break
                res = self.execute_query(query)
                print("-" * 10)
                if res is not None:
                    if isinstance(res, list):
                        if res:
                            for row in res:
                                print(row)
                        else:
                            print("Query executed successfully, no results to display.")
                    else:
                        print(f"Query executed successfully. Affected rows: {res}")
                else:
                    print("Query failed or returned no data.")
                print("-" * 10)







# class MySQLConnector:
#     def __init__(self):
#         self.host = os.getenv("DB_HOST")
#         self.port = 3306 # Default to 3306 if not set
#         self.username = os.getenv("DB_USERNAME")
#         self.password = os.getenv("DB_PASSWORD")
#         self.database = os.getenv("DB_DATABASE")
#         self.connection = None

#     def connect(self):
#         try:
#             self.connection = mysql.connector.connect(
#                 host=self.host,
#                 port=self.port,
#                 user=self.username,
#                 password=self.password,
#                 database=self.database
#             )
#             print("Connected to the database successfully!")
#         except mysql.connector.Error as err:
#             print(f"Error connecting to the database: {err}")

#     def disconnect(self):
#         if self.connection:
#             self.connection.close()
#             print("Disconnected from the database.")

#     def test_connection(self):
#         if self.connection:
#             print("Connection test successful.")
#         else:
#             print("No active connection. Please connect first.")

#     def test(self):
#         self.connect()
#         self.test_connection()
#         self.disconnect()

#     def execute_query(self, sql_query, params=None):
#         """
#         Executes a SQL query and returns the results.
#         Use params for parameterized queries to prevent SQL injection.
#         """

#         if not self.connection or not self.connection.is_connected():
#             print("Not connected to the database. Please connect first.")
#             return None

#         cursor = None # Initialize cursor to None
#         try:
#             cursor = self.connection.cursor(dictionary=True)
#             cursor.execute(sql_query, params)
#             if sql_query.strip().upper().startswith(("SELECT", "SHOW", "DESCRIBE")):
#                 result = cursor.fetchall()
#             else:
#                 self.connection.commit() # Commit changes for INSERT, UPDATE, DELETE
#                 result = cursor.rowcount # Return number of affected rows
#             return result

#         except mysql.connector.Error as err:
#             print(f"Error executing query: {err}")
#             return None

#         finally:
#             if cursor:
#                 cursor.close() 

#     def create_book(self, book_data):
#         try:
#             cursor = self.connection.cursor()
#             insert_query = """
#                 INSERT INTO cliperest_book
#                 (user_id, name, slug, rendered, version, category_id, modified, addEnd, coverImage, sharing,
#                 coverColor, dollarsGiven, privacy, type, created, coverHexColor, numLikers, description,
#                 tags, thumbnailImage, numClips, numViews, userLanguage, embed_code, thumbnailImageSmall,
#                 humanModified, coverV3, typeFilters)
#                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#             """
#             cursor.execute(insert_query, tuple(book_data.values()))
#             self.connection.commit()
#             book_id = cursor.lastrowid  # Obtiene el ID del registro insertado
#             print(f"Book record inserted successfully with ID: {book_id}")
#             return book_id  # Devuelve el ID del libro insertado
#         except mysql.connector.Error as err:
#             print(f"Error creating book: {err}")
#             return None  # Devuelve None si hay un error
    
#     def create_clipping(self, clipping_data):
#         try:
#             cursor = self.connection.cursor()
#             insert_query = """
#             INSERT INTO cliperest_clipping
#             (book_id, caption, text, thumbnail, useThumbnail, type, url, created, num, migratedS3, modified)
#             VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
#             """

#             cursor.execute(insert_query, tuple(clipping_data.values()))
#             self.connection.commit()
#             print("Clipping record inserted successfully!")
#             clip_id = cursor.lastrowid  # Obtiene el ID del registro insertado   
#             return clip_id  # Devuelve el ID del libro insertado            
#         except mysql.connector.Error as err:
#             print(f"Error creating clipping: {err}")
#             return None
    
#     def create_clippings_batch(self, clippings_data_list):
#         if not clippings_data_list:
#             print("No clippings data provided for batch insert.")
#             return None

#         try:
#             cursor = self.connection.cursor()
#             insert_query = """
#             INSERT INTO cliperest_clipping
#             (book_id, caption, text, thumbnail, useThumbnail, type, url, created, num, migratedS3, modified)
#             VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
#             """
#             # Prepare the list of tuples for executemany
#             values_to_insert = [tuple(clipping_data.values()) for clipping_data in clippings_data_list]
            
#             cursor.executemany(insert_query, values_to_insert)
#             self.connection.commit()
#             print(f"Batch insert of {len(clippings_data_list)} clipping records successful!")
#             return cursor.rowcount # Return number of affected rows
#         except mysql.connector.Error as err:
#             print(f"Error creating clippings in batch: {err}")
#             return None
    
#     def update_book_num_clips(self, book_id, num_clips):
#         """
#         Updates the numClips field for a given book_id.
#         """
#         try:
#             cursor = self.connection.cursor()
#             update_query = "UPDATE cliperest_book SET numClips = %s WHERE id = %s"
#             cursor.execute(update_query, (num_clips, book_id))
#             self.connection.commit()
#             print(f"Updated numClips for book_id {book_id} to {num_clips}.")
#             return True
#         except mysql.connector.Error as err:
#             print(f"Error updating numClips for book_id {book_id}: {err}")
#             return False
#         finally:
#             if cursor:
#                 cursor.close()

#     def get_book_id_for_date(self, target_date_str, slug_prefix):
#         """
#         Checks if a book exists for a given date (YYYY-MM-DD) and a slug prefix, and returns its ID.
#         """
#         try:
#             cursor = self.connection.cursor(dictionary=True)
#             # Dynamically construct the slug pattern for today's date (DD-MM-YYYY)
#             today_date_for_slug = datetime.datetime.now().strftime("%d-%m-%Y")
#             slug_pattern = f"{slug_prefix}-{today_date_for_slug}"
            
#             # We are checking for a book where the 'created' datetime starts with the target date string
#             # AND the slug exactly matches the constructed slug_pattern.
#             select_query = "SELECT id FROM cliperest_book WHERE DATE(created) = %s AND slug = %s LIMIT 1"
#             cursor.execute(select_query, (target_date_str, slug_pattern))
#             result = cursor.fetchone()
#             if result:
#                 print(f"Found existing book for {target_date_str} with ID: {result['id']} and slug prefix '{slug_prefix}'")
#                 return result['id']
#             else:
#                 print(f"No existing book found for {target_date_str} with slug prefix '{slug_prefix}'.")
#                 return None
#         except mysql.connector.Error as err:
#             print(f"Error checking for existing book: {err}")
#             return None
#         finally:
#             if cursor:
#                 cursor.close()

#     def get_or_create_book_for_today(self, book_data_template, slug_prefix):
#         """
#         Gets the ID of an existing book for today based on slug prefix, or creates a new one if it doesn't exist.
#         """
#         today_date_str = datetime.datetime.now().strftime("%Y-%m-%d")
#         existing_book_id = self.get_book_id_for_date(today_date_str, slug_prefix)

#         if existing_book_id:
#             return existing_book_id
#         else:
#             print("Creating a new book for today.")
#             # Generate date strings for the new book (without hour and minute for relevant fields)
#             fecha_hoy = datetime.datetime.now()
#             fecha_str_day = fecha_hoy.strftime("%d/%m/%Y") # For name and description
#             fecha_str_americana_day = fecha_hoy.strftime("%Y-%m-%d") # For created and modified
#             fecha_str_slug_day = fecha_hoy.strftime("%d-%m-%Y") # For slug

#             # Update book_data_template with current date info (without hour/minute in relevant fields)
#             book_data_template["name"] = "Fotos de Deportes " + fecha_str_day
#             book_data_template["slug"] = slug_prefix + "-" + fecha_str_slug_day
#             book_data_template["modified"] = fecha_str_americana_day + " 00:00:00" # Set to beginning of day
#             book_data_template["created"] = fecha_str_americana_day + " 00:00:00" # Set to beginning of day
#             book_data_template["description"] = f"Fotos de Deportes - {fecha_str_day}"
#             book_data_template["humanModified"] = fecha_str_americana_day + " 00:00:00" # Set to beginning of day
#             book_data_template["numClips"] = 0 # Initialize with 0 clips

#             return self.create_book(book_data_template)

#     # Interactive SQL Querier
#     def SQL_queries(self):
#         if self.connection and self.connection.is_connected():
#             print("\n--- Interactive Query Executor ---")

#             while True:
#                 query = input("Enter SQL query (or press ENTER to exit): ").strip()
#                 if not query:
#                     break
#                 res = self.execute_query(query)
#                 print("-" * 10)
#                 if res is not None:
#                     if isinstance(res, list):
#                         if res:
#                             for row in res:
#                                 print(row)
#                         else:
#                             print("Query executed successfully, no results to display.")
#                     else:
#                         print(f"Query executed successfully. Affected rows: {res}")
#                 else:
#                     print("Query failed or returned no data.")
#                 print("-" * 10)
